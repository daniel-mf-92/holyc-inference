#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoValidateDimsCheckedNoPartial (IQ-1153)."""

from __future__ import annotations

import random
from pathlib import Path

GGUF_TENSOR_PARSE_OK = 0
GGUF_TENSOR_PARSE_ERR_NULL_PTR = 1
GGUF_TENSOR_PARSE_ERR_BAD_DIMS = 5
GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE = 6
GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW = 7
GGUF_TENSOR_PARSE_ERR_BAD_TYPE = 8

GGUF_TENSOR_MAX_DIMS = 8
U64_MAX = (1 << 64) - 1

SCALAR_TYPES = {0, 1, 24, 25, 26, 27, 28, 30}
Q32_TYPES = {2, 3, 6, 7, 8, 9, 31, 32, 33}
Q256_TYPES = {
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    29,
    34,
    35,
}
KNOWN_TYPES = SCALAR_TYPES | Q32_TYPES | Q256_TYPES


def tensor_type_block_elems_checked(ggml_type: int, out_block: list[int] | None) -> int:
    if out_block is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    if ggml_type in SCALAR_TYPES:
        out_block[0] = 1
        return GGUF_TENSOR_PARSE_OK
    if ggml_type in Q32_TYPES:
        out_block[0] = 32
        return GGUF_TENSOR_PARSE_OK
    if ggml_type in Q256_TYPES:
        out_block[0] = 256
        return GGUF_TENSOR_PARSE_OK

    return GGUF_TENSOR_PARSE_ERR_BAD_TYPE


def u64_mul_checked(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a != 0 and b > U64_MAX // a:
        return None
    return a * b


def validate_dims_checked(
    n_dims: int,
    dims: list[int] | None,
    ggml_type: int,
    out_n_elements: list[int] | None,
) -> int:
    if dims is None or out_n_elements is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if n_dims == 0 or n_dims > GGUF_TENSOR_MAX_DIMS:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    if ggml_type not in KNOWN_TYPES:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    n_elements = 1
    for i in range(n_dims):
        d = dims[i]
        if d == 0:
            return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
        next_n_elements = u64_mul_checked(n_elements, d)
        if next_n_elements is None:
            return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
        n_elements = next_n_elements

    block = [0]
    status = tensor_type_block_elems_checked(ggml_type, block)
    if status != GGUF_TENSOR_PARSE_OK:
        return status
    if block[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    if n_elements % block[0] != 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE

    out_n_elements[0] = n_elements
    return GGUF_TENSOR_PARSE_OK


def validate_dims_checked_nopartial(
    n_dims: int,
    dims: list[int] | None,
    ggml_type: int,
    out_n_elements: list[int] | None,
) -> int:
    if dims is None or out_n_elements is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    snapshot_n_dims = n_dims
    snapshot_dims = dims
    snapshot_ggml_type = ggml_type

    staged = [0]
    status = validate_dims_checked(snapshot_n_dims, snapshot_dims, snapshot_ggml_type, staged)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    if snapshot_n_dims != n_dims or snapshot_dims is not dims or snapshot_ggml_type != ggml_type:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    out_n_elements[0] = staged[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1153_function_and_contract() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoValidateDimsCheckedNoPartial(U32 n_dims,"
    assert sig in source
    body = source.split(sig, 1)[1].split("Bool GGUFTensorTryAddU64", 1)[0]

    assert "snapshot_n_dims = n_dims;" in body
    assert "snapshot_dims_ptr = dims;" in body
    assert "snapshot_ggml_type = ggml_type;" in body
    assert "status = GGUFTensorInfoValidateDimsChecked(snapshot_n_dims," in body
    assert "if (snapshot_n_dims != n_dims ||" in body
    assert "snapshot_dims_ptr != dims ||" in body
    assert "snapshot_ggml_type != ggml_type)" in body
    assert "*out_n_elements = staged_n_elements;" in body


def test_nopartial_error_paths_preserve_output() -> None:
    out = [0xDEADBEEF]

    status = validate_dims_checked_nopartial(2, None, 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out == [0xDEADBEEF]

    status = validate_dims_checked_nopartial(0, [32], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    assert out == [0xDEADBEEF]

    status = validate_dims_checked_nopartial(2, [32, 0], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
    assert out == [0xDEADBEEF]

    status = validate_dims_checked_nopartial(2, [1 << 63, 3], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    assert out == [0xDEADBEEF]

    status = validate_dims_checked_nopartial(2, [32, 32], 999, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    assert out == [0xDEADBEEF]


def test_nopartial_success_vectors_and_randomized_parity() -> None:
    out = [0]

    status = validate_dims_checked_nopartial(3, [32, 16, 8], 2, out)
    assert status == GGUF_TENSOR_PARSE_OK
    assert out == [4096]

    status = validate_dims_checked_nopartial(1, [13], 0, out)
    assert status == GGUF_TENSOR_PARSE_OK
    assert out == [13]

    rng = random.Random(20260423_1153)
    type_pool = sorted(KNOWN_TYPES)

    for _ in range(400):
        ggml_type = rng.choice(type_pool)
        block = [0]
        assert tensor_type_block_elems_checked(ggml_type, block) == GGUF_TENSOR_PARSE_OK

        n_dims = rng.randint(1, GGUF_TENSOR_MAX_DIMS)
        dims = [1] * n_dims
        remaining = rng.randint(1, 1 << 20)

        for i in range(n_dims - 1):
            factor = rng.randint(1, 64)
            dims[i] = factor
            remaining = max(1, remaining // factor)

        dims[-1] = max(block[0], remaining * block[0])

        expect = [123456]
        status_expect = validate_dims_checked(n_dims, dims, ggml_type, expect)

        got = [789012]
        status = validate_dims_checked_nopartial(n_dims, dims, ggml_type, got)

        assert status == status_expect == GGUF_TENSOR_PARSE_OK
        assert got == expect


if __name__ == "__main__":
    test_source_contains_iq1153_function_and_contract()
    test_nopartial_error_paths_preserve_output()
    test_nopartial_success_vectors_and_randomized_parity()
    print("ok")
