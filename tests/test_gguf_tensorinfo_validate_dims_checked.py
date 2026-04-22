#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoValidateDimsChecked (IQ-1138)."""

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


def test_source_contains_iq1138_validate_dims_checked() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoValidateDimsChecked(U32 n_dims,"
    assert sig in source
    body = source.split(sig, 1)[1].split("Bool GGUFTensorTryAddU64", 1)[0]

    assert "if (!dims || !out_n_elements)" in body
    assert "if (!n_dims || n_dims > GGUF_TENSOR_MAX_DIMS)" in body
    assert "if (!GGUFTensorTypeKnown(ggml_type))" in body
    assert "if (!GGUFTensorTryMulU64(n_elements, dims[i], &next_n_elements))" in body
    assert "status = GGUFTensorTypeBlockElemsChecked(ggml_type, &block_elems);" in body
    assert "if (n_elements % block_elems)" in body
    assert "*out_n_elements = n_elements;" in body


def test_validate_dims_checked_error_contracts_and_no_partial_publish() -> None:
    out = [777]

    status = validate_dims_checked(2, None, 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out == [777]

    status = validate_dims_checked(0, [32], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    assert out == [777]

    status = validate_dims_checked(GGUF_TENSOR_MAX_DIMS + 1, [32] * (GGUF_TENSOR_MAX_DIMS + 1), 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    assert out == [777]

    status = validate_dims_checked(2, [32, 32], 999, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    assert out == [777]

    status = validate_dims_checked(2, [32, 0], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
    assert out == [777]

    status = validate_dims_checked(2, [1 << 63, 3], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    assert out == [777]

    status = validate_dims_checked(2, [33, 1], 2, out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
    assert out == [777]


def test_validate_dims_checked_success_fixed_and_randomized() -> None:
    out = [0]
    status = validate_dims_checked(3, [32, 16, 8], 2, out)
    assert status == GGUF_TENSOR_PARSE_OK
    assert out == [4096]

    status = validate_dims_checked(1, [13], 0, out)
    assert status == GGUF_TENSOR_PARSE_OK
    assert out == [13]

    rng = random.Random(1138)
    type_pool = sorted(KNOWN_TYPES)

    for _ in range(300):
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

        target = max(block[0], remaining * block[0])
        dims[-1] = target

        out = [123]
        status = validate_dims_checked(n_dims, dims, ggml_type, out)
        assert status == GGUF_TENSOR_PARSE_OK

        expected = 1
        for d in dims:
            expected *= d
        assert out == [expected]
        assert expected % block[0] == 0
