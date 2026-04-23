#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoValidateDimsCheckedCommitOnlyPreflightOnlyParity (IQ-1197)."""

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


def validate_dims_checked_commit_only(
    n_dims: int,
    dims: list[int] | None,
    ggml_type: int,
    out_n_elements: list[int] | None,
    out_block_elems: list[int] | None,
) -> int:
    if dims is None or out_n_elements is None or out_block_elems is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if out_n_elements is out_block_elems:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    snapshot_n_dims = n_dims
    snapshot_dims = dims
    snapshot_ggml_type = ggml_type

    staged_n = [0]
    status = validate_dims_checked(snapshot_n_dims, snapshot_dims, snapshot_ggml_type, staged_n)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    staged_b = [0]
    status = tensor_type_block_elems_checked(snapshot_ggml_type, staged_b)
    if status != GGUF_TENSOR_PARSE_OK:
        return status
    if staged_b[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    if staged_n[0] % staged_b[0] != 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE

    parity_n = [0]
    status = validate_dims_checked(n_dims, dims, ggml_type, parity_n)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    parity_b = [0]
    status = tensor_type_block_elems_checked(ggml_type, parity_b)
    if status != GGUF_TENSOR_PARSE_OK:
        return status
    if parity_b[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    if parity_n[0] % parity_b[0] != 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE

    if snapshot_n_dims != n_dims or snapshot_dims is not dims or snapshot_ggml_type != ggml_type:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    if staged_n[0] != parity_n[0] or staged_b[0] != parity_b[0]:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    out_n_elements[0] = staged_n[0]
    out_block_elems[0] = staged_b[0]
    return GGUF_TENSOR_PARSE_OK


def validate_dims_checked_commit_only_preflight_only_parity(
    n_dims: int,
    dims: list[int] | None,
    ggml_type: int,
    out_n_elements: list[int] | None,
    out_block_elems: list[int] | None,
) -> int:
    if dims is None or out_n_elements is None or out_block_elems is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if out_n_elements is out_block_elems:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR

    snapshot_n_dims = n_dims
    snapshot_dims = dims
    snapshot_ggml_type = ggml_type

    staged_n = [0]
    staged_b = [0]
    status = validate_dims_checked_commit_only(snapshot_n_dims, snapshot_dims, snapshot_ggml_type, staged_n, staged_b)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    parity_n = [0]
    status = validate_dims_checked(snapshot_n_dims, snapshot_dims, snapshot_ggml_type, parity_n)
    if status != GGUF_TENSOR_PARSE_OK:
        return status

    parity_b = [0]
    status = tensor_type_block_elems_checked(snapshot_ggml_type, parity_b)
    if status != GGUF_TENSOR_PARSE_OK:
        return status
    if parity_b[0] == 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    if parity_n[0] % parity_b[0] != 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE

    if snapshot_n_dims != n_dims or snapshot_dims is not dims or snapshot_ggml_type != ggml_type:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    if staged_n[0] != parity_n[0] or staged_b[0] != parity_b[0]:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    out_n_elements[0] = staged_n[0]
    out_block_elems[0] = staged_b[0]
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq1197_validate_dims_commit_only_preflight_only_parity() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoValidateDimsCheckedCommitOnlyPreflightOnlyParity(U32 n_dims,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorTryAddU64", 1)[0]

    assert "if (!dims || !out_n_elements || !out_block_elems)" in body
    assert "if (out_n_elements == out_block_elems)" in body
    assert "status = GGUFTensorInfoValidateDimsCheckedCommitOnly(snapshot_n_dims," in body
    assert "status = GGUFTensorInfoValidateDimsChecked(snapshot_n_dims," in body
    assert "status = GGUFTensorTypeBlockElemsChecked(snapshot_ggml_type," in body
    assert "if (snapshot_n_dims != n_dims ||" in body
    assert "if (staged_n_elements != parity_n_elements ||" in body
    assert "*out_n_elements = staged_n_elements;" in body
    assert "*out_block_elems = staged_block_elems;" in body


def test_commit_only_preflight_only_parity_error_contracts_and_no_partial_publish() -> None:
    n_out = [111]
    b_out = [222]

    status = validate_dims_checked_commit_only_preflight_only_parity(2, None, 2, n_out, b_out)
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert n_out == [111]
    assert b_out == [222]

    alias_out = [333]
    status = validate_dims_checked_commit_only_preflight_only_parity(2, [32, 1], 2, alias_out, alias_out)
    assert status == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert alias_out == [333]

    status = validate_dims_checked_commit_only_preflight_only_parity(2, [0, 32], 2, n_out, b_out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
    assert n_out == [111]
    assert b_out == [222]

    status = validate_dims_checked_commit_only_preflight_only_parity(2, [1 << 63, 3], 2, n_out, b_out)
    assert status == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    assert n_out == [111]
    assert b_out == [222]

    status = validate_dims_checked_commit_only_preflight_only_parity(2, [32, 1], 999, n_out, b_out)
    assert status == GGUF_TENSOR_PARSE_ERR_BAD_TYPE
    assert n_out == [111]
    assert b_out == [222]


def test_commit_only_preflight_only_parity_success_and_randomized_vectors() -> None:
    status = validate_dims_checked_commit_only_preflight_only_parity(2, [32, 4], 2, [0], [0])
    assert status == GGUF_TENSOR_PARSE_OK

    rng = random.Random(1197)
    type_pool = sorted(KNOWN_TYPES)

    for _ in range(500):
        ggml_type = rng.choice(type_pool)
        block = [0]
        assert tensor_type_block_elems_checked(ggml_type, block) == GGUF_TENSOR_PARSE_OK

        n_dims = rng.randint(1, GGUF_TENSOR_MAX_DIMS)
        dims = [1] * n_dims
        n_elements = block[0]

        for i in range(n_dims - 1):
            max_factor = min(64, max(1, U64_MAX // n_elements))
            factor = rng.randint(1, max_factor)
            dims[i] = factor
            n_elements *= factor

        remain_limit = max(1, U64_MAX // n_elements)
        if remain_limit == 0:
            remain_limit = 1
        tail = rng.randint(1, min(1024, remain_limit))
        dims[-1] = tail * block[0]

        out_n = [987654321]
        out_b = [123456789]
        status = validate_dims_checked_commit_only_preflight_only_parity(
            n_dims,
            dims,
            ggml_type,
            out_n,
            out_b,
        )
        assert status == GGUF_TENSOR_PARSE_OK

        expected_n = 1
        for d in dims:
            expected_n *= d
        assert out_n == [expected_n]
        assert out_b == block
        assert expected_n % out_b[0] == 0
