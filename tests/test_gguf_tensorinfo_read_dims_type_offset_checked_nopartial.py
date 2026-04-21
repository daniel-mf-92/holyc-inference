#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial (IQ-999)."""

from __future__ import annotations

import random
import struct
from pathlib import Path

GGUF_TENSOR_PARSE_OK = 0
GGUF_TENSOR_PARSE_ERR_NULL_PTR = 1
GGUF_TENSOR_PARSE_ERR_TRUNCATED = 2
GGUF_TENSOR_PARSE_ERR_BAD_DIMS = 5
GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE = 6
GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW = 7
GGUF_TENSOR_PARSE_ERR_BAD_TYPE = 8

GGUF_TENSOR_MAX_DIMS = 8
I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1

KNOWN_TYPES = {
    0,
    1,
    2,
    3,
    6,
    7,
    8,
    9,
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
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
}


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def u64_add(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a > U64_MAX - b:
        return None
    return a + b


def u64_mul(a: int, b: int) -> int | None:
    if a < 0 or b < 0:
        return None
    if a != 0 and b > U64_MAX // a:
        return None
    return a * b


def dims_type_offset_entry(dims: list[int], ggml_type: int, offset: int) -> bytes:
    payload = [u32(len(dims))]
    payload.extend(u64(d) for d in dims)
    payload.extend([u32(ggml_type), u64(offset)])
    return b"".join(payload)


def parse_dims_type_offset_checked_nopartial(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_value: dict | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or out_value is None
        or out_next_cursor is None
    ):
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if size > I64_MAX or cursor > I64_MAX:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    c = cursor
    if c + 4 > size or c + 4 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (n_dims,) = struct.unpack_from("<I", buf, c)
    c += 4
    if n_dims == 0 or n_dims > GGUF_TENSOR_MAX_DIMS:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    dims: list[int] = []
    n_elements = 1
    for _ in range(n_dims):
        if c + 8 > size or c + 8 > len(buf):
            return GGUF_TENSOR_PARSE_ERR_TRUNCATED
        (dim,) = struct.unpack_from("<Q", buf, c)
        c += 8
        if dim == 0:
            return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
        if n_elements > U64_MAX // dim:
            return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
        dims.append(dim)
        n_elements *= dim

    if c + 4 > size or c + 4 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (ggml_type,) = struct.unpack_from("<I", buf, c)
    c += 4
    if ggml_type not in KNOWN_TYPES:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    if c + 8 > size or c + 8 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (offset,) = struct.unpack_from("<Q", buf, c)
    c += 8

    checked_end = u64_add(cursor, 4)
    if checked_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    dims_bytes = u64_mul(n_dims, 8)
    if dims_bytes is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    checked_end = u64_add(checked_end, dims_bytes)
    if checked_end is None:
        return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
    checked_end = u64_add(checked_end, 4)
    if checked_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    checked_end = u64_add(checked_end, 8)
    if checked_end is None:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    if checked_end > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    if c != checked_end:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    staged_dims = dims + [0] * (GGUF_TENSOR_MAX_DIMS - len(dims))
    out_value.clear()
    out_value.update(
        {
            "n_dims": n_dims,
            "dims": staged_dims,
            "n_elements": n_elements,
            "ggml_type": ggml_type,
            "offset": offset,
        }
    )
    out_next_cursor[0] = checked_end
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_iq999_dims_type_offset_parser_and_parseone_composition() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(U8 *buf,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFTensorParseOne(", 1)[0]

    assert "GGUFTensorReadU32LE(buf, (I64)size, &staged_cursor, &staged_n_dims)" in body
    assert "GGUFTensorTryMulU64(staged_n_dims, 8, &dims_bytes)" in body
    assert "if (!GGUFTensorTypeKnown(staged_ggml_type))" in body
    assert "if ((U64)staged_cursor != checked_end)" in body
    assert "*out_n_dims = staged_n_dims;" in body
    assert "*out_next_cursor = checked_end;" in body

    parse_one_body = source.split("I64 GGUFTensorParseOne(", 1)[1]
    assert "GGUFTensorInfoReadDimsTypeOffsetCheckedNoPartial(buf," in parse_one_body


def test_null_and_no_partial_publish_on_failure() -> None:
    entry = dims_type_offset_entry([8, 8], 2, 16)
    out = {"sentinel": 999}
    next_cursor = [77]

    err = parse_dims_type_offset_checked_nopartial(None, len(entry), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out == {"sentinel": 999}
    assert next_cursor == [77]

    truncated = entry[:-3]
    err = parse_dims_type_offset_checked_nopartial(truncated, len(truncated), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out == {"sentinel": 999}
    assert next_cursor == [77]


def test_adversarial_dim_type_and_offset_overflow_vectors() -> None:
    out = {"old": 1}
    next_cursor = [7]

    bad_dims_count = dims_type_offset_entry([], 2, 0)
    err = parse_dims_type_offset_checked_nopartial(
        bad_dims_count,
        len(bad_dims_count),
        0,
        out,
        next_cursor,
    )
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIMS
    assert out == {"old": 1}
    assert next_cursor == [7]

    too_many_dims = u32(GGUF_TENSOR_MAX_DIMS + 1)
    err = parse_dims_type_offset_checked_nopartial(too_many_dims, len(too_many_dims), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    zero_dim = dims_type_offset_entry([4, 0, 9], 2, 0)
    err = parse_dims_type_offset_checked_nopartial(zero_dim, len(zero_dim), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE

    dim_overflow = dims_type_offset_entry([1 << 63, 3], 2, 0)
    err = parse_dims_type_offset_checked_nopartial(dim_overflow, len(dim_overflow), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    bad_type = dims_type_offset_entry([16, 16], 999, 0)
    err = parse_dims_type_offset_checked_nopartial(bad_type, len(bad_type), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    # Offset field truncation (vector specifically targeting offset tail parsing).
    trunc_offset = dims_type_offset_entry([32, 32], 8, 1234)[:-5]
    err = parse_dims_type_offset_checked_nopartial(trunc_offset, len(trunc_offset), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED

    # Arithmetic overflow vector: cursor so high that checked_end computation overflows.
    # Uses large logical size but tiny backing payload; overflow is detected before publish.
    tail = dims_type_offset_entry([1], 2, 0)
    err = parse_dims_type_offset_checked_nopartial(tail, U64_MAX, U64_MAX - 1, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def test_success_and_randomized_cursor_parity() -> None:
    fixed = dims_type_offset_entry([32000, 2048], 2, 917504)
    out: dict = {"old": 1}
    next_cursor = [5]
    err = parse_dims_type_offset_checked_nopartial(fixed, len(fixed), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_OK
    assert out["n_dims"] == 2
    assert out["dims"][0] == 32000
    assert out["dims"][1] == 2048
    assert out["n_elements"] == 32000 * 2048
    assert out["ggml_type"] == 2
    assert out["offset"] == 917504
    assert next_cursor[0] == len(fixed)

    rng = random.Random(20260422999)
    for i in range(2000):
        n_dims = rng.randint(1, GGUF_TENSOR_MAX_DIMS)
        dims: list[int] = []
        prod = 1
        for _ in range(n_dims):
            max_dim = max(1, min(1 << 14, U64_MAX // prod))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            prod *= dim

        ggml_type = rng.choice(tuple(KNOWN_TYPES))
        offset = rng.randint(0, 1 << 42)

        prefix_len = rng.randint(0, 24)
        prefix = bytes(rng.randint(0, 255) for _ in range(prefix_len))
        payload = prefix + dims_type_offset_entry(dims, ggml_type, offset)

        out = {"keep": i}
        next_cursor = [0]
        err = parse_dims_type_offset_checked_nopartial(payload, len(payload), prefix_len, out, next_cursor)
        assert err == GGUF_TENSOR_PARSE_OK
        assert out["n_dims"] == n_dims
        assert out["dims"][:n_dims] == dims
        assert out["dims"][n_dims:] == [0] * (GGUF_TENSOR_MAX_DIMS - n_dims)
        assert out["n_elements"] == prod
        assert out["ggml_type"] == ggml_type
        assert out["offset"] == offset
        assert next_cursor[0] == len(payload)


if __name__ == "__main__":
    test_source_contains_iq999_dims_type_offset_parser_and_parseone_composition()
    test_null_and_no_partial_publish_on_failure()
    test_adversarial_dim_type_and_offset_overflow_vectors()
    test_success_and_randomized_cursor_parity()
    print("gguf_tensorinfo_read_dims_type_offset_checked_nopartial=ok")
