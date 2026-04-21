#!/usr/bin/env python3
"""Parity harness for GGUFTensorInfoReadCheckedNoPartial (IQ-987)."""

from __future__ import annotations

import random
import struct
from pathlib import Path

GGUF_TENSOR_PARSE_OK = 0
GGUF_TENSOR_PARSE_ERR_NULL_PTR = 1
GGUF_TENSOR_PARSE_ERR_TRUNCATED = 2
GGUF_TENSOR_PARSE_ERR_BAD_COUNT = 3
GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN = 4
GGUF_TENSOR_PARSE_ERR_BAD_DIMS = 5
GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE = 6
GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW = 7
GGUF_TENSOR_PARSE_ERR_BAD_TYPE = 8

GGUF_TENSOR_MAX_NAME_BYTES = 1 << 20
GGUF_TENSOR_MAX_DIMS = 8
U64_MAX = (1 << 64) - 1

KNOWN_TYPES = {
    0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35,
}


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def gguf_string_bytes(raw: bytes) -> bytes:
    return u64(len(raw)) + raw


def tensor_entry(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    name_raw = name.encode("utf-8")
    payload = [gguf_string_bytes(name_raw), u32(len(dims))]
    payload.extend(u64(d) for d in dims)
    payload.extend([u32(ggml_type), u64(offset)])
    return b"".join(payload)


def parse_one_checked_nopartial(
    buf: bytes | None,
    size: int,
    cursor: int,
    out_value: dict | None,
    out_next_cursor: list[int] | None,
) -> int:
    if buf is None or out_value is None or out_next_cursor is None:
        return GGUF_TENSOR_PARSE_ERR_NULL_PTR
    if cursor > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED

    staged = {}
    c = cursor

    if c + 8 > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (name_len,) = struct.unpack_from("<Q", buf, c)
    c += 8
    if name_len > GGUF_TENSOR_MAX_NAME_BYTES:
        return GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN

    if c + name_len > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    name_raw = buf[c : c + name_len]
    c += name_len

    if c + 4 > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (n_dims,) = struct.unpack_from("<I", buf, c)
    c += 4
    if n_dims == 0 or n_dims > GGUF_TENSOR_MAX_DIMS:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    dims: list[int] = []
    n_elements = 1
    for _ in range(n_dims):
        if c + 8 > size:
            return GGUF_TENSOR_PARSE_ERR_TRUNCATED
        (dim,) = struct.unpack_from("<Q", buf, c)
        c += 8
        if dim == 0:
            return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE
        if n_elements > U64_MAX // dim:
            return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW
        dims.append(dim)
        n_elements *= dim

    if c + 4 > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (ggml_type,) = struct.unpack_from("<I", buf, c)
    c += 4
    if ggml_type not in KNOWN_TYPES:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE

    if c + 8 > size:
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED
    (offset,) = struct.unpack_from("<Q", buf, c)
    c += 8

    staged["name_raw"] = name_raw
    staged["name_len"] = name_len
    staged["n_dims"] = n_dims
    staged["dims"] = dims
    staged["n_elements"] = n_elements
    staged["ggml_type"] = ggml_type
    staged["offset"] = offset

    out_value.clear()
    out_value.update(staged)
    out_next_cursor[0] = c
    return GGUF_TENSOR_PARSE_OK


def test_source_contains_checked_nopartial_parser_and_table_composition() -> None:
    source = Path("src/gguf/tensorinfo.HC").read_text(encoding="utf-8")
    sig = "I64 GGUFTensorInfoReadCheckedNoPartial(U8 *buf,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 GGUFParseTensorInfo(", 1)[0]

    assert "status = GGUFTensorParseOne(buf," in body
    assert "GGUFTensorTryAddU64(cursor, 8, &checked_end)" in body
    assert "GGUFTensorTryMulU64(staged.n_dims, 8, &dims_bytes)" in body
    assert "if ((U64)staged_cursor != offset_end)" in body
    assert "*out = staged;" in body
    assert "*out_next_cursor = staged_cursor;" in body

    parse_table_body = source.split("I64 GGUFParseTensorInfo(", 1)[1]
    assert "GGUFTensorInfoReadCheckedNoPartial(buf," in parse_table_body


def test_null_and_no_partial_publish_on_fail() -> None:
    entry = tensor_entry("w", [8, 8], 2, 16)
    out = {"sentinel": 123}
    next_cursor = [77]

    err = parse_one_checked_nopartial(None, len(entry), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_NULL_PTR
    assert out == {"sentinel": 123}
    assert next_cursor == [77]

    # Truncated offset field: parser must fail and keep outputs unchanged.
    truncated = entry[:-3]
    err = parse_one_checked_nopartial(truncated, len(truncated), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED
    assert out == {"sentinel": 123}
    assert next_cursor == [77]


def test_adversarial_name_dims_and_offset_vectors() -> None:
    # Name length beyond hard cap.
    too_long_hdr = u64(GGUF_TENSOR_MAX_NAME_BYTES + 1)
    out = {"x": 1}
    next_cursor = [9]
    err = parse_one_checked_nopartial(too_long_hdr, len(too_long_hdr), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN
    assert out == {"x": 1}
    assert next_cursor == [9]

    # Bad dim-count (9 > GGUF_TENSOR_MAX_DIMS).
    name = gguf_string_bytes(b"bad.dims")
    bad_dims = name + u32(9)
    err = parse_one_checked_nopartial(bad_dims, len(bad_dims), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIMS

    # Dim product overflow.
    ovf = tensor_entry("ovf", [1 << 63, 3], 2, 0)
    err = parse_one_checked_nopartial(ovf, len(ovf), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW

    # Offset field truncation.
    trunc_offset = tensor_entry("trunc.offset", [32, 64], 8, 1234)[:-5]
    err = parse_one_checked_nopartial(trunc_offset, len(trunc_offset), 0, out, next_cursor)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def test_success_and_randomized_cursor_parity() -> None:
    first = tensor_entry("tok", [32000, 2048], 2, 0)
    second = tensor_entry("blk.0.attn_q.weight", [2048, 2048], 8, 917504)
    buf = first + second

    out1: dict = {}
    cur1 = [0]
    err = parse_one_checked_nopartial(buf, len(buf), 0, out1, cur1)
    assert err == GGUF_TENSOR_PARSE_OK
    assert out1["name_raw"].decode("utf-8") == "tok"
    assert out1["n_elements"] == 32000 * 2048

    out2: dict = {}
    cur2 = [0]
    err = parse_one_checked_nopartial(buf, len(buf), cur1[0], out2, cur2)
    assert err == GGUF_TENSOR_PARSE_OK
    assert out2["name_raw"].decode("utf-8") == "blk.0.attn_q.weight"
    assert cur2[0] == len(buf)

    rng = random.Random(20260422_987)
    for _ in range(2000):
        name_len = rng.randint(1, 24)
        name_raw = bytes(rng.randint(97, 122) for _ in range(name_len))
        n_dims = rng.randint(1, GGUF_TENSOR_MAX_DIMS)
        dims: list[int] = []
        prod = 1
        for _d in range(n_dims):
            max_dim = max(1, min(512, U64_MAX // prod))
            dim = rng.randint(1, max_dim)
            dims.append(dim)
            prod *= dim

        ggml_type = rng.choice(tuple(KNOWN_TYPES))
        offset = rng.randint(0, 1 << 40)

        entry = gguf_string_bytes(name_raw)
        entry += u32(n_dims)
        for dim in dims:
            entry += u64(dim)
        entry += u32(ggml_type)
        entry += u64(offset)

        out: dict = {"old": 1}
        nxt = [111]
        err = parse_one_checked_nopartial(entry, len(entry), 0, out, nxt)
        assert err == GGUF_TENSOR_PARSE_OK
        assert out["name_raw"] == name_raw
        assert out["n_dims"] == n_dims
        assert out["dims"] == dims
        assert out["ggml_type"] == ggml_type
        assert out["offset"] == offset
        assert nxt[0] == len(entry)


if __name__ == "__main__":
    test_source_contains_checked_nopartial_parser_and_table_composition()
    test_null_and_no_partial_publish_on_fail()
    test_adversarial_name_dims_and_offset_vectors()
    test_success_and_randomized_cursor_parity()
    print("gguf_tensor_info_read_checked_nopartial=ok")
