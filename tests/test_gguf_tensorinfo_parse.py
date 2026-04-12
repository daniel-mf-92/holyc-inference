#!/usr/bin/env python3
"""Reference checks for GGUF tensor-info parsing semantics."""

from __future__ import annotations

import struct

GGUF_TENSOR_PARSE_OK = 0
GGUF_TENSOR_PARSE_ERR_NULL_PTR = 1
GGUF_TENSOR_PARSE_ERR_TRUNCATED = 2
GGUF_TENSOR_PARSE_ERR_BAD_COUNT = 3
GGUF_TENSOR_PARSE_ERR_BAD_NAME_LEN = 4
GGUF_TENSOR_PARSE_ERR_BAD_DIMS = 5
GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE = 6
GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW = 7
GGUF_TENSOR_PARSE_ERR_BAD_TYPE = 8

GGUF_TENSOR_MAX_DIMS = 8
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


def gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return u64(len(raw)) + raw


def tensor_entry(name: str, dims: list[int], ggml_type: int, offset: int) -> bytes:
    payload = [gguf_string(name), u32(len(dims))]
    payload.extend(u64(d) for d in dims)
    payload.extend([u32(ggml_type), u64(offset)])
    return b"".join(payload)


def parse_one(buf: bytes, cursor: int):
    if cursor + 8 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, cursor, None
    (name_len,) = struct.unpack_from("<Q", buf, cursor)
    cursor += 8

    if cursor + name_len > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, cursor, None
    name = buf[cursor : cursor + name_len].decode("utf-8")
    cursor += name_len

    if cursor + 4 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, cursor, None
    (n_dims,) = struct.unpack_from("<I", buf, cursor)
    cursor += 4

    if n_dims == 0 or n_dims > GGUF_TENSOR_MAX_DIMS:
        return GGUF_TENSOR_PARSE_ERR_BAD_DIMS, cursor, None

    dims: list[int] = []
    n_elements = 1
    for _ in range(n_dims):
        if cursor + 8 > len(buf):
            return GGUF_TENSOR_PARSE_ERR_TRUNCATED, cursor, None
        (dim,) = struct.unpack_from("<Q", buf, cursor)
        cursor += 8
        if dim == 0:
            return GGUF_TENSOR_PARSE_ERR_BAD_DIM_SIZE, cursor, None
        if n_elements > U64_MAX // dim:
            return GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW, cursor, None

        dims.append(dim)
        n_elements *= dim

    if cursor + 4 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, cursor, None
    (ggml_type,) = struct.unpack_from("<I", buf, cursor)
    cursor += 4
    if ggml_type not in KNOWN_TYPES:
        return GGUF_TENSOR_PARSE_ERR_BAD_TYPE, cursor, None

    if cursor + 8 > len(buf):
        return GGUF_TENSOR_PARSE_ERR_TRUNCATED, cursor, None
    (offset,) = struct.unpack_from("<Q", buf, cursor)
    cursor += 8

    return (
        GGUF_TENSOR_PARSE_OK,
        cursor,
        {
            "name": name,
            "n_dims": n_dims,
            "dims": dims,
            "n_elements": n_elements,
            "ggml_type": ggml_type,
            "offset": offset,
        },
    )


def parse_table(buf: bytes, count: int):
    if count <= 0:
        return GGUF_TENSOR_PARSE_ERR_BAD_COUNT, 0, []

    cursor = 0
    items = []
    for _ in range(count):
        err, cursor, item = parse_one(buf, cursor)
        if err != GGUF_TENSOR_PARSE_OK:
            return err, cursor, items
        items.append(item)

    return GGUF_TENSOR_PARSE_OK, cursor, items


def test_tensor_table_success() -> None:
    buf = b"".join(
        [
            tensor_entry("token_embd.weight", [32000, 2048], 2, 0),
            tensor_entry("blk.0.attn_q.weight", [2048, 2048], 8, 917504),
        ]
    )

    err, cursor, items = parse_table(buf, 2)
    assert err == GGUF_TENSOR_PARSE_OK
    assert cursor == len(buf)
    assert len(items) == 2
    assert items[0]["name"] == "token_embd.weight"
    assert items[0]["n_elements"] == 32000 * 2048
    assert items[1]["ggml_type"] == 8


def test_reject_bad_dims_count() -> None:
    bad = gguf_string("bad") + u32(0)
    err, _cursor, _items = parse_table(bad, 1)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_DIMS


def test_reject_dim_overflow() -> None:
    huge = tensor_entry("overflow", [1 << 63, 3], 2, 0)
    err, _cursor, _items = parse_table(huge, 1)
    assert err == GGUF_TENSOR_PARSE_ERR_DIM_OVERFLOW


def test_reject_unknown_type() -> None:
    bad_type = tensor_entry("type.bad", [64, 64], 999, 0)
    err, _cursor, _items = parse_table(bad_type, 1)
    assert err == GGUF_TENSOR_PARSE_ERR_BAD_TYPE


def test_truncation_detected() -> None:
    trunc = tensor_entry("trunc", [128, 128], 2, 17)[:-5]
    err, _cursor, _items = parse_table(trunc, 1)
    assert err == GGUF_TENSOR_PARSE_ERR_TRUNCATED


def run() -> None:
    test_tensor_table_success()
    test_reject_bad_dims_count()
    test_reject_dim_overflow()
    test_reject_unknown_type()
    test_truncation_detected()
    print("gguf_tensorinfo_reference_checks=ok")


if __name__ == "__main__":
    run()
