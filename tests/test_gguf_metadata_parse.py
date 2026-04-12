#!/usr/bin/env python3
"""Focused GGUF metadata parser parity fixtures.

This is host-side validation only. It mirrors GGUF metadata binary layout and
checks that expected decode/validation behavior matches HolyC parser intent.
"""

from __future__ import annotations

import struct

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

GGUF_META_PARSE_OK = 0
GGUF_META_PARSE_ERR_TRUNCATED = 2
GGUF_META_PARSE_ERR_BAD_TYPE = 3
GGUF_META_PARSE_ERR_NESTED_ARRAY = 7


def u8(v: int) -> bytes:
    return struct.pack("<B", v)


def u16(v: int) -> bytes:
    return struct.pack("<H", v)


def u32(v: int) -> bytes:
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def i32(v: int) -> bytes:
    return struct.pack("<i", v)


def i64(v: int) -> bytes:
    return struct.pack("<q", v)


def f32(v: float) -> bytes:
    return struct.pack("<f", v)


def gguf_string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return u64(len(raw)) + raw


def meta_entry(key: str, value_type: int, payload: bytes) -> bytes:
    return gguf_string(key) + u32(value_type) + payload


def scalar_payload(t: int, v):
    if t in (GGUF_TYPE_UINT8, GGUF_TYPE_BOOL):
        return u8(int(v))
    if t == GGUF_TYPE_INT8:
        return struct.pack("<b", int(v))
    if t == GGUF_TYPE_UINT16:
        return u16(int(v))
    if t == GGUF_TYPE_INT16:
        return struct.pack("<h", int(v))
    if t == GGUF_TYPE_UINT32:
        return u32(int(v))
    if t == GGUF_TYPE_INT32:
        return i32(int(v))
    if t == GGUF_TYPE_UINT64:
        return u64(int(v))
    if t == GGUF_TYPE_INT64:
        return i64(int(v))
    if t == GGUF_TYPE_FLOAT32:
        return f32(float(v))
    if t == GGUF_TYPE_FLOAT64:
        return struct.pack("<d", float(v))
    if t == GGUF_TYPE_STRING:
        return gguf_string(str(v))
    raise ValueError(t)


def array_payload(elem_type: int, values: list) -> bytes:
    payload = b"".join(
        scalar_payload(elem_type, value) if elem_type != GGUF_TYPE_STRING else gguf_string(value)
        for value in values
    )
    return u32(elem_type) + u64(len(values)) + payload


def parse_gguf_string(buf: bytes, cursor: int):
    if cursor + 8 > len(buf):
        return GGUF_META_PARSE_ERR_TRUNCATED, cursor, None
    (n,) = struct.unpack_from("<Q", buf, cursor)
    cursor += 8
    if cursor + n > len(buf):
        return GGUF_META_PARSE_ERR_TRUNCATED, cursor, None
    s = buf[cursor : cursor + n].decode("utf-8")
    cursor += n
    return GGUF_META_PARSE_OK, cursor, s


def skip_value_by_type(buf: bytes, cursor: int, t: int):
    if t in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        return (GGUF_META_PARSE_OK, cursor + 1) if cursor + 1 <= len(buf) else (GGUF_META_PARSE_ERR_TRUNCATED, cursor)
    if t in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return (GGUF_META_PARSE_OK, cursor + 2) if cursor + 2 <= len(buf) else (GGUF_META_PARSE_ERR_TRUNCATED, cursor)
    if t in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return (GGUF_META_PARSE_OK, cursor + 4) if cursor + 4 <= len(buf) else (GGUF_META_PARSE_ERR_TRUNCATED, cursor)
    if t in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return (GGUF_META_PARSE_OK, cursor + 8) if cursor + 8 <= len(buf) else (GGUF_META_PARSE_ERR_TRUNCATED, cursor)
    if t == GGUF_TYPE_STRING:
        err, cursor2, _ = parse_gguf_string(buf, cursor)
        return err, cursor2
    if t == GGUF_TYPE_ARRAY:
        return GGUF_META_PARSE_ERR_NESTED_ARRAY, cursor
    return GGUF_META_PARSE_ERR_BAD_TYPE, cursor


def parse_value(buf: bytes, cursor: int):
    if cursor + 4 > len(buf):
        return GGUF_META_PARSE_ERR_TRUNCATED, cursor, None
    (t,) = struct.unpack_from("<I", buf, cursor)
    cursor += 4

    if t == GGUF_TYPE_ARRAY:
        if cursor + 12 > len(buf):
            return GGUF_META_PARSE_ERR_TRUNCATED, cursor, None
        elem_type, n = struct.unpack_from("<IQ", buf, cursor)
        cursor += 12
        if elem_type == GGUF_TYPE_ARRAY:
            return GGUF_META_PARSE_ERR_NESTED_ARRAY, cursor, None
        if elem_type > GGUF_TYPE_FLOAT64:
            return GGUF_META_PARSE_ERR_BAD_TYPE, cursor, None

        payload_start = cursor
        for _ in range(n):
            err, cursor = skip_value_by_type(buf, cursor, elem_type)
            if err != GGUF_META_PARSE_OK:
                return err, cursor, None

        return GGUF_META_PARSE_OK, cursor, {
            "type": t,
            "array_elem_type": elem_type,
            "array_len": n,
            "array_payload_off": payload_start,
            "array_payload_bytes": cursor - payload_start,
        }

    if t > GGUF_TYPE_FLOAT64:
        return GGUF_META_PARSE_ERR_BAD_TYPE, cursor, None

    if t == GGUF_TYPE_STRING:
        err, cursor, value = parse_gguf_string(buf, cursor)
        if err != GGUF_META_PARSE_OK:
            return err, cursor, None
        return GGUF_META_PARSE_OK, cursor, {"type": t, "value": value}

    err, cursor2 = skip_value_by_type(buf, cursor, t)
    if err != GGUF_META_PARSE_OK:
        return err, cursor2, None

    raw = buf[cursor:cursor2]
    return GGUF_META_PARSE_OK, cursor2, {"type": t, "raw": raw}


def parse_metadata_table(buf: bytes, count: int):
    items = []
    cursor = 0
    for _ in range(count):
        err, cursor, key = parse_gguf_string(buf, cursor)
        if err != GGUF_META_PARSE_OK:
            return err, cursor, items

        err, cursor, value = parse_value(buf, cursor)
        if err != GGUF_META_PARSE_OK:
            return err, cursor, items

        items.append((key, value))

    return GGUF_META_PARSE_OK, cursor, items


def test_mixed_scalar_and_arrays() -> None:
    buf = b"".join(
        [
            meta_entry("general.architecture", GGUF_TYPE_STRING, gguf_string("llama")),
            meta_entry("llama.block_count", GGUF_TYPE_UINT32, scalar_payload(GGUF_TYPE_UINT32, 22)),
            meta_entry("tokenizer.ggml.bos_token_id", GGUF_TYPE_INT32, scalar_payload(GGUF_TYPE_INT32, 1)),
            meta_entry("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, scalar_payload(GGUF_TYPE_BOOL, 1)),
            meta_entry(
                "tokenizer.ggml.tokens",
                GGUF_TYPE_ARRAY,
                array_payload(GGUF_TYPE_STRING, ["<unk>", "<s>", "</s>"]),
            ),
            meta_entry(
                "llama.rope.freq_base_train",
                GGUF_TYPE_FLOAT32,
                scalar_payload(GGUF_TYPE_FLOAT32, 10000.0),
            ),
        ]
    )

    err, cursor, items = parse_metadata_table(buf, 6)
    assert err == GGUF_META_PARSE_OK
    assert cursor == len(buf)
    assert len(items) == 6
    assert items[0][0] == "general.architecture"
    assert items[0][1]["value"] == "llama"
    assert items[4][1]["array_elem_type"] == GGUF_TYPE_STRING
    assert items[4][1]["array_len"] == 3
    assert items[4][1]["array_payload_bytes"] > 0


def test_nested_array_rejected() -> None:
    bad_nested = meta_entry(
        "bad.nested",
        GGUF_TYPE_ARRAY,
        u32(GGUF_TYPE_ARRAY) + u64(1) + b"\x00",
    )
    err, _cursor, _items = parse_metadata_table(bad_nested, 1)
    assert err == GGUF_META_PARSE_ERR_NESTED_ARRAY


def test_truncation_detected() -> None:
    truncated = meta_entry("x", GGUF_TYPE_STRING, gguf_string("abc"))[:-1]
    err, _cursor, _items = parse_metadata_table(truncated, 1)
    assert err == GGUF_META_PARSE_ERR_TRUNCATED


def run() -> None:
    test_mixed_scalar_and_arrays()
    test_nested_array_rejected()
    test_truncation_detected()
    print("gguf_metadata_reference_checks=ok")


if __name__ == "__main__":
    run()
