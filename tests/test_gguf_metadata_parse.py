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
GGUF_META_PARSE_ERR_NOT_FOUND = 9
GGUF_META_PARSE_ERR_TYPE_MISMATCH = 10

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


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


def decode_scalar_by_type(raw: bytes, t: int):
    if t == GGUF_TYPE_UINT8:
        return raw[0]
    if t == GGUF_TYPE_INT8:
        return int.from_bytes(raw[:1], "little", signed=True)
    if t == GGUF_TYPE_UINT16:
        return int.from_bytes(raw[:2], "little", signed=False)
    if t == GGUF_TYPE_INT16:
        return int.from_bytes(raw[:2], "little", signed=True)
    if t == GGUF_TYPE_UINT32:
        return int.from_bytes(raw[:4], "little", signed=False)
    if t == GGUF_TYPE_INT32:
        return int.from_bytes(raw[:4], "little", signed=True)
    if t == GGUF_TYPE_UINT64:
        return int.from_bytes(raw[:8], "little", signed=False)
    if t == GGUF_TYPE_INT64:
        return int.from_bytes(raw[:8], "little", signed=True)
    if t == GGUF_TYPE_BOOL:
        return raw[0] != 0
    if t == GGUF_TYPE_FLOAT32:
        return int.from_bytes(raw[:4], "little", signed=False)
    if t == GGUF_TYPE_FLOAT64:
        return int.from_bytes(raw[:8], "little", signed=False)
    raise ValueError(t)


def decode_array_payload(buf: bytes, parsed_value: dict):
    elem_type = parsed_value["array_elem_type"]
    cursor = parsed_value["array_payload_off"]
    end = cursor + parsed_value["array_payload_bytes"]

    values = []
    while cursor < end:
        if elem_type == GGUF_TYPE_STRING:
            err, cursor2, value = parse_gguf_string(buf, cursor)
            assert err == GGUF_META_PARSE_OK
            cursor = cursor2
            values.append(value)
            continue

        width = {
            GGUF_TYPE_UINT8: 1,
            GGUF_TYPE_INT8: 1,
            GGUF_TYPE_BOOL: 1,
            GGUF_TYPE_UINT16: 2,
            GGUF_TYPE_INT16: 2,
            GGUF_TYPE_UINT32: 4,
            GGUF_TYPE_INT32: 4,
            GGUF_TYPE_FLOAT32: 4,
            GGUF_TYPE_UINT64: 8,
            GGUF_TYPE_INT64: 8,
            GGUF_TYPE_FLOAT64: 8,
        }[elem_type]
        raw = buf[cursor : cursor + width]
        assert len(raw) == width
        values.append(decode_scalar_by_type(raw, elem_type))
        cursor += width

    assert cursor == end
    assert len(values) == parsed_value["array_len"]
    return values


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


def metadata_table_span_validate_checked(
    table_start: int,
    table_nbytes: int,
    file_nbytes: int,
):
    if table_start > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, 0
    if table_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, 0
    if file_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, 0

    if table_start > file_nbytes:
        return GGUF_META_TABLE_ERR_BAD_PARAM, 0
    if table_nbytes > U64_MAX - table_start:
        return GGUF_META_TABLE_ERR_OVERFLOW, 0

    table_end = table_start + table_nbytes
    if table_end > file_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, 0
    return GGUF_META_TABLE_OK, table_end


def metadata_table_cursor_advance_checked(
    table_start: int,
    table_nbytes: int,
    file_nbytes: int,
    cursor_advance: int,
):
    err, table_end = metadata_table_span_validate_checked(
        table_start=table_start,
        table_nbytes=table_nbytes,
        file_nbytes=file_nbytes,
    )
    if err != GGUF_META_TABLE_OK:
        return err, 0, 0

    if cursor_advance > table_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, 0, 0
    if table_start > U64_MAX - cursor_advance:
        return GGUF_META_TABLE_ERR_OVERFLOW, 0, 0

    cursor = table_start + cursor_advance
    if cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, 0, 0

    return GGUF_META_TABLE_OK, cursor, table_end


def meta_find_by_key(items, key: str):
    for k, v in items:
        if k == key:
            return v
    return None


def meta_get_u64(items, key: str):
    v = meta_find_by_key(items, key)
    if v is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND, None
    if v["type"] not in (GGUF_TYPE_UINT8, GGUF_TYPE_UINT16, GGUF_TYPE_UINT32, GGUF_TYPE_UINT64):
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH, None
    return GGUF_META_PARSE_OK, int.from_bytes(v["raw"], "little", signed=False)


def meta_get_i64(items, key: str):
    v = meta_find_by_key(items, key)
    if v is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND, None
    ty = v["type"]
    if ty not in (GGUF_TYPE_INT8, GGUF_TYPE_INT16, GGUF_TYPE_INT32, GGUF_TYPE_INT64):
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH, None
    size = {GGUF_TYPE_INT8: 1, GGUF_TYPE_INT16: 2, GGUF_TYPE_INT32: 4, GGUF_TYPE_INT64: 8}[ty]
    return GGUF_META_PARSE_OK, int.from_bytes(v["raw"][:size], "little", signed=True)


def meta_get_bool(items, key: str):
    v = meta_find_by_key(items, key)
    if v is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND, None
    if v["type"] != GGUF_TYPE_BOOL:
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH, None
    return GGUF_META_PARSE_OK, v["raw"][0] != 0


def meta_get_string(items, key: str):
    v = meta_find_by_key(items, key)
    if v is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND, None
    if v["type"] != GGUF_TYPE_STRING:
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH, None
    return GGUF_META_PARSE_OK, v["value"]


def meta_get_f32_bits(items, key: str):
    v = meta_find_by_key(items, key)
    if v is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND, None
    if v["type"] != GGUF_TYPE_FLOAT32:
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH, None
    return GGUF_META_PARSE_OK, int.from_bytes(v["raw"], "little", signed=False)


def meta_get_f64_bits(items, key: str):
    v = meta_find_by_key(items, key)
    if v is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND, None
    if v["type"] != GGUF_TYPE_FLOAT64:
        return GGUF_META_PARSE_ERR_TYPE_MISMATCH, None
    return GGUF_META_PARSE_OK, int.from_bytes(v["raw"], "little", signed=False)


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
    assert decode_array_payload(buf, items[4][1]) == ["<unk>", "<s>", "</s>"]


def test_all_scalar_widths_and_signs_round_trip() -> None:
    expected_f32_bits = int.from_bytes(struct.pack("<f", -13.5), "little", signed=False)
    expected_f64_bits = int.from_bytes(struct.pack("<d", 0.03125), "little", signed=False)

    fixtures = [
        ("u8", GGUF_TYPE_UINT8, 255),
        ("i8", GGUF_TYPE_INT8, -127),
        ("u16", GGUF_TYPE_UINT16, 65535),
        ("i16", GGUF_TYPE_INT16, -12345),
        ("u32", GGUF_TYPE_UINT32, 0xFEDCBA98),
        ("i32", GGUF_TYPE_INT32, -123456789),
        ("u64", GGUF_TYPE_UINT64, 0xFEDCBA9876543210),
        ("i64", GGUF_TYPE_INT64, -0x0123456789ABCDEF),
        ("bool_true", GGUF_TYPE_BOOL, 1),
        ("f32", GGUF_TYPE_FLOAT32, -13.5),
        ("f64", GGUF_TYPE_FLOAT64, 0.03125),
        ("txt", GGUF_TYPE_STRING, "temple"),
    ]

    buf = b"".join(meta_entry(key, t, scalar_payload(t, v)) for key, t, v in fixtures)
    err, cursor, items = parse_metadata_table(buf, len(fixtures))
    assert err == GGUF_META_PARSE_OK
    assert cursor == len(buf)

    expected = {
        "u8": 255,
        "i8": -127,
        "u16": 65535,
        "i16": -12345,
        "u32": 0xFEDCBA98,
        "i32": -123456789,
        "u64": 0xFEDCBA9876543210,
        "i64": -0x0123456789ABCDEF,
        "bool_true": True,
        "f32": expected_f32_bits,
        "f64": expected_f64_bits,
        "txt": "temple",
    }

    for key, value in items:
        if value["type"] == GGUF_TYPE_STRING:
            got = value["value"]
        else:
            got = decode_scalar_by_type(value["raw"], value["type"])
        assert got == expected[key]


def test_array_payload_offsets_and_byte_sizes_for_numeric_types() -> None:
    f32_bits_one = int.from_bytes(struct.pack("<f", 1.5), "little", signed=False)
    f32_bits_two = int.from_bytes(struct.pack("<f", -2.25), "little", signed=False)

    buf = b"".join(
        [
            meta_entry("arr.u8", GGUF_TYPE_ARRAY, array_payload(GGUF_TYPE_UINT8, [2, 7, 9, 255])),
            meta_entry("arr.i16", GGUF_TYPE_ARRAY, array_payload(GGUF_TYPE_INT16, [-1, 22, -32768])),
            meta_entry(
                "arr.f32",
                GGUF_TYPE_ARRAY,
                array_payload(GGUF_TYPE_FLOAT32, [1.5, -2.25]),
            ),
        ]
    )

    err, cursor, items = parse_metadata_table(buf, 3)
    assert err == GGUF_META_PARSE_OK
    assert cursor == len(buf)

    arr_u8 = meta_find_by_key(items, "arr.u8")
    arr_i16 = meta_find_by_key(items, "arr.i16")
    arr_f32 = meta_find_by_key(items, "arr.f32")
    assert arr_u8 is not None and arr_i16 is not None and arr_f32 is not None

    assert arr_u8["array_payload_bytes"] == 4
    assert arr_i16["array_payload_bytes"] == 6
    assert arr_f32["array_payload_bytes"] == 8

    assert decode_array_payload(buf, arr_u8) == [2, 7, 9, 255]
    assert decode_array_payload(buf, arr_i16) == [-1, 22, -32768]
    assert decode_array_payload(buf, arr_f32) == [f32_bits_one, f32_bits_two]


def test_invalid_array_element_type_rejected() -> None:
    payload = u32(99) + u64(1) + u32(9)
    nested = meta_entry("bad.array.elem_type", GGUF_TYPE_ARRAY, payload)
    err, _cursor, _items = parse_metadata_table(nested, 1)
    assert err == GGUF_META_PARSE_ERR_BAD_TYPE


def test_array_payload_truncation_detected() -> None:
    full = meta_entry("arr.u64", GGUF_TYPE_ARRAY, array_payload(GGUF_TYPE_UINT64, [1, 2]))
    truncated = full[:-3]
    err, _cursor, _items = parse_metadata_table(truncated, 1)
    assert err == GGUF_META_PARSE_ERR_TRUNCATED


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


def test_lookup_and_scalar_extractors() -> None:
    buf = b"".join(
        [
            meta_entry("general.architecture", GGUF_TYPE_STRING, gguf_string("llama")),
            meta_entry("llama.block_count", GGUF_TYPE_UINT32, scalar_payload(GGUF_TYPE_UINT32, 22)),
            meta_entry("tokenizer.ggml.bos_token_id", GGUF_TYPE_INT32, scalar_payload(GGUF_TYPE_INT32, 1)),
            meta_entry("tokenizer.ggml.add_bos_token", GGUF_TYPE_BOOL, scalar_payload(GGUF_TYPE_BOOL, 1)),
            meta_entry("llama.rope.freq_base_train", GGUF_TYPE_FLOAT32, scalar_payload(GGUF_TYPE_FLOAT32, 10000.0)),
            meta_entry("llama.attention.scale_hint", GGUF_TYPE_FLOAT64, scalar_payload(GGUF_TYPE_FLOAT64, 0.125)),
        ]
    )

    err, cursor, items = parse_metadata_table(buf, 6)
    assert err == GGUF_META_PARSE_OK
    assert cursor == len(buf)

    err, value = meta_get_u64(items, "llama.block_count")
    assert err == GGUF_META_PARSE_OK and value == 22

    err, value = meta_get_i64(items, "tokenizer.ggml.bos_token_id")
    assert err == GGUF_META_PARSE_OK and value == 1

    err, value = meta_get_bool(items, "tokenizer.ggml.add_bos_token")
    assert err == GGUF_META_PARSE_OK and value is True

    err, value = meta_get_string(items, "general.architecture")
    assert err == GGUF_META_PARSE_OK and value == "llama"

    err, f32_bits = meta_get_f32_bits(items, "llama.rope.freq_base_train")
    assert err == GGUF_META_PARSE_OK
    assert f32_bits == int.from_bytes(struct.pack("<f", 10000.0), "little", signed=False)

    err, f64_bits = meta_get_f64_bits(items, "llama.attention.scale_hint")
    assert err == GGUF_META_PARSE_OK
    assert f64_bits == int.from_bytes(struct.pack("<d", 0.125), "little", signed=False)

    err, _ = meta_get_u64(items, "general.architecture")
    assert err == GGUF_META_PARSE_ERR_TYPE_MISMATCH

    err, _ = meta_get_string(items, "missing.key")
    assert err == GGUF_META_PARSE_ERR_NOT_FOUND


def test_metadata_table_span_validate_known_good() -> None:
    err, table_end = metadata_table_span_validate_checked(64, 128, 4096)
    assert err == GGUF_META_TABLE_OK
    assert table_end == 192


def test_metadata_table_span_validate_zero_length() -> None:
    err, table_end = metadata_table_span_validate_checked(256, 0, 4096)
    assert err == GGUF_META_TABLE_OK
    assert table_end == 256


def test_metadata_table_span_validate_bad_param_start_past_eof() -> None:
    err, _ = metadata_table_span_validate_checked(5000, 0, 4096)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM


def test_metadata_table_span_validate_overflow_inputs() -> None:
    err, _ = metadata_table_span_validate_checked(I64_MAX + 1, 0, I64_MAX)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW

    err, _ = metadata_table_span_validate_checked(0, I64_MAX + 1, I64_MAX)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW

    err, _ = metadata_table_span_validate_checked(0, 0, I64_MAX + 1)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW


def test_metadata_table_span_validate_unsigned_add_wrap() -> None:
    err, _ = metadata_table_span_validate_checked(U64_MAX - 1, 4, U64_MAX)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW


def test_metadata_table_span_validate_out_of_bounds() -> None:
    err, _ = metadata_table_span_validate_checked(1024, 2048, 2047)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS


def test_metadata_table_cursor_advance_known_good() -> None:
    err, cursor, table_end = metadata_table_cursor_advance_checked(64, 128, 4096, 32)
    assert err == GGUF_META_TABLE_OK
    assert cursor == 96
    assert table_end == 192


def test_metadata_table_cursor_advance_allows_end_cursor() -> None:
    err, cursor, table_end = metadata_table_cursor_advance_checked(1000, 24, 4096, 24)
    assert err == GGUF_META_TABLE_OK
    assert cursor == 1024
    assert table_end == 1024


def test_metadata_table_cursor_advance_rejects_past_table_end() -> None:
    err, _, _ = metadata_table_cursor_advance_checked(256, 32, 4096, 33)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS


def test_metadata_table_cursor_advance_propagates_span_error() -> None:
    err, _, _ = metadata_table_cursor_advance_checked(5000, 10, 4096, 0)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM


def test_metadata_table_cursor_advance_unsigned_add_wrap() -> None:
    err, _, _ = metadata_table_cursor_advance_checked(U64_MAX - 1, 1, U64_MAX, 2)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW


def run() -> None:
    test_mixed_scalar_and_arrays()
    test_all_scalar_widths_and_signs_round_trip()
    test_array_payload_offsets_and_byte_sizes_for_numeric_types()
    test_nested_array_rejected()
    test_invalid_array_element_type_rejected()
    test_array_payload_truncation_detected()
    test_truncation_detected()
    test_lookup_and_scalar_extractors()
    test_metadata_table_span_validate_known_good()
    test_metadata_table_span_validate_zero_length()
    test_metadata_table_span_validate_bad_param_start_past_eof()
    test_metadata_table_span_validate_overflow_inputs()
    test_metadata_table_span_validate_unsigned_add_wrap()
    test_metadata_table_span_validate_out_of_bounds()
    test_metadata_table_cursor_advance_known_good()
    test_metadata_table_cursor_advance_allows_end_cursor()
    test_metadata_table_cursor_advance_rejects_past_table_end()
    test_metadata_table_cursor_advance_propagates_span_error()
    test_metadata_table_cursor_advance_unsigned_add_wrap()
    print("gguf_metadata_reference_checks=ok")


if __name__ == "__main__":
    run()
