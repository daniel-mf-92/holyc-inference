#!/usr/bin/env python3
"""Parity harness for GGUFMetadataCursorAdvanceChecked (IQ-1137)."""

from __future__ import annotations

import random
from pathlib import Path

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

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

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


def cursor_can_advance_checked(cursor: int, need: int, table_end: int):
    if cursor > I64_MAX or need > I64_MAX or table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if cursor > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None
    if cursor > U64_MAX - need:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None

    return GGUF_META_TABLE_OK, next_cursor


def value_fixed_width_bytes_checked(value_type: int):
    if value_type in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        return GGUF_META_TABLE_OK, 1
    if value_type in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        return GGUF_META_TABLE_OK, 2
    if value_type in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        return GGUF_META_TABLE_OK, 4
    if value_type in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        return GGUF_META_TABLE_OK, 8
    return GGUF_META_TABLE_ERR_BAD_PARAM, None


def cursor_typed_span_bytes_checked(value_type: int, variable_payload_bytes: int):
    if variable_payload_bytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    if value_type == GGUF_TYPE_STRING:
        span_bytes = 8
        if span_bytes > U64_MAX - variable_payload_bytes:
            return GGUF_META_TABLE_ERR_OVERFLOW, None
        span_bytes += variable_payload_bytes
    elif value_type == GGUF_TYPE_ARRAY:
        span_bytes = 12
        if span_bytes > U64_MAX - variable_payload_bytes:
            return GGUF_META_TABLE_ERR_OVERFLOW, None
        span_bytes += variable_payload_bytes
    else:
        if variable_payload_bytes != 0:
            return GGUF_META_TABLE_ERR_BAD_PARAM, None
        err, fixed = value_fixed_width_bytes_checked(value_type)
        if err != GGUF_META_TABLE_OK:
            return err, None
        span_bytes = fixed

    if span_bytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    return GGUF_META_TABLE_OK, span_bytes


def metadata_cursor_advance_checked(
    cursor_ref: list[int] | None,
    table_end: int,
    value_type: int,
    variable_payload_bytes: int,
    out_span_start_ref: list[int] | None,
    out_span_bytes_ref: list[int] | None,
    out_span_end_ref: list[int] | None,
) -> int:
    if (
        cursor_ref is None
        or out_span_start_ref is None
        or out_span_bytes_ref is None
        or out_span_end_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cursor = cursor_ref[0]
    if cursor > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    err, span_bytes = cursor_typed_span_bytes_checked(value_type, variable_payload_bytes)
    if err != GGUF_META_TABLE_OK:
        return err

    err, span_end = cursor_can_advance_checked(cursor, span_bytes, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    out_span_start_ref[0] = cursor
    out_span_bytes_ref[0] = span_bytes
    out_span_end_ref[0] = span_end
    cursor_ref[0] = span_end
    return GGUF_META_TABLE_OK


def test_source_contains_iq1137_functions() -> None:
    source = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")

    typed_sig = "I32 GGUFMetadataCursorTypedSpanBytesChecked(U32 value_type,"
    advance_sig = "I32 GGUFMetadataCursorAdvanceChecked(U64 *cursor,"
    assert typed_sig in source
    assert advance_sig in source

    typed_body = source.split(typed_sig, 1)[1].split(advance_sig, 1)[0]
    assert "case GGUF_TYPE_STRING:" in typed_body
    assert "span_bytes = 8;" in typed_body
    assert "case GGUF_TYPE_ARRAY:" in typed_body
    assert "span_bytes = 12;" in typed_body
    assert "GGUFMetadataValueFixedWidthBytesChecked(value_type," in typed_body

    assert "GGUFMetadataCursorTypedSpanBytesChecked(value_type," in source
    assert "GGUFMetadataCursorCanAdvanceChecked(cur," in source
    assert "*cursor = span_end;" in source
    assert "if (variable_payload_bytes != 0)" in source


def test_fixed_width_type_accounting() -> None:
    vectors = [
        (GGUF_TYPE_UINT8, 1),
        (GGUF_TYPE_INT16, 2),
        (GGUF_TYPE_FLOAT32, 4),
        (GGUF_TYPE_UINT64, 8),
    ]

    for value_type, expected_span in vectors:
        cursor = [100]
        start = [0xAAAA]
        span = [0xBBBB]
        end = [0xCCCC]

        err = metadata_cursor_advance_checked(cursor, 200, value_type, 0, start, span, end)
        assert err == GGUF_META_TABLE_OK
        assert start[0] == 100
        assert span[0] == expected_span
        assert end[0] == 100 + expected_span
        assert cursor[0] == end[0]


def test_string_and_array_variable_payload_accounting() -> None:
    cursor = [10]
    start = [0]
    span = [0]
    end = [0]
    err = metadata_cursor_advance_checked(cursor, 200, GGUF_TYPE_STRING, 5, start, span, end)
    assert err == GGUF_META_TABLE_OK
    assert (start[0], span[0], end[0], cursor[0]) == (10, 13, 23, 23)

    cursor = [10]
    start = [0]
    span = [0]
    end = [0]
    err = metadata_cursor_advance_checked(cursor, 200, GGUF_TYPE_ARRAY, 17, start, span, end)
    assert err == GGUF_META_TABLE_OK
    assert (start[0], span[0], end[0], cursor[0]) == (10, 29, 39, 39)


def test_no_partial_writes_on_failure() -> None:
    cursor = [33]
    start = [0x1234]
    span = [0x5678]
    end = [0x9ABC]

    err = metadata_cursor_advance_checked(cursor, 100, 99, 0, start, span, end)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 33
    assert start[0] == 0x1234
    assert span[0] == 0x5678
    assert end[0] == 0x9ABC

    err = metadata_cursor_advance_checked(cursor, 35, GGUF_TYPE_UINT64, 0, start, span, end)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 33
    assert start[0] == 0x1234
    assert span[0] == 0x5678
    assert end[0] == 0x9ABC

    err = metadata_cursor_advance_checked(cursor, 100, GGUF_TYPE_UINT16, 4, start, span, end)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 33
    assert start[0] == 0x1234
    assert span[0] == 0x5678
    assert end[0] == 0x9ABC


def test_null_and_overflow_contracts() -> None:
    start = [0]
    span = [0]
    end = [0]

    err = metadata_cursor_advance_checked(None, 100, GGUF_TYPE_UINT8, 0, start, span, end)
    assert err == GGUF_META_TABLE_ERR_NULL_PTR

    cursor = [0]
    err = metadata_cursor_advance_checked(cursor, I64_MAX + 1, GGUF_TYPE_UINT8, 0, start, span, end)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0

    err = metadata_cursor_advance_checked(cursor, 100, GGUF_TYPE_ARRAY, I64_MAX + 1, start, span, end)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0


def test_typed_span_overflow_edges() -> None:
    cursor = [0]
    start = [0]
    span = [0]
    end = [0]

    # STRING span = 8 + payload must stay inside signed-I64 metadata cursor math.
    err = metadata_cursor_advance_checked(cursor, I64_MAX, GGUF_TYPE_STRING, I64_MAX - 7, start, span, end)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0

    # ARRAY span = 12 + payload must stay inside signed-I64 metadata cursor math.
    err = metadata_cursor_advance_checked(cursor, I64_MAX, GGUF_TYPE_ARRAY, I64_MAX - 11, start, span, end)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0


def test_fuzz_reference() -> None:
    rng = random.Random(20260422_1137)
    value_types = [
        GGUF_TYPE_UINT8,
        GGUF_TYPE_INT8,
        GGUF_TYPE_UINT16,
        GGUF_TYPE_INT16,
        GGUF_TYPE_UINT32,
        GGUF_TYPE_INT32,
        GGUF_TYPE_FLOAT32,
        GGUF_TYPE_BOOL,
        GGUF_TYPE_STRING,
        GGUF_TYPE_ARRAY,
        GGUF_TYPE_UINT64,
        GGUF_TYPE_INT64,
        GGUF_TYPE_FLOAT64,
        99,
    ]

    for _ in range(8000):
        table_end = rng.randint(0, 1 << 20)
        cursor0 = rng.randint(0, table_end + 16)
        payload = rng.randint(0, 1 << 20)
        value_type = rng.choice(value_types)

        cursor_ref = [cursor0]
        start = [0xAA]
        span = [0xBB]
        end = [0xCC]

        err = metadata_cursor_advance_checked(
            cursor_ref,
            table_end,
            value_type,
            payload,
            start,
            span,
            end,
        )

        if err == GGUF_META_TABLE_OK:
            assert start[0] == cursor0
            assert end[0] == cursor0 + span[0]
            assert end[0] <= table_end
            assert cursor_ref[0] == end[0]
        else:
            assert cursor_ref[0] == cursor0
            assert start[0] == 0xAA
            assert span[0] == 0xBB
            assert end[0] == 0xCC


def run() -> None:
    test_source_contains_iq1137_functions()
    test_fixed_width_type_accounting()
    test_string_and_array_variable_payload_accounting()
    test_no_partial_writes_on_failure()
    test_null_and_overflow_contracts()
    test_typed_span_overflow_edges()
    test_fuzz_reference()
    print("gguf_metadata_cursor_advance_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
