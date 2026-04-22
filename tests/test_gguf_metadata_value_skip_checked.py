#!/usr/bin/env python3
"""Harness for GGUFMetadataValueSkipChecked (IQ-1152)."""

from __future__ import annotations

from pathlib import Path

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
GGUF_MAX_ARRAY_ELEMS = 1 << 24

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

_FIXED_WIDTH = {
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
}


def _u32le(v: int) -> bytes:
    return int(v & 0xFFFFFFFF).to_bytes(4, "little", signed=False)


def _u64le(v: int) -> bytes:
    return int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)


def _string_payload(data: bytes) -> bytes:
    return _u64le(len(data)) + data


def _ensure_advance(cur: int, need: int, table_end: int, buf_nbytes: int) -> tuple[int, int | None]:
    nxt = cur + need
    if nxt > table_end or nxt > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None
    return GGUF_META_TABLE_OK, nxt


def _cursor_typed_span_bytes(value_type: int, variable_payload_bytes: int) -> tuple[int, int | None]:
    if variable_payload_bytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    if value_type == GGUF_TYPE_STRING:
        if variable_payload_bytes > I64_MAX - 8:
            return GGUF_META_TABLE_ERR_OVERFLOW, None
        return GGUF_META_TABLE_OK, 8 + variable_payload_bytes

    if value_type == GGUF_TYPE_ARRAY:
        if variable_payload_bytes > I64_MAX - 12:
            return GGUF_META_TABLE_ERR_OVERFLOW, None
        return GGUF_META_TABLE_OK, 12 + variable_payload_bytes

    if variable_payload_bytes != 0:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None

    width = _FIXED_WIDTH.get(value_type)
    if width is None:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None
    return GGUF_META_TABLE_OK, width


def _cursor_advance_checked(
    cursor: int,
    table_end: int,
    buf_nbytes: int,
    value_type: int,
    variable_payload_bytes: int,
) -> tuple[int, int | None]:
    if table_end > I64_MAX or buf_nbytes > I64_MAX or cursor > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    err, span_bytes = _cursor_typed_span_bytes(value_type, variable_payload_bytes)
    if err != GGUF_META_TABLE_OK:
        return err, None
    assert span_bytes is not None

    if cursor > I64_MAX - span_bytes:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    end = cursor + span_bytes

    if end > table_end or end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None
    return GGUF_META_TABLE_OK, end


def _read_u32le(buf: list[int], cur: int, table_end: int, buf_nbytes: int) -> tuple[int, int | None, int | None]:
    err, nxt = _ensure_advance(cur, 4, table_end, buf_nbytes)
    if err != GGUF_META_TABLE_OK:
        return err, None, None
    assert nxt is not None
    value = buf[cur] | (buf[cur + 1] << 8) | (buf[cur + 2] << 16) | (buf[cur + 3] << 24)
    return GGUF_META_TABLE_OK, value, nxt


def _read_u64le(buf: list[int], cur: int, table_end: int, buf_nbytes: int) -> tuple[int, int | None, int | None]:
    err, nxt = _ensure_advance(cur, 8, table_end, buf_nbytes)
    if err != GGUF_META_TABLE_OK:
        return err, None, None
    assert nxt is not None
    value = 0
    for i in range(8):
        value |= buf[cur + i] << (8 * i)
    return GGUF_META_TABLE_OK, value, nxt


def _read_array_header(
    buf: list[int],
    cur: int,
    table_end: int,
    buf_nbytes: int,
) -> tuple[int, int | None, int | None, int | None]:
    err, elem_type, after_type = _read_u32le(buf, cur, table_end, buf_nbytes)
    if err != GGUF_META_TABLE_OK:
        return err, None, None, None
    assert elem_type is not None and after_type is not None

    if elem_type == GGUF_TYPE_ARRAY or elem_type < GGUF_TYPE_UINT8 or elem_type > GGUF_TYPE_FLOAT64:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None, None, None

    err, elem_count, after_count = _read_u64le(buf, after_type, table_end, buf_nbytes)
    if err != GGUF_META_TABLE_OK:
        return err, None, None, None
    assert elem_count is not None and after_count is not None

    if elem_count > GGUF_MAX_ARRAY_ELEMS:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None, None, None

    return GGUF_META_TABLE_OK, elem_type, elem_count, after_count


def gguf_metadata_value_skip_checked_reference(
    *,
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    value_type: int,
) -> int:
    if buf is None or cursor_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if buf_nbytes > I64_MAX or table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cur = cursor_ref[0]
    if cur > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    if value_type in _FIXED_WIDTH:
        err, end = _cursor_advance_checked(cur, table_end, buf_nbytes, value_type, 0)
        if err != GGUF_META_TABLE_OK:
            return err
        assert end is not None
        cursor_ref[0] = end
        return GGUF_META_TABLE_OK

    if value_type == GGUF_TYPE_STRING:
        err, str_len, payload_cursor = _read_u64le(buf, cur, table_end, buf_nbytes)
        if err != GGUF_META_TABLE_OK:
            return err
        assert str_len is not None and payload_cursor is not None

        err, end = _cursor_advance_checked(cur, table_end, buf_nbytes, GGUF_TYPE_STRING, str_len)
        if err != GGUF_META_TABLE_OK:
            return err
        assert end is not None

        if payload_cursor > end or payload_cursor + str_len != end:
            return GGUF_META_TABLE_ERR_OVERFLOW

        cursor_ref[0] = end
        return GGUF_META_TABLE_OK

    if value_type == GGUF_TYPE_ARRAY:
        err, elem_type, elem_count, payload_cursor = _read_array_header(buf, cur, table_end, buf_nbytes)
        if err != GGUF_META_TABLE_OK:
            return err
        assert elem_type is not None and elem_count is not None and payload_cursor is not None

        if elem_type in _FIXED_WIDTH:
            width = _FIXED_WIDTH[elem_type]
            payload_bytes = elem_count * width
            if payload_bytes > I64_MAX:
                return GGUF_META_TABLE_ERR_OVERFLOW
            payload_end = payload_cursor + payload_bytes
            if payload_end > table_end or payload_end > buf_nbytes:
                return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            if payload_end < payload_cursor:
                return GGUF_META_TABLE_ERR_OVERFLOW

            err, end = _cursor_advance_checked(cur, table_end, buf_nbytes, GGUF_TYPE_ARRAY, payload_bytes)
            if err != GGUF_META_TABLE_OK:
                return err
            assert end is not None
            if end != payload_end:
                return GGUF_META_TABLE_ERR_OVERFLOW

            cursor_ref[0] = end
            return GGUF_META_TABLE_OK

        if elem_type == GGUF_TYPE_STRING:
            lane_payload_cursor = payload_cursor
            for _ in range(elem_count):
                err, lane_len, lane_after_len = _read_u64le(buf, lane_payload_cursor, table_end, buf_nbytes)
                if err != GGUF_META_TABLE_OK:
                    return err
                assert lane_len is not None and lane_after_len is not None

                err, lane_end = _cursor_advance_checked(
                    lane_payload_cursor,
                    table_end,
                    buf_nbytes,
                    GGUF_TYPE_STRING,
                    lane_len,
                )
                if err != GGUF_META_TABLE_OK:
                    return err
                assert lane_end is not None

                if lane_after_len > lane_end or lane_after_len + lane_len != lane_end:
                    return GGUF_META_TABLE_ERR_OVERFLOW

                lane_payload_cursor = lane_end

            payload_bytes = lane_payload_cursor - payload_cursor
            err, end = _cursor_advance_checked(cur, table_end, buf_nbytes, GGUF_TYPE_ARRAY, payload_bytes)
            if err != GGUF_META_TABLE_OK:
                return err
            assert end is not None
            if end != lane_payload_cursor:
                return GGUF_META_TABLE_ERR_OVERFLOW

            cursor_ref[0] = end
            return GGUF_META_TABLE_OK

        return GGUF_META_TABLE_ERR_BAD_PARAM

    return GGUF_META_TABLE_ERR_BAD_PARAM


def test_source_contains_iq1152_functions() -> None:
    source = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")

    sig = "I32 GGUFMetadataValueSkipChecked(U8 *buf,"
    sig_def = "I32 GGUFMetadataValueSkipChecked(U8 *buf,\n                                 U64 buf_nbytes,\n                                 U64 *cursor,\n                                 U64 table_end,\n                                 U32 value_type)\n{"
    alias_sig = "I32 GGUFMetadataSkipValuePayloadChecked(U8 *buf,"
    assert sig in source
    assert sig_def in source
    assert alias_sig in source

    body = source.split(sig_def, 1)[1].split("// Backward-compatible alias retained", 1)[0]
    assert "GGUFMetadataCursorAdvanceChecked(&cur," in body
    assert "case GGUF_TYPE_ARRAY:" in body
    assert "GGUFMetadataReadArrayHeaderChecked(" in body
    assert "case GGUF_TYPE_STRING:" in body
    assert "GGUFMetadataCursorAdvanceChecked(&lane_start," in body


def test_scalar_skip_widths() -> None:
    buf = list(range(128))
    cursor = [5]

    err = gguf_metadata_value_skip_checked_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_UINT16,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 7

    err = gguf_metadata_value_skip_checked_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_FLOAT64,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 15


def test_string_skip_and_no_partial() -> None:
    payload = _string_payload(b"terry")
    buf = [0xAB] * 16 + list(payload) + [0xCD] * 16

    cursor = [16]
    err = gguf_metadata_value_skip_checked_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_STRING,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 16 + len(payload)

    truncated = buf[: (16 + len(payload) - 1)]
    cursor = [16]
    err = gguf_metadata_value_skip_checked_reference(
        buf=truncated,
        buf_nbytes=len(truncated),
        cursor_ref=cursor,
        table_end=len(truncated),
        value_type=GGUF_TYPE_STRING,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 16


def test_array_fixed_elem_skip() -> None:
    elems = [1, 2, 3, 4]
    payload = _u32le(GGUF_TYPE_UINT16) + _u64le(len(elems)) + b"".join(x.to_bytes(2, "little") for x in elems)
    buf = [0] * 11 + list(payload) + [0] * 7

    cursor = [11]
    err = gguf_metadata_value_skip_checked_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_ARRAY,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 11 + len(payload)


def test_array_string_elem_skip() -> None:
    strings = [b"ab", b"c", b"holy"]
    payload = _u32le(GGUF_TYPE_STRING) + _u64le(len(strings)) + b"".join(_string_payload(s) for s in strings)
    buf = [0] * 9 + list(payload) + [0] * 9

    cursor = [9]
    err = gguf_metadata_value_skip_checked_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_ARRAY,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 9 + len(payload)


def test_null_and_overflow_contracts() -> None:
    cursor = [0]

    err = gguf_metadata_value_skip_checked_reference(
        buf=None,
        buf_nbytes=0,
        cursor_ref=cursor,
        table_end=0,
        value_type=GGUF_TYPE_UINT8,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR

    err = gguf_metadata_value_skip_checked_reference(
        buf=[0],
        buf_nbytes=I64_MAX + 1,
        cursor_ref=cursor,
        table_end=1,
        value_type=GGUF_TYPE_UINT8,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW

    cursor = [I64_MAX + 1]
    err = gguf_metadata_value_skip_checked_reference(
        buf=[0],
        buf_nbytes=1,
        cursor_ref=cursor,
        table_end=1,
        value_type=GGUF_TYPE_UINT8,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
