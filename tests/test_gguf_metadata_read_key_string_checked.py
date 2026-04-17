#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadKeyStringChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

GGUF_MAX_STRING_BYTES = 1 << 20
I64_MAX = (1 << 63) - 1


def gguf_metadata_cursor_can_advance_checked(
    cursor: int,
    need: int,
    table_end: int,
) -> tuple[int, int | None]:
    if cursor > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if need > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if cursor > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None
    if cursor + need > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None
    return GGUF_META_TABLE_OK, cursor + need


def gguf_metadata_read_u8_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cur = cursor_ref[0]
    err, next_cursor = gguf_metadata_cursor_can_advance_checked(cur, 1, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert next_cursor is not None
    if next_cursor > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    out_value_ref[0] = buf[cur]
    cursor_ref[0] = next_cursor
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    b = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= b[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_u64_header_and_bounds_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_len_ref: list[int] | None,
    out_payload_end_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_len_ref is None
        or out_payload_end_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    str_len = [0]

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, str_len)
    if err != GGUF_META_TABLE_OK:
        return err

    err, payload_end = gguf_metadata_cursor_can_advance_checked(cur[0], str_len[0], table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert payload_end is not None
    if payload_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    if str_len[0] > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_len_ref[0] = str_len[0]
    out_payload_end_ref[0] = payload_end
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_len_u64_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_len_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_len_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    str_len = [0]
    payload_end = [0]

    err = gguf_metadata_read_string_u64_header_and_bounds_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        str_len,
        payload_end,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_len_ref[0] = str_len[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_bytes_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    str_len: int,
    out_bytes: list[int] | None,
    out_nbytes: int,
) -> int:
    if buf is None or cursor_ref is None or out_bytes is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if str_len > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if out_nbytes < str_len:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    cur = cursor_ref[0]
    err, next_cursor = gguf_metadata_cursor_can_advance_checked(cur, str_len, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert next_cursor is not None
    if next_cursor > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    for i in range(str_len):
        out_bytes[i] = buf[cur + i]

    cursor_ref[0] = next_cursor
    return GGUF_META_TABLE_OK


def gguf_metadata_read_key_string_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_key_bytes: list[int] | None,
    out_key_nbytes: int,
    out_key_len_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_key_bytes is None
        or out_key_len_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    key_len = [0]

    err = gguf_metadata_read_string_len_u64_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        key_len,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if key_len[0] == 0:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if out_key_nbytes == 0:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if key_len[0] >= out_key_nbytes:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    for i in range(key_len[0]):
        if buf[cur[0] + i] == 0:
            return GGUF_META_TABLE_ERR_BAD_PARAM

    err = gguf_metadata_read_string_bytes_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        key_len[0],
        out_key_bytes,
        out_key_nbytes - 1,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_key_bytes[key_len[0]] = 0
    out_key_len_ref[0] = key_len[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def le_u64(x: int) -> list[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    buf = le_u64(3) + [ord("k"), ord("e"), ord("y")]
    cursor = [0]
    out_key = [0x77] * 8
    out_len = [123]

    assert (
        gguf_metadata_read_key_string_checked(
            None, len(buf), cursor, len(buf), out_key, len(out_key), out_len
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [0]
    assert out_key == [0x77] * 8
    assert out_len == [123]

    assert (
        gguf_metadata_read_key_string_checked(
            buf, len(buf), cursor, len(buf), out_key, len(out_key), None
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [0]
    assert out_key == [0x77] * 8


def test_rejects_empty_key_and_embedded_nul() -> None:
    # Empty key.
    buf = le_u64(0)
    cursor = [0]
    out_key = [0x55] * 4
    out_len = [9]

    assert (
        gguf_metadata_read_key_string_checked(
            buf, len(buf), cursor, len(buf), out_key, len(out_key), out_len
        )
        == GGUF_META_TABLE_ERR_BAD_PARAM
    )
    assert cursor == [0]
    assert out_key == [0x55] * 4
    assert out_len == [9]

    # Embedded NUL inside non-empty key.
    key_bytes = [ord("a"), 0, ord("b")]
    buf = le_u64(len(key_bytes)) + key_bytes
    cursor = [0]
    out_key = [0x44] * 8
    out_len = [7]

    assert (
        gguf_metadata_read_key_string_checked(
            buf, len(buf), cursor, len(buf), out_key, len(out_key), out_len
        )
        == GGUF_META_TABLE_ERR_BAD_PARAM
    )
    assert cursor == [0]
    assert out_key == [0x44] * 8
    assert out_len == [7]


def test_rejects_too_small_output_buffer() -> None:
    key_bytes = [ord("k"), ord("e"), ord("y")]
    buf = le_u64(len(key_bytes)) + key_bytes

    cursor = [0]
    out_key = [0x33] * 3
    out_len = [0]

    assert (
        gguf_metadata_read_key_string_checked(
            buf, len(buf), cursor, len(buf), out_key, len(out_key), out_len
        )
        == GGUF_META_TABLE_ERR_BAD_PARAM
    )
    assert cursor == [0]
    assert out_key == [0x33] * 3
    assert out_len == [0]


def test_success_reads_key_and_appends_nul() -> None:
    key = b"general.architecture"
    buf = le_u64(len(key)) + list(key) + [0x99, 0xAB]

    cursor = [0]
    out_key = [0] * 64
    out_len = [0]

    err = gguf_metadata_read_key_string_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_key,
        len(out_key),
        out_len,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_len == [len(key)]
    assert cursor == [8 + len(key)]
    assert bytes(out_key[: len(key)]) == key
    assert out_key[len(key)] == 0


def test_randomized_valid_keys_and_bounds() -> None:
    rng = random.Random(0xB16B00B5)

    for _ in range(300):
        key_len = rng.randint(1, 64)
        key = [rng.randint(1, 255) for _ in range(key_len)]
        trailer = [rng.randint(0, 255) for _ in range(rng.randint(0, 16))]
        buf = le_u64(key_len) + key + trailer

        cursor = [0]
        out_key = [0xA5] * (key_len + 1 + rng.randint(0, 8))
        out_len = [999]

        err = gguf_metadata_read_key_string_checked(
            buf,
            len(buf),
            cursor,
            8 + key_len,
            out_key,
            len(out_key),
            out_len,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_len == [key_len]
        assert cursor == [8 + key_len]
        assert out_key[key_len] == 0
        assert out_key[:key_len] == key


def test_table_or_buffer_bounds_failure_keeps_outputs_unchanged() -> None:
    key = b"abcd"
    buf = le_u64(len(key)) + list(key)

    # table_end excludes final payload byte.
    cursor = [0]
    out_key = [0x11] * 16
    out_len = [42]
    assert (
        gguf_metadata_read_key_string_checked(
            buf,
            len(buf),
            cursor,
            8 + len(key) - 1,
            out_key,
            len(out_key),
            out_len,
        )
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [0]
    assert out_key == [0x11] * 16
    assert out_len == [42]

    # buf_nbytes excludes final payload byte.
    cursor = [0]
    out_key = [0x22] * 16
    out_len = [24]
    assert (
        gguf_metadata_read_key_string_checked(
            buf,
            len(buf) - 1,
            cursor,
            len(buf),
            out_key,
            len(out_key),
            out_len,
        )
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [0]
    assert out_key == [0x22] * 16
    assert out_len == [24]
