#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadStringLenU64Checked semantics."""

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
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def encode_u64le(x: int) -> list[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_len = [123]

    assert (
        gguf_metadata_read_string_len_u64_checked(None, 16, cursor, 16, out_len)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3
    assert out_len[0] == 123

    assert (
        gguf_metadata_read_string_len_u64_checked([0] * 16, 16, None, 16, out_len)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_len[0] == 123

    assert (
        gguf_metadata_read_string_len_u64_checked([0] * 16, 16, cursor, 16, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3


def test_short_header_and_payload_bounds_fail_without_mutation() -> None:
    out_len = [77]

    for table_end in range(0, 8):
        buf = [0xAA] * 32
        cursor = [0]
        err = gguf_metadata_read_string_len_u64_checked(
            buf, len(buf), cursor, table_end, out_len
        )
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out_len[0] == 77

    # Header decodes (len=16), but table payload window is too short.
    buf = encode_u64le(16) + [0x42] * 32
    cursor = [0]
    err = gguf_metadata_read_string_len_u64_checked(buf, len(buf), cursor, 12, out_len)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 0
    assert out_len[0] == 77


def test_success_returns_len_and_advances_to_payload_start() -> None:
    text = b"temple"
    payload = encode_u64le(len(text)) + list(text) + [0x99, 0x88]

    cursor = [0]
    out_len = [0]

    err = gguf_metadata_read_string_len_u64_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        out_len,
    )

    assert err == GGUF_META_TABLE_OK
    assert out_len[0] == len(text)
    assert cursor[0] == 8


def test_rejects_len_above_global_cap() -> None:
    over = GGUF_MAX_STRING_BYTES + 1
    buf = encode_u64le(over) + [0x11] * over

    cursor = [0]
    out_len = [999]

    # Give enough table/buffer space so only the max-length gate rejects.
    err = gguf_metadata_read_string_len_u64_checked(buf, len(buf), cursor, len(buf), out_len)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_len[0] == 999


def test_randomized_parity() -> None:
    rng = random.Random(20260417_200)

    for _ in range(4000):
        n = rng.randint(16, 512)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_len = [0xBEEF]

        err = gguf_metadata_read_string_len_u64_checked(buf, n, cursor, table_end, out_len)

        can_header = cursor0 + 8 <= table_end and cursor0 + 8 <= n
        if not can_header:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert cursor[0] == cursor0
            assert out_len[0] == 0xBEEF
            continue

        raw_len = 0
        for i in range(8):
            raw_len |= buf[cursor0 + i] << (8 * i)

        payload_start = cursor0 + 8

        if raw_len > I64_MAX:
            assert err == GGUF_META_TABLE_ERR_OVERFLOW
            assert cursor[0] == cursor0
            assert out_len[0] == 0xBEEF
            continue

        if payload_start + raw_len > table_end:
            assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            assert cursor[0] == cursor0
            assert out_len[0] == 0xBEEF
            continue

        if payload_start + raw_len > n:
            assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            assert cursor[0] == cursor0
            assert out_len[0] == 0xBEEF
            continue

        if raw_len > GGUF_MAX_STRING_BYTES:
            assert err == GGUF_META_TABLE_ERR_BAD_PARAM
            assert cursor[0] == cursor0
            assert out_len[0] == 0xBEEF
            continue

        assert err == GGUF_META_TABLE_OK
        assert cursor[0] == payload_start
        assert out_len[0] == raw_len


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_header_and_payload_bounds_fail_without_mutation()
    test_success_returns_len_and_advances_to_payload_start()
    test_rejects_len_above_global_cap()
    test_randomized_parity()
    print("gguf_metadata_read_string_len_u64_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
