#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadStringU64HeaderAndBoundsChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
GGUF_MAX_STRING_BYTES = 1 << 20


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

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None

    return GGUF_META_TABLE_OK, next_cursor


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
    bytes_out = [[0] for _ in range(8)]

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, bytes_out[i])
        if err != GGUF_META_TABLE_OK:
            return err

    value = 0
    for i in range(8):
        value |= bytes_out[i][0] << (8 * i)

    out_value_ref[0] = value
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


def le_u64(x: int) -> list[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    buf = le_u64(3) + [65, 66, 67]
    cursor = [0]
    out_len = [111]
    out_payload_end = [222]

    assert (
        gguf_metadata_read_string_u64_header_and_bounds_checked(
            None, len(buf), cursor, len(buf), out_len, out_payload_end
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [0]
    assert out_len == [111]
    assert out_payload_end == [222]

    assert (
        gguf_metadata_read_string_u64_header_and_bounds_checked(
            buf, len(buf), cursor, len(buf), None, out_payload_end
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [0]
    assert out_payload_end == [222]


def test_truncated_u64_header_and_payload_window_bounds() -> None:
    cursor = [0]
    out_len = [7]
    out_payload_end = [9]
    buf_short = [1, 2, 3, 4, 5, 6, 7]

    assert (
        gguf_metadata_read_string_u64_header_and_bounds_checked(
            buf_short, len(buf_short), cursor, len(buf_short), out_len, out_payload_end
        )
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [0]
    assert out_len == [7]
    assert out_payload_end == [9]

    # Header decodes, but payload exceeds table_end.
    buf = le_u64(5) + [11, 22, 33]
    cursor = [0]
    out_len = [7]
    out_payload_end = [9]
    assert (
        gguf_metadata_read_string_u64_header_and_bounds_checked(
            buf, len(buf), cursor, 10, out_len, out_payload_end
        )
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [0]
    assert out_len == [7]
    assert out_payload_end == [9]


def test_rejects_oversized_string_len() -> None:
    too_big = GGUF_MAX_STRING_BYTES + 1
    buf = le_u64(too_big) + [0] * too_big

    cursor = [0]
    out_len = [1]
    out_payload_end = [2]

    err = gguf_metadata_read_string_u64_header_and_bounds_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_len,
        out_payload_end,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_len == [1]
    assert out_payload_end == [2]


def test_success_decodes_len_sets_payload_end_and_advances_to_payload() -> None:
    buf = le_u64(4) + [ord("T"), ord("E"), ord("R"), ord("R"), 0x99]
    cursor = [0]
    out_len = [0]
    out_payload_end = [0]

    err = gguf_metadata_read_string_u64_header_and_bounds_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_len,
        out_payload_end,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_len[0] == 4
    assert cursor[0] == 8
    assert out_payload_end[0] == 12


def test_randomized_parity() -> None:
    rng = random.Random(20260417_204)

    for _ in range(4000):
        n = rng.randint(8, 300)
        buf = [rng.randint(0, 255) for _ in range(n)]

        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)
        cursor = [cursor0]

        out_len = [0xAAAA]
        out_payload_end = [0xBBBB]

        err = gguf_metadata_read_string_u64_header_and_bounds_checked(
            buf,
            n,
            cursor,
            table_end,
            out_len,
            out_payload_end,
        )

        if cursor0 + 8 <= table_end and cursor0 + 8 <= n:
            str_len = 0
            for i in range(8):
                str_len |= buf[cursor0 + i] << (8 * i)

            payload_end = cursor0 + 8 + str_len
            if (
                str_len <= GGUF_MAX_STRING_BYTES
                and payload_end <= table_end
                and payload_end <= n
            ):
                assert err == GGUF_META_TABLE_OK
                assert out_len[0] == str_len
                assert cursor[0] == cursor0 + 8
                assert out_payload_end[0] == payload_end
                continue

        assert err in (
            GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
            GGUF_META_TABLE_ERR_BAD_PARAM,
            GGUF_META_TABLE_ERR_OVERFLOW,
        )
        assert cursor[0] == cursor0
        assert out_len[0] == 0xAAAA
        assert out_payload_end[0] == 0xBBBB


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_truncated_u64_header_and_payload_window_bounds()
    test_rejects_oversized_string_len()
    test_success_decodes_len_sets_payload_end_and_advances_to_payload()
    test_randomized_parity()
    print("gguf_metadata_read_string_u64_header_and_bounds_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
