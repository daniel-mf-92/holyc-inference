#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI32PairCheckedNoPartial semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1


def _to_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - (1 << 32)
    return value


def _le_i32(value: int) -> list[int]:
    raw = value & 0xFFFFFFFF
    return [
        raw & 0xFF,
        (raw >> 8) & 0xFF,
        (raw >> 16) & 0xFF,
        (raw >> 24) & 0xFF,
    ]


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
    if cursor > ((1 << 64) - 1) - need:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

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


def gguf_metadata_read_u32le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    b0 = [0]
    b1 = [0]
    b2 = [0]
    b3 = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b0)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b1)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b2)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b3)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = b0[0] | (b1[0] << 8) | (b2[0] << 16) | (b3[0] << 24)
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    raw_u32 = [0]
    cur = [cursor_ref[0]]

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, raw_u32)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = _to_i32(raw_u32[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_pair_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_pair_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]

    err = gguf_metadata_read_i32_pair_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [7]
    out_first = [101]
    out_second = [202]

    err = gguf_metadata_read_i32_pair_checked_no_partial(
        None,
        32,
        cursor,
        32,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [7]
    assert out_first == [101]
    assert out_second == [202]


def test_success_reads_two_i32_and_advances() -> None:
    first = -2147483648
    second = 2147483647
    buf = [0xEE] + _le_i32(first) + _le_i32(second) + [0x7A]

    cursor = [1]
    out_first = [0]
    out_second = [0]

    err = gguf_metadata_read_i32_pair_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [first]
    assert out_second == [second]
    assert cursor == [9]


def test_truncation_frontiers_keep_cursor_and_outputs() -> None:
    first = -42
    second = 31337
    full = _le_i32(first) + _le_i32(second)

    for n in range(0, len(full)):
        buf = full[:n]
        cur = [0]
        out_first = [0x11111111]
        out_second = [0x22222222]

        err = gguf_metadata_read_i32_pair_checked_no_partial(
            buf,
            len(buf),
            cur,
            len(buf),
            out_first,
            out_second,
        )
        assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        assert cur == [0]
        assert out_first == [0x11111111]
        assert out_second == [0x22222222]


def test_explicit_end_truncation_frontiers_keep_cursor_and_outputs() -> None:
    first = -123456789
    second = 987654321
    buf = _le_i32(first) + _le_i32(second) + [0xAA, 0xBB, 0xCC]

    for table_end in range(0, 8):
        cur = [0]
        out_first = [0x55555555]
        out_second = [0xAAAAAAAA]

        err = gguf_metadata_read_i32_pair_checked_no_partial(
            buf,
            len(buf),
            cur,
            table_end,
            out_first,
            out_second,
        )
        assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        assert cur == [0]
        assert out_first == [0x55555555]
        assert out_second == [0xAAAAAAAA]


def test_randomized_parity_vs_checked_core_with_explicit_end() -> None:
    rng = random.Random(20260418_349)

    for _ in range(5000):
        n = rng.randint(0, 256)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cur_no_partial = [cursor0]
        cur_checked = [cursor0]
        out_no_partial = [rng.randint(-(1 << 31), (1 << 31) - 1)]
        out_no_partial_2 = [rng.randint(-(1 << 31), (1 << 31) - 1)]
        out_checked = [out_no_partial[0]]
        out_checked_2 = [out_no_partial_2[0]]

        err_no_partial = gguf_metadata_read_i32_pair_checked_no_partial(
            buf,
            n,
            cur_no_partial,
            table_end,
            out_no_partial,
            out_no_partial_2,
        )
        err_checked = gguf_metadata_read_i32_pair_checked(
            buf,
            n,
            cur_checked,
            table_end,
            out_checked,
            out_checked_2,
        )

        assert err_no_partial == err_checked
        assert cur_no_partial == cur_checked
        assert out_no_partial == out_checked
        assert out_no_partial_2 == out_checked_2


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_success_reads_two_i32_and_advances()
    test_truncation_frontiers_keep_cursor_and_outputs()
    test_explicit_end_truncation_frontiers_keep_cursor_and_outputs()
    test_randomized_parity_vs_checked_core_with_explicit_end()
    print("gguf_metadata_read_i32_pair_checked_no_partial=ok")


if __name__ == "__main__":
    run()
