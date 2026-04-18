#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadU8PairCheckedDefault."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

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


def gguf_metadata_read_u8_pair_checked(
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

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u8_pair_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u8_pair_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [77]
    out_second = [88]

    assert (
        gguf_metadata_read_u8_pair_checked_default(
            None,
            16,
            cursor,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [5]
    assert out_first == [77]
    assert out_second == [88]


def test_success_reads_pair_and_advances() -> None:
    buf = [0xEE, 0x11, 0x22, 0xDD]
    cursor = [1]
    out_first = [0]
    out_second = [0]

    err = gguf_metadata_read_u8_pair_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [0x11]
    assert out_second == [0x22]
    assert cursor == [3]


def test_truncated_default_end_no_partial_write() -> None:
    buf = [0xAA, 0xBB, 0xCC]
    cursor = [2]
    out_first = [0x44]
    out_second = [0x55]

    err = gguf_metadata_read_u8_pair_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [2]
    assert out_first == [0x44]
    assert out_second == [0x55]


def test_default_matches_checked_with_explicit_end_randomized() -> None:
    rng = random.Random(20260418_329)

    for _ in range(5000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        cur_default = [cursor0]
        first_default = [111]
        second_default = [222]

        cur_checked = [cursor0]
        first_checked = [111]
        second_checked = [222]

        err_default = gguf_metadata_read_u8_pair_checked_default(
            buf,
            n,
            cur_default,
            first_default,
            second_default,
        )
        err_checked = gguf_metadata_read_u8_pair_checked(
            buf,
            n,
            cur_checked,
            n,
            first_checked,
            second_checked,
        )

        assert err_default == err_checked
        assert cur_default == cur_checked
        assert first_default == first_checked
        assert second_default == second_checked


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_success_reads_pair_and_advances()
    test_truncated_default_end_no_partial_write()
    test_default_matches_checked_with_explicit_end_randomized()
    print("gguf_metadata_read_u8_pair_checked_default_reference_checks=ok")


if __name__ == "__main__":
    run()
