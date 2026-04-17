#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU8QuadChecked semantics."""

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


def gguf_metadata_read_u8_triple_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u8_quad_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
        or out_fourth_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]
    fourth = [0]

    err = gguf_metadata_read_u8_triple_checked(
        buf, buf_nbytes, cur, table_end, first, second, third
    )
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [1]
    out_first = [11]
    out_second = [22]
    out_third = [33]
    out_fourth = [44]

    assert (
        gguf_metadata_read_u8_quad_checked(
            None,
            8,
            cursor,
            8,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [1]
    assert out_first == [11]
    assert out_second == [22]
    assert out_third == [33]
    assert out_fourth == [44]


def test_fourth_scalar_fail_does_not_commit_prior_lanes_or_cursor() -> None:
    buf = [0xA1, 0xA2, 0xA3]
    cursor = [0]
    out_first = [0x11]
    out_second = [0x22]
    out_third = [0x33]
    out_fourth = [0x44]

    err = gguf_metadata_read_u8_quad_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x11]
    assert out_second == [0x22]
    assert out_third == [0x33]
    assert out_fourth == [0x44]


def test_success_reads_four_bytes_and_advances() -> None:
    buf = [9, 10, 20, 30, 40, 50]
    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u8_quad_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == 10
    assert out_second[0] == 20
    assert out_third[0] == 30
    assert out_fourth[0] == 40
    assert cursor[0] == 5


def test_randomized_parity() -> None:
    rng = random.Random(20260417_259)

    for _ in range(5000):
        n = rng.randint(1, 256)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [77]
        out_second = [88]
        out_third = [99]
        out_fourth = [111]

        err = gguf_metadata_read_u8_quad_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )

        if cursor0 + 4 <= table_end and cursor0 + 4 <= n:
            assert err == GGUF_META_TABLE_OK
            assert out_first[0] == buf[cursor0]
            assert out_second[0] == buf[cursor0 + 1]
            assert out_third[0] == buf[cursor0 + 2]
            assert out_fourth[0] == buf[cursor0 + 3]
            assert cursor[0] == cursor0 + 4
        else:
            assert err in (
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
                GGUF_META_TABLE_ERR_BAD_PARAM,
                GGUF_META_TABLE_ERR_OVERFLOW,
            )
            assert out_first[0] == 77
            assert out_second[0] == 88
            assert out_third[0] == 99
            assert out_fourth[0] == 111
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_fourth_scalar_fail_does_not_commit_prior_lanes_or_cursor()
    test_success_reads_four_bytes_and_advances()
    test_randomized_parity()
    print("gguf_metadata_read_u8_quad_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
