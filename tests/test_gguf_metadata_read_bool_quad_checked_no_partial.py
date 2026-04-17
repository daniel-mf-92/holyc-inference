#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadBoolQuadCheckedNoPartial semantics."""

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


def gguf_metadata_read_bool_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[bool] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    if raw[0] > 1:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_value_ref[0] = raw[0] == 1
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_bool_quad_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
    out_third_ref: list[bool] | None,
    out_fourth_ref: list[bool] | None,
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
    first = [False]
    second = [False]
    third = [False]
    fourth = [False]

    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_bool_quad_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
    out_third_ref: list[bool] | None,
    out_fourth_ref: list[bool] | None,
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
    first = [False]
    second = [False]
    third = [False]
    fourth = [False]

    err = gguf_metadata_read_bool_quad_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
        third,
        fourth,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out_first = [True]
    out_second = [False]
    out_third = [True]
    out_fourth = [False]

    err = gguf_metadata_read_bool_quad_checked_no_partial(
        None,
        8,
        cursor,
        8,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [2]
    assert out_first == [True]
    assert out_second == [False]
    assert out_third == [True]
    assert out_fourth == [False]


def test_noncanonical_fourth_lane_and_short_read_do_not_commit() -> None:
    cursor = [0]
    out_first = [False]
    out_second = [False]
    out_third = [False]
    out_fourth = [False]

    err = gguf_metadata_read_bool_quad_checked_no_partial(
        [1, 0, 1, 2, 1],
        5,
        cursor,
        5,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [False]
    assert out_third == [False]
    assert out_fourth == [False]

    cursor = [1]
    out_first = [True]
    out_second = [True]
    out_third = [True]
    out_fourth = [True]
    err = gguf_metadata_read_bool_quad_checked_no_partial(
        [7, 1, 0, 1],
        4,
        cursor,
        4,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [1]
    assert out_first == [True]
    assert out_second == [True]
    assert out_third == [True]
    assert out_fourth == [True]


def test_success_reads_four_bools_and_advances() -> None:
    buf = [9, 1, 0, 1, 1, 0]
    cursor = [1]
    out_first = [False]
    out_second = [True]
    out_third = [False]
    out_fourth = [False]

    err = gguf_metadata_read_bool_quad_checked_no_partial(
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
    assert out_first[0] is True
    assert out_second[0] is False
    assert out_third[0] is True
    assert out_fourth[0] is True
    assert cursor[0] == 5


def test_randomized_parity_vs_checked_quad() -> None:
    rng = random.Random(0x268B00)

    for _ in range(700):
        n = rng.randint(0, 32)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, 32)
        cursor0 = rng.randint(0, 32)

        cursor_a = [cursor0]
        cursor_b = [cursor0]
        a1 = [True]
        a2 = [False]
        a3 = [True]
        a4 = [False]
        b1 = [True]
        b2 = [False]
        b3 = [True]
        b4 = [False]

        err_a = gguf_metadata_read_bool_quad_checked_no_partial(
            buf,
            n,
            cursor_a,
            table_end,
            a1,
            a2,
            a3,
            a4,
        )
        err_b = gguf_metadata_read_bool_quad_checked(
            buf,
            n,
            cursor_b,
            table_end,
            b1,
            b2,
            b3,
            b4,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert a1 == b1
        assert a2 == b2
        assert a3 == b3
        assert a4 == b4


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_noncanonical_fourth_lane_and_short_read_do_not_commit()
    test_success_reads_four_bools_and_advances()
    test_randomized_parity_vs_checked_quad()
    print("ok")
