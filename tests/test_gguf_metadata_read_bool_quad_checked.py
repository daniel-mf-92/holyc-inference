#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadBoolQuadChecked semantics."""

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

    out_value_ref[0] = raw[0] != 0
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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_first = [True]
    out_second = [False]
    out_third = [True]
    out_fourth = [False]

    assert (
        gguf_metadata_read_bool_quad_checked(
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
    assert cursor == [3]
    assert out_first == [True]
    assert out_second == [False]
    assert out_third == [True]
    assert out_fourth == [False]


def test_fourth_lane_noncanonical_and_short_read_do_not_commit() -> None:
    cursor = [0]
    out_first = [False]
    out_second = [False]
    out_third = [False]
    out_fourth = [False]

    err = gguf_metadata_read_bool_quad_checked(
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
    err = gguf_metadata_read_bool_quad_checked(
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

    err = gguf_metadata_read_bool_quad_checked(
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


def test_overflow_passthrough() -> None:
    buf = [0, 1, 0, 1, 0, 1]

    cursor = [0]
    out_first = [False]
    out_second = [False]
    out_third = [False]
    out_fourth = [False]
    err = gguf_metadata_read_bool_quad_checked(
        buf,
        I64_MAX + 1,
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [False]
    assert out_third == [False]
    assert out_fourth == [False]

    cursor = [I64_MAX + 1]
    out_first = [True]
    out_second = [True]
    out_third = [True]
    out_fourth = [True]
    err = gguf_metadata_read_bool_quad_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [I64_MAX + 1]
    assert out_first == [True]
    assert out_second == [True]
    assert out_third == [True]
    assert out_fourth == [True]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_255)

    for _ in range(8000):
        n = rng.randint(1, 320)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [True]
        out_second = [False]
        out_third = [True]
        out_fourth = [False]

        err = gguf_metadata_read_bool_quad_checked(
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
            raw_first = buf[cursor0]
            raw_second = buf[cursor0 + 1]
            raw_third = buf[cursor0 + 2]
            raw_fourth = buf[cursor0 + 3]
            if raw_first <= 1 and raw_second <= 1 and raw_third <= 1 and raw_fourth <= 1:
                assert err == GGUF_META_TABLE_OK
                assert out_first[0] == (raw_first == 1)
                assert out_second[0] == (raw_second == 1)
                assert out_third[0] == (raw_third == 1)
                assert out_fourth[0] == (raw_fourth == 1)
                assert cursor[0] == cursor0 + 4
            else:
                assert err == GGUF_META_TABLE_ERR_BAD_PARAM
                assert cursor[0] == cursor0
                assert out_first[0] is True
                assert out_second[0] is False
                assert out_third[0] is True
                assert out_fourth[0] is False
        else:
            if cursor0 + 1 > table_end or cursor0 + 1 > n:
                expected_err = GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            elif buf[cursor0] > 1:
                expected_err = GGUF_META_TABLE_ERR_BAD_PARAM
            elif cursor0 + 2 > table_end or cursor0 + 2 > n:
                expected_err = GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            elif buf[cursor0 + 1] > 1:
                expected_err = GGUF_META_TABLE_ERR_BAD_PARAM
            elif cursor0 + 3 > table_end or cursor0 + 3 > n:
                expected_err = GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            elif buf[cursor0 + 2] > 1:
                expected_err = GGUF_META_TABLE_ERR_BAD_PARAM
            elif cursor0 + 4 > table_end or cursor0 + 4 > n:
                expected_err = GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            else:
                expected_err = GGUF_META_TABLE_ERR_BAD_PARAM
            assert err == expected_err
            assert cursor[0] == cursor0
            assert out_first[0] is True
            assert out_second[0] is False
            assert out_third[0] is True
            assert out_fourth[0] is False


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_lane_noncanonical_and_short_read_do_not_commit()
    test_success_reads_four_bools_and_advances()
    test_overflow_passthrough()
    test_randomized_parity()
    print("ok")
