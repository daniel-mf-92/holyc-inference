#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadBoolTripleCheckedNoPartial semantics."""

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


def gguf_metadata_read_bool_triple_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
    out_third_ref: list[bool] | None,
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
    first = [False]
    second = [False]
    third = [False]

    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_bool_triple_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
    out_third_ref: list[bool] | None,
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
    first = [False]
    second = [False]
    third = [False]

    err = gguf_metadata_read_bool_triple_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
        third,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_first = [True]
    out_second = [False]
    out_third = [True]

    assert (
        gguf_metadata_read_bool_triple_checked_no_partial(
            None,
            8,
            cursor,
            8,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [3]
    assert out_first == [True]
    assert out_second == [False]
    assert out_third == [True]


def test_third_lane_noncanonical_and_short_read_do_not_commit() -> None:
    cursor = [0]
    out_first = [False]
    out_second = [False]
    out_third = [False]

    err = gguf_metadata_read_bool_triple_checked_no_partial(
        [1, 0, 2, 1],
        4,
        cursor,
        4,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [False]
    assert out_third == [False]

    cursor = [1]
    out_first = [True]
    out_second = [True]
    out_third = [True]
    err = gguf_metadata_read_bool_triple_checked_no_partial(
        [9, 1, 0],
        3,
        cursor,
        3,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [1]
    assert out_first == [True]
    assert out_second == [True]
    assert out_third == [True]


def test_success_reads_three_bools_and_advances() -> None:
    buf = [7, 0, 1, 1, 9]
    cursor = [1]
    out_first = [True]
    out_second = [False]
    out_third = [False]

    err = gguf_metadata_read_bool_triple_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [False]
    assert out_second == [True]
    assert out_third == [True]
    assert cursor == [4]


def test_randomized_wrapper_parity_with_checked_triple() -> None:
    rng = random.Random(20260417_282)

    for _ in range(5000):
        n = rng.randint(1, 96)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor_checked = [cursor0]
        out_checked_first = [False]
        out_checked_second = [False]
        out_checked_third = [False]
        err_checked = gguf_metadata_read_bool_triple_checked(
            buf,
            n,
            cursor_checked,
            table_end,
            out_checked_first,
            out_checked_second,
            out_checked_third,
        )

        cursor_wrapped = [cursor0]
        out_wrapped_first = [False]
        out_wrapped_second = [False]
        out_wrapped_third = [False]
        err_wrapped = gguf_metadata_read_bool_triple_checked_no_partial(
            buf,
            n,
            cursor_wrapped,
            table_end,
            out_wrapped_first,
            out_wrapped_second,
            out_wrapped_third,
        )

        assert err_wrapped == err_checked
        assert cursor_wrapped == cursor_checked
        assert out_wrapped_first == out_checked_first
        assert out_wrapped_second == out_checked_second
        assert out_wrapped_third == out_checked_third


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_third_lane_noncanonical_and_short_read_do_not_commit()
    test_success_reads_three_bools_and_advances()
    test_randomized_wrapper_parity_with_checked_triple()
    print("gguf_metadata_read_bool_triple_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
