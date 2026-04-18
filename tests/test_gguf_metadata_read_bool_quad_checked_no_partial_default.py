#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadBoolQuadCheckedNoPartialDefault."""

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
    lane = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, lane)
    if err != GGUF_META_TABLE_OK:
        return err

    if lane[0] > 1:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_value_ref[0] = lane[0] == 1
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
    return gguf_metadata_read_bool_quad_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        table_end,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def gguf_metadata_read_bool_quad_checked_no_partial_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
    out_third_ref: list[bool] | None,
    out_fourth_ref: list[bool] | None,
) -> int:
    return gguf_metadata_read_bool_quad_checked_no_partial(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out_first = [True]
    out_second = [False]
    out_third = [True]
    out_fourth = [False]

    err = gguf_metadata_read_bool_quad_checked_no_partial_default(
        None,
        32,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [4]
    assert out_first == [True]
    assert out_second == [False]
    assert out_third == [True]
    assert out_fourth == [False]


def test_success_matches_explicit_no_partial_core() -> None:
    buf = [0x77, 1, 1, 0, 1, 0x88]

    cur_default = [1]
    cur_core = [1]
    out_default = [[False], [False], [False], [False]]
    out_core = [[False], [False], [False], [False]]

    err_default = gguf_metadata_read_bool_quad_checked_no_partial_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_core = gguf_metadata_read_bool_quad_checked_no_partial(
        buf,
        len(buf),
        cur_core,
        len(buf),
        out_core[0],
        out_core[1],
        out_core[2],
        out_core[3],
    )

    assert err_default == err_core == GGUF_META_TABLE_OK
    assert cur_default == cur_core == [5]
    assert out_default == out_core == [[True], [True], [False], [True]]


def test_truncation_bad_bool_and_no_commit_parity() -> None:
    trunc_buf = [1, 0, 1]

    cur_default = [0]
    cur_core = [0]
    out_default = [[False], [True], [False], [True]]
    out_core = [[False], [True], [False], [True]]

    err_default = gguf_metadata_read_bool_quad_checked_no_partial_default(
        trunc_buf,
        len(trunc_buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_core = gguf_metadata_read_bool_quad_checked_no_partial(
        trunc_buf,
        len(trunc_buf),
        cur_core,
        len(trunc_buf),
        out_core[0],
        out_core[1],
        out_core[2],
        out_core[3],
    )

    assert err_default == err_core == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cur_default == cur_core == [0]
    assert out_default == out_core == [[False], [True], [False], [True]]

    bad_buf = [1, 0, 2, 1]
    cur_default = [0]
    cur_core = [0]
    out_default = [[False], [True], [False], [True]]
    out_core = [[False], [True], [False], [True]]

    err_default = gguf_metadata_read_bool_quad_checked_no_partial_default(
        bad_buf,
        len(bad_buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_core = gguf_metadata_read_bool_quad_checked_no_partial(
        bad_buf,
        len(bad_buf),
        cur_core,
        len(bad_buf),
        out_core[0],
        out_core[1],
        out_core[2],
        out_core[3],
    )

    assert err_default == err_core == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cur_default == cur_core == [0]
    assert out_default == out_core == [[False], [True], [False], [True]]


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 16

    cur_default = [3]
    cur_core = [3]
    out_default = [[True], [False], [True], [False]]
    out_core = [[True], [False], [True], [False]]

    err_default = gguf_metadata_read_bool_quad_checked_no_partial_default(
        buf,
        I64_MAX + 1,
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_core = gguf_metadata_read_bool_quad_checked_no_partial(
        buf,
        I64_MAX + 1,
        cur_core,
        I64_MAX + 1,
        out_core[0],
        out_core[1],
        out_core[2],
        out_core[3],
    )

    assert err_default == err_core == GGUF_META_TABLE_ERR_OVERFLOW
    assert cur_default == cur_core == [3]
    assert out_default == out_core == [[True], [False], [True], [False]]


def test_randomized_default_wrapper_parity() -> None:
    rng = random.Random(20260418_324)

    for _ in range(10000):
        n = rng.randint(1, 320)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        cur_default = [cursor0]
        cur_core = [cursor0]
        out_default = [[True], [False], [True], [False]]
        out_core = [[True], [False], [True], [False]]

        err_default = gguf_metadata_read_bool_quad_checked_no_partial_default(
            buf,
            n,
            cur_default,
            out_default[0],
            out_default[1],
            out_default[2],
            out_default[3],
        )
        err_core = gguf_metadata_read_bool_quad_checked_no_partial(
            buf,
            n,
            cur_core,
            n,
            out_core[0],
            out_core[1],
            out_core[2],
            out_core[3],
        )

        assert err_default == err_core
        assert cur_default == cur_core
        assert out_default == out_core


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_success_matches_explicit_no_partial_core()
    test_truncation_bad_bool_and_no_commit_parity()
    test_overflow_passthrough_and_no_commit()
    test_randomized_default_wrapper_parity()
    print("gguf_metadata_read_bool_quad_checked_no_partial_default_parity=ok")


if __name__ == "__main__":
    run()
