#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadBoolQuadCheckedDefault."""

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


def gguf_metadata_read_bool_quad_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
    out_third_ref: list[bool] | None,
    out_fourth_ref: list[bool] | None,
) -> int:
    return gguf_metadata_read_bool_quad_checked(
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
    cursor = [3]
    out_first = [True]
    out_second = [False]
    out_third = [True]
    out_fourth = [False]

    err = gguf_metadata_read_bool_quad_checked_default(
        None,
        16,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [3]
    assert out_first == [True]
    assert out_second == [False]
    assert out_third == [True]
    assert out_fourth == [False]


def test_success_uses_default_table_end() -> None:
    buf = [0xA5, 1, 0, 1, 1, 0x5A]

    cur_default = [1]
    out_default_a = [False]
    out_default_b = [False]
    out_default_c = [False]
    out_default_d = [False]

    cur_checked = [1]
    out_checked_a = [False]
    out_checked_b = [False]
    out_checked_c = [False]
    out_checked_d = [False]

    err_default = gguf_metadata_read_bool_quad_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default_a,
        out_default_b,
        out_default_c,
        out_default_d,
    )
    err_checked = gguf_metadata_read_bool_quad_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked_a,
        out_checked_b,
        out_checked_c,
        out_checked_d,
    )

    assert err_default == GGUF_META_TABLE_OK
    assert err_checked == GGUF_META_TABLE_OK
    assert cur_default == cur_checked == [5]
    assert out_default_a == out_checked_a == [True]
    assert out_default_b == out_checked_b == [False]
    assert out_default_c == out_checked_c == [True]
    assert out_default_d == out_checked_d == [True]


def test_truncation_and_bad_bool_no_partial_commit() -> None:
    short_buf = [1, 0, 1]
    cursor = [0]
    out_first = [False]
    out_second = [True]
    out_third = [False]
    out_fourth = [True]

    err = gguf_metadata_read_bool_quad_checked_default(
        short_buf,
        len(short_buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [True]
    assert out_third == [False]
    assert out_fourth == [True]

    bad_bool_buf = [1, 0, 2, 1]
    cursor = [0]
    out_first = [False]
    out_second = [True]
    out_third = [False]
    out_fourth = [True]
    err = gguf_metadata_read_bool_quad_checked_default(
        bad_bool_buf,
        len(bad_bool_buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [True]
    assert out_third == [False]
    assert out_fourth == [True]


def test_randomized_parity_against_checked_with_buf_end() -> None:
    rng = random.Random(20260418_309)

    for _ in range(10000):
        n = rng.randint(1, 320)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        cur_default = [cursor0]
        out_default_a = [True]
        out_default_b = [False]
        out_default_c = [True]
        out_default_d = [False]

        err_default = gguf_metadata_read_bool_quad_checked_default(
            buf,
            n,
            cur_default,
            out_default_a,
            out_default_b,
            out_default_c,
            out_default_d,
        )

        cur_checked = [cursor0]
        out_checked_a = [True]
        out_checked_b = [False]
        out_checked_c = [True]
        out_checked_d = [False]

        err_checked = gguf_metadata_read_bool_quad_checked(
            buf,
            n,
            cur_checked,
            n,
            out_checked_a,
            out_checked_b,
            out_checked_c,
            out_checked_d,
        )

        assert err_default == err_checked
        assert cur_default == cur_checked
        assert out_default_a == out_checked_a
        assert out_default_b == out_checked_b
        assert out_default_c == out_checked_c
        assert out_default_d == out_checked_d


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_uses_default_table_end()
    test_truncation_and_bad_bool_no_partial_commit()
    test_randomized_parity_against_checked_with_buf_end()
    print("ok")
