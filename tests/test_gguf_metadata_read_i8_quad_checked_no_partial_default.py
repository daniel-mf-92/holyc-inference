#!/usr/bin/env python3
"""Parity harness for GGUFMetadataReadI8QuadCheckedNoPartialDefault contract."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


def reinterpret_u8_as_i8(value: int) -> int:
    value &= 0xFF
    if value >= 0x80:
        return value - 0x100
    return value


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
    if cursor > U64_MAX - need:
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


def gguf_metadata_read_i8_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u8_as_i8(raw[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i8_triple_checked(
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

    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i8_quad_checked(
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

    err = gguf_metadata_read_i8_triple_checked(
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

    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i8_quad_checked_no_partial(
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

    err = gguf_metadata_read_i8_quad_checked(
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


def gguf_metadata_read_i8_quad_checked_no_partial_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i8_quad_checked_no_partial(
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
    cursor = [7]
    out_first = [11]
    out_second = [22]
    out_third = [33]
    out_fourth = [44]

    err = gguf_metadata_read_i8_quad_checked_no_partial_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )

    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [7]
    assert out_first == [11]
    assert out_second == [22]
    assert out_third == [33]
    assert out_fourth == [44]


def test_default_end_short_read_no_partial_commit() -> None:
    cursor = [0]
    out_first = [101]
    out_second = [102]
    out_third = [103]
    out_fourth = [104]

    err = gguf_metadata_read_i8_quad_checked_no_partial_default(
        [0x7F, 0x80, 0x01],
        3,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )

    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [101]
    assert out_second == [102]
    assert out_third == [103]
    assert out_fourth == [104]


def test_success_path_signed_values() -> None:
    cursor = [0]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_i8_quad_checked_no_partial_default(
        [0x80, 0xFF, 0x00, 0x7F],
        4,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )

    assert err == GGUF_META_TABLE_OK
    assert cursor == [4]
    assert out_first == [-128]
    assert out_second == [-1]
    assert out_third == [0]
    assert out_fourth == [127]


def test_randomized_parity_vs_explicit_no_partial() -> None:
    rng = random.Random(0x1B323)

    for _ in range(400):
        buf_len = rng.randint(0, 128)
        buf = [rng.randint(0, 255) for _ in range(buf_len)]
        cursor0 = rng.randint(0, 128)

        cursor_default = [cursor0]
        cursor_explicit = [cursor0]

        out_default_1 = [rng.randint(-128, 127)]
        out_default_2 = [rng.randint(-128, 127)]
        out_default_3 = [rng.randint(-128, 127)]
        out_default_4 = [rng.randint(-128, 127)]

        out_explicit_1 = [out_default_1[0]]
        out_explicit_2 = [out_default_2[0]]
        out_explicit_3 = [out_default_3[0]]
        out_explicit_4 = [out_default_4[0]]

        err_default = gguf_metadata_read_i8_quad_checked_no_partial_default(
            buf,
            buf_len,
            cursor_default,
            out_default_1,
            out_default_2,
            out_default_3,
            out_default_4,
        )
        err_explicit = gguf_metadata_read_i8_quad_checked_no_partial(
            buf,
            buf_len,
            cursor_explicit,
            buf_len,
            out_explicit_1,
            out_explicit_2,
            out_explicit_3,
            out_explicit_4,
        )

        assert err_default == err_explicit
        assert cursor_default == cursor_explicit
        assert out_default_1 == out_explicit_1
        assert out_default_2 == out_explicit_2
        assert out_default_3 == out_explicit_3
        assert out_default_4 == out_explicit_4
