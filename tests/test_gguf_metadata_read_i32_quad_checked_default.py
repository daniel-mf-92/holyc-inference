#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI32QuadCheckedDefault."""

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
        return value - 0x100000000
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

    cur = [cursor_ref[0]]
    raw_u32 = [0]

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, raw_u32)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = _to_i32(raw_u32[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_quad_checked(
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

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_quad_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i32_quad_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def _le_i32(value: int) -> list[int]:
    raw = value & 0xFFFFFFFF
    return [
        raw & 0xFF,
        (raw >> 8) & 0xFF,
        (raw >> 16) & 0xFF,
        (raw >> 24) & 0xFF,
    ]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [9]
    out_first = [11]
    out_second = [22]
    out_third = [33]
    out_fourth = [44]

    assert (
        gguf_metadata_read_i32_quad_checked_default(
            None,
            32,
            cursor,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [9]
    assert out_first == [11]
    assert out_second == [22]
    assert out_third == [33]
    assert out_fourth == [44]


def test_uses_default_end_and_no_commit_on_short_payload() -> None:
    first = -7
    second = 12
    third = -1024
    buf = _le_i32(first) + _le_i32(second) + _le_i32(third) + [0xAA, 0xBB, 0xCC]

    cursor = [0]
    out_first = [101]
    out_second = [202]
    out_third = [303]
    out_fourth = [404]

    err = gguf_metadata_read_i32_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [101]
    assert out_second == [202]
    assert out_third == [303]
    assert out_fourth == [404]


def test_success_reads_four_i32_and_advances() -> None:
    first = -2147483648
    second = -1
    third = 0
    fourth = 2147483647
    prefix = [0xDD, 0xEE]
    buf = prefix + _le_i32(first) + _le_i32(second) + _le_i32(third) + _le_i32(fourth)

    cursor = [2]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_i32_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == first
    assert out_second[0] == second
    assert out_third[0] == third
    assert out_fourth[0] == fourth
    assert cursor[0] == 18


def test_randomized_success_matches_base_checked() -> None:
    rng = random.Random(0xC0DEC0DE)

    for _ in range(400):
        values = [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(4)]
        prefix_len = rng.randint(0, 8)
        suffix_len = rng.randint(0, 8)

        prefix = [rng.randrange(256) for _ in range(prefix_len)]
        suffix = [rng.randrange(256) for _ in range(suffix_len)]
        payload: list[int] = []
        for value in values:
            payload.extend(_le_i32(value))
        buf = prefix + payload + suffix

        cursor_default = [prefix_len]
        cursor_base = [prefix_len]

        out_default_a = [0]
        out_default_b = [0]
        out_default_c = [0]
        out_default_d = [0]

        out_base_a = [0]
        out_base_b = [0]
        out_base_c = [0]
        out_base_d = [0]

        err_default = gguf_metadata_read_i32_quad_checked_default(
            buf,
            len(buf),
            cursor_default,
            out_default_a,
            out_default_b,
            out_default_c,
            out_default_d,
        )

        err_base = gguf_metadata_read_i32_quad_checked(
            buf,
            len(buf),
            cursor_base,
            len(buf),
            out_base_a,
            out_base_b,
            out_base_c,
            out_base_d,
        )

        assert err_default == GGUF_META_TABLE_OK
        assert err_default == err_base
        assert cursor_default == cursor_base
        assert out_default_a == out_base_a == [values[0]]
        assert out_default_b == out_base_b == [values[1]]
        assert out_default_c == out_base_c == [values[2]]
        assert out_default_d == out_base_d == [values[3]]


def test_table_end_guard_parity_against_base_checked() -> None:
    first = 0x12345678
    second = -1
    third = -2147483648
    fourth = 0x7FFFFFFF
    buf = _le_i32(first) + _le_i32(second) + _le_i32(third) + _le_i32(fourth)

    cursor_default = [0]
    cursor_base = [0]

    out_default_a = [1]
    out_default_b = [2]
    out_default_c = [3]
    out_default_d = [4]

    out_base_a = [5]
    out_base_b = [6]
    out_base_c = [7]
    out_base_d = [8]

    err_default = gguf_metadata_read_i32_quad_checked_default(
        buf,
        len(buf),
        cursor_default,
        out_default_a,
        out_default_b,
        out_default_c,
        out_default_d,
    )

    err_base = gguf_metadata_read_i32_quad_checked(
        buf,
        len(buf),
        cursor_base,
        len(buf) - 1,
        out_base_a,
        out_base_b,
        out_base_c,
        out_base_d,
    )

    assert err_default == GGUF_META_TABLE_OK
    assert err_base == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor_default == [16]
    assert cursor_base == [0]
    assert out_default_a == [first]
    assert out_default_b == [second]
    assert out_default_c == [third]
    assert out_default_d == [fourth]
    assert out_base_a == [5]
    assert out_base_b == [6]
    assert out_base_c == [7]
    assert out_base_d == [8]
