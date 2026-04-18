#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadU16QuadCheckedDefault."""

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


def gguf_metadata_read_u16le_checked(
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

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b0)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b1)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = b0[0] | (b1[0] << 8)
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u16_quad_checked(
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

    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u16_quad_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u16_quad_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out_first = [0x1111]
    out_second = [0x2222]
    out_third = [0x3333]
    out_fourth = [0x4444]

    assert (
        gguf_metadata_read_u16_quad_checked_default(
            None,
            16,
            cursor,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [4]
    assert out_first == [0x1111]
    assert out_second == [0x2222]
    assert out_third == [0x3333]
    assert out_fourth == [0x4444]


def test_uses_default_end_and_no_commit_on_short_payload() -> None:
    first = 0xAAAA
    second = 0xBBBB
    third = 0xCCCC
    buf = _le_u16(first) + _le_u16(second) + _le_u16(third) + [0xEE]

    cursor = [0]
    out_first = [1]
    out_second = [2]
    out_third = [3]
    out_fourth = [4]

    err = gguf_metadata_read_u16_quad_checked_default(
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
    assert out_first == [1]
    assert out_second == [2]
    assert out_third == [3]
    assert out_fourth == [4]


def test_success_reads_four_u16_and_advances() -> None:
    values = [0xBEEF, 0x1234, 0xCAFE, 0x0F0F]
    prefix = [0x77]
    buf = prefix
    for value in values:
        buf += _le_u16(value)
    buf += [0x88]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u16_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == values[0]
    assert out_second[0] == values[1]
    assert out_third[0] == values[2]
    assert out_fourth[0] == values[3]
    assert cursor[0] == 9


def test_randomized_success_matches_base_checked() -> None:
    rng = random.Random(0xC0DE16)

    for _ in range(500):
        values = [rng.randint(0, 0xFFFF) for _ in range(4)]
        prefix_len = rng.randint(0, 8)
        suffix_len = rng.randint(0, 8)

        prefix = [rng.randrange(256) for _ in range(prefix_len)]
        suffix = [rng.randrange(256) for _ in range(suffix_len)]
        payload: list[int] = []
        for value in values:
            payload.extend(_le_u16(value))
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

        err_default = gguf_metadata_read_u16_quad_checked_default(
            buf,
            len(buf),
            cursor_default,
            out_default_a,
            out_default_b,
            out_default_c,
            out_default_d,
        )

        err_base = gguf_metadata_read_u16_quad_checked(
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


def test_default_end_differs_from_tighter_base_end() -> None:
    values = [0x1234, 0xABCD, 0x00EF, 0xFFFF]
    buf: list[int] = []
    for value in values:
        buf.extend(_le_u16(value))

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

    err_default = gguf_metadata_read_u16_quad_checked_default(
        buf,
        len(buf),
        cursor_default,
        out_default_a,
        out_default_b,
        out_default_c,
        out_default_d,
    )

    err_base = gguf_metadata_read_u16_quad_checked(
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
    assert cursor_default == [8]
    assert cursor_base == [0]
    assert out_default_a == [values[0]]
    assert out_default_b == [values[1]]
    assert out_default_c == [values[2]]
    assert out_default_d == [values[3]]
    assert out_base_a == [5]
    assert out_base_b == [6]
    assert out_base_c == [7]
    assert out_base_d == [8]


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_uses_default_end_and_no_commit_on_short_payload()
    test_success_reads_four_u16_and_advances()
    test_randomized_success_matches_base_checked()
    test_default_end_differs_from_tighter_base_end()
    print("ok")
