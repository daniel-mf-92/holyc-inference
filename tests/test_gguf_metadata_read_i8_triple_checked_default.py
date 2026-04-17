#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI8TripleCheckedDefault."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U8_MASK = (1 << 8) - 1


def reinterpret_u8_as_i8(value: int) -> int:
    value &= U8_MASK
    if value >= (1 << 7):
        return value - (1 << 8)
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
    raw_u8 = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, raw_u8)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u8_as_i8(raw_u8[0])
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


def gguf_metadata_read_i8_triple_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i8_triple_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [11]
    out_second = [22]
    out_third = [33]

    assert (
        gguf_metadata_read_i8_triple_checked_default(
            None,
            16,
            cursor,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [5]
    assert out_first == [11]
    assert out_second == [22]
    assert out_third == [33]


def test_success_uses_default_table_end() -> None:
    buf = [0x55, 0x7F, 0x80, 0xFF, 0x01]
    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_i8_triple_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == 127
    assert out_second[0] == -128
    assert out_third[0] == -1
    assert cursor[0] == 4


def test_default_end_short_read_no_partial_commit() -> None:
    buf = [0x10, 0x20, 0x30]
    cursor = [1]
    out_first = [101]
    out_second = [202]
    out_third = [99]

    err = gguf_metadata_read_i8_triple_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [1]
    assert out_first == [101]
    assert out_second == [202]
    assert out_third == [99]


def test_randomized_parity_against_checked_with_buf_end() -> None:
    rng = random.Random(20260418_290)

    for _ in range(9000):
        n = rng.randint(1, 320)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        cursor_default = [cursor0]
        out_default_a = [11]
        out_default_b = [22]
        out_default_c = [33]

        err_default = gguf_metadata_read_i8_triple_checked_default(
            buf,
            n,
            cursor_default,
            out_default_a,
            out_default_b,
            out_default_c,
        )

        cursor_checked = [cursor0]
        out_checked_a = [11]
        out_checked_b = [22]
        out_checked_c = [33]

        err_checked = gguf_metadata_read_i8_triple_checked(
            buf,
            n,
            cursor_checked,
            n,
            out_checked_a,
            out_checked_b,
            out_checked_c,
        )

        assert err_default == err_checked
        assert cursor_default == cursor_checked
        assert out_default_a == out_checked_a
        assert out_default_b == out_checked_b
        assert out_default_c == out_checked_c


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_uses_default_table_end()
    test_default_end_short_read_no_partial_commit()
    test_randomized_parity_against_checked_with_buf_end()
    print("ok")
