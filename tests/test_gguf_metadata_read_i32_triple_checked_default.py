#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI32TripleCheckedDefault."""

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


def gguf_metadata_read_i32_triple_checked(
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

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_triple_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i32_triple_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
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
    cursor = [4]
    out_first = [111]
    out_second = [222]
    out_third = [333]

    assert (
        gguf_metadata_read_i32_triple_checked_default(
            None,
            20,
            cursor,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [4]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]


def test_default_end_short_read_no_partial_commit() -> None:
    first = -2147483648
    second = 2147483647
    buf = _le_i32(first) + _le_i32(second) + [0xAA, 0xBB, 0xCC]

    cursor = [0]
    out_first = [0x11111111]
    out_second = [0x22222222]
    out_third = [0x33333333]

    err = gguf_metadata_read_i32_triple_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x11111111]
    assert out_second == [0x22222222]
    assert out_third == [0x33333333]


def test_success_reads_three_i32_and_advances() -> None:
    first = -123456789
    second = 0
    third = 123456789
    buf = [0xAB] + _le_i32(first) + _le_i32(second) + _le_i32(third) + [0xCD]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_i32_triple_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [first]
    assert out_second == [second]
    assert out_third == [third]
    assert cursor == [13]


def test_randomized_parity_against_checked_with_buf_end() -> None:
    rng = random.Random(20260418_293)

    for _ in range(10000):
        n = rng.randint(1, 512)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        cursor_default = [cursor0]
        default_first = [0x11111111]
        default_second = [0x22222222]
        default_third = [0x33333333]

        err_default = gguf_metadata_read_i32_triple_checked_default(
            buf,
            n,
            cursor_default,
            default_first,
            default_second,
            default_third,
        )

        cursor_core = [cursor0]
        core_first = [0x11111111]
        core_second = [0x22222222]
        core_third = [0x33333333]

        err_core = gguf_metadata_read_i32_triple_checked(
            buf,
            n,
            cursor_core,
            n,
            core_first,
            core_second,
            core_third,
        )

        assert err_default == err_core
        assert cursor_default == cursor_core
        assert default_first == core_first
        assert default_second == core_second
        assert default_third == core_third


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_default_end_short_read_no_partial_commit()
    test_success_reads_three_i32_and_advances()
    test_randomized_parity_against_checked_with_buf_end()
    print("ok")
