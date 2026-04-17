#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI32QuadCheckedNoPartial semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U32_MASK = (1 << 32) - 1


def _to_i32(value: int) -> int:
    value &= U32_MASK
    if value >= 0x8000_0000:
        return value - (1 << 32)
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

    raw_u32 = [0]
    cur = [cursor_ref[0]]

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


def gguf_metadata_read_i32_quad_checked_no_partial(
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

    err = gguf_metadata_read_i32_quad_checked(
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


def _le_i32(value: int) -> list[int]:
    raw = value & U32_MASK
    return [
        raw & 0xFF,
        (raw >> 8) & 0xFF,
        (raw >> 16) & 0xFF,
        (raw >> 24) & 0xFF,
    ]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [111]
    out_second = [222]
    out_third = [333]
    out_fourth = [444]

    assert (
        gguf_metadata_read_i32_quad_checked_no_partial(
            None,
            16,
            cursor,
            16,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [5]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]
    assert out_fourth == [444]


def test_fourth_scalar_fail_does_not_commit_anything() -> None:
    first = -(1 << 31)
    second = -1
    third = (1 << 31) - 1
    buf = _le_i32(first) + _le_i32(second) + _le_i32(third) + [0xAA, 0xBB, 0xCC]

    cursor = [0]
    out_first = [0x111]
    out_second = [0x222]
    out_third = [0x333]
    out_fourth = [0x444]

    err = gguf_metadata_read_i32_quad_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x111]
    assert out_second == [0x222]
    assert out_third == [0x333]
    assert out_fourth == [0x444]


def test_success_reads_four_i32_and_advances() -> None:
    first = -(1 << 31)
    second = -123456789
    third = 123456789
    fourth = (1 << 31) - 1

    buf = [0xEE] + _le_i32(first) + _le_i32(second) + _le_i32(third) + _le_i32(fourth) + [0x77]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_i32_quad_checked_no_partial(
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
    assert out_first[0] == first
    assert out_second[0] == second
    assert out_third[0] == third
    assert out_fourth[0] == fourth
    assert cursor == [1 + 16]


def test_randomized_parity_against_quad_checked() -> None:
    random.seed(264)
    for _ in range(300):
        values = [random.randint(-(1 << 31), (1 << 31) - 1) for _ in range(4)]
        prefix_len = random.randint(0, 4)
        suffix_len = random.randint(0, 4)

        payload: list[int] = [random.randint(0, 255) for _ in range(prefix_len)]
        for value in values:
            payload.extend(_le_i32(value))
        payload.extend(random.randint(0, 255) for _ in range(suffix_len))

        cursor0 = [prefix_len]
        cursor1 = [prefix_len]
        out0 = [[0], [0], [0], [0]]
        out1 = [[0], [0], [0], [0]]

        err0 = gguf_metadata_read_i32_quad_checked(
            payload,
            len(payload),
            cursor0,
            len(payload),
            out0[0],
            out0[1],
            out0[2],
            out0[3],
        )
        err1 = gguf_metadata_read_i32_quad_checked_no_partial(
            payload,
            len(payload),
            cursor1,
            len(payload),
            out1[0],
            out1[1],
            out1[2],
            out1[3],
        )

        assert err1 == err0
        assert cursor1 == cursor0
        assert [out1[i][0] for i in range(4)] == [out0[i][0] for i in range(4)]


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_scalar_fail_does_not_commit_anything()
    test_success_reads_four_i32_and_advances()
    test_randomized_parity_against_quad_checked()
    print("gguf_metadata_read_i32_quad_checked_no_partial: ok")
