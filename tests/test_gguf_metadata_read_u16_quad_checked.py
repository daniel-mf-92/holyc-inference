#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU16QuadChecked semantics."""

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


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out_first = [0x1111]
    out_second = [0x2222]
    out_third = [0x3333]
    out_fourth = [0x4444]

    assert (
        gguf_metadata_read_u16_quad_checked(
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
    assert cursor == [4]
    assert out_first == [0x1111]
    assert out_second == [0x2222]
    assert out_third == [0x3333]
    assert out_fourth == [0x4444]


def test_fourth_scalar_fail_does_not_commit_prior_lanes_or_cursor() -> None:
    first = 0x1111
    second = 0x2222
    third = 0x3333
    buf = _le_u16(first) + _le_u16(second) + _le_u16(third) + [0xAB]

    cursor = [0]
    out_first = [0xAAAA]
    out_second = [0xBBBB]
    out_third = [0xCCCC]
    out_fourth = [0xDDDD]

    err = gguf_metadata_read_u16_quad_checked(
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
    assert out_first == [0xAAAA]
    assert out_second == [0xBBBB]
    assert out_third == [0xCCCC]
    assert out_fourth == [0xDDDD]


def test_success_reads_four_u16_and_advances() -> None:
    first = 0xBEEF
    second = 0x1234
    third = 0xCAFE
    fourth = 0x0F0F
    buf = (
        [0x77]
        + _le_u16(first)
        + _le_u16(second)
        + _le_u16(third)
        + _le_u16(fourth)
        + [0x88]
    )

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u16_quad_checked(
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
    assert cursor[0] == 9


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 20

    cursor = [0]
    out_first = [1]
    out_second = [2]
    out_third = [3]
    out_fourth = [4]
    err = gguf_metadata_read_u16_quad_checked(
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
    assert out_first == [1]
    assert out_second == [2]
    assert out_third == [3]
    assert out_fourth == [4]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_246)

    for _ in range(6000):
        n = rng.randint(1, 320)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]

        if table_end < 8:
            cursor0 = 0
        else:
            cursor0 = rng.randint(0, table_end - 8)

        cursor_a = [cursor0]
        cursor_b = [cursor0]

        out_a0 = [rng.randint(0, 0xFFFF)]
        out_a1 = [rng.randint(0, 0xFFFF)]
        out_a2 = [rng.randint(0, 0xFFFF)]
        out_a3 = [rng.randint(0, 0xFFFF)]
        out_b0 = [out_a0[0]]
        out_b1 = [out_a1[0]]
        out_b2 = [out_a2[0]]
        out_b3 = [out_a3[0]]

        err_a = gguf_metadata_read_u16_quad_checked(
            buf,
            n,
            cursor_a,
            table_end,
            out_a0,
            out_a1,
            out_a2,
            out_a3,
        )

        cur = [cursor0]
        t0 = [out_b0[0]]
        t1 = [out_b1[0]]
        t2 = [out_b2[0]]
        t3 = [out_b3[0]]

        err_b = gguf_metadata_read_u16le_checked(buf, n, cur, table_end, t0)
        if err_b == GGUF_META_TABLE_OK:
            err_b = gguf_metadata_read_u16le_checked(buf, n, cur, table_end, t1)
        if err_b == GGUF_META_TABLE_OK:
            err_b = gguf_metadata_read_u16le_checked(buf, n, cur, table_end, t2)
        if err_b == GGUF_META_TABLE_OK:
            err_b = gguf_metadata_read_u16le_checked(buf, n, cur, table_end, t3)

        if err_b == GGUF_META_TABLE_OK:
            out_b0[0] = t0[0]
            out_b1[0] = t1[0]
            out_b2[0] = t2[0]
            out_b3[0] = t3[0]
            cursor_b[0] = cur[0]

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert out_a0 == out_b0
        assert out_a1 == out_b1
        assert out_a2 == out_b2
        assert out_a3 == out_b3


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_scalar_fail_does_not_commit_prior_lanes_or_cursor()
    test_success_reads_four_u16_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_randomized_parity()
    print("ok")
