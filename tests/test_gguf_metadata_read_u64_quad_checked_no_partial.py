#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU64QuadCheckedNoPartial semantics."""

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


def gguf_metadata_read_u64le_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    b = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= b[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64_quad_checked_no_partial(
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

    err = gguf_metadata_read_u64le_checked_no_partial(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u64le_checked_no_partial(
        buf,
        buf_nbytes,
        cur,
        table_end,
        second,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u64le_checked_no_partial(
        buf,
        buf_nbytes,
        cur,
        table_end,
        third,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u64le_checked_no_partial(
        buf,
        buf_nbytes,
        cur,
        table_end,
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


def _le_u64(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [0x1111111111111111]
    out_second = [0x2222222222222222]
    out_third = [0x3333333333333333]
    out_fourth = [0x4444444444444444]

    assert (
        gguf_metadata_read_u64_quad_checked_no_partial(
            None,
            80,
            cursor,
            80,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [5]
    assert out_first == [0x1111111111111111]
    assert out_second == [0x2222222222222222]
    assert out_third == [0x3333333333333333]
    assert out_fourth == [0x4444444444444444]


def test_fourth_scalar_fail_does_not_commit_anything() -> None:
    first = 0x0123456789ABCDEF
    second = 0xFFEEDDCCBBAA9988
    third = 0x7FFFFFFFFFFFFFFF
    buf = _le_u64(first) + _le_u64(second) + _le_u64(third) + [0xAA] * 7

    cursor = [0]
    out_first = [0xAAAAAAAAAAAAAAAA]
    out_second = [0xBBBBBBBBBBBBBBBB]
    out_third = [0xCCCCCCCCCCCCCCCC]
    out_fourth = [0xDDDDDDDDDDDDDDDD]

    err = gguf_metadata_read_u64_quad_checked_no_partial(
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
    assert out_first == [0xAAAAAAAAAAAAAAAA]
    assert out_second == [0xBBBBBBBBBBBBBBBB]
    assert out_third == [0xCCCCCCCCCCCCCCCC]
    assert out_fourth == [0xDDDDDDDDDDDDDDDD]


def test_success_reads_four_u64_and_advances() -> None:
    first = 0x0123456789ABCDEF
    second = 0xFFEEDDCCBBAA9988
    third = 0x7FFFFFFFFFFFFFFF
    fourth = 0xDEADBEEF00C0FFEE
    buf = [0xEE] + _le_u64(first) + _le_u64(second) + _le_u64(third) + _le_u64(fourth) + [0x77]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u64_quad_checked_no_partial(
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
    assert cursor[0] == 33


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0x44] * 96
    out_first_seed = 0x0BADF00D
    out_second_seed = 0xDEADC0DE
    out_third_seed = 0xCAFEBABE
    out_fourth_seed = 0x0D15EA5E

    cursor = [0]
    out_first = [out_first_seed]
    out_second = [out_second_seed]
    out_third = [out_third_seed]
    out_fourth = [out_fourth_seed]

    err = gguf_metadata_read_u64_quad_checked_no_partial(
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
    assert out_first == [out_first_seed]
    assert out_second == [out_second_seed]
    assert out_third == [out_third_seed]
    assert out_fourth == [out_fourth_seed]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_247)

    for _ in range(6000):
        n = rng.randint(1, 640)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [111]
        out_second = [222]
        out_third = [333]
        out_fourth = [444]

        err = gguf_metadata_read_u64_quad_checked_no_partial(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )

        if cursor0 + 32 <= table_end and cursor0 + 32 <= n:
            expect_first = 0
            expect_second = 0
            expect_third = 0
            expect_fourth = 0
            for i in range(8):
                expect_first |= buf[cursor0 + i] << (8 * i)
                expect_second |= buf[cursor0 + 8 + i] << (8 * i)
                expect_third |= buf[cursor0 + 16 + i] << (8 * i)
                expect_fourth |= buf[cursor0 + 24 + i] << (8 * i)
            assert err == GGUF_META_TABLE_OK
            assert out_first[0] == expect_first
            assert out_second[0] == expect_second
            assert out_third[0] == expect_third
            assert out_fourth[0] == expect_fourth
            assert cursor[0] == cursor0 + 32
        else:
            assert err in (
                GGUF_META_TABLE_ERR_BAD_PARAM,
                GGUF_META_TABLE_ERR_OVERFLOW,
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
            )
            assert out_first[0] == 111
            assert out_second[0] == 222
            assert out_third[0] == 333
            assert out_fourth[0] == 444
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_fourth_scalar_fail_does_not_commit_anything()
    test_success_reads_four_u64_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_randomized_parity()
    print("gguf_metadata_read_u64_quad_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
