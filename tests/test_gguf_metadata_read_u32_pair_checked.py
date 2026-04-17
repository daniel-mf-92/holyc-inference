#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU32PairChecked semantics."""

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


def gguf_metadata_read_u32_pair_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def _le_u32(value: int) -> list[int]:
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out_first = [0xAAAAAAAA]
    out_second = [0xBBBBBBBB]

    assert (
        gguf_metadata_read_u32_pair_checked(
            None,
            16,
            cursor,
            16,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [2]
    assert out_first == [0xAAAAAAAA]
    assert out_second == [0xBBBBBBBB]

    assert (
        gguf_metadata_read_u32_pair_checked(
            [0] * 16,
            16,
            None,
            16,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_first == [0xAAAAAAAA]
    assert out_second == [0xBBBBBBBB]


def test_second_scalar_fail_does_not_commit_first_or_cursor() -> None:
    first = 0x01234567
    buf = _le_u32(first) + [0xAA, 0xBB, 0xCC]

    cursor = [0]
    out_first = [0xDEADBEEF]
    out_second = [0xFEEDFACE]

    err = gguf_metadata_read_u32_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0xDEADBEEF]
    assert out_second == [0xFEEDFACE]


def test_success_reads_two_u32_and_advances() -> None:
    first = 0x89ABCDEF
    second = 0x10203040
    buf = [0xEE] + _le_u32(first) + _le_u32(second) + [0x77]

    cursor = [1]
    out_first = [0]
    out_second = [0]

    err = gguf_metadata_read_u32_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == first
    assert out_second[0] == second
    assert cursor[0] == 9


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 16

    cursor = [0]
    out_first = [1]
    out_second = [2]
    err = gguf_metadata_read_u32_pair_checked(
        buf,
        I64_MAX + 1,
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_first == [1]
    assert out_second == [2]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_217)

    for _ in range(5000):
        n = rng.randint(1, 512)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [0x11111111]
        out_second = [0x22222222]

        err = gguf_metadata_read_u32_pair_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
        )

        if cursor0 + 8 <= table_end and cursor0 + 8 <= n:
            assert err == GGUF_META_TABLE_OK
            expect_first = (
                buf[cursor0]
                | (buf[cursor0 + 1] << 8)
                | (buf[cursor0 + 2] << 16)
                | (buf[cursor0 + 3] << 24)
            )
            expect_second = (
                buf[cursor0 + 4]
                | (buf[cursor0 + 5] << 8)
                | (buf[cursor0 + 6] << 16)
                | (buf[cursor0 + 7] << 24)
            )
            assert out_first[0] == expect_first
            assert out_second[0] == expect_second
            assert cursor[0] == cursor0 + 8
        else:
            assert err in (
                GGUF_META_TABLE_ERR_BAD_PARAM,
                GGUF_META_TABLE_ERR_OVERFLOW,
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
            )
            assert out_first[0] == 0x11111111
            assert out_second[0] == 0x22222222
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_second_scalar_fail_does_not_commit_first_or_cursor()
    test_success_reads_two_u32_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_randomized_parity()
    print("gguf_metadata_read_u32_pair_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
