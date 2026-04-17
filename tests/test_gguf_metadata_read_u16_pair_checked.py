#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU16PairChecked semantics."""

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


def gguf_metadata_read_u16_pair_checked(
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

    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out_first = [0xAAAA]
    out_second = [0xBBBB]

    assert (
        gguf_metadata_read_u16_pair_checked(
            None,
            8,
            cursor,
            8,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [2]
    assert out_first == [0xAAAA]
    assert out_second == [0xBBBB]

    assert (
        gguf_metadata_read_u16_pair_checked(
            [0] * 8,
            8,
            None,
            8,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_first == [0xAAAA]
    assert out_second == [0xBBBB]


def test_second_scalar_fail_does_not_commit_first_or_cursor() -> None:
    first = 0x1234
    buf = _le_u16(first) + [0xAA]

    cursor = [0]
    out_first = [0xDEAD]
    out_second = [0xBEEF]

    err = gguf_metadata_read_u16_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0xDEAD]
    assert out_second == [0xBEEF]


def test_success_reads_two_u16_and_advances() -> None:
    first = 0xBEEF
    second = 0x1234
    buf = [0x99] + _le_u16(first) + _le_u16(second) + [0x77]

    cursor = [1]
    out_first = [0]
    out_second = [0]

    err = gguf_metadata_read_u16_pair_checked(
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
    assert cursor[0] == 5


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 16

    cursor = [0]
    out_first = [1]
    out_second = [2]
    err = gguf_metadata_read_u16_pair_checked(
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

    cursor = [I64_MAX + 1]
    out_first = [3]
    out_second = [4]
    err = gguf_metadata_read_u16_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [I64_MAX + 1]
    assert out_first == [3]
    assert out_second == [4]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_221)

    for _ in range(5000):
        n = rng.randint(1, 256)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [0x1111]
        out_second = [0x2222]

        err = gguf_metadata_read_u16_pair_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
        )

        if cursor0 + 4 <= table_end and cursor0 + 4 <= n:
            expect_first = buf[cursor0] | (buf[cursor0 + 1] << 8)
            expect_second = buf[cursor0 + 2] | (buf[cursor0 + 3] << 8)
            assert err == GGUF_META_TABLE_OK
            assert out_first[0] == expect_first
            assert out_second[0] == expect_second
            assert cursor[0] == cursor0 + 4
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out_first[0] == 0x1111
            assert out_second[0] == 0x2222
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_second_scalar_fail_does_not_commit_first_or_cursor()
    test_success_reads_two_u16_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_randomized_parity()
    print("gguf_metadata_read_u16_pair_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
