#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU64PairChecked semantics."""

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


def gguf_metadata_read_u64le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u64le_checked_no_partial(
        buf,
        buf_nbytes,
        cursor_ref,
        table_end,
        out_value_ref,
    )


def gguf_metadata_read_u64_pair_checked(
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

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def _le_u64(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_first = [0xAAAAAAAAAAAAAAAA]
    out_second = [0xBBBBBBBBBBBBBBBB]

    assert (
        gguf_metadata_read_u64_pair_checked(
            None,
            64,
            cursor,
            64,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [3]
    assert out_first == [0xAAAAAAAAAAAAAAAA]
    assert out_second == [0xBBBBBBBBBBBBBBBB]

    assert (
        gguf_metadata_read_u64_pair_checked(
            [0] * 64,
            64,
            None,
            64,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_first == [0xAAAAAAAAAAAAAAAA]
    assert out_second == [0xBBBBBBBBBBBBBBBB]


def test_second_scalar_fail_does_not_commit_first_or_cursor() -> None:
    first = 0x0123456789ABCDEF
    buf = _le_u64(first) + [0xAA] * 7

    cursor = [0]
    out_first = [0x1111111111111111]
    out_second = [0x2222222222222222]

    err = gguf_metadata_read_u64_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x1111111111111111]
    assert out_second == [0x2222222222222222]


def test_success_reads_two_u64_and_advances() -> None:
    first = 0xFFEEDDCCBBAA9988
    second = 0x0123456789ABCDEF
    buf = [0xEE] + _le_u64(first) + _le_u64(second) + [0x77]

    cursor = [1]
    out_first = [0]
    out_second = [0]

    err = gguf_metadata_read_u64_pair_checked(
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
    assert cursor[0] == 17


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 24

    cursor = [0]
    out_first = [1]
    out_second = [2]
    err = gguf_metadata_read_u64_pair_checked(
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
    rng = random.Random(20260417_265)

    for _ in range(5000):
        n = rng.randint(1, 512)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [0x1111111111111111]
        out_second = [0x2222222222222222]

        err = gguf_metadata_read_u64_pair_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
        )

        if cursor0 + 16 <= table_end:
            assert err == GGUF_META_TABLE_OK
            first = 0
            second = 0
            for i in range(8):
                first |= buf[cursor0 + i] << (8 * i)
                second |= buf[cursor0 + 8 + i] << (8 * i)
            assert out_first[0] == first
            assert out_second[0] == second
            assert cursor[0] == cursor0 + 16
        else:
            assert err in (
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
                GGUF_META_TABLE_ERR_BAD_PARAM,
            )
            assert cursor[0] == cursor0
            assert out_first[0] == 0x1111111111111111
            assert out_second[0] == 0x2222222222222222
