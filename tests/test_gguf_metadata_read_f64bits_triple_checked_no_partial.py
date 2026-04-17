#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadF64BitsTripleCheckedNoPartial semantics."""

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


def gguf_metadata_read_u64le_checked(
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


def gguf_metadata_read_f64bitsle_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_bits_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_bits_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]
    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    out_bits_ref[0] = raw[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bits_pair_checked(
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

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bits_triple_checked(
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

    err = gguf_metadata_read_f64bits_pair_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bits_triple_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_f64bits_triple_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        table_end,
        out_first_ref,
        out_second_ref,
        out_third_ref,
    )


def _le_u64(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [0xAAAA]
    out_second = [0xBBBB]
    out_third = [0xCCCC]

    assert (
        gguf_metadata_read_f64bits_triple_checked_no_partial(
            None,
            64,
            cursor,
            64,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [5]
    assert out_first == [0xAAAA]
    assert out_second == [0xBBBB]
    assert out_third == [0xCCCC]

    assert (
        gguf_metadata_read_f64bits_triple_checked_no_partial(
            [0] * 64,
            64,
            None,
            64,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_first == [0xAAAA]
    assert out_second == [0xBBBB]
    assert out_third == [0xCCCC]


def test_third_scalar_fail_does_not_commit_anything() -> None:
    first = 0x3FF0000000000000
    second = 0x4000000000000000
    buf = _le_u64(first) + _le_u64(second) + [0xAB] * 7

    cursor = [0]
    out_first = [0x111]
    out_second = [0x222]
    out_third = [0x333]

    err = gguf_metadata_read_f64bits_triple_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x111]
    assert out_second == [0x222]
    assert out_third == [0x333]


def test_success_reads_three_f64_bitpatterns_and_advances() -> None:
    first = 0x3FF199999999999A
    second = 0xC004000000000000
    third = 0x0000000000000001

    buf = _le_u64(first) + _le_u64(second) + _le_u64(third) + [0xEE, 0xDD]

    cursor = [0]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_f64bits_triple_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [first]
    assert out_second == [second]
    assert out_third == [third]
    assert cursor == [24]


def test_randomized_success_parity() -> None:
    rng = random.Random(0xF64B17)

    for _ in range(200):
        first = rng.getrandbits(64)
        second = rng.getrandbits(64)
        third = rng.getrandbits(64)
        prefix = [rng.getrandbits(8) for _ in range(rng.randrange(0, 5))]
        suffix = [rng.getrandbits(8) for _ in range(rng.randrange(0, 5))]
        payload = _le_u64(first) + _le_u64(second) + _le_u64(third)
        buf = prefix + payload + suffix

        start = len(prefix)
        cursor = [start]
        out_first = [0]
        out_second = [0]
        out_third = [0]

        err = gguf_metadata_read_f64bits_triple_checked_no_partial(
            buf,
            len(buf),
            cursor,
            len(buf),
            out_first,
            out_second,
            out_third,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_first == [first]
        assert out_second == [second]
        assert out_third == [third]
        assert cursor == [start + 24]


def test_table_end_truncation_is_no_partial() -> None:
    first = 0xAAAAAAAAAAAAAAAA
    second = 0xBBBBBBBBBBBBBBBB
    third = 0xCCCCCCCCCCCCCCCC
    buf = _le_u64(first) + _le_u64(second) + _le_u64(third)

    cursor = [0]
    out_first = [1]
    out_second = [2]
    out_third = [3]

    err = gguf_metadata_read_f64bits_triple_checked_no_partial(
        buf,
        len(buf),
        cursor,
        23,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [1]
    assert out_second == [2]
    assert out_third == [3]
