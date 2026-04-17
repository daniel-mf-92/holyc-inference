#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI16TripleChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U16_MASK = (1 << 16) - 1


def reinterpret_u16_as_i16(value: int) -> int:
    value &= U16_MASK
    if value >= (1 << 15):
        return value - (1 << 16)
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


def gguf_metadata_read_i16le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]

    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u16_as_i16(raw[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i16_triple_checked(
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

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i16le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_first = [111]
    out_second = [222]
    out_third = [333]

    assert (
        gguf_metadata_read_i16_triple_checked(
            None,
            10,
            cursor,
            10,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [3]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]


def test_third_scalar_fail_does_not_commit_prior_lanes_or_cursor() -> None:
    first_raw = 0xFFAA
    second_raw = 0x8001
    buf = _le_u16(first_raw) + _le_u16(second_raw) + [0x55]

    cursor = [0]
    out_first = [-1]
    out_second = [-2]
    out_third = [-3]

    err = gguf_metadata_read_i16_triple_checked(
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
    assert out_first == [-1]
    assert out_second == [-2]
    assert out_third == [-3]


def test_success_reads_three_i16_and_advances() -> None:
    first_raw = 0x7FFF
    second_raw = 0x8000
    third_raw = 0xBEEF

    buf = [0x42] + _le_u16(first_raw) + _le_u16(second_raw) + _le_u16(third_raw) + [0x99]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_i16_triple_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == reinterpret_u16_as_i16(first_raw)
    assert out_second[0] == reinterpret_u16_as_i16(second_raw)
    assert out_third[0] == reinterpret_u16_as_i16(third_raw)
    assert cursor[0] == 7


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 20

    cursor = [0]
    out_first = [1]
    out_second = [2]
    out_third = [3]
    err = gguf_metadata_read_i16_triple_checked(
        buf,
        I64_MAX + 1,
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_first == [1]
    assert out_second == [2]
    assert out_third == [3]


def test_cursor_beyond_table_end_passthrough() -> None:
    buf = [0] * 16

    cursor = [9]
    out_first = [10]
    out_second = [20]
    out_third = [30]

    err = gguf_metadata_read_i16_triple_checked(
        buf,
        len(buf),
        cursor,
        8,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [9]
    assert out_first == [10]
    assert out_second == [20]
    assert out_third == [30]


def test_randomized_reference_vectors() -> None:
    rng = random.Random(0x1A16_7333)

    for _ in range(500):
        start = rng.randint(0, 3)
        raw_vals = [rng.randint(0, 0xFFFF) for _ in range(3)]

        prefix = [rng.randrange(0, 256) for _ in range(start)]
        body: list[int] = []
        for value in raw_vals:
            body.extend(_le_u16(value))
        suffix = [rng.randrange(0, 256) for _ in range(rng.randint(0, 4))]

        buf = prefix + body + suffix
        table_end = len(prefix) + len(body)

        cursor = [start]
        out_first = [0]
        out_second = [0]
        out_third = [0]

        err = gguf_metadata_read_i16_triple_checked(
            buf,
            len(buf),
            cursor,
            table_end,
            out_first,
            out_second,
            out_third,
        )

        assert err == GGUF_META_TABLE_OK
        assert out_first[0] == reinterpret_u16_as_i16(raw_vals[0])
        assert out_second[0] == reinterpret_u16_as_i16(raw_vals[1])
        assert out_third[0] == reinterpret_u16_as_i16(raw_vals[2])
        assert cursor[0] == start + 6


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_third_scalar_fail_does_not_commit_prior_lanes_or_cursor()
    test_success_reads_three_i16_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_cursor_beyond_table_end_passthrough()
    test_randomized_reference_vectors()
    print("gguf_metadata_read_i16_triple_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
