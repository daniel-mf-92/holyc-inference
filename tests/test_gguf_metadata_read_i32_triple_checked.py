#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI32TripleChecked semantics."""

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


def _le_i32(value: int) -> list[int]:
    raw = value & 0xFFFFFFFF
    return [
        raw & 0xFF,
        (raw >> 8) & 0xFF,
        (raw >> 16) & 0xFF,
        (raw >> 24) & 0xFF,
    ]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [7]
    out_first = [111]
    out_second = [222]
    out_third = [333]

    assert (
        gguf_metadata_read_i32_triple_checked(
            None,
            16,
            cursor,
            16,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [7]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]


def test_third_scalar_fail_does_not_commit_prior_lanes_or_cursor() -> None:
    first = -2147483648
    second = 2147483647
    buf = _le_i32(first) + _le_i32(second) + [0x11, 0x22, 0x33]

    cursor = [0]
    out_first = [-1]
    out_second = [-2]
    out_third = [-3]

    err = gguf_metadata_read_i32_triple_checked(
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


def test_success_reads_three_i32_and_advances() -> None:
    first = -123456789
    second = 0
    third = 123456789
    buf = [0xAB] + _le_i32(first) + _le_i32(second) + _le_i32(third) + [0xCD]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_i32_triple_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == first
    assert out_second[0] == second
    assert out_third[0] == third
    assert cursor[0] == 13


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 16

    cursor = [0]
    out_first = [1]
    out_second = [2]
    out_third = [3]
    err = gguf_metadata_read_i32_triple_checked(
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


def test_random_reference_parity() -> None:
    rng = random.Random(0x1324_5678)

    for _ in range(300):
        first = rng.randint(-(1 << 31), (1 << 31) - 1)
        second = rng.randint(-(1 << 31), (1 << 31) - 1)
        third = rng.randint(-(1 << 31), (1 << 31) - 1)

        prefix = [rng.randint(0, 255) for _ in range(rng.randint(0, 6))]
        suffix = [rng.randint(0, 255) for _ in range(rng.randint(0, 6))]
        buf = prefix + _le_i32(first) + _le_i32(second) + _le_i32(third) + suffix

        cursor = [len(prefix)]
        out_first = [0]
        out_second = [0]
        out_third = [0]

        err = gguf_metadata_read_i32_triple_checked(
            buf,
            len(buf),
            cursor,
            len(buf),
            out_first,
            out_second,
            out_third,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_first[0] == first
        assert out_second[0] == second
        assert out_third[0] == third
        assert cursor[0] == len(prefix) + 12


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_third_scalar_fail_does_not_commit_prior_lanes_or_cursor()
    test_success_reads_three_i32_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_random_reference_parity()
    print("ok")
