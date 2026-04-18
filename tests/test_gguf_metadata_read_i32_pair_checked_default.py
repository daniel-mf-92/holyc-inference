#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI32PairCheckedDefault."""

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


def gguf_metadata_read_i32_pair_checked(
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

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_pair_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i32_pair_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
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
    cursor = [11]
    out_first = [777]
    out_second = [888]

    assert (
        gguf_metadata_read_i32_pair_checked_default(
            None,
            64,
            cursor,
            out_first,
            out_second,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [11]
    assert out_first == [777]
    assert out_second == [888]


def test_success_reads_pair_and_advances() -> None:
    first = -2147483648
    second = 2147483647
    prefix = [0xAA, 0xBB]
    buf = prefix + _le_i32(first) + _le_i32(second) + [0xDD]

    cursor = [2]
    out_first = [0]
    out_second = [0]

    err = gguf_metadata_read_i32_pair_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
    )

    assert err == GGUF_META_TABLE_OK
    assert cursor == [10]
    assert out_first == [first]
    assert out_second == [second]


def test_truncation_matches_checked_and_no_commit() -> None:
    first = -99
    buf = _le_i32(first) + [0x01, 0x02, 0x03]

    d_cursor = [0]
    c_cursor = [0]
    d_out0 = [1234]
    d_out1 = [5678]
    c_out0 = [1234]
    c_out1 = [5678]

    err_default = gguf_metadata_read_i32_pair_checked_default(
        buf,
        len(buf),
        d_cursor,
        d_out0,
        d_out1,
    )
    err_checked = gguf_metadata_read_i32_pair_checked(
        buf,
        len(buf),
        c_cursor,
        len(buf),
        c_out0,
        c_out1,
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert d_cursor == c_cursor == [0]
    assert d_out0 == c_out0 == [1234]
    assert d_out1 == c_out1 == [5678]


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0] * 16

    d_cursor = [4]
    c_cursor = [4]
    d_out0 = [11]
    d_out1 = [22]
    c_out0 = [11]
    c_out1 = [22]

    err_default = gguf_metadata_read_i32_pair_checked_default(
        buf,
        I64_MAX + 1,
        d_cursor,
        d_out0,
        d_out1,
    )
    err_checked = gguf_metadata_read_i32_pair_checked(
        buf,
        I64_MAX + 1,
        c_cursor,
        I64_MAX + 1,
        c_out0,
        c_out1,
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OVERFLOW
    assert d_cursor == c_cursor == [4]
    assert d_out0 == c_out0 == [11]
    assert d_out1 == c_out1 == [22]


def test_randomized_default_vs_checked_equivalence() -> None:
    rng = random.Random(20260418_319)

    for _ in range(6000):
        n = rng.randint(0, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]

        cursor0 = rng.randint(0, max(0, n + 10))
        out0 = _to_i32(rng.getrandbits(32))
        out1 = _to_i32(rng.getrandbits(32))

        d_cursor = [cursor0]
        c_cursor = [cursor0]
        d_out0 = [out0]
        d_out1 = [out1]
        c_out0 = [out0]
        c_out1 = [out1]

        err_default = gguf_metadata_read_i32_pair_checked_default(
            buf,
            n,
            d_cursor,
            d_out0,
            d_out1,
        )
        err_checked = gguf_metadata_read_i32_pair_checked(
            buf,
            n,
            c_cursor,
            n,
            c_out0,
            c_out1,
        )

        assert err_default == err_checked
        assert d_cursor == c_cursor
        assert d_out0 == c_out0
        assert d_out1 == c_out1


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_reads_pair_and_advances()
    test_truncation_matches_checked_and_no_commit()
    test_overflow_passthrough_and_no_commit()
    test_randomized_default_vs_checked_equivalence()
    print("ok")
