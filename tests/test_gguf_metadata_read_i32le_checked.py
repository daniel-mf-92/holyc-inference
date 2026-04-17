#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI32LEChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U32_MASK = (1 << 32) - 1


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


def reinterpret_u32_as_i32(value: int) -> int:
    value &= U32_MASK
    if value >= (1 << 31):
        return value - (1 << 32)
    return value


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
    raw = [0]

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u32_as_i32(raw[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out = [31415926]

    assert (
        gguf_metadata_read_i32le_checked(None, 32, cursor, 16, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 4
    assert out[0] == 31415926

    assert (
        gguf_metadata_read_i32le_checked([0] * 32, 32, None, 16, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out[0] == 31415926

    assert (
        gguf_metadata_read_i32le_checked([0] * 32, 32, cursor, 16, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 4


def test_short_reads_do_not_advance_or_write() -> None:
    buf = [0xAA] * 32
    out = [27182818]

    for table_end in range(0, 4):
        cursor = [0]
        err = gguf_metadata_read_i32le_checked(buf, len(buf), cursor, table_end, out)
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out[0] == 27182818


def test_success_signed_interpretation_and_cursor_advance() -> None:
    # 0xFFFFFFAA at offset 1 -> -86 as I32.
    buf = [0x11, 0xAA, 0xFF, 0xFF, 0xFF, 0x77]
    cursor = [1]
    out = [0]

    err = gguf_metadata_read_i32le_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == -86
    assert cursor[0] == 5


def test_min_i32_boundary() -> None:
    # 0x80000000 little-endian => -2147483648
    buf = [0x00, 0x00, 0x00, 0x80]
    cursor = [0]
    out = [1]

    err = gguf_metadata_read_i32le_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == -(1 << 31)
    assert cursor[0] == 4


def test_randomized_parity() -> None:
    rng = random.Random(20260417_202)

    for _ in range(5000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out = [-123456789]
        err = gguf_metadata_read_i32le_checked(buf, n, cursor, table_end, out)

        if cursor0 + 4 <= table_end and cursor0 + 4 <= n:
            raw = 0
            for i in range(4):
                raw |= buf[cursor0 + i] << (8 * i)
            expect = reinterpret_u32_as_i32(raw)
            assert err == GGUF_META_TABLE_OK
            assert out[0] == expect
            assert cursor[0] == cursor0 + 4
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out[0] == -123456789
            assert cursor[0] == cursor0


def test_table_end_overflow_surface() -> None:
    buf = [0x11, 0x22, 0x33, 0x44]
    cursor = [0]
    out = [7]

    err = gguf_metadata_read_i32le_checked(buf, len(buf), cursor, I64_MAX + 1, out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out[0] == 7


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_reads_do_not_advance_or_write()
    test_success_signed_interpretation_and_cursor_advance()
    test_min_i32_boundary()
    test_randomized_parity()
    test_table_end_overflow_surface()
    print("gguf_metadata_read_i32le_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
