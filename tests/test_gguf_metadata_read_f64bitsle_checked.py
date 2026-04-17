#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadF64BitsLEChecked semantics."""

from __future__ import annotations

import random
import struct

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
    raw_bits = [0]

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, raw_bits)
    if err != GGUF_META_TABLE_OK:
        return err

    out_bits_ref[0] = raw_bits[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_bits = [0xBAD0BAD0BAD0BAD0]

    assert (
        gguf_metadata_read_f64bitsle_checked(None, 16, cursor, 16, out_bits)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 5
    assert out_bits[0] == 0xBAD0BAD0BAD0BAD0

    assert (
        gguf_metadata_read_f64bitsle_checked([0] * 16, 16, None, 16, out_bits)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_bits[0] == 0xBAD0BAD0BAD0BAD0

    assert (
        gguf_metadata_read_f64bitsle_checked([0] * 16, 16, cursor, 16, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 5


def test_short_reads_do_not_advance_or_write() -> None:
    buf = [0xFF] * 16
    out_bits = [0xCAFEBABECAFEBABE]

    for table_end in range(0, 8):
        cursor = [0]
        err = gguf_metadata_read_f64bitsle_checked(
            buf,
            len(buf),
            cursor,
            table_end,
            out_bits,
        )
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out_bits[0] == 0xCAFEBABECAFEBABE


def test_success_preserves_known_ieee754_bit_patterns() -> None:
    values = [0.0, -0.0, 1.0, -2.5, float("inf"), float("-inf")]
    payload = bytearray([0xAA, 0x55])
    expected_bits = []

    for value in values:
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        expected_bits.append(bits)
        payload.extend(struct.pack("<Q", bits))

    cursor = [2]
    out_bits = [0]

    for expected in expected_bits:
        err = gguf_metadata_read_f64bitsle_checked(
            list(payload),
            len(payload),
            cursor,
            len(payload),
            out_bits,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_bits[0] == expected

    assert cursor[0] == 2 + 8 * len(expected_bits)


def test_nan_payload_bits_are_preserved_exactly() -> None:
    # Quiet NaN with non-trivial payload bits.
    nan_bits = 0x7FF8_1234_89AB_CDEF
    payload = list(struct.pack("<Q", nan_bits))

    cursor = [0]
    out_bits = [0]

    err = gguf_metadata_read_f64bitsle_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        out_bits,
    )

    assert err == GGUF_META_TABLE_OK
    assert out_bits[0] == nan_bits
    assert cursor[0] == 8




def test_cursor_beyond_table_end_is_bad_param_and_no_commit() -> None:
    buf = [0x11] * 16
    cursor = [9]
    out_bits = [0xDEADBEEFDEADBEEF]

    err = gguf_metadata_read_f64bitsle_checked(
        buf,
        len(buf),
        cursor,
        8,
        out_bits,
    )

    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 9
    assert out_bits[0] == 0xDEADBEEFDEADBEEF


def test_i64_domain_overflow_surfaces_and_no_commit() -> None:
    buf = [0x22] * 16
    out_seed = 0x123456789ABCDEF0

    cursor = [0]
    out_bits = [out_seed]
    err = gguf_metadata_read_f64bitsle_checked(buf, (1 << 63), cursor, 16, out_bits)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out_bits[0] == out_seed

    cursor = [0]
    out_bits = [out_seed]
    err = gguf_metadata_read_f64bitsle_checked(buf, len(buf), cursor, (1 << 63), out_bits)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out_bits[0] == out_seed

    cursor = [(1 << 63)]
    out_bits = [out_seed]
    err = gguf_metadata_read_f64bitsle_checked(buf, len(buf), cursor, len(buf), out_bits)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == (1 << 63)
    assert out_bits[0] == out_seed
def test_randomized_parity() -> None:
    rng = random.Random(20260417_198)

    for _ in range(5000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_bits = [0xA1A2A3A4A5A6A7A8]

        err = gguf_metadata_read_f64bitsle_checked(buf, n, cursor, table_end, out_bits)

        if cursor0 + 8 <= table_end and cursor0 + 8 <= n:
            expect = 0
            for i in range(8):
                expect |= buf[cursor0 + i] << (8 * i)
            assert err == GGUF_META_TABLE_OK
            assert out_bits[0] == expect
            assert cursor[0] == cursor0 + 8
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out_bits[0] == 0xA1A2A3A4A5A6A7A8
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_reads_do_not_advance_or_write()
    test_success_preserves_known_ieee754_bit_patterns()
    test_nan_payload_bits_are_preserved_exactly()
    test_cursor_beyond_table_end_is_bad_param_and_no_commit()
    test_i64_domain_overflow_surfaces_and_no_commit()
    test_randomized_parity()
    print("gguf_metadata_read_f64bitsle_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
