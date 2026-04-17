#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadF32BitsLECheckedNoPartial semantics."""

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


def gguf_metadata_read_f32bitsle_checked_no_partial(
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

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, raw_bits)
    if err != GGUF_META_TABLE_OK:
        return err

    out_bits_ref[0] = raw_bits[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f32bitsle_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_bits_ref: list[int] | None,
) -> int:
    # HolyC entrypoint delegates directly to no-partial helper.
    return gguf_metadata_read_f32bitsle_checked_no_partial(
        buf,
        buf_nbytes,
        cursor_ref,
        table_end,
        out_bits_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_bits = [0xDEADBEEF]

    assert (
        gguf_metadata_read_f32bitsle_checked_no_partial(None, 16, cursor, 16, out_bits)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3
    assert out_bits[0] == 0xDEADBEEF

    assert (
        gguf_metadata_read_f32bitsle_checked_no_partial(
            [0] * 16,
            16,
            None,
            16,
            out_bits,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_bits[0] == 0xDEADBEEF

    assert (
        gguf_metadata_read_f32bitsle_checked_no_partial(
            [0] * 16,
            16,
            cursor,
            16,
            None,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3


def test_short_reads_do_not_advance_or_write() -> None:
    buf = [0xFF] * 16
    out_bits = [0xABCD1234]

    for table_end in range(0, 4):
        cursor = [0]
        err = gguf_metadata_read_f32bitsle_checked_no_partial(
            buf,
            len(buf),
            cursor,
            table_end,
            out_bits,
        )
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out_bits[0] == 0xABCD1234


def test_success_decodes_little_endian_and_advances() -> None:
    buf = [
        0xAA,
        0xBB,
        0x01,
        0x23,
        0x45,
        0x67,
        0x89,
        0x11,
    ]

    cursor = [2]
    out_bits = [0]
    err = gguf_metadata_read_f32bitsle_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_bits,
    )

    assert err == GGUF_META_TABLE_OK
    assert out_bits[0] == 0x67452301
    assert cursor[0] == 6


def test_known_ieee754_payloads_and_nan_bits_preserved() -> None:
    values = [0.0, -0.0, 1.0, -2.5, float("inf"), float("-inf")]
    payload = bytearray([0xAA, 0x55])
    expected_bits = []

    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        expected_bits.append(bits)
        payload.extend(struct.pack("<I", bits))

    nan_bits = 0x7FC12345
    expected_bits.append(nan_bits)
    payload.extend(struct.pack("<I", nan_bits))

    cursor = [2]
    out_bits = [0]

    for expected in expected_bits:
        err = gguf_metadata_read_f32bitsle_checked_no_partial(
            list(payload),
            len(payload),
            cursor,
            len(payload),
            out_bits,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_bits[0] == expected

    assert cursor[0] == 2 + 4 * len(expected_bits)


def test_wrapper_delegates_identically() -> None:
    rng = random.Random(20260417_225)

    for _ in range(2500):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor_core = [cursor0]
        out_core = [0xC0DEC0DE]
        err_core = gguf_metadata_read_f32bitsle_checked_no_partial(
            buf,
            n,
            cursor_core,
            table_end,
            out_core,
        )

        cursor_wrap = [cursor0]
        out_wrap = [0xC0DEC0DE]
        err_wrap = gguf_metadata_read_f32bitsle_checked(
            buf,
            n,
            cursor_wrap,
            table_end,
            out_wrap,
        )

        assert err_wrap == err_core
        assert cursor_wrap[0] == cursor_core[0]
        assert out_wrap[0] == out_core[0]


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_reads_do_not_advance_or_write()
    test_success_decodes_little_endian_and_advances()
    test_known_ieee754_payloads_and_nan_bits_preserved()
    test_wrapper_delegates_identically()
    print("gguf_metadata_read_f32bitsle_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
