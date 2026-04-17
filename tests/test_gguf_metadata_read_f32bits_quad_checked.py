#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadF32BitsQuadChecked semantics."""

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


def gguf_metadata_read_f32bits_quad_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
        or out_fourth_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]
    fourth = [0]

    err = gguf_metadata_read_f32bitsle_checked_no_partial(
        buf, buf_nbytes, cur, table_end, first
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f32bitsle_checked_no_partial(
        buf, buf_nbytes, cur, table_end, second
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f32bitsle_checked_no_partial(
        buf, buf_nbytes, cur, table_end, third
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f32bitsle_checked_no_partial(
        buf, buf_nbytes, cur, table_end, fourth
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def _le_u32(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(4)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [7]
    out_first = [0xAAAAAAAA]
    out_second = [0xBBBBBBBB]
    out_third = [0xCCCCCCCC]
    out_fourth = [0xDDDDDDDD]

    assert (
        gguf_metadata_read_f32bits_quad_checked(
            None,
            32,
            cursor,
            32,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [7]
    assert out_first == [0xAAAAAAAA]
    assert out_second == [0xBBBBBBBB]
    assert out_third == [0xCCCCCCCC]
    assert out_fourth == [0xDDDDDDDD]


def test_fourth_lane_fail_does_not_commit_prior_or_cursor() -> None:
    first = struct.unpack("<I", struct.pack("<f", 1.5))[0]
    second = struct.unpack("<I", struct.pack("<f", -0.75))[0]
    third = struct.unpack("<I", struct.pack("<f", 9.125))[0]
    buf = _le_u32(first) + _le_u32(second) + _le_u32(third) + [0x11, 0x22, 0x33]

    cursor = [0]
    out_first = [0x11111111]
    out_second = [0x22222222]
    out_third = [0x33333333]
    out_fourth = [0x44444444]

    err = gguf_metadata_read_f32bits_quad_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x11111111]
    assert out_second == [0x22222222]
    assert out_third == [0x33333333]
    assert out_fourth == [0x44444444]


def test_success_reads_four_raw_bit_patterns_and_advances() -> None:
    first = struct.unpack("<I", struct.pack("<f", 0.0))[0]
    second = struct.unpack("<I", struct.pack("<f", -2.5))[0]
    third = struct.unpack("<I", struct.pack("<f", 123.25))[0]
    fourth = struct.unpack("<I", struct.pack("<f", -77.125))[0]

    buf = [0xAB] + _le_u32(first) + _le_u32(second) + _le_u32(third) + _le_u32(fourth) + [0xCD]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_f32bits_quad_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == first
    assert out_second[0] == second
    assert out_third[0] == third
    assert out_fourth[0] == fourth
    assert cursor[0] == 17


def test_i64_domain_overflow_surfaces_and_no_commit() -> None:
    buf = [0x44] * 64
    out_first_seed = 0x0BADF00D
    out_second_seed = 0xDEADC0DE
    out_third_seed = 0xCAFEBABE
    out_fourth_seed = 0xFEEDFACE

    cursor = [0]
    out_first = [out_first_seed]
    out_second = [out_second_seed]
    out_third = [out_third_seed]
    out_fourth = [out_fourth_seed]

    err = gguf_metadata_read_f32bits_quad_checked(
        buf,
        1 << 63,
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_first == [out_first_seed]
    assert out_second == [out_second_seed]
    assert out_third == [out_third_seed]
    assert out_fourth == [out_fourth_seed]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_253)

    for _ in range(7000):
        n = rng.randint(1, 320)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, n)

        cursor = [cursor0]
        out_first = [0x12345678]
        out_second = [0x9ABCDEF0]
        out_third = [0x0BADF00D]
        out_fourth = [0xFEEDFACE]

        err = gguf_metadata_read_f32bits_quad_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )

        if err == GGUF_META_TABLE_OK:
            assert cursor[0] == cursor0 + 16
            expected = []
            for lane in range(4):
                base = cursor0 + lane * 4
                expected.append(
                    buf[base]
                    | (buf[base + 1] << 8)
                    | (buf[base + 2] << 16)
                    | (buf[base + 3] << 24)
                )
            assert out_first[0] == expected[0]
            assert out_second[0] == expected[1]
            assert out_third[0] == expected[2]
            assert out_fourth[0] == expected[3]
        else:
            assert cursor[0] == cursor0
            assert out_first[0] == 0x12345678
            assert out_second[0] == 0x9ABCDEF0
            assert out_third[0] == 0x0BADF00D
            assert out_fourth[0] == 0xFEEDFACE
            assert err in (
                GGUF_META_TABLE_ERR_BAD_PARAM,
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
            )


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_lane_fail_does_not_commit_prior_or_cursor()
    test_success_reads_four_raw_bit_patterns_and_advances()
    test_i64_domain_overflow_surfaces_and_no_commit()
    test_randomized_parity()
    print("ok")
