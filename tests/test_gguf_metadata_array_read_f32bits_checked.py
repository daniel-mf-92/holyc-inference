#!/usr/bin/env python3
"""Parity checks for GGUFMetadataArrayReadF32BitsChecked semantics."""

from __future__ import annotations

import random
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_gguf_metadata_array_payload_span_from_fixed_type_checked import (
    GGUF_META_TABLE_ERR_BAD_PARAM,
    GGUF_META_TABLE_ERR_NULL_PTR,
    GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
    GGUF_META_TABLE_ERR_OVERFLOW,
    GGUF_META_TABLE_OK,
    GGUF_TYPE_FLOAT32,
    I64_MAX,
    gguf_metadata_array_payload_span_from_fixed_type_checked,
)

GGUF_MAX_ARRAY_ELEMS = 1 << 24


def gguf_metadata_array_read_f32bits_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    elem_count: int,
    out_bits: list[int] | None,
    out_bits_capacity: int,
) -> int:
    if buf is None or cursor_ref is None or out_bits is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if (
        buf_nbytes > I64_MAX
        or table_end > I64_MAX
        or elem_count > I64_MAX
        or out_bits_capacity > I64_MAX
    ):
        return GGUF_META_TABLE_ERR_OVERFLOW

    if elem_count > GGUF_MAX_ARRAY_ELEMS:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if out_bits_capacity < elem_count:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    cur = cursor_ref[0]
    if cur > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    if elem_count and (cur & 3):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    payload_bytes_ref = [0]
    payload_end_ref = [0]
    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_FLOAT32,
        elem_count,
        cur,
        table_end,
        buf_nbytes,
        payload_bytes_ref,
        payload_end_ref,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    expected_payload_bytes = elem_count << 2
    if payload_bytes_ref[0] != expected_payload_bytes:
        return GGUF_META_TABLE_ERR_OVERFLOW

    payload_end = payload_end_ref[0]
    if elem_count and (payload_end & 3):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    for lane in range(elem_count):
        lane_base = cur + (lane << 2)
        out_bits[lane] = (
            buf[lane_base]
            | (buf[lane_base + 1] << 8)
            | (buf[lane_base + 2] << 16)
            | (buf[lane_base + 3] << 24)
        )

    cursor_ref[0] = payload_end
    return GGUF_META_TABLE_OK


def explicit_checked_composition(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    elem_count: int,
    out_bits: list[int] | None,
    out_bits_capacity: int,
) -> int:
    if buf is None or cursor_ref is None or out_bits is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if (
        buf_nbytes > I64_MAX
        or table_end > I64_MAX
        or elem_count > I64_MAX
        or out_bits_capacity > I64_MAX
    ):
        return GGUF_META_TABLE_ERR_OVERFLOW

    if elem_count > GGUF_MAX_ARRAY_ELEMS:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if out_bits_capacity < elem_count:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    cur = cursor_ref[0]
    if cur > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    if elem_count and (cur & 3):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    payload_bytes_ref = [0]
    payload_end_ref = [0]
    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_FLOAT32,
        elem_count,
        cur,
        table_end,
        buf_nbytes,
        payload_bytes_ref,
        payload_end_ref,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    expected_payload_bytes = elem_count << 2
    if payload_bytes_ref[0] != expected_payload_bytes:
        return GGUF_META_TABLE_ERR_OVERFLOW

    payload_end = payload_end_ref[0]
    if elem_count and (payload_end & 3):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    staged = [0] * elem_count
    for lane in range(elem_count):
        lane_base = cur + (lane << 2)
        staged[lane] = (
            buf[lane_base]
            | (buf[lane_base + 1] << 8)
            | (buf[lane_base + 2] << 16)
            | (buf[lane_base + 3] << 24)
        )

    for lane in range(elem_count):
        out_bits[lane] = staged[lane]

    cursor_ref[0] = payload_end
    return GGUF_META_TABLE_OK


def _le_u32(v: int) -> list[int]:
    return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF]


def test_null_ptr_and_no_partial_outputs() -> None:
    out = [0xDEAD_BEEF, 0xABCD_1234]
    cursor = [4]

    err = gguf_metadata_array_read_f32bits_checked(None, 16, cursor, 16, 1, out, 2)
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [4]
    assert out == [0xDEAD_BEEF, 0xABCD_1234]

    err = gguf_metadata_array_read_f32bits_checked([0] * 16, 16, None, 16, 1, out, 2)
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert out == [0xDEAD_BEEF, 0xABCD_1234]


def test_alignment_and_capacity_guards() -> None:
    out = [0xFACE_CAFE, 0x1234_5678]

    cursor = [2]
    err = gguf_metadata_array_read_f32bits_checked([0] * 32, 32, cursor, 32, 1, out, 2)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [2]
    assert out == [0xFACE_CAFE, 0x1234_5678]

    cursor = [0]
    err = gguf_metadata_array_read_f32bits_checked([0] * 32, 32, cursor, 32, 3, out, 2)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out == [0xFACE_CAFE, 0x1234_5678]

    err = gguf_metadata_array_read_f32bits_checked(
        [0] * 32,
        32,
        cursor,
        32,
        GGUF_MAX_ARRAY_ELEMS + 1,
        out,
        GGUF_MAX_ARRAY_ELEMS + 1,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out == [0xFACE_CAFE, 0x1234_5678]


def test_overflow_and_out_of_bounds_surfaces() -> None:
    out = [0x1111_1111, 0x2222_2222]
    cursor = [0]

    err = gguf_metadata_array_read_f32bits_checked([0] * 8, 1 << 63, cursor, 8, 1, out, 2)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out == [0x1111_1111, 0x2222_2222]

    err = gguf_metadata_array_read_f32bits_checked([0] * 8, 8, [1 << 63], 8, 1, out, 2)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out == [0x1111_1111, 0x2222_2222]

    cursor = [0]
    err = gguf_metadata_array_read_f32bits_checked([0] * 7, 7, cursor, 7, 2, out, 2)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out == [0x1111_1111, 0x2222_2222]


def test_success_reads_f32_bit_patterns() -> None:
    bits = [
        struct.unpack("<I", struct.pack("<f", 1.0))[0],
        struct.unpack("<I", struct.pack("<f", -2.5))[0],
        struct.unpack("<I", struct.pack("<f", 0.03125))[0],
    ]
    buf = [0xAA, 0xBB, 0xCC, 0xDD] + _le_u32(bits[0]) + _le_u32(bits[1]) + _le_u32(bits[2])

    out = [0, 0, 0, 0]
    cursor = [4]
    err = gguf_metadata_array_read_f32bits_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        3,
        out,
        len(out),
    )
    assert err == GGUF_META_TABLE_OK
    assert out[:3] == bits
    assert out[3] == 0
    assert cursor == [16]


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260420_683)

    for _ in range(4000):
        buf_nbytes = rng.randint(0, 2048)
        buf = [rng.randrange(0, 256) for _ in range(buf_nbytes)]

        table_end = rng.randint(0, 2048)
        cursor_val = rng.randint(0, 2048)
        elem_count = rng.randint(0, 64)
        out_cap = rng.randint(0, 96)

        # Align some vectors so success paths are exercised.
        if rng.random() < 0.35:
            cursor_val &= ~0x3
        if rng.random() < 0.35:
            table_end &= ~0x3

        out_a = [0xA5A5_A5A5] * max(out_cap, 1)
        out_b = out_a.copy()

        cursor_a = [cursor_val]
        cursor_b = [cursor_val]

        err_a = gguf_metadata_array_read_f32bits_checked(
            buf,
            buf_nbytes,
            cursor_a,
            table_end,
            elem_count,
            out_a,
            out_cap,
        )
        err_b = explicit_checked_composition(
            buf,
            buf_nbytes,
            cursor_b,
            table_end,
            elem_count,
            out_b,
            out_cap,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert out_a == out_b


if __name__ == "__main__":
    test_null_ptr_and_no_partial_outputs()
    test_alignment_and_capacity_guards()
    test_overflow_and_out_of_bounds_surfaces()
    test_success_reads_f32_bit_patterns()
    test_randomized_parity_against_explicit_checked_composition()
    print("ok")
