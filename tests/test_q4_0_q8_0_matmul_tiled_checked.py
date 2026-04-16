#!/usr/bin/env python3
"""Reference checks for checked tiled mixed Q4_0 x Q8_0 matmul semantics."""

from __future__ import annotations

import random
import struct

Q4_0_Q8_0_OK = 0
Q4_0_Q8_0_ERR_NULL_PTR = 1
Q4_0_Q8_0_ERR_BAD_DST_LEN = 2
Q4_0_Q8_0_ERR_OVERFLOW = 3
Q4_0_Q8_0_I64_MAX = (1 << 63) - 1
Q4_0_Q8_0_I64_MIN = -(1 << 63)


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    round_bias = 1 << (shift - 1)
    return (value + round_bias) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


def f16_to_q16(fp16_bits: int) -> int:
    sign_bit = (fp16_bits >> 15) & 1
    exponent_bits = (fp16_bits >> 10) & 0x1F
    fraction_bits = fp16_bits & 0x03FF

    if exponent_bits == 0:
        if fraction_bits == 0:
            return 0
        magnitude_q16 = round_shift_right_unsigned(fraction_bits, 8)
        return -magnitude_q16 if sign_bit else magnitude_q16

    if exponent_bits == 0x1F:
        return -(0x3FFFFFFFFFFFFFFF) if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9
    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        magnitude_q16 = round_shift_right_unsigned(mantissa, -shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def dot_q32_to_q16(dot_q32: int) -> int:
    return round_shift_right_signed(dot_q32, 16)


def pack_q4_from_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    out = bytearray()
    for idx in range(0, 32, 2):
        lo = vals[idx] + 8
        hi = vals[idx + 1] + 8
        assert 0 <= lo <= 15 and 0 <= hi <= 15
        out.append((lo & 0x0F) | ((hi & 0x0F) << 4))
    return bytes(out)


def pack_q8_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    return bytes((v + 256) % 256 for v in vals)


def nibble_to_signed(packed: int, upper_nibble: bool) -> int:
    if upper_nibble:
        q_unsigned = (packed >> 4) & 0x0F
    else:
        q_unsigned = packed & 0x0F
    return q_unsigned - 8


def unpack_q4_signed(qs: bytes) -> list[int]:
    out: list[int] = []
    for packed in qs:
        out.append(nibble_to_signed(packed, False))
        out.append(nibble_to_signed(packed, True))
    return out


def unpack_q8_signed(qs: bytes) -> list[int]:
    return [struct.unpack("<b", bytes([byte]))[0] for byte in qs]


def dot_product_block_q32(lhs_scale_fp16: int, lhs_q4: bytes, rhs_scale_fp16: int, rhs_q8: bytes) -> tuple[int, int]:
    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16

    lhs_signed = unpack_q4_signed(lhs_q4)
    rhs_signed = unpack_q8_signed(rhs_q8)
    q_dot_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))
    return Q4_0_Q8_0_OK, scale_prod_q32 * q_dot_q0


def try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > Q4_0_Q8_0_I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < Q4_0_Q8_0_I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def dot_product_blocks_q16_accumulate_checked(lhs_blocks, rhs_blocks, initial_accum_q16: int) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total_q16 = initial_accum_q16
    for (l_scale, l_q4), (r_scale, r_q8) in zip(lhs_blocks, rhs_blocks):
        err, block_dot_q32 = dot_product_block_q32(l_scale, l_q4, r_scale, r_q8)
        if err != Q4_0_Q8_0_OK:
            return err, 0
        block_dot_q16 = dot_q32_to_q16(block_dot_q32)
        ok, checked = try_add_i64(total_q16, block_dot_q16)
        if not ok:
            return Q4_0_Q8_0_ERR_OVERFLOW, 0
        total_q16 = checked

    return Q4_0_Q8_0_OK, total_q16


def q4_0_q8_0_matmul_q16_tiled_checked(
    lhs_q4_blocks,
    lhs_q4_block_capacity: int,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    rhs_q8_block_capacity: int,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    tile_rows: int,
    tile_cols: int,
    out_cell_capacity: int,
    out_row_stride_cells: int,
):
    if lhs_q4_blocks is None or rhs_q8_col_blocks is None:
        return Q4_0_Q8_0_ERR_NULL_PTR, []
    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    if row_count > 0 and out_row_stride_cells < col_count:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    lhs_required = row_count * lhs_row_stride_blocks
    rhs_required = col_count * rhs_col_stride_blocks
    out_required = row_count * out_row_stride_cells

    if lhs_required > Q4_0_Q8_0_I64_MAX:
        return Q4_0_Q8_0_ERR_OVERFLOW, []
    if rhs_required > Q4_0_Q8_0_I64_MAX:
        return Q4_0_Q8_0_ERR_OVERFLOW, []
    if out_required > Q4_0_Q8_0_I64_MAX:
        return Q4_0_Q8_0_ERR_OVERFLOW, []

    if lhs_required > lhs_q4_block_capacity:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if rhs_required > rhs_q8_block_capacity:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if out_required > out_cell_capacity:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    out_cells_q16 = [0] * out_required

    row_tile_start = 0
    while row_tile_start < row_count:
        row_tile_end = min(row_count, row_tile_start + tile_rows)

        col_tile_start = 0
        while col_tile_start < col_count:
            col_tile_end = min(col_count, col_tile_start + tile_cols)

            for row_index in range(row_tile_start, row_tile_end):
                lhs_row_base = row_index * lhs_row_stride_blocks
                out_row_base = row_index * out_row_stride_cells
                lhs_row_slice = lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count]

                for col_index in range(col_tile_start, col_tile_end):
                    rhs_col_base = col_index * rhs_col_stride_blocks
                    rhs_col_slice = rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
                    out_index = out_row_base + col_index

                    err, cell_dot_q16 = dot_product_blocks_q16_accumulate_checked(
                        lhs_row_slice,
                        rhs_col_slice,
                        0,
                    )
                    if err != Q4_0_Q8_0_OK:
                        return err, []

                    out_cells_q16[out_index] = cell_dot_q16

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q4_0_Q8_0_OK, out_cells_q16


def q4_0_q8_0_matmul_q16_reference_untiled(
    lhs_q4_blocks,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
):
    out_required = row_count * out_row_stride_cells
    out = [0] * out_required

    for row_index in range(row_count):
        lhs_row_base = row_index * lhs_row_stride_blocks
        lhs_row_slice = lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count]
        if len(lhs_row_slice) != k_block_count:
            return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

        for col_index in range(col_count):
            rhs_col_base = col_index * rhs_col_stride_blocks
            rhs_col_slice = rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
            if len(rhs_col_slice) != k_block_count:
                return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

            err, dot_q16 = dot_product_blocks_q16_accumulate_checked(lhs_row_slice, rhs_col_slice, 0)
            if err != Q4_0_Q8_0_OK:
                return err, []

            out[row_index * out_row_stride_cells + col_index] = dot_q16

    return Q4_0_Q8_0_OK, out


def make_q4_block(rng: random.Random, *, scale: float | None = None):
    if scale is None:
        scale = rng.uniform(-2.0, 2.0)
    qs = [rng.randint(-8, 7) for _ in range(32)]
    return half_bits(scale), pack_q4_from_signed(qs)


def make_q8_block(rng: random.Random, *, scale: float | None = None):
    if scale is None:
        scale = rng.uniform(-2.0, 2.0)
    qs = [rng.randint(-128, 127) for _ in range(32)]
    return half_bits(scale), pack_q8_signed(qs)


def test_tiled_matches_untiled_randomized() -> None:
    rng = random.Random(2026041611)

    for _ in range(220):
        row_count = rng.randint(1, 7)
        col_count = rng.randint(1, 7)
        k_block_count = rng.randint(1, 6)

        lhs_row_stride_blocks = k_block_count + rng.randint(0, 3)
        rhs_col_stride_blocks = k_block_count + rng.randint(0, 3)
        out_row_stride_cells = col_count + rng.randint(0, 3)

        lhs_capacity = row_count * lhs_row_stride_blocks
        rhs_capacity = col_count * rhs_col_stride_blocks
        out_capacity = row_count * out_row_stride_cells

        lhs_q4_blocks = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs_q8_blocks = [make_q8_block(rng) for _ in range(rhs_capacity)]

        tile_rows = rng.randint(1, max(1, row_count + 2))
        tile_cols = rng.randint(1, max(1, col_count + 2))

        err_tiled, out_tiled = q4_0_q8_0_matmul_q16_tiled_checked(
            lhs_q4_blocks,
            lhs_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_q8_blocks,
            rhs_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            tile_rows,
            tile_cols,
            out_capacity,
            out_row_stride_cells,
        )
        assert err_tiled == Q4_0_Q8_0_OK

        err_ref, out_ref = q4_0_q8_0_matmul_q16_reference_untiled(
            lhs_q4_blocks,
            row_count,
            lhs_row_stride_blocks,
            rhs_q8_blocks,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            out_row_stride_cells,
        )
        assert err_ref == Q4_0_Q8_0_OK
        assert out_tiled == out_ref


def test_rejects_bad_dimensions_and_capacities() -> None:
    rng = random.Random(2026041612)
    lhs_blocks = [make_q4_block(rng) for _ in range(6)]
    rhs_blocks = [make_q8_block(rng) for _ in range(6)]

    err, _ = q4_0_q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        -1,
        2,
        3,
        rhs_blocks,
        6,
        2,
        3,
        2,
        1,
        1,
        6,
        2,
    )
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = q4_0_q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        6,
        2,
        1,
        rhs_blocks,
        6,
        2,
        3,
        2,
        1,
        1,
        6,
        2,
    )
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = q4_0_q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        6,
        2,
        3,
        rhs_blocks,
        6,
        2,
        3,
        2,
        0,
        1,
        6,
        2,
    )
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = q4_0_q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        5,
        2,
        3,
        rhs_blocks,
        6,
        2,
        3,
        2,
        1,
        1,
        6,
        2,
    )
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_reports_extent_multiply_overflow() -> None:
    rng = random.Random(2026041613)
    lhs_blocks = [make_q4_block(rng)]
    rhs_blocks = [make_q8_block(rng)]

    err, _ = q4_0_q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        Q4_0_Q8_0_I64_MAX,
        Q4_0_Q8_0_I64_MAX,
        2,
        rhs_blocks,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert err == Q4_0_Q8_0_ERR_OVERFLOW


def test_reports_cell_accumulator_overflow() -> None:
    scale_pos_max = 0x7C00
    plus_one_q8 = pack_q8_signed([1] + [0] * 31)
    plus_seven_q4 = pack_q4_from_signed([7] + [0] * 31)

    vec_col_blocks = [
        (scale_pos_max, plus_one_q8),
        (scale_pos_max, plus_one_q8),
    ]
    lhs_blocks = [
        (scale_pos_max, plus_seven_q4),
        (scale_pos_max, plus_seven_q4),
    ]

    err, _ = q4_0_q8_0_matmul_q16_tiled_checked(
        lhs_blocks,
        2,
        1,
        2,
        vec_col_blocks,
        2,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
    )
    assert err == Q4_0_Q8_0_ERR_OVERFLOW


def run() -> None:
    test_tiled_matches_untiled_randomized()
    test_rejects_bad_dimensions_and_capacities()
    test_reports_extent_multiply_overflow()
    test_reports_cell_accumulator_overflow()
    print("q4_0_q8_0_matmul_tiled_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
