#!/usr/bin/env python3
"""Reference checks for Q4_0Q8_0MatMulQ32TiledAVX2Checked semantics."""

from __future__ import annotations

import random

Q4_0_Q8_0_AVX2_VALUES_PER_BLOCK = 32
Q4_0_Q8_0_AVX2_Q4_PACKED_BYTES = 16

Q4_0_Q8_0_AVX2_OK = 0
Q4_0_Q8_0_AVX2_ERR_NULL_PTR = 1
Q4_0_Q8_0_AVX2_ERR_BAD_LEN = 2
Q4_0_Q8_0_AVX2_ERR_OVERFLOW = 3

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def try_mul_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def try_add_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > I64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def f16_to_q16(fp16_bits: int) -> int:
    sign_bit = (fp16_bits >> 15) & 1
    exponent_bits = (fp16_bits >> 10) & 0x1F
    fraction_bits = fp16_bits & 0x03FF

    if exponent_bits == 0:
        if fraction_bits == 0:
            return 0
        magnitude_q16 = (fraction_bits + (1 << 7)) >> 8
        return -magnitude_q16 if sign_bit else magnitude_q16

    if exponent_bits == 0x1F:
        return -0x3FFFFFFFFFFFFFFF if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9
    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        rounding = 1 << ((-shift_amount) - 1)
        magnitude_q16 = (mantissa + rounding) >> (-shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def q4_nibble_to_signed(packed: int, high_nibble: bool) -> int:
    if high_nibble:
        return ((packed >> 4) & 0x0F) - 8
    return (packed & 0x0F) - 8


def q4_pack_signed(values: list[int]) -> bytes:
    assert len(values) == Q4_0_Q8_0_AVX2_VALUES_PER_BLOCK
    packed = bytearray(Q4_0_Q8_0_AVX2_Q4_PACKED_BYTES)
    for index in range(0, Q4_0_Q8_0_AVX2_VALUES_PER_BLOCK, 2):
        low = values[index] + 8
        high = values[index + 1] + 8
        assert 0 <= low <= 15
        assert 0 <= high <= 15
        packed[index >> 1] = low | (high << 4)
    return bytes(packed)


def q8_pack_signed(values: list[int]) -> bytes:
    assert len(values) == Q4_0_Q8_0_AVX2_VALUES_PER_BLOCK
    return bytes(v & 0xFF for v in values)


def q4_unpack_signed(qs_packed: bytes) -> list[int]:
    out: list[int] = []
    for packed in qs_packed:
        out.append(q4_nibble_to_signed(packed, False))
        out.append(q4_nibble_to_signed(packed, True))
    return out


def q8_unpack_signed(qs: bytes) -> list[int]:
    return [q - 256 if q >= 128 else q for q in qs]


def dot_block_avx2_q32_checked(lhs_block, rhs_block) -> tuple[int, int]:
    lhs_scale_fp16, lhs_q4_packed = lhs_block
    rhs_scale_fp16, rhs_q8_packed = rhs_block

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)

    ok, scale_prod_q32 = try_mul_i64(lhs_scale_q16, rhs_scale_q16)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, 0

    lhs_vals = q4_unpack_signed(lhs_q4_packed)
    rhs_vals = q8_unpack_signed(rhs_q8_packed)
    q_dot_q0 = sum(int(a) * int(b) for a, b in zip(lhs_vals, rhs_vals))

    ok, block_dot_q32 = try_mul_i64(scale_prod_q32, q_dot_q0)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q4_0_Q8_0_AVX2_OK, block_dot_q32


def dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count: int) -> tuple[int, int]:
    if lhs_blocks is None or rhs_blocks is None:
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, 0
    if len(lhs_blocks) < block_count or len(rhs_blocks) < block_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, 0

    total = 0
    for index in range(block_count):
        err, block_dot_q32 = dot_block_avx2_q32_checked(lhs_blocks[index], rhs_blocks[index])
        if err != Q4_0_Q8_0_AVX2_OK:
            return err, 0
        ok, total = try_add_i64(total, block_dot_q32)
        if not ok:
            return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q4_0_Q8_0_AVX2_OK, total


def q4_0_q8_0_matmul_tiled_avx2_q32_checked(
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
        return Q4_0_Q8_0_AVX2_ERR_NULL_PTR, []
    if lhs_q4_block_capacity < 0 or rhs_q8_block_capacity < 0 or out_cell_capacity < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []
    if row_count < 0 or col_count < 0 or k_block_count < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []
    if lhs_row_stride_blocks < 0 or rhs_col_stride_blocks < 0 or out_row_stride_cells < 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []

    if k_block_count > lhs_row_stride_blocks or k_block_count > rhs_col_stride_blocks:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []
    if row_count > 0 and out_row_stride_cells < col_count:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []

    ok, lhs_required = try_mul_i64_nonneg(row_count, lhs_row_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []
    ok, rhs_required = try_mul_i64_nonneg(col_count, rhs_col_stride_blocks)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []
    ok, out_required = try_mul_i64_nonneg(row_count, out_row_stride_cells)
    if not ok:
        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []

    if lhs_required > lhs_q4_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []
    if rhs_required > rhs_q8_block_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []
    if out_required > out_cell_capacity:
        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []

    out_cells_q32 = [0] * out_cell_capacity

    row_tile_start = 0
    while row_tile_start < row_count:
        ok, row_tile_end = try_add_i64_nonneg(row_tile_start, tile_rows)
        if not ok:
            return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []
        row_tile_end = min(row_tile_end, row_count)

        col_tile_start = 0
        while col_tile_start < col_count:
            ok, col_tile_end = try_add_i64_nonneg(col_tile_start, tile_cols)
            if not ok:
                return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []
            col_tile_end = min(col_tile_end, col_count)

            for row_index in range(row_tile_start, row_tile_end):
                ok, lhs_row_base = try_mul_i64_nonneg(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []
                ok, out_row_base = try_mul_i64_nonneg(row_index, out_row_stride_cells)
                if not ok:
                    return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []

                lhs_row_slice = lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count]
                if len(lhs_row_slice) != k_block_count:
                    return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []

                for col_index in range(col_tile_start, col_tile_end):
                    ok, rhs_col_base = try_mul_i64_nonneg(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []
                    ok, out_index = try_add_i64_nonneg(out_row_base, col_index)
                    if not ok:
                        return Q4_0_Q8_0_AVX2_ERR_OVERFLOW, []

                    rhs_col_slice = rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count]
                    if len(rhs_col_slice) != k_block_count:
                        return Q4_0_Q8_0_AVX2_ERR_BAD_LEN, []

                    err, cell_dot_q32 = dot_blocks_avx2_q32_checked(lhs_row_slice, rhs_col_slice, k_block_count)
                    if err != Q4_0_Q8_0_AVX2_OK:
                        return err, []

                    out_cells_q32[out_index] = cell_dot_q32

            col_tile_start = col_tile_end

        row_tile_start = row_tile_end

    return Q4_0_Q8_0_AVX2_OK, out_cells_q32


def q4_0_q8_0_matmul_scalar_reference(
    lhs_q4_blocks,
    row_count: int,
    lhs_row_stride_blocks: int,
    rhs_q8_col_blocks,
    col_count: int,
    rhs_col_stride_blocks: int,
    k_block_count: int,
    out_row_stride_cells: int,
):
    out = [0] * (row_count * out_row_stride_cells)

    for row_index in range(row_count):
        lhs_row_base = row_index * lhs_row_stride_blocks
        lhs_row_slice = lhs_q4_blocks[lhs_row_base : lhs_row_base + k_block_count]

        for col_index in range(col_count):
            rhs_col_base = col_index * rhs_col_stride_blocks
            rhs_col_slice = rhs_q8_col_blocks[rhs_col_base : rhs_col_base + k_block_count]

            err, dot_q32 = dot_blocks_avx2_q32_checked(lhs_row_slice, rhs_col_slice, k_block_count)
            if err != Q4_0_Q8_0_AVX2_OK:
                return err, []

            out[row_index * out_row_stride_cells + col_index] = dot_q32

    return Q4_0_Q8_0_AVX2_OK, out


def make_q4_block(rng: random.Random):
    scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xB800, 0xBC00]
    vals = [rng.randint(-8, 7) for _ in range(Q4_0_Q8_0_AVX2_VALUES_PER_BLOCK)]
    return rng.choice(scales), q4_pack_signed(vals)


def make_q8_block(rng: random.Random):
    scales = [0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0x4400, 0xB800, 0xBC00]
    vals = [rng.randint(-128, 127) for _ in range(Q4_0_Q8_0_AVX2_VALUES_PER_BLOCK)]
    return rng.choice(scales), q8_pack_signed(vals)


def test_known_small_matches_scalar_reference() -> None:
    lhs = [
        (0x3C00, q4_pack_signed([((i % 16) - 8) for i in range(32)])),
        (0x3800, q4_pack_signed([7 - (i % 16) for i in range(32)])),
        (0x3000, q4_pack_signed([(-1 if (i % 2) else 1) * (i % 8) for i in range(32)])),
        (0x3400, q4_pack_signed([(3 - (i % 7)) for i in range(32)])),
    ]
    rhs = [
        (0x3C00, q8_pack_signed([2 * ((i % 17) - 8) for i in range(32)])),
        (0x3400, q8_pack_signed([((i % 13) - 6) * 5 for i in range(32)])),
        (0x3000, q8_pack_signed([(-1 if (i % 2) else 1) * (i % 19) for i in range(32)])),
        (0x4000, q8_pack_signed([(9 - (i % 9)) * 3 for i in range(32)])),
    ]

    err, got = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs,
        lhs_q4_block_capacity=len(lhs),
        row_count=2,
        lhs_row_stride_blocks=2,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=len(rhs),
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        tile_rows=1,
        tile_cols=2,
        out_cell_capacity=4,
        out_row_stride_cells=2,
    )
    assert err == Q4_0_Q8_0_AVX2_OK

    err, ref = q4_0_q8_0_matmul_scalar_reference(
        lhs,
        2,
        2,
        rhs,
        2,
        2,
        2,
        2,
    )
    assert err == Q4_0_Q8_0_AVX2_OK
    assert got == ref


def test_randomized_tiling_matches_scalar_many_shapes() -> None:
    rng = random.Random(2026041618)

    for _ in range(220):
        row_count = rng.randint(1, 7)
        col_count = rng.randint(1, 7)
        k_block_count = rng.randint(1, 6)

        lhs_stride = k_block_count + rng.randint(0, 2)
        rhs_stride = k_block_count + rng.randint(0, 2)
        out_stride = col_count + rng.randint(0, 2)
        out_capacity = row_count * out_stride

        lhs = [make_q4_block(rng) for _ in range(row_count * lhs_stride)]
        rhs = [make_q8_block(rng) for _ in range(col_count * rhs_stride)]

        err, got = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
            lhs_q4_blocks=lhs,
            lhs_q4_block_capacity=len(lhs),
            row_count=row_count,
            lhs_row_stride_blocks=lhs_stride,
            rhs_q8_col_blocks=rhs,
            rhs_q8_block_capacity=len(rhs),
            col_count=col_count,
            rhs_col_stride_blocks=rhs_stride,
            k_block_count=k_block_count,
            tile_rows=rng.randint(1, 4),
            tile_cols=rng.randint(1, 4),
            out_cell_capacity=out_capacity,
            out_row_stride_cells=out_stride,
        )
        assert err == Q4_0_Q8_0_AVX2_OK

        err, ref = q4_0_q8_0_matmul_scalar_reference(
            lhs,
            row_count,
            lhs_stride,
            rhs,
            col_count,
            rhs_stride,
            k_block_count,
            out_stride,
        )
        assert err == Q4_0_Q8_0_AVX2_OK
        assert got == ref


def test_error_paths() -> None:
    rng = random.Random(2026041619)
    lhs = [make_q4_block(rng) for _ in range(6)]
    rhs = [make_q8_block(rng) for _ in range(6)]

    err, _ = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=None,
        lhs_q4_block_capacity=0,
        row_count=0,
        lhs_row_stride_blocks=0,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=len(rhs),
        col_count=0,
        rhs_col_stride_blocks=0,
        k_block_count=0,
        tile_rows=1,
        tile_cols=1,
        out_cell_capacity=0,
        out_row_stride_cells=0,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs,
        lhs_q4_block_capacity=len(lhs),
        row_count=2,
        lhs_row_stride_blocks=1,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=len(rhs),
        col_count=2,
        rhs_col_stride_blocks=2,
        k_block_count=2,
        tile_rows=1,
        tile_cols=1,
        out_cell_capacity=8,
        out_row_stride_cells=4,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs,
        lhs_q4_block_capacity=len(lhs),
        row_count=2,
        lhs_row_stride_blocks=3,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=len(rhs),
        col_count=2,
        rhs_col_stride_blocks=3,
        k_block_count=2,
        tile_rows=1,
        tile_cols=1,
        out_cell_capacity=3,
        out_row_stride_cells=2,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_BAD_LEN


def test_overflow_guard_propagates_from_dot_kernel() -> None:
    huge_scale = 0x7C00
    lhs = [(huge_scale, q4_pack_signed([7] * 32))]
    rhs = [(huge_scale, q8_pack_signed([127] * 32))]

    err, _ = q4_0_q8_0_matmul_tiled_avx2_q32_checked(
        lhs_q4_blocks=lhs,
        lhs_q4_block_capacity=1,
        row_count=1,
        lhs_row_stride_blocks=1,
        rhs_q8_col_blocks=rhs,
        rhs_q8_block_capacity=1,
        col_count=1,
        rhs_col_stride_blocks=1,
        k_block_count=1,
        tile_rows=1,
        tile_cols=1,
        out_cell_capacity=1,
        out_row_stride_cells=1,
    )
    assert err == Q4_0_Q8_0_AVX2_ERR_OVERFLOW


if __name__ == "__main__":
    test_known_small_matches_scalar_reference()
    test_randomized_tiling_matches_scalar_many_shapes()
    test_error_paths()
    test_overflow_guard_propagates_from_dot_kernel()
