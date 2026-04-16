#!/usr/bin/env python3
"""Reference checks for Q8_0MatMulTiledAVX2Q32Checked semantics."""

from __future__ import annotations

import random

Q8_0_AVX2_VALUES_PER_BLOCK = 32

Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2
Q8_0_AVX2_ERR_OVERFLOW = 3

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def q8_0_try_add_i64(lhs: int, rhs: int):
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def q8_0_try_mul_i64(lhs: int, rhs: int):
    if lhs == 0 or rhs == 0:
        return True, 0
    product = lhs * rhs
    if product < I64_MIN or product > I64_MAX:
        return False, 0
    return True, product


def compute_tile_end_checked(tile_start: int, tile_span: int, axis_len: int):
    if tile_start < 0 or tile_span < 0 or axis_len < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    if tile_start > axis_len:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, tile_end = q8_0_try_add_i64(tile_start, tile_span)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, min(tile_end, axis_len)


def compute_out_index_checked(out_row_base: int, col_index: int):
    if out_row_base < 0 or col_index < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    ok, out_index = q8_0_try_add_i64(out_row_base, col_index)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, out_index


def q8_0_f16_to_q16(fp16_bits: int) -> int:
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


def q8_0_dot_blocks_avx2_q32_checked(lhs_blocks, rhs_blocks, block_count):
    if lhs_blocks is None or rhs_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0
    if block_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, 0
    if len(lhs_blocks) < block_count or len(rhs_blocks) < block_count:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    total = 0
    for i in range(block_count):
        q_dot = sum(int(a) * int(b) for a, b in zip(lhs_blocks[i]["qs"], rhs_blocks[i]["qs"]))
        lhs_scale_q16 = q8_0_f16_to_q16(lhs_blocks[i]["d_fp16"])
        rhs_scale_q16 = q8_0_f16_to_q16(rhs_blocks[i]["d_fp16"])

        ok, scale_prod_q32 = q8_0_try_mul_i64(lhs_scale_q16, rhs_scale_q16)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

        ok, block_dot_q32 = q8_0_try_mul_i64(scale_prod_q32, q_dot)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

        ok, total = q8_0_try_add_i64(total, block_dot_q32)
        if not ok:
            return Q8_0_AVX2_ERR_OVERFLOW, 0

    return Q8_0_AVX2_OK, total


def q8_0_matmul_tiled_avx2_q32_checked(
    lhs_matrix_blocks,
    lhs_block_capacity,
    lhs_rows,
    lhs_row_stride_blocks,
    rhs_col_blocks,
    rhs_block_capacity,
    rhs_cols,
    rhs_col_stride_blocks,
    k_block_count,
    tile_rows,
    tile_cols,
    out_capacity,
    out_row_stride_cols,
):
    if lhs_matrix_blocks is None or rhs_col_blocks is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_capacity < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if lhs_rows < 0 or lhs_row_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if rhs_cols < 0 or rhs_col_stride_blocks < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if k_block_count < 0 or out_row_stride_cols < 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if tile_rows <= 0 or tile_cols <= 0:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    if lhs_rows > 0 and lhs_row_stride_blocks < k_block_count:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if rhs_cols > 0 and rhs_col_stride_blocks < k_block_count:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if lhs_rows > 0 and out_row_stride_cols < rhs_cols:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    ok, required_lhs_blocks = q8_0_try_mul_i64(lhs_rows, lhs_row_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    if required_lhs_blocks > lhs_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    ok, required_rhs_blocks = q8_0_try_mul_i64(rhs_cols, rhs_col_stride_blocks)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    if required_rhs_blocks > rhs_block_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    ok, required_out_cells = q8_0_try_mul_i64(lhs_rows, out_row_stride_cols)
    if not ok:
        return Q8_0_AVX2_ERR_OVERFLOW, []
    if required_out_cells > out_capacity:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    out = [0] * out_capacity

    tile_row_start = 0
    while tile_row_start < lhs_rows:
        err, tile_row_end = compute_tile_end_checked(tile_row_start, tile_rows, lhs_rows)
        if err != Q8_0_AVX2_OK:
            return err, []

        tile_col_start = 0
        while tile_col_start < rhs_cols:
            err, tile_col_end = compute_tile_end_checked(tile_col_start, tile_cols, rhs_cols)
            if err != Q8_0_AVX2_OK:
                return err, []

            for row_index in range(tile_row_start, tile_row_end):
                ok, lhs_row_base = q8_0_try_mul_i64(row_index, lhs_row_stride_blocks)
                if not ok:
                    return Q8_0_AVX2_ERR_OVERFLOW, []

                for col_index in range(tile_col_start, tile_col_end):
                    ok, rhs_col_base = q8_0_try_mul_i64(col_index, rhs_col_stride_blocks)
                    if not ok:
                        return Q8_0_AVX2_ERR_OVERFLOW, []

                    err, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(
                        lhs_matrix_blocks[lhs_row_base : lhs_row_base + k_block_count],
                        rhs_col_blocks[rhs_col_base : rhs_col_base + k_block_count],
                        k_block_count,
                    )
                    if err != Q8_0_AVX2_OK:
                        return err, []

                    ok, out_index = q8_0_try_mul_i64(row_index, out_row_stride_cols)
                    if not ok:
                        return Q8_0_AVX2_ERR_OVERFLOW, []
                    err, out_index = compute_out_index_checked(out_index, col_index)
                    if err != Q8_0_AVX2_OK:
                        return err, []

                    out[out_index] = dot_q32

            tile_col_start = tile_col_end

        tile_row_start = tile_row_end

    return Q8_0_AVX2_OK, out


def q8_0_matmul_scalar_reference(lhs_matrix_blocks, lhs_rows, lhs_row_stride_blocks, rhs_col_blocks, rhs_cols, rhs_col_stride_blocks, k_block_count, out_row_stride_cols):
    out = [0] * (lhs_rows * out_row_stride_cols)
    for row in range(lhs_rows):
        lhs_base = row * lhs_row_stride_blocks
        for col in range(rhs_cols):
            rhs_base = col * rhs_col_stride_blocks
            _, dot_q32 = q8_0_dot_blocks_avx2_q32_checked(
                lhs_matrix_blocks[lhs_base : lhs_base + k_block_count],
                rhs_col_blocks[rhs_base : rhs_base + k_block_count],
                k_block_count,
            )
            out[row * out_row_stride_cols + col] = dot_q32
    return out


def make_block(d_fp16: int, qs):
    assert len(qs) == 32
    return {"d_fp16": d_fp16 & 0xFFFF, "qs": [int(x) for x in qs]}


def build_matrix_rows_as_blocks(rows: int, row_stride_blocks: int, k_blocks: int, rng: random.Random):
    fp16_scales = [0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0xB800, 0xBC00]
    out = []
    for _ in range(rows):
        for block_index in range(row_stride_blocks):
            if block_index < k_blocks:
                out.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))
            else:
                out.append(make_block(0x0000, [0] * 32))
    return out


def build_matrix_cols_as_blocks(cols: int, col_stride_blocks: int, k_blocks: int, rng: random.Random):
    fp16_scales = [0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3A00, 0x3C00, 0x4000, 0xB800, 0xBC00]
    out = []
    for _ in range(cols):
        for block_index in range(col_stride_blocks):
            if block_index < k_blocks:
                out.append(make_block(rng.choice(fp16_scales), [rng.randint(-128, 127) for _ in range(32)]))
            else:
                out.append(make_block(0x0000, [0] * 32))
    return out


def test_known_small_matches_scalar_reference() -> None:
    rng = random.Random(2026041602)
    lhs_rows = 3
    rhs_cols = 4
    k_blocks = 2
    lhs_row_stride = 3
    rhs_col_stride = 3
    out_row_stride = 5

    lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride, k_blocks, rng)
    rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride, k_blocks, rng)

    err, out = q8_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        lhs_rows,
        lhs_row_stride,
        rhs,
        len(rhs),
        rhs_cols,
        rhs_col_stride,
        k_blocks,
        2,
        3,
        lhs_rows * out_row_stride,
        out_row_stride,
    )
    assert err == Q8_0_AVX2_OK

    expected = q8_0_matmul_scalar_reference(
        lhs,
        lhs_rows,
        lhs_row_stride,
        rhs,
        rhs_cols,
        rhs_col_stride,
        k_blocks,
        out_row_stride,
    )
    assert out == expected


def test_randomized_tiling_matches_scalar_many_shapes() -> None:
    rng = random.Random(2026041603)

    for _ in range(180):
        lhs_rows = rng.randint(1, 6)
        rhs_cols = rng.randint(1, 6)
        k_blocks = rng.randint(1, 6)
        lhs_row_stride = k_blocks + rng.randint(0, 3)
        rhs_col_stride = k_blocks + rng.randint(0, 3)
        out_row_stride = rhs_cols + rng.randint(0, 3)
        tile_rows = rng.randint(1, 4)
        tile_cols = rng.randint(1, 4)

        lhs = build_matrix_rows_as_blocks(lhs_rows, lhs_row_stride, k_blocks, rng)
        rhs = build_matrix_cols_as_blocks(rhs_cols, rhs_col_stride, k_blocks, rng)

        err, out = q8_0_matmul_tiled_avx2_q32_checked(
            lhs,
            len(lhs),
            lhs_rows,
            lhs_row_stride,
            rhs,
            len(rhs),
            rhs_cols,
            rhs_col_stride,
            k_blocks,
            tile_rows,
            tile_cols,
            lhs_rows * out_row_stride,
            out_row_stride,
        )
        assert err == Q8_0_AVX2_OK

        expected = q8_0_matmul_scalar_reference(
            lhs,
            lhs_rows,
            lhs_row_stride,
            rhs,
            rhs_cols,
            rhs_col_stride,
            k_blocks,
            out_row_stride,
        )
        assert out == expected


def test_error_paths() -> None:
    rng = random.Random(2026041604)
    lhs = build_matrix_rows_as_blocks(2, 2, 2, rng)
    rhs = build_matrix_cols_as_blocks(2, 2, 2, rng)

    err, _ = q8_0_matmul_tiled_avx2_q32_checked(
        None,
        0,
        0,
        0,
        rhs,
        len(rhs),
        2,
        2,
        2,
        1,
        1,
        0,
        0,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        2,
        1,
        rhs,
        len(rhs),
        2,
        2,
        2,
        1,
        1,
        8,
        4,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        2,
        2,
        rhs,
        len(rhs),
        2,
        2,
        2,
        0,
        1,
        8,
        4,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_matmul_tiled_avx2_q32_checked(
        lhs,
        len(lhs),
        2,
        2,
        rhs,
        len(rhs),
        2,
        2,
        2,
        1,
        1,
        7,
        4,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def test_compute_tile_end_checked_contract() -> None:
    err, tile_end = compute_tile_end_checked(1, 4, 9)
    assert err == Q8_0_AVX2_OK
    assert tile_end == 5

    err, tile_end = compute_tile_end_checked(8, 4, 9)
    assert err == Q8_0_AVX2_OK
    assert tile_end == 9

    err, _ = compute_tile_end_checked(-1, 2, 9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    err, _ = compute_tile_end_checked(1, -2, 9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    err, _ = compute_tile_end_checked(1, 2, -9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    err, _ = compute_tile_end_checked(10, 1, 9)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = compute_tile_end_checked(I64_MAX - 2, 5, I64_MAX)
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_known_small_matches_scalar_reference()
    test_randomized_tiling_matches_scalar_many_shapes()
    test_error_paths()
    test_compute_tile_end_checked_contract()
    print("q8_0_matmul_tiled_avx2_q32_reference_checks=ok")


if __name__ == "__main__":
    run()
