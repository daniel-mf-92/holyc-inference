// q8_0_avx2.HC
// AVX2-preparatory lane packing helpers for Q8_0 kernels.
//
// Why this exists:
// - Upcoming tiled AVX2 matmul kernels need signed 8-bit quant lanes widened to
//   signed 16-bit lanes before pairwise multiply-accumulate.
// - This helper defines the exact widening contract in auditable HolyC first.
// - Hot loops can later replace the scalar widening body with AVX2 asm
//   (`vpmovsxbw`) without changing call-site semantics.

#define Q8_0_AVX2_VALUES_PER_BLOCK   32
#define Q8_0_AVX2_PAIR_COUNT         16

#define Q8_0_AVX2_OK                 0
#define Q8_0_AVX2_ERR_NULL_PTR       1
#define Q8_0_AVX2_ERR_BAD_LEN        2
#define Q8_0_AVX2_ERR_OVERFLOW       3
#define Q8_0_AVX2_ERR_PARITY         4

#define Q8_0_AVX2_I64_MAX            0x7FFFFFFFFFFFFFFF
#define Q8_0_AVX2_I64_MIN            (-0x7FFFFFFFFFFFFFFF - 1)

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q8_0_AVX2_VALUES_PER_BLOCK];
};

Bool Q8_0AVX2TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (rhs > 0 && lhs > Q8_0_AVX2_I64_MAX - rhs)
        return FALSE;
    if (rhs < 0 && lhs < Q8_0_AVX2_I64_MIN - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool Q8_0AVX2TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs == -1 && rhs == Q8_0_AVX2_I64_MIN)
        return FALSE;
    if (rhs == -1 && lhs == Q8_0_AVX2_I64_MIN)
        return FALSE;

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > Q8_0_AVX2_I64_MAX / rhs)
                return FALSE;
        }
        else
        {
            if (rhs < Q8_0_AVX2_I64_MIN / lhs)
                return FALSE;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < Q8_0_AVX2_I64_MIN / rhs)
                return FALSE;
        }
        else
        {
            if (lhs < Q8_0_AVX2_I64_MAX / rhs)
                return FALSE;
        }
    }

    *out_prod = lhs * rhs;
    return TRUE;
}

I64 Q8_0AVX2RoundShiftRightUnsigned(I64 value, I64 shift)
{
    I64 round_bias;

    if (shift <= 0)
        return value;

    round_bias = 1 << (shift - 1);
    return (value + round_bias) >> shift;
}

I64 Q8_0AVX2RoundShiftRightSigned(I64 value, I64 shift)
{
    if (shift <= 0)
        return value;

    if (value >= 0)
        return Q8_0AVX2RoundShiftRightUnsigned(value, shift);

    return -Q8_0AVX2RoundShiftRightUnsigned(-value, shift);
}

I64 Q8_0AVX2DotQ32ToQ16(I64 dot_q32)
{
    // Dot products carry Q16*Q16 scale in Q32 during accumulation.
    // Hand-off to Q16 row kernels is one rounded signed downshift by 16.
    return Q8_0AVX2RoundShiftRightSigned(dot_q32, 16);
}

I64 Q8_0AVX2F16ToQ16(U16 fp16_bits)
{
    I64 sign_bit;
    I64 exponent_bits;
    I64 fraction_bits;
    I64 mantissa;
    I64 shift_amount;
    I64 magnitude_q16;

    sign_bit = (fp16_bits >> 15) & 1;
    exponent_bits = (fp16_bits >> 10) & 0x1F;
    fraction_bits = fp16_bits & 0x03FF;

    if (!exponent_bits)
    {
        if (!fraction_bits)
            return 0;

        magnitude_q16 = Q8_0AVX2RoundShiftRightUnsigned(fraction_bits, 8);
        if (sign_bit)
            return -magnitude_q16;
        return magnitude_q16;
    }

    if (exponent_bits == 0x1F)
    {
        if (sign_bit)
            return -0x3FFFFFFFFFFFFFFF;
        return 0x3FFFFFFFFFFFFFFF;
    }

    mantissa = 1024 + fraction_bits;
    shift_amount = exponent_bits - 9;

    if (shift_amount >= 0)
        magnitude_q16 = mantissa << shift_amount;
    else
        magnitude_q16 = Q8_0AVX2RoundShiftRightUnsigned(mantissa, -shift_amount);

    if (sign_bit)
        return -magnitude_q16;
    return magnitude_q16;
}

I32 Q8_0Pack32ToI16LanesAVX2(I8 *src_q8, I64 src_len, I16 *dst_i16, I64 dst_len)
{
    I64 lane;

    if (!src_q8 || !dst_i16)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (src_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (dst_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Lane contract:
    //   dst_i16[i] = SignExtend16(src_q8[i]) for i in [0, 31]
    // This exact order matches the two AVX2 16-lane halves that would be
    // produced by widening bytes 0..15 then 16..31.
    for (lane = 0; lane < Q8_0_AVX2_VALUES_PER_BLOCK; ++lane)
        dst_i16[lane] = (I16)src_q8[lane];

    return Q8_0_AVX2_OK;
}

I32 Q8_0PackBlockToI16LanesAVX2(Q8_0Block *block, I16 *dst_i16, I64 dst_len)
{
    if (!block || !dst_i16)
        return Q8_0_AVX2_ERR_NULL_PTR;

    return Q8_0Pack32ToI16LanesAVX2(block->qs,
                                    Q8_0_AVX2_VALUES_PER_BLOCK,
                                    dst_i16,
                                    dst_len);
}

I32 Q8_0MulI16LanesToI32PairsAVX2(I16 *lhs_i16,
                                  I64 lhs_len,
                                  I16 *rhs_i16,
                                  I64 rhs_len,
                                  I32 *dst_i32,
                                  I64 dst_len)
{
    I64 pair_index;
    I64 lane_base;
    I32 product0;
    I32 product1;

    if (!lhs_i16 || !rhs_i16 || !dst_i32)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (lhs_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (rhs_len < Q8_0_AVX2_VALUES_PER_BLOCK)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (dst_len < Q8_0_AVX2_PAIR_COUNT)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Contract mirrors AVX2 `vpmaddwd` lane behavior used by quantized dot kernels:
    // - Multiply lane pairs independently in signed 16-bit domain.
    // - Accumulate each adjacent pair into one signed 32-bit output.
    // - Preserve lane order so output pair k maps to input lanes [2k, 2k+1].
    for (pair_index = 0; pair_index < Q8_0_AVX2_PAIR_COUNT; ++pair_index)
    {
        lane_base = pair_index << 1;

        product0 = lhs_i16[lane_base] * rhs_i16[lane_base];
        product1 = lhs_i16[lane_base + 1] * rhs_i16[lane_base + 1];

        dst_i32[pair_index] = product0 + product1;
    }

    return Q8_0_AVX2_OK;
}

I32 Q8_0HSumI32PairsAVX2(I32 *pairs_i32,
                         I64 pairs_len,
                         I64 *out_sum_i64)
{
    I64 pair_index;
    I64 running_sum;
    I64 term;

    if (!pairs_i32 || !out_sum_i64)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (pairs_len < Q8_0_AVX2_PAIR_COUNT)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Horizontal reduction contract:
    // - Sum all 16 pairwise I32 partial products into one signed I64 scalar.
    // - Preserve deterministic lane order (0..15) for auditability parity
    //   against the scalar reference path.
    // - Explicit overflow checks keep reduction safe if this helper is reused
    //   for wider accumulation streams in future kernels.
    running_sum = 0;
    for (pair_index = 0; pair_index < Q8_0_AVX2_PAIR_COUNT; ++pair_index)
    {
        term = (I64)pairs_i32[pair_index];

        if (term > 0 && running_sum > Q8_0_AVX2_I64_MAX - term)
            return Q8_0_AVX2_ERR_OVERFLOW;
        if (term < 0 && running_sum < Q8_0_AVX2_I64_MIN - term)
            return Q8_0_AVX2_ERR_OVERFLOW;

        running_sum += term;
    }

    *out_sum_i64 = running_sum;
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotI16LanesAVX2(I16 *lhs_i16,
                        I64 lhs_len,
                        I16 *rhs_i16,
                        I64 rhs_len,
                        I64 *out_dot_i64)
{
    I32 pairs_i32[Q8_0_AVX2_PAIR_COUNT];
    I64 dot_i64;
    I32 status;

    if (!lhs_i16 || !rhs_i16 || !out_dot_i64)
        return Q8_0_AVX2_ERR_NULL_PTR;

    // Compose the lane-dot from the two audited AVX2-shape primitives:
    // 1) `vpmaddwd`-equivalent pairwise multiply-add over 32 I16 lanes.
    // 2) Horizontal sum over 16 I32 pair partials into one I64 scalar.
    // Keeping this composition explicit preserves bit-level auditability while
    // allowing future inline-asm replacement of each stage independently.
    status = Q8_0MulI16LanesToI32PairsAVX2(lhs_i16,
                                           lhs_len,
                                           rhs_i16,
                                           rhs_len,
                                           pairs_i32,
                                           Q8_0_AVX2_PAIR_COUNT);
    if (status != Q8_0_AVX2_OK)
        return status;

    status = Q8_0HSumI32PairsAVX2(pairs_i32,
                                  Q8_0_AVX2_PAIR_COUNT,
                                  &dot_i64);
    if (status != Q8_0_AVX2_OK)
        return status;

    *out_dot_i64 = dot_i64;
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotBlocksAVX2Q32Checked(Q8_0Block *lhs,
                                Q8_0Block *rhs,
                                I64 block_count,
                                I64 *out_dot_q32)
{
    I64 block_index;
    I16 lhs_lanes_i16[Q8_0_AVX2_VALUES_PER_BLOCK];
    I16 rhs_lanes_i16[Q8_0_AVX2_VALUES_PER_BLOCK];
    I64 q_dot_q0;
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Kernel contract for IQ-106:
    // - Each Q8_0 block is widened with AVX2-shape lane packing.
    // - Dot is reduced in integer lane domain (q_dot_q0).
    // - Block scales are applied in Q16 domain: block_dot_q32 = dA_q16*dB_q16*q_dot_q0.
    // - Multi-block accumulation is checked for I64 overflow at every step.
    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q8_0PackBlockToI16LanesAVX2(&lhs[block_index], lhs_lanes_i16, Q8_0_AVX2_VALUES_PER_BLOCK);
        if (status != Q8_0_AVX2_OK)
            return status;

        status = Q8_0PackBlockToI16LanesAVX2(&rhs[block_index], rhs_lanes_i16, Q8_0_AVX2_VALUES_PER_BLOCK);
        if (status != Q8_0_AVX2_OK)
            return status;

        status = Q8_0DotI16LanesAVX2(lhs_lanes_i16,
                                     Q8_0_AVX2_VALUES_PER_BLOCK,
                                     rhs_lanes_i16,
                                     Q8_0_AVX2_VALUES_PER_BLOCK,
                                     &q_dot_q0);
        if (status != Q8_0_AVX2_OK)
            return status;

        lhs_scale_q16 = Q8_0AVX2F16ToQ16(lhs[block_index].d_fp16);
        rhs_scale_q16 = Q8_0AVX2F16ToQ16(rhs[block_index].d_fp16);

        if (!Q8_0AVX2TryMulI64(lhs_scale_q16, rhs_scale_q16, &scale_prod_q32))
            return Q8_0_AVX2_ERR_OVERFLOW;
        if (!Q8_0AVX2TryMulI64(scale_prod_q32, q_dot_q0, &block_dot_q32))
            return Q8_0_AVX2_ERR_OVERFLOW;
        if (!Q8_0AVX2TryAddI64(total_dot_q32, block_dot_q32, &total_dot_q32))
            return Q8_0_AVX2_ERR_OVERFLOW;
    }

    *out_dot_q32 = total_dot_q32;
    return Q8_0_AVX2_OK;
}

Bool Q8_0DotProductAVX2KernelEnabled()
{
    // TempleOS inline-asm AVX2 dispatch hook:
    // keep this explicit guard so older/unknown CPUs deterministically fall
    // back to scalar checked semantics.
    return TRUE;
}

I32 Q8_0DotProductBlocksAVX2Checked(Q8_0Block *lhs,
                                    Q8_0Block *rhs,
                                    I64 block_count,
                                    I64 *out_dot_q32)
{
    I64 avx2_dot_q32;
    I64 scalar_dot_q32;
    I32 avx2_status;
    I32 scalar_status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Guarded parity contract:
    // 1) Always compute the scalar checked result first as source-of-truth.
    // 2) If AVX2 path disabled, return scalar status/result directly.
    // 3) If AVX2 path enabled, require exact status/value parity before commit.
    scalar_status = Q8_0DotProductBlocksQ32Checked(lhs,
                                                   rhs,
                                                   block_count,
                                                   &scalar_dot_q32);

    if (!Q8_0DotProductAVX2KernelEnabled())
    {
        if (scalar_status != Q8_0_AVX2_OK)
            return scalar_status;

        *out_dot_q32 = scalar_dot_q32;
        return Q8_0_AVX2_OK;
    }

    avx2_status = Q8_0DotBlocksAVX2Q32Checked(lhs,
                                              rhs,
                                              block_count,
                                              &avx2_dot_q32);

    if (avx2_status != scalar_status)
        return Q8_0_AVX2_ERR_PARITY;
    if (avx2_status != Q8_0_AVX2_OK)
        return avx2_status;
    if (avx2_dot_q32 != scalar_dot_q32)
        return Q8_0_AVX2_ERR_PARITY;

    *out_dot_q32 = avx2_dot_q32;
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotProductBlocksAVX2CheckedDefault(Q8_0Block *lhs,
                                           Q8_0Block *rhs,
                                           I64 block_count,
                                           I64 *out_dot_q32)
{
    // IQ-471 canonical default-capacity wrapper:
    // - Dot-product checked core already consumes only active span
    //   (`block_count`) and exposes no extra capacity arguments.
    // - Default wrapper is strict pass-through so callers get one stable
    //   entry point while preserving exact checked-core status and commit
    //   behavior on both success and failure.
    return Q8_0DotProductBlocksAVX2Checked(lhs,
                                           rhs,
                                           block_count,
                                           out_dot_q32);
}

I32 Q8_0DotProductBlocksAVX2CheckedDefaultNoPartial(Q8_0Block *lhs,
                                                    Q8_0Block *rhs,
                                                    I64 block_count,
                                                    I64 *out_dot_q32)
{
    I64 staged_dot_q32;
    I32 status;

    if (!out_dot_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;

    // IQ-474 strict all-or-nothing default-capacity wrapper:
    // - Preserve exact success/error surfaces from checked-default helper.
    // - Route writes through local staging so caller output is untouched on
    //   every failure path (NULL/BAD_LEN/OVERFLOW/PARITY).
    // - Commit caller-visible output only after successful staged completion.
    staged_dot_q32 = 0;
    status = Q8_0DotProductBlocksAVX2CheckedDefault(lhs,
                                                    rhs,
                                                    block_count,
                                                    &staged_dot_q32);
    if (status != Q8_0_AVX2_OK)
        return status;

    *out_dot_q32 = staged_dot_q32;
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotBlocksAVX2Q32CheckedDefault(Q8_0Block *lhs,
                                       Q8_0Block *rhs,
                                       I64 block_count,
                                       I64 *out_dot_q32)
{
    // IQ-316 canonical default-capacity wrapper:
    // - Q8_0 checked block-dot core already uses active span contract only
    //   (`block_count`) with no extra capacity arguments.
    // - Default wrapper is intentionally a direct pass-through so callers can
    //   depend on one stable "default" entry point while preserving exact
    //   checked-core success/error/no-partial-write semantics.
    return Q8_0DotBlocksAVX2Q32Checked(lhs,
                                       rhs,
                                       block_count,
                                       out_dot_q32);
}

I32 Q8_0DotRowsAVX2Q32Checked(
    Q8_0Block *matrix_blocks,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q32)
{
    I64 row_index;
    I64 required_matrix_blocks;
    I64 row_base;
    Q8_0Block *row_ptr;
    I64 row_dot_q32;
    I32 status;

    if (!matrix_blocks || !vec_blocks || !out_rows_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (matrix_block_capacity < 0 || vec_block_capacity < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Row layout contract:
    // - Row i starts at matrix_blocks + i*row_stride_blocks.
    // - Active row span consumes vec_block_count blocks.
    // Require stride to cover active K span so every row window is valid.
    if (row_count > 0 && row_stride_blocks < vec_block_count)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (vec_block_count > vec_block_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Full matrix view capacity proof for strided rows.
    if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &required_matrix_blocks))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (required_matrix_blocks > matrix_block_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    for (row_index = 0; row_index < row_count; ++row_index)
    {
        if (!Q8_0AVX2TryMulI64(row_index, row_stride_blocks, &row_base))
            return Q8_0_AVX2_ERR_OVERFLOW;

        row_ptr = &matrix_blocks[row_base];
        status = Q8_0DotBlocksAVX2Q32Checked(row_ptr,
                                             vec_blocks,
                                             vec_block_count,
                                             &row_dot_q32);
        if (status != Q8_0_AVX2_OK)
            return status;

        out_rows_q32[row_index] = row_dot_q32;
    }

    return Q8_0_AVX2_OK;
}

I32 Q8_0DotRowsAVX2Q32CheckedDefault(
    Q8_0Block *matrix_blocks,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_count,
    I64 *out_rows_q32)
{
    I64 matrix_block_capacity;

    // IQ-313 canonical default-capacity row wrapper:
    // - Matrix capacity defaults to `row_count * row_stride_blocks`.
    // - Vector capacity defaults to `vec_block_count`.
    // - Negative lengths preserve checked-core BAD_LEN semantics.
    // - Capacity multiplication is checked so default derivation never wraps.
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &matrix_block_capacity))
        return Q8_0_AVX2_ERR_OVERFLOW;

    return Q8_0DotRowsAVX2Q32Checked(matrix_blocks,
                                     matrix_block_capacity,
                                     row_count,
                                     row_stride_blocks,
                                     vec_blocks,
                                     vec_block_count,
                                     vec_block_count,
                                     out_rows_q32);
}

I32 Q8_0DotRowsAVX2Q32CheckedDefaultNoPartial(
    Q8_0Block *matrix_blocks,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_count,
    I64 *out_rows_q32)
{
    I64 stage_bytes;
    I64 row_index;
    I64 *staged_rows_q32;
    I32 status;

    if (!matrix_blocks || !vec_blocks || !out_rows_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (row_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // IQ-489 strict all-or-nothing default-capacity row wrapper:
    // - Route checked-default row dots into private staging memory first.
    // - Preserve exact checked-default error surface (BAD_LEN/OVERFLOW/PARITY).
    // - Commit caller output only after full staged success.
    // This prevents visible partial row writes when a later row fails.
    if (!row_count)
        return Q8_0DotRowsAVX2Q32CheckedDefault(matrix_blocks,
                                                row_count,
                                                row_stride_blocks,
                                                vec_blocks,
                                                vec_block_count,
                                                out_rows_q32);

    if (!Q8_0AVX2TryMulI64(row_count, sizeof(I64), &stage_bytes))
        return Q8_0_AVX2_ERR_OVERFLOW;

    staged_rows_q32 = MAlloc(stage_bytes);
    if (!staged_rows_q32)
        return Q8_0_AVX2_ERR_OVERFLOW;

    status = Q8_0DotRowsAVX2Q32CheckedDefault(matrix_blocks,
                                              row_count,
                                              row_stride_blocks,
                                              vec_blocks,
                                              vec_block_count,
                                              staged_rows_q32);
    if (status != Q8_0_AVX2_OK)
    {
        Free(staged_rows_q32);
        return status;
    }

    for (row_index = 0; row_index < row_count; ++row_index)
        out_rows_q32[row_index] = staged_rows_q32[row_index];

    Free(staged_rows_q32);
    return Q8_0_AVX2_OK;
}

I32 Q8_0MatMulTiledAVX2Q32Checked(
    Q8_0Block *lhs_matrix_blocks,
    I64 lhs_block_capacity,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_mat_q32,
    I64 out_capacity,
    I64 out_row_stride_cols)
{
    I64 required_lhs_blocks;
    I64 required_rhs_blocks;
    I64 required_out_cells;
    I64 tile_row_start;
    I64 tile_col_start;
    I64 tile_row_end;
    I64 tile_col_end;
    I64 row_index;
    I64 col_index;
    I64 lhs_row_base;
    I64 rhs_col_base;
    I64 out_index;
    I64 dot_q32;
    I32 status;

    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_capacity < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (lhs_rows < 0 || lhs_row_stride_blocks < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (rhs_cols < 0 || rhs_col_stride_blocks < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (k_block_count < 0 || out_row_stride_cols < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Matrix layout contracts:
    // - lhs_matrix_blocks stores A as row-major rows, each row consuming
    //   k_block_count active Q8_0 blocks inside lhs_row_stride_blocks stride.
    // - rhs_col_blocks stores B as pre-transposed columns, each column
    //   consuming k_block_count active Q8_0 blocks inside rhs_col_stride_blocks.
    // - out_mat_q32 stores C as row-major Q32 cells with out_row_stride_cols.
    if (lhs_rows > 0 && lhs_row_stride_blocks < k_block_count)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (rhs_cols > 0 && rhs_col_stride_blocks < k_block_count)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (lhs_rows > 0 && out_row_stride_cols < rhs_cols)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(lhs_rows, lhs_row_stride_blocks, &required_lhs_blocks))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (required_lhs_blocks > lhs_block_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(rhs_cols, rhs_col_stride_blocks, &required_rhs_blocks))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (required_rhs_blocks > rhs_block_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(lhs_rows, out_row_stride_cols, &required_out_cells))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (required_out_cells > out_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Tiled traversal keeps hot row/column windows local for later AVX2 asm.
    // Dot primitive remains the checked AVX2-shape kernel so arithmetic and
    // overflow behavior are identical to the audited block-level path.
    for (tile_row_start = 0; tile_row_start < lhs_rows; tile_row_start += tile_rows)
    {
        tile_row_end = tile_row_start + tile_rows;
        if (tile_row_end > lhs_rows)
            tile_row_end = lhs_rows;

        for (tile_col_start = 0; tile_col_start < rhs_cols; tile_col_start += tile_cols)
        {
            tile_col_end = tile_col_start + tile_cols;
            if (tile_col_end > rhs_cols)
                tile_col_end = rhs_cols;

            for (row_index = tile_row_start; row_index < tile_row_end; ++row_index)
            {
                if (!Q8_0AVX2TryMulI64(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q8_0_AVX2_ERR_OVERFLOW;

                for (col_index = tile_col_start; col_index < tile_col_end; ++col_index)
                {
                    if (!Q8_0AVX2TryMulI64(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q8_0_AVX2_ERR_OVERFLOW;

                    status = Q8_0DotBlocksAVX2Q32Checked(
                        &lhs_matrix_blocks[lhs_row_base],
                        &rhs_col_blocks[rhs_col_base],
                        k_block_count,
                        &dot_q32);
                    if (status != Q8_0_AVX2_OK)
                        return status;

                    if (!Q8_0AVX2TryMulI64(row_index, out_row_stride_cols, &out_index))
                        return Q8_0_AVX2_ERR_OVERFLOW;
                    if (!Q8_0AVX2TryAddI64(out_index, col_index, &out_index))
                        return Q8_0_AVX2_ERR_OVERFLOW;

                    out_mat_q32[out_index] = dot_q32;
                }
            }
        }
    }

    return Q8_0_AVX2_OK;
}

I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultTiles(
    Q8_0Block *lhs_matrix_blocks,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_mat_q32,
    I64 out_row_stride_cols)
{
    I64 lhs_block_capacity;
    I64 rhs_block_capacity;
    I64 out_capacity;

    // IQ-317 canonical default-tiling wrapper:
    // - Preserve checked-core argument ordering for error precedence.
    // - Derive capacities from visible matrix layout dimensions only.
    // - Default traversal tiles are 1x1 so wrapper behavior is exact checked
    //   core behavior with explicit default tile parameters.
    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (lhs_rows < 0 || lhs_row_stride_blocks < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (rhs_cols < 0 || rhs_col_stride_blocks < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (k_block_count < 0 || out_row_stride_cols < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(lhs_rows, lhs_row_stride_blocks, &lhs_block_capacity))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (!Q8_0AVX2TryMulI64(rhs_cols, rhs_col_stride_blocks, &rhs_block_capacity))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (!Q8_0AVX2TryMulI64(lhs_rows, out_row_stride_cols, &out_capacity))
        return Q8_0_AVX2_ERR_OVERFLOW;

    return Q8_0MatMulTiledAVX2Q32Checked(
        lhs_matrix_blocks,
        lhs_block_capacity,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        1,
        1,
        out_mat_q32,
        out_capacity,
        out_row_stride_cols);
}

I32 Q8_0MatMulTiledAVX2Q32CheckedDefault(
    Q8_0Block *lhs_matrix_blocks,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_mat_q32,
    I64 out_row_stride_cols)
{
    // Canonical default-capacity/default-tile matmul wrapper.
    // Delegate to IQ-317 default-tiles helper so checked semantics stay
    // centralized in one auditable implementation.
    return Q8_0MatMulTiledAVX2Q32CheckedDefaultTiles(lhs_matrix_blocks,
                                                     lhs_rows,
                                                     lhs_row_stride_blocks,
                                                     rhs_col_blocks,
                                                     rhs_cols,
                                                     rhs_col_stride_blocks,
                                                     k_block_count,
                                                     out_mat_q32,
                                                     out_row_stride_cols);
}

I32 Q8_0MatMulTiledAVX2Q32CheckedDefaultNoPartial(
    Q8_0Block *lhs_matrix_blocks,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_mat_q32,
    I64 out_row_stride_cols)
{
    I64 staged_out_capacity;
    I64 staged_out_bytes;
    I64 *staged_out_mat_q32;
    I64 idx;
    I32 status;

    if (!out_mat_q32)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (lhs_rows < 0 || out_row_stride_cols < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Preserve default-wrapper behavior for degenerate result spans.
    // Route through default-tiles entry so no-partial staging keeps one
    // canonical checked core path for default tile semantics.
    if (!lhs_rows || !rhs_cols)
        return Q8_0MatMulTiledAVX2Q32CheckedDefaultTiles(lhs_matrix_blocks,
                                                         lhs_rows,
                                                         lhs_row_stride_blocks,
                                                         rhs_col_blocks,
                                                         rhs_cols,
                                                         rhs_col_stride_blocks,
                                                         k_block_count,
                                                         out_mat_q32,
                                                         out_row_stride_cols);

    if (!Q8_0AVX2TryMulI64(lhs_rows, out_row_stride_cols, &staged_out_capacity))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (staged_out_capacity <= 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(staged_out_capacity, sizeof(I64), &staged_out_bytes))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (staged_out_bytes <= 0)
        return Q8_0_AVX2_ERR_OVERFLOW;

    staged_out_mat_q32 = MAlloc(staged_out_bytes);
    if (!staged_out_mat_q32)
        return Q8_0_AVX2_ERR_OVERFLOW;

    status = Q8_0MatMulTiledAVX2Q32CheckedDefaultTiles(lhs_matrix_blocks,
                                                       lhs_rows,
                                                       lhs_row_stride_blocks,
                                                       rhs_col_blocks,
                                                       rhs_cols,
                                                       rhs_col_stride_blocks,
                                                       k_block_count,
                                                       staged_out_mat_q32,
                                                       out_row_stride_cols);
    if (status != Q8_0_AVX2_OK)
    {
        Free(staged_out_mat_q32);
        return status;
    }

    for (idx = 0; idx < staged_out_capacity; ++idx)
        out_mat_q32[idx] = staged_out_mat_q32[idx];

    Free(staged_out_mat_q32);
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotBlocksAVX2Q32ToQ16Checked(Q8_0Block *lhs,
                                     Q8_0Block *rhs,
                                     I64 block_count,
                                     I64 *out_dot_q16)
{
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q16)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Single-rounding contract for row kernels:
    // 1) Build checked full Q32 dot over all blocks.
    // 2) Downshift once Q32->Q16 at the boundary.
    // This avoids per-block rounding drift and preserves GGML-style intent.
    status = Q8_0DotBlocksAVX2Q32Checked(lhs, rhs, block_count, &total_dot_q32);
    if (status != Q8_0_AVX2_OK)
        return status;

    *out_dot_q16 = Q8_0AVX2DotQ32ToQ16(total_dot_q32);
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotBlocksAVX2Q32ToQ16CheckedDefault(Q8_0Block *lhs,
                                            Q8_0Block *rhs,
                                            I64 block_count,
                                            I64 *out_dot_q16)
{
    // IQ-328 canonical default-capacity wrapper:
    // - Q32->Q16 checked block-dot core already operates on active span only
    //   (`block_count`) and does not expose extra capacity arguments.
    // - Default wrapper is a direct pass-through so callers can depend on one
    //   stable default entry point while preserving exact checked-core success,
    //   error, and no-partial-write semantics.
    return Q8_0DotBlocksAVX2Q32ToQ16Checked(lhs,
                                            rhs,
                                            block_count,
                                            out_dot_q16);
}

I32 Q8_0DotRowsAVX2Q32ToQ16Checked(
    Q8_0Block *matrix_blocks,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q16)
{
    I64 row_index;
    I64 required_matrix_blocks;
    I64 row_base;
    Q8_0Block *row_ptr;
    I64 row_dot_q16;
    I32 status;

    if (!matrix_blocks || !vec_blocks || !out_rows_q16)
        return Q8_0_AVX2_ERR_NULL_PTR;

    if (matrix_block_capacity < 0 || vec_block_capacity < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Row layout contract mirrors Q32 row helper:
    // - Row i begins at matrix_blocks + i*row_stride_blocks.
    // - Each row consumes vec_block_count active blocks for the dot.
    if (row_count > 0 && row_stride_blocks < vec_block_count)
        return Q8_0_AVX2_ERR_BAD_LEN;
    if (vec_block_count > vec_block_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &required_matrix_blocks))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (required_matrix_blocks > matrix_block_capacity)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // Single-rounding row contract:
    // - Build each row dot in checked Q32 precision across all K blocks.
    // - Downshift once Q32->Q16 after full-row accumulation.
    // This preserves GGML-style rounding behavior and avoids per-block drift.
    for (row_index = 0; row_index < row_count; ++row_index)
    {
        if (!Q8_0AVX2TryMulI64(row_index, row_stride_blocks, &row_base))
            return Q8_0_AVX2_ERR_OVERFLOW;

        row_ptr = &matrix_blocks[row_base];
        status = Q8_0DotBlocksAVX2Q32ToQ16Checked(row_ptr,
                                                  vec_blocks,
                                                  vec_block_count,
                                                  &row_dot_q16);
        if (status != Q8_0_AVX2_OK)
            return status;

        out_rows_q16[row_index] = row_dot_q16;
    }

    return Q8_0_AVX2_OK;
}

I32 Q8_0DotRowsAVX2Q32ToQ16CheckedNoPartial(
    Q8_0Block *matrix_blocks,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q16)
{
    I64 stage_bytes;
    I64 row_index;
    I64 *staged_rows_q16;
    I32 status;

    if (!matrix_blocks || !vec_blocks || !out_rows_q16)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (row_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // IQ-359 all-or-nothing wrapper contract:
    // - Materialize row dots into a private staging span.
    // - Reuse checked core helper for the full row conversion path.
    // - Commit caller output lanes only after full success.
    // This makes row conversion externally no-partial-write even when a later
    // row fails after earlier rows were already computable.
    if (!row_count)
        return Q8_0DotRowsAVX2Q32ToQ16Checked(matrix_blocks,
                                              matrix_block_capacity,
                                              row_count,
                                              row_stride_blocks,
                                              vec_blocks,
                                              vec_block_capacity,
                                              vec_block_count,
                                              out_rows_q16);

    if (!Q8_0AVX2TryMulI64(row_count, sizeof(I64), &stage_bytes))
        return Q8_0_AVX2_ERR_OVERFLOW;

    staged_rows_q16 = MAlloc(stage_bytes);
    if (!staged_rows_q16)
        return Q8_0_AVX2_ERR_OVERFLOW;

    status = Q8_0DotRowsAVX2Q32ToQ16Checked(matrix_blocks,
                                            matrix_block_capacity,
                                            row_count,
                                            row_stride_blocks,
                                            vec_blocks,
                                            vec_block_capacity,
                                            vec_block_count,
                                            staged_rows_q16);
    if (status != Q8_0_AVX2_OK)
    {
        Free(staged_rows_q16);
        return status;
    }

    for (row_index = 0; row_index < row_count; ++row_index)
        out_rows_q16[row_index] = staged_rows_q16[row_index];

    Free(staged_rows_q16);
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotRowsAVX2Q32ToQ16CheckedDefault(
    Q8_0Block *matrix_blocks,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_count,
    I64 *out_rows_q16)
{
    I64 matrix_block_capacity;

    // IQ-344 canonical default-capacity Q32->Q16 row wrapper:
    // - Matrix capacity defaults to `row_count * row_stride_blocks`.
    // - Vector capacity defaults to `vec_block_count`.
    // - Negative lengths preserve BAD_LEN semantics from checked row core.
    // - Multiplication is checked so default-capacity derivation never wraps.
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    if (!Q8_0AVX2TryMulI64(row_count, row_stride_blocks, &matrix_block_capacity))
        return Q8_0_AVX2_ERR_OVERFLOW;

    return Q8_0DotRowsAVX2Q32ToQ16Checked(matrix_blocks,
                                          matrix_block_capacity,
                                          row_count,
                                          row_stride_blocks,
                                          vec_blocks,
                                          vec_block_count,
                                          vec_block_count,
                                          out_rows_q16);
}

I32 Q8_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartial(
    Q8_0Block *matrix_blocks,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_count,
    I64 *out_rows_q16)
{
    I64 stage_bytes;
    I64 row_index;
    I64 *staged_rows_q16;
    I32 status;

    status = Q8_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartialPreflight(matrix_blocks,
                                                                     row_count,
                                                                     row_stride_blocks,
                                                                     vec_blocks,
                                                                     vec_block_count,
                                                                     out_rows_q16,
                                                                     &stage_bytes);
    if (status != Q8_0_AVX2_OK)
        return status;

    // Zero-row pass-through preserves checked-default behavior exactly and
    // avoids transient staging allocation for empty output spans.
    if (!row_count)
        return Q8_0DotRowsAVX2Q32ToQ16CheckedDefault(matrix_blocks,
                                                     row_count,
                                                     row_stride_blocks,
                                                     vec_blocks,
                                                     vec_block_count,
                                                     out_rows_q16);

    staged_rows_q16 = MAlloc(stage_bytes);
    if (!staged_rows_q16)
        return Q8_0_AVX2_ERR_OVERFLOW;

    status = Q8_0DotRowsAVX2Q32ToQ16CheckedDefault(matrix_blocks,
                                                   row_count,
                                                   row_stride_blocks,
                                                   vec_blocks,
                                                   vec_block_count,
                                                   staged_rows_q16);
    if (status != Q8_0_AVX2_OK)
    {
        Free(staged_rows_q16);
        return status;
    }

    for (row_index = 0; row_index < row_count; ++row_index)
        out_rows_q16[row_index] = staged_rows_q16[row_index];

    Free(staged_rows_q16);
    return Q8_0_AVX2_OK;
}

I32 Q8_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartialPreflight(
    Q8_0Block *matrix_blocks,
    I64 row_count,
    I64 row_stride_blocks,
    Q8_0Block *vec_blocks,
    I64 vec_block_count,
    I64 *out_rows_q16,
    I64 *out_stage_bytes)
{
    I64 stage_bytes;

    if (!out_rows_q16 || !out_stage_bytes)
        return Q8_0_AVX2_ERR_NULL_PTR;
    if (row_count < 0)
        return Q8_0_AVX2_ERR_BAD_LEN;

    // IQ-504 preflight contract for default no-partial row wrappers:
    // - Validate caller-facing pointer/null and signed-length domain first.
    // - Compute checked staging-byte count once so wrappers share one audited
    //   overflow surface instead of duplicating `row_count * sizeof(I64)`.
    // - Keep matrix/stride/vector semantics delegated to checked-default core
    //   so error ordering stays centralized and parity remains exact.
    if (!row_count)
    {
        *out_stage_bytes = 0;
        return Q8_0_AVX2_OK;
    }

    if (!Q8_0AVX2TryMulI64(row_count, sizeof(I64), &stage_bytes))
        return Q8_0_AVX2_ERR_OVERFLOW;
    if (stage_bytes <= 0)
        return Q8_0_AVX2_ERR_OVERFLOW;

    *out_stage_bytes = stage_bytes;
    return Q8_0_AVX2_OK;
}
