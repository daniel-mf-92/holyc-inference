// q4_0_dot_avx2.HC
// Checked Q4_0 AVX2-shape block dot kernel with guarded scalar fallback.
//
// IQ-449 contract:
// - Public entry point is Q4_0DotProductBlocksAVX2Checked.
// - Uses AVX2-lane semantics for nibble unpack + signed dot accumulation.
// - Maintains a guarded parity path against scalar checked reference.
// - Keeps every multiply/add in integer domain (Q4_0 nibble dot in Q0,
//   block scales in Q16, output accumulation in Q32).

#define Q4_0_DOT_AVX2_VALUES_PER_BLOCK 32
#define Q4_0_DOT_AVX2_PACKED_BYTES     16

#define Q4_0_DOT_AVX2_OK               0
#define Q4_0_DOT_AVX2_ERR_NULL_PTR     1
#define Q4_0_DOT_AVX2_ERR_BAD_LEN      2
#define Q4_0_DOT_AVX2_ERR_OVERFLOW     3
#define Q4_0_DOT_AVX2_ERR_PARITY       4

#define Q4_0_DOT_AVX2_I64_MAX          0x7FFFFFFFFFFFFFFF
#define Q4_0_DOT_AVX2_I64_MIN          (-0x7FFFFFFFFFFFFFFF - 1)

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_DOT_AVX2_PACKED_BYTES];
};

Bool Q4_0DotAVX2TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if ((rhs > 0 && lhs > Q4_0_DOT_AVX2_I64_MAX - rhs) ||
        (rhs < 0 && lhs < Q4_0_DOT_AVX2_I64_MIN - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool Q4_0DotAVX2TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs == -1 && rhs == Q4_0_DOT_AVX2_I64_MIN)
        return FALSE;
    if (rhs == -1 && lhs == Q4_0_DOT_AVX2_I64_MIN)
        return FALSE;

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > Q4_0_DOT_AVX2_I64_MAX / rhs)
                return FALSE;
        }
        else
        {
            if (rhs < Q4_0_DOT_AVX2_I64_MIN / lhs)
                return FALSE;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < Q4_0_DOT_AVX2_I64_MIN / rhs)
                return FALSE;
        }
        else
        {
            if (lhs < Q4_0_DOT_AVX2_I64_MAX / rhs)
                return FALSE;
        }
    }

    *out_prod = lhs * rhs;
    return TRUE;
}

I64 Q4_0DotAVX2RoundShiftRightUnsigned(I64 value, I64 shift)
{
    I64 round_bias;

    if (shift <= 0)
        return value;

    round_bias = 1 << (shift - 1);
    return (value + round_bias) >> shift;
}

I64 Q4_0DotAVX2F16ToQ16(U16 fp16_bits)
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

        magnitude_q16 = Q4_0DotAVX2RoundShiftRightUnsigned(fraction_bits, 8);
        return sign_bit ? -magnitude_q16 : magnitude_q16;
    }

    if (exponent_bits == 0x1F)
        return sign_bit ? -0x3FFFFFFFFFFFFFFF : 0x3FFFFFFFFFFFFFFF;

    mantissa = 1024 + fraction_bits;
    shift_amount = exponent_bits - 9;

    if (shift_amount >= 0)
        magnitude_q16 = mantissa << shift_amount;
    else
        magnitude_q16 = Q4_0DotAVX2RoundShiftRightUnsigned(mantissa, -shift_amount);

    return sign_bit ? -magnitude_q16 : magnitude_q16;
}

I64 Q4_0DotAVX2NibbleToSigned(U8 packed, Bool upper)
{
    I64 q_unsigned;

    if (upper)
        q_unsigned = (packed >> 4) & 0x0F;
    else
        q_unsigned = packed & 0x0F;

    return q_unsigned - 8;
}

I32 Q4_0DotAVX2UnpackBlockToI16Lanes(U8 *packed_qs,
                                     I64 packed_len,
                                     I16 *dst_lanes,
                                     I64 dst_len)
{
    I64 byte_index;
    I64 lane_base;

    if (!packed_qs || !dst_lanes)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;
    if (packed_len < Q4_0_DOT_AVX2_PACKED_BYTES)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;
    if (dst_len < Q4_0_DOT_AVX2_VALUES_PER_BLOCK)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;

    // AVX2 unpack contract:
    // packed byte i -> lane (2*i) low nibble, lane (2*i+1) high nibble.
    for (byte_index = 0; byte_index < Q4_0_DOT_AVX2_PACKED_BYTES; ++byte_index)
    {
        lane_base = byte_index << 1;
        dst_lanes[lane_base] = (I16)Q4_0DotAVX2NibbleToSigned(packed_qs[byte_index], FALSE);
        dst_lanes[lane_base + 1] = (I16)Q4_0DotAVX2NibbleToSigned(packed_qs[byte_index], TRUE);
    }

    return Q4_0_DOT_AVX2_OK;
}

I32 Q4_0DotAVX2I16LanesToQ0(I16 *lhs_lanes,
                            I64 lhs_len,
                            I16 *rhs_lanes,
                            I64 rhs_len,
                            I64 *out_q0)
{
    I32 pair_partials[16];
    I64 pair_index;
    I64 lane0;
    I64 lane1;
    I64 lane_base;
    I64 q0_accum;
    I64 term;

    if (!lhs_lanes || !rhs_lanes || !out_q0)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;
    if (lhs_len < Q4_0_DOT_AVX2_VALUES_PER_BLOCK)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;
    if (rhs_len < Q4_0_DOT_AVX2_VALUES_PER_BLOCK)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;

    // AVX2 vpmaddwd-shape reduction:
    // - pair_partials[i] accumulates lanes (2*i,2*i+1) as one I32 term.
    // - horizontal sum over 16 terms yields q_dot_q0.
    for (pair_index = 0; pair_index < 16; ++pair_index)
    {
        lane_base = pair_index << 1;
        lane0 = ((I64)lhs_lanes[lane_base]) * ((I64)rhs_lanes[lane_base]);
        lane1 = ((I64)lhs_lanes[lane_base + 1]) * ((I64)rhs_lanes[lane_base + 1]);
        pair_partials[pair_index] = (I32)(lane0 + lane1);
    }

    q0_accum = 0;
    for (pair_index = 0; pair_index < 16; ++pair_index)
    {
        term = (I64)pair_partials[pair_index];
        if (!Q4_0DotAVX2TryAddI64(q0_accum, term, &q0_accum))
            return Q4_0_DOT_AVX2_ERR_OVERFLOW;
    }

    *out_q0 = q0_accum;
    return Q4_0_DOT_AVX2_OK;
}

Bool Q4_0DotAVX2KernelEnabled()
{
    // TempleOS inline-asm AVX2 dispatch hook:
    // keep this helper as explicit guard so air-gapped/older CPUs
    // deterministically use the scalar checked path.
    return TRUE;
}

I32 Q4_0DotProductBlockQ32Checked(Q4_0Block *lhs, Q4_0Block *rhs, I64 *out_dot_q32)
{
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 nibble_dot_q0;
    I64 byte_index;
    U8 lhs_packed;
    U8 rhs_packed;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;

    lhs_scale_q16 = Q4_0DotAVX2F16ToQ16(lhs->d_fp16);
    rhs_scale_q16 = Q4_0DotAVX2F16ToQ16(rhs->d_fp16);

    if (!Q4_0DotAVX2TryMulI64(lhs_scale_q16, rhs_scale_q16, &scale_prod_q32))
        return Q4_0_DOT_AVX2_ERR_OVERFLOW;

    nibble_dot_q0 = 0;
    for (byte_index = 0; byte_index < Q4_0_DOT_AVX2_PACKED_BYTES; ++byte_index)
    {
        lhs_packed = lhs->qs[byte_index];
        rhs_packed = rhs->qs[byte_index];

        if (!Q4_0DotAVX2TryAddI64(nibble_dot_q0,
                                  Q4_0DotAVX2NibbleToSigned(lhs_packed, FALSE) * Q4_0DotAVX2NibbleToSigned(rhs_packed, FALSE),
                                  &nibble_dot_q0))
            return Q4_0_DOT_AVX2_ERR_OVERFLOW;

        if (!Q4_0DotAVX2TryAddI64(nibble_dot_q0,
                                  Q4_0DotAVX2NibbleToSigned(lhs_packed, TRUE) * Q4_0DotAVX2NibbleToSigned(rhs_packed, TRUE),
                                  &nibble_dot_q0))
            return Q4_0_DOT_AVX2_ERR_OVERFLOW;
    }

    if (!Q4_0DotAVX2TryMulI64(scale_prod_q32, nibble_dot_q0, out_dot_q32))
        return Q4_0_DOT_AVX2_ERR_OVERFLOW;

    return Q4_0_DOT_AVX2_OK;
}

I32 Q4_0DotProductBlockQ32AVX2CoreChecked(Q4_0Block *lhs,
                                          Q4_0Block *rhs,
                                          I64 *out_dot_q32)
{
    I16 lhs_lanes[Q4_0_DOT_AVX2_VALUES_PER_BLOCK];
    I16 rhs_lanes[Q4_0_DOT_AVX2_VALUES_PER_BLOCK];
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 dot_q0;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;

    status = Q4_0DotAVX2UnpackBlockToI16Lanes(lhs->qs,
                                              Q4_0_DOT_AVX2_PACKED_BYTES,
                                              lhs_lanes,
                                              Q4_0_DOT_AVX2_VALUES_PER_BLOCK);
    if (status != Q4_0_DOT_AVX2_OK)
        return status;

    status = Q4_0DotAVX2UnpackBlockToI16Lanes(rhs->qs,
                                              Q4_0_DOT_AVX2_PACKED_BYTES,
                                              rhs_lanes,
                                              Q4_0_DOT_AVX2_VALUES_PER_BLOCK);
    if (status != Q4_0_DOT_AVX2_OK)
        return status;

    status = Q4_0DotAVX2I16LanesToQ0(lhs_lanes,
                                     Q4_0_DOT_AVX2_VALUES_PER_BLOCK,
                                     rhs_lanes,
                                     Q4_0_DOT_AVX2_VALUES_PER_BLOCK,
                                     &dot_q0);
    if (status != Q4_0_DOT_AVX2_OK)
        return status;

    lhs_scale_q16 = Q4_0DotAVX2F16ToQ16(lhs->d_fp16);
    rhs_scale_q16 = Q4_0DotAVX2F16ToQ16(rhs->d_fp16);

    if (!Q4_0DotAVX2TryMulI64(lhs_scale_q16, rhs_scale_q16, &scale_prod_q32))
        return Q4_0_DOT_AVX2_ERR_OVERFLOW;
    if (!Q4_0DotAVX2TryMulI64(scale_prod_q32, dot_q0, out_dot_q32))
        return Q4_0_DOT_AVX2_ERR_OVERFLOW;

    return Q4_0_DOT_AVX2_OK;
}

I32 Q4_0DotProductBlocksQ32Checked(Q4_0Block *lhs,
                                   Q4_0Block *rhs,
                                   I64 block_count,
                                   I64 *out_dot_q32)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;

    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DotProductBlockQ32Checked(&lhs[block_index], &rhs[block_index], &block_dot_q32);
        if (status != Q4_0_DOT_AVX2_OK)
            return status;

        if (!Q4_0DotAVX2TryAddI64(total_dot_q32, block_dot_q32, &total_dot_q32))
            return Q4_0_DOT_AVX2_ERR_OVERFLOW;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_DOT_AVX2_OK;
}

I32 Q4_0DotProductBlocksQ32AVX2CoreChecked(Q4_0Block *lhs,
                                           Q4_0Block *rhs,
                                           I64 block_count,
                                           I64 *out_dot_q32)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;

    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DotProductBlockQ32AVX2CoreChecked(&lhs[block_index],
                                                       &rhs[block_index],
                                                       &block_dot_q32);
        if (status != Q4_0_DOT_AVX2_OK)
            return status;

        if (!Q4_0DotAVX2TryAddI64(total_dot_q32, block_dot_q32, &total_dot_q32))
            return Q4_0_DOT_AVX2_ERR_OVERFLOW;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_DOT_AVX2_OK;
}

I32 Q4_0DotProductBlocksAVX2Checked(Q4_0Block *lhs,
                                    Q4_0Block *rhs,
                                    I64 block_count,
                                    I64 *out_dot_q32)
{
    I64 avx2_dot_q32;
    I64 scalar_dot_q32;
    I32 avx2_status;
    I32 scalar_status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_DOT_AVX2_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_DOT_AVX2_ERR_BAD_LEN;

    // Guarded fallback parity path:
    // 1) Run scalar checked kernel always (single source of correctness).
    // 2) Run AVX2-shape kernel when enabled.
    // 3) Require exact status/result parity before committing output.
    scalar_status = Q4_0DotProductBlocksQ32Checked(lhs,
                                                   rhs,
                                                   block_count,
                                                   &scalar_dot_q32);
    if (!Q4_0DotAVX2KernelEnabled())
    {
        if (scalar_status != Q4_0_DOT_AVX2_OK)
            return scalar_status;

        *out_dot_q32 = scalar_dot_q32;
        return Q4_0_DOT_AVX2_OK;
    }

    avx2_status = Q4_0DotProductBlocksQ32AVX2CoreChecked(lhs,
                                                         rhs,
                                                         block_count,
                                                         &avx2_dot_q32);

    if (avx2_status != scalar_status)
        return Q4_0_DOT_AVX2_ERR_PARITY;
    if (avx2_status != Q4_0_DOT_AVX2_OK)
        return avx2_status;
    if (avx2_dot_q32 != scalar_dot_q32)
        return Q4_0_DOT_AVX2_ERR_PARITY;

    *out_dot_q32 = avx2_dot_q32;
    return Q4_0_DOT_AVX2_OK;
}

I32 Q4_0DotProductBlocksAVX2CheckedDefault(Q4_0Block *lhs,
                                           Q4_0Block *rhs,
                                           I64 block_count,
                                           I64 *out_dot_q32)
{
    // Canonical default wrapper:
    // - Core checked kernel already consumes only active span (`block_count`).
    // - Keep this as a direct pass-through default entrypoint so callers can
    //   rely on one stable symbol while preserving exact checked-core
    //   success/error/no-partial-write semantics.
    return Q4_0DotProductBlocksAVX2Checked(lhs,
                                           rhs,
                                           block_count,
                                           out_dot_q32);
}
