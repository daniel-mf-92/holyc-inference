// q4_0_dot.HC
// Naive integer-only Q4_0 dot product kernels for TempleOS.
//
// Math:
//   For each block, GGML Q4_0 stores 32 4-bit values and one fp16 scale d.
//   Decode uses: x_i = d * (q_i - 8)
//
// Integer HolyC form:
//   d_q16   = Round(d * 2^16)
//   q_s     = nibble - 8   (in [-8, 7])
//   x_q16   = d_q16 * q_s
//   dot_q32 = Sum(x_q16 * y_q16)
//           = (dA_q16 * dB_q16) * Sum(qA_s * qB_s)
//
// dot_q32 keeps 32 fractional bits; caller may round to Q16 as needed.

#define Q4_0_VALUES_PER_BLOCK 32
#define Q4_0_PACKED_BYTES     16

#define Q4_0_OK               0
#define Q4_0_ERR_NULL_PTR     1
#define Q4_0_ERR_BAD_DST_LEN  2

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_PACKED_BYTES];
};

I64 Q4_0RoundShiftRightUnsigned(I64 value, I64 shift)
{
    I64 round_bias;

    if (shift <= 0)
        return value;

    round_bias = 1 << (shift - 1);
    return (value + round_bias) >> shift;
}

I64 Q4_0RoundShiftRightSigned(I64 value, I64 shift)
{
    if (shift <= 0)
        return value;

    if (value >= 0)
        return Q4_0RoundShiftRightUnsigned(value, shift);

    return -Q4_0RoundShiftRightUnsigned(-value, shift);
}

I64 Q4_0F16ToQ16(U16 fp16_bits)
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

        magnitude_q16 = Q4_0RoundShiftRightUnsigned(fraction_bits, 8);
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
        magnitude_q16 = Q4_0RoundShiftRightUnsigned(mantissa, -shift_amount);

    if (sign_bit)
        return -magnitude_q16;
    return magnitude_q16;
}

I64 Q4_0NibbleToSigned(U8 packed, Bool upper_nibble)
{
    I64 q_unsigned;

    if (upper_nibble)
        q_unsigned = (packed >> 4) & 0x0F;
    else
        q_unsigned = packed & 0x0F;

    return q_unsigned - 8;
}

I32 Q4_0DotProductBlockQ32(Q4_0Block *lhs, Q4_0Block *rhs, I64 *out_dot_q32)
{
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 nibble_dot_q0;
    I64 byte_index;
    U8 lhs_packed;
    U8 rhs_packed;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_ERR_NULL_PTR;

    lhs_scale_q16 = Q4_0F16ToQ16(lhs->d_fp16);
    rhs_scale_q16 = Q4_0F16ToQ16(rhs->d_fp16);
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16;

    nibble_dot_q0 = 0;
    for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
    {
        lhs_packed = lhs->qs[byte_index];
        rhs_packed = rhs->qs[byte_index];

        nibble_dot_q0 += Q4_0NibbleToSigned(lhs_packed, FALSE) * Q4_0NibbleToSigned(rhs_packed, FALSE);
        nibble_dot_q0 += Q4_0NibbleToSigned(lhs_packed, TRUE) * Q4_0NibbleToSigned(rhs_packed, TRUE);
    }

    *out_dot_q32 = scale_prod_q32 * nibble_dot_q0;
    return Q4_0_OK;
}

I32 Q4_0DotProductBlocksQ32(Q4_0Block *lhs, Q4_0Block *rhs, I64 block_count, I64 *out_dot_q32)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DotProductBlockQ32(&lhs[block_index], &rhs[block_index], &block_dot_q32);
        if (status != Q4_0_OK)
            return status;

        total_dot_q32 += block_dot_q32;
    }

    *out_dot_q32 = total_dot_q32;
    return Q4_0_OK;
}

I32 Q4_0DotProductBlocksQ32ToQ16(Q4_0Block *lhs, Q4_0Block *rhs, I64 block_count, I64 *out_dot_q16)
{
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    // WS4 matmul callers can keep a widened I64 Q32 path for the full dot,
    // then round once to Q16 at the hand-off boundary.
    status = Q4_0DotProductBlocksQ32(lhs, rhs, block_count, &total_dot_q32);
    if (status != Q4_0_OK)
        return status;

    *out_dot_q16 = Q4_0DotQ32ToQ16(total_dot_q32);
    return Q4_0_OK;
}

I32 Q4_0DotRowBlocksQ16(Q4_0Block *row_blocks, Q4_0Block *vec_blocks, I64 block_count, I64 *out_dot_q16)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q16;
    I32 status;

    if (!row_blocks || !vec_blocks || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    // Matmul row kernels usually keep accumulators in Q16 lanes.
    // Convert each block dot Q32->Q16 with explicit rounding before summation,
    // so every 32-value block contribution stays auditable and deterministic.
    total_dot_q16 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DotProductBlockQ32(&row_blocks[block_index], &vec_blocks[block_index], &block_dot_q32);
        if (status != Q4_0_OK)
            return status;

        total_dot_q16 += Q4_0DotQ32ToQ16(block_dot_q32);
    }

    *out_dot_q16 = total_dot_q16;
    return Q4_0_OK;
}

I64 Q4_0DotQ32ToQ16(I64 dot_q32)
{
    return Q4_0RoundShiftRightSigned(dot_q32, 16);
}
