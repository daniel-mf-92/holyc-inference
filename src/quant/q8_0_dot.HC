// q8_0_dot.HC
// Naive integer-only Q8_0 dot product kernels for TempleOS.
//
// Math:
//   GGML Q8_0 stores 32 signed 8-bit values and one fp16 scale d per block.
//   Decode uses: x_i = d * q_i
//
// Integer HolyC form:
//   d_q16    = Round(d * 2^16)
//   x_q16    = d_q16 * q_i
//   dot_q32  = Sum(x_q16 * y_q16)
//            = (dA_q16 * dB_q16) * Sum(qA_i * qB_i)
//
// dot_q32 keeps 32 fractional bits; caller may round to Q16 as needed.

#define Q8_0_VALUES_PER_BLOCK 32
#define Q8_0_PACKED_BYTES     32

#define Q8_0_OK               0
#define Q8_0_ERR_NULL_PTR     1
#define Q8_0_ERR_BAD_DST_LEN  2

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q8_0_PACKED_BYTES];
};

I64 Q8_0RoundShiftRightUnsigned(I64 value, I64 shift)
{
    I64 round_bias;

    if (shift <= 0)
        return value;

    round_bias = 1 << (shift - 1);
    return (value + round_bias) >> shift;
}

I64 Q8_0RoundShiftRightSigned(I64 value, I64 shift)
{
    if (shift <= 0)
        return value;

    if (value >= 0)
        return Q8_0RoundShiftRightUnsigned(value, shift);

    return -Q8_0RoundShiftRightUnsigned(-value, shift);
}

I64 Q8_0F16ToQ16(U16 fp16_bits)
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

        magnitude_q16 = Q8_0RoundShiftRightUnsigned(fraction_bits, 8);
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
        magnitude_q16 = Q8_0RoundShiftRightUnsigned(mantissa, -shift_amount);

    if (sign_bit)
        return -magnitude_q16;
    return magnitude_q16;
}

I32 Q8_0DotProductBlockQ32(Q8_0Block *lhs, Q8_0Block *rhs, I64 *out_dot_q32)
{
    I64 lhs_scale_q16;
    I64 rhs_scale_q16;
    I64 scale_prod_q32;
    I64 byte_index;
    I64 q_dot_q0;

    if (!lhs || !rhs || !out_dot_q32)
        return Q8_0_ERR_NULL_PTR;

    lhs_scale_q16 = Q8_0F16ToQ16(lhs->d_fp16);
    rhs_scale_q16 = Q8_0F16ToQ16(rhs->d_fp16);
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16;

    // Keep the integer dot explicit and auditable:
    // q_dot_q0 = Sum(qA_i * qB_i), with q in signed byte domain [-128, 127].
    q_dot_q0 = 0;
    for (byte_index = 0; byte_index < Q8_0_PACKED_BYTES; ++byte_index)
        q_dot_q0 += lhs->qs[byte_index] * rhs->qs[byte_index];

    *out_dot_q32 = scale_prod_q32 * q_dot_q0;
    return Q8_0_OK;
}

I32 Q8_0DotProductBlocksQ32(Q8_0Block *lhs, Q8_0Block *rhs, I64 block_count, I64 *out_dot_q32)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_dot_q32;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32)
        return Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;

    total_dot_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q8_0DotProductBlockQ32(&lhs[block_index], &rhs[block_index], &block_dot_q32);
        if (status != Q8_0_OK)
            return status;

        total_dot_q32 += block_dot_q32;
    }

    *out_dot_q32 = total_dot_q32;
    return Q8_0_OK;
}

I32 Q8_0DotProductBlocksQ16Accumulate(Q8_0Block *lhs, Q8_0Block *rhs, I64 block_count, I64 *io_accum_q16)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 total_q16;
    I32 status;

    if (!lhs || !rhs || !io_accum_q16)
        return Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;

    total_q16 = *io_accum_q16;

    // Matmul callers usually hold accumulators in Q16 lanes.
    // Keep per-block rounding explicit: Q32 block dot -> rounded Q16 -> add.
    // This avoids hidden scale changes and makes every block contribution auditable.
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q8_0DotProductBlockQ32(&lhs[block_index], &rhs[block_index], &block_dot_q32);
        if (status != Q8_0_OK)
            return status;

        total_q16 += Q8_0DotQ32ToQ16(block_dot_q32);
    }

    *io_accum_q16 = total_q16;
    return Q8_0_OK;
}

I64 Q8_0DotQ32ToQ16(I64 dot_q32)
{
    return Q8_0RoundShiftRightSigned(dot_q32, 16);
}
