// q8_0.HC
// Q8_0 block layout + integer-only dequantization helpers for TempleOS.
//
// GGML Q8_0 block layout (34 bytes total):
//   - d_fp16 : fp16 scale
//   - qs[32] : signed 8-bit quantized values
//
// Decode equation (reference):
//   x = d * q
//
// Integer HolyC form:
//   d_q16 = Round(d_fp16 * 2^16)
//   x_q16 = d_q16 * q_signed
//
// Runtime path is integer-only after one fp16->Q16 conversion per block.

#define Q8_0_VALUES_PER_BLOCK 32
#define Q8_0_PACKED_BYTES     32
#define Q8_0_BLOCK_BYTES      34

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

    // Subnormal and zero:
    // value = fraction * 2^-24, so Q16 = fraction * 2^-8.
    if (!exponent_bits)
    {
        if (!fraction_bits)
            return 0;

        magnitude_q16 = Q8_0RoundShiftRightUnsigned(fraction_bits, 8);
        if (sign_bit)
            return -magnitude_q16;
        return magnitude_q16;
    }

    // INF/NAN: saturate instead of trapping.
    if (exponent_bits == 0x1F)
    {
        if (sign_bit)
            return -0x3FFFFFFFFFFFFFFF;
        return 0x3FFFFFFFFFFFFFFF;
    }

    // Normal numbers:
    // value = (1024 + fraction) * 2^(exp-25)
    // Q16  = (1024 + fraction) * 2^(exp-9)
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

I32 Q8_0UnpackBlockSigned(Q8_0Block *block, I64 *dst_signed, I64 dst_len)
{
    I64 index;

    if (!block || !dst_signed)
        return Q8_0_ERR_NULL_PTR;
    if (dst_len < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    for (index = 0; index < Q8_0_VALUES_PER_BLOCK; ++index)
        dst_signed[index] = block->qs[index];

    return Q8_0_OK;
}

I32 Q8_0DequantizeBlockQ16(Q8_0Block *block, I64 *dst_q16, I64 dst_len)
{
    I64 scale_q16;
    I64 index;

    if (!block || !dst_q16)
        return Q8_0_ERR_NULL_PTR;
    if (dst_len < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    scale_q16 = Q8_0F16ToQ16(block->d_fp16);

    // Explicit element-wise integer decode keeps math auditable:
    // x_q16 = d_q16 * q_signed.
    for (index = 0; index < Q8_0_VALUES_PER_BLOCK; ++index)
        dst_q16[index] = scale_q16 * block->qs[index];

    return Q8_0_OK;
}

I32 Q8_0DequantizeBlocksQ16(Q8_0Block *blocks, I64 block_count, I64 *dst_q16, I64 dst_len)
{
    I64 block_index;
    I64 dst_offset;
    I32 status;

    if (!blocks || !dst_q16)
        return Q8_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (dst_len < block_count * Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    dst_offset = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q8_0DequantizeBlockQ16(&blocks[block_index], &dst_q16[dst_offset], Q8_0_VALUES_PER_BLOCK);
        if (status != Q8_0_OK)
            return status;

        dst_offset += Q8_0_VALUES_PER_BLOCK;
    }

    return Q8_0_OK;
}

