// q4_0.HC
// Q4_0 block layout + integer-only dequantization helpers for TempleOS.
//
// GGML Q4_0 block layout (18 bytes total):
//   - d_fp16 : fp16 scale
//   - qs[16] : packed nibbles (32 values, 2 values per byte)
//
// Decode equation (reference):
//   x = d * (q - 8)
//
// Integer HolyC form:
//   d_q16  = Round(d_fp16 * 2^16)
//   x_q16  = d_q16 * q_signed
//
// No float ops are used in runtime decode.

#define Q4_0_VALUES_PER_BLOCK 32
#define Q4_0_PACKED_BYTES     16
#define Q4_0_BLOCK_BYTES      18

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

    // Subnormal and zero numbers:
    // value = fraction * 2^-24, so Q16 = fraction * 2^-8.
    if (!exponent_bits)
    {
        if (!fraction_bits)
            return 0;

        magnitude_q16 = Q4_0RoundShiftRightUnsigned(fraction_bits, 8);
        if (sign_bit)
            return -magnitude_q16;
        return magnitude_q16;
    }

    // INF/NAN: saturate to a very large signed magnitude instead of trapping.
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

I32 Q4_0UnpackBlockSigned(Q4_0Block *block, I64 *dst_signed, I64 dst_len)
{
    I64 byte_index;
    I64 out_index;
    U8 packed;

    if (!block || !dst_signed)
        return Q4_0_ERR_NULL_PTR;
    if (dst_len < Q4_0_VALUES_PER_BLOCK)
        return Q4_0_ERR_BAD_DST_LEN;

    for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
    {
        packed = block->qs[byte_index];

        out_index = byte_index << 1;
        dst_signed[out_index] = Q4_0NibbleToSigned(packed, FALSE);
        dst_signed[out_index + 1] = Q4_0NibbleToSigned(packed, TRUE);
    }

    return Q4_0_OK;
}

I32 Q4_0DequantizeBlockQ16(Q4_0Block *block, I64 *dst_q16, I64 dst_len)
{
    I64 scale_q16;
    I64 byte_index;
    I64 out_index;
    U8 packed;
    I64 q_signed;

    if (!block || !dst_q16)
        return Q4_0_ERR_NULL_PTR;
    if (dst_len < Q4_0_VALUES_PER_BLOCK)
        return Q4_0_ERR_BAD_DST_LEN;

    scale_q16 = Q4_0F16ToQ16(block->d_fp16);

    for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
    {
        packed = block->qs[byte_index];
        out_index = byte_index << 1;

        // Lower nibble is even index; upper nibble is odd index.
        q_signed = Q4_0NibbleToSigned(packed, FALSE);
        dst_q16[out_index] = scale_q16 * q_signed;

        q_signed = Q4_0NibbleToSigned(packed, TRUE);
        dst_q16[out_index + 1] = scale_q16 * q_signed;
    }

    return Q4_0_OK;
}

I32 Q4_0DequantizeBlocksQ16(Q4_0Block *blocks, I64 block_count, I64 *dst_q16, I64 dst_len)
{
    I64 block_index;
    I64 dst_offset;
    I32 status;

    if (!blocks || !dst_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;
    if (dst_len < block_count * Q4_0_VALUES_PER_BLOCK)
        return Q4_0_ERR_BAD_DST_LEN;

    dst_offset = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DequantizeBlockQ16(&blocks[block_index], &dst_q16[dst_offset], Q4_0_VALUES_PER_BLOCK);
        if (status != Q4_0_OK)
            return status;

        dst_offset += Q4_0_VALUES_PER_BLOCK;
    }

    return Q4_0_OK;
}
