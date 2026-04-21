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
#define Q8_0_ERR_OVERFLOW     3
#define Q8_0_I64_MAX          0x7FFFFFFFFFFFFFFF

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q8_0_PACKED_BYTES];
};

Bool Q8_0TryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs > Q8_0_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q8_0TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (rhs > 0 && lhs > Q8_0_I64_MAX - rhs)
        return FALSE;
    if (rhs < 0 && lhs < (-Q8_0_I64_MAX - 1) - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

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

// Checked strided Q8_0 dequantizer with strict no-partial semantics.
//
// Layout:
//   src block index = block_idx * src_block_stride
//   dst base index  = block_idx * dst_block_stride_values
//   dst lane index  = dst base + lane (lane in [0,32))
//
// The function preflights all capacity/overflow constraints first, then
// performs decode commits. Any preflight failure leaves dst_q16 untouched.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArray(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values)
{
    I64 block_idx;
    I64 lane;
    I64 src_index;
    I64 dst_base;
    I64 src_last_offset;
    I64 dst_last_base;
    I64 src_required_blocks;
    I64 dst_required_values;
    I64 scale_q16;

    if (!src_blocks || !dst_q16)
        return Q8_0_ERR_NULL_PTR;

    if (src_block_capacity < 0 || dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (src_block_stride <= 0 || dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;

    // Per-block decoded vectors are 32 contiguous values; smaller strides
    // would overlap destination rows and hide partial-write faults.
    if (dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!block_count)
        return Q8_0_OK;

    if (!Q8_0TryMulI64NonNeg(block_count - 1, src_block_stride, &src_last_offset))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(block_count - 1, dst_block_stride_values, &dst_last_base))
        return Q8_0_ERR_OVERFLOW;

    if (!Q8_0TryAddI64(src_last_offset, 1, &src_required_blocks))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryAddI64(dst_last_base, Q8_0_VALUES_PER_BLOCK, &dst_required_values))
        return Q8_0_ERR_OVERFLOW;

    if (src_required_blocks > src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (dst_required_values > dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    // Commit pass: decode each block only after full preflight success.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q8_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(block_idx, dst_block_stride_values, &dst_base))
            return Q8_0_ERR_OVERFLOW;

        scale_q16 = Q8_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (lane = 0; lane < Q8_0_VALUES_PER_BLOCK; ++lane)
            dst_q16[dst_base + lane] = scale_q16 * src_blocks[src_index].qs[lane];
    }

    return Q8_0_OK;
}
