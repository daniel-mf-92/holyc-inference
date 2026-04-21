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
#define Q4_0_BLOCK_SIZE       Q4_0_VALUES_PER_BLOCK
#define Q4_0_PACKED_BYTES     16
#define Q4_0_BLOCK_BYTES      18

#define Q4_0_OK               0
#define Q4_0_ERR_NULL_PTR     1
#define Q4_0_ERR_BAD_DST_LEN  2
#define Q4_0_ERR_OVERFLOW     3
#define Q4_0_I64_MAX          0x7FFFFFFFFFFFFFFF
#define Q4_0_I64_MIN          -0x8000000000000000
#define Q4_0_ERR_OVERFLOW     3
#define Q4_0_I64_MAX          0x7FFFFFFFFFFFFFFF
#define Q4_0_I64_MIN          -0x8000000000000000

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_PACKED_BYTES];
};

Bool Q4_0TryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
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

    if (lhs > Q4_0_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q4_0TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (rhs > 0 && lhs > Q4_0_I64_MAX - rhs)
        return FALSE;
    if (rhs < 0 && lhs < Q4_0_I64_MIN - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool Q4_0TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    I64 abs_lhs;
    I64 abs_rhs;
    Bool neg_result;

    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if ((lhs == Q4_0_I64_MIN && rhs == -1) ||
        (rhs == Q4_0_I64_MIN && lhs == -1))
        return FALSE;

    abs_lhs = lhs;
    if (abs_lhs < 0)
        abs_lhs = -abs_lhs;

    abs_rhs = rhs;
    if (abs_rhs < 0)
        abs_rhs = -abs_rhs;

    if (abs_lhs > Q4_0_I64_MAX / abs_rhs)
        return FALSE;

    neg_result = ((lhs < 0) != (rhs < 0));
    *out_prod = abs_lhs * abs_rhs;
    if (neg_result)
        *out_prod = -*out_prod;
    return TRUE;
}

Bool Q4_0TryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
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

    if (lhs > Q4_0_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q4_0TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (rhs > 0 && lhs > Q4_0_I64_MAX - rhs)
        return FALSE;
    if (rhs < 0 && lhs < Q4_0_I64_MIN - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool Q4_0TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    I64 abs_lhs;
    I64 abs_rhs;

    if (!out_prod)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if ((lhs == Q4_0_I64_MIN && rhs == -1) ||
        (rhs == Q4_0_I64_MIN && lhs == -1))
        return FALSE;

    abs_lhs = (lhs < 0) ? -lhs : lhs;
    abs_rhs = (rhs < 0) ? -rhs : rhs;

    if (abs_lhs > Q4_0_I64_MAX / abs_rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

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

// Checked strided Q4_0 dequantizer with canonical contiguous output stride.
// No-partial contract: preflight all bounds/overflow first, then commit decode.
I32 Q4_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride(
    Q4_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity)
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
    I64 q_signed;
    I64 decode_q16;
    I64 dst_block_stride_values;
    I64 byte_index;
    U8 packed;

    if (!src_blocks || !dst_q16)
        return Q4_0_ERR_NULL_PTR;

    if (src_block_capacity < 0 || dst_q16_capacity < 0)
        return Q4_0_ERR_BAD_DST_LEN;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;
    if (src_block_stride <= 0)
        return Q4_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    dst_block_stride_values = Q4_0_VALUES_PER_BLOCK;

    if (!block_count)
        return Q4_0_OK;

    if (!Q4_0TryMulI64NonNeg(block_count - 1, src_block_stride, &src_last_offset))
        return Q4_0_ERR_OVERFLOW;
    if (!Q4_0TryMulI64NonNeg(block_count - 1, dst_block_stride_values, &dst_last_base))
        return Q4_0_ERR_OVERFLOW;

    if (!Q4_0TryAddI64(src_last_offset, 1, &src_required_blocks))
        return Q4_0_ERR_OVERFLOW;
    if (!Q4_0TryAddI64(dst_last_base, Q4_0_VALUES_PER_BLOCK, &dst_required_values))
        return Q4_0_ERR_OVERFLOW;

    if (src_required_blocks > src_block_capacity)
        return Q4_0_ERR_BAD_DST_LEN;
    if (dst_required_values > dst_q16_capacity)
        return Q4_0_ERR_BAD_DST_LEN;

    // Overflow preflight pass (zero-write): every lane decode must be safe
    // before any destination element is committed.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q4_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q4_0_ERR_OVERFLOW;

        scale_q16 = Q4_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
        {
            packed = src_blocks[src_index].qs[byte_index];

            q_signed = Q4_0NibbleToSigned(packed, FALSE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;

            q_signed = Q4_0NibbleToSigned(packed, TRUE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;
        }
    }

    // Commit pass: decode each block only after the full preflight succeeds.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q4_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q4_0_ERR_OVERFLOW;
        if (!Q4_0TryMulI64NonNeg(block_idx, dst_block_stride_values, &dst_base))
            return Q4_0_ERR_OVERFLOW;

        scale_q16 = Q4_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
        {
            packed = src_blocks[src_index].qs[byte_index];
            lane = byte_index << 1;

            q_signed = Q4_0NibbleToSigned(packed, FALSE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;
            dst_q16[dst_base + lane] = decode_q16;

            q_signed = Q4_0NibbleToSigned(packed, TRUE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;
            dst_q16[dst_base + lane + 1] = decode_q16;
        }
    }

    return Q4_0_OK;
}

// Checked + no-partial dequantization over a strided Q4_0 block array.
//
// Preflight validates geometry/capacity and every per-lane scale*q_signed
// multiplication for overflow before any destination cell is written.
I32 Q4_0DequantizeBlockQ16CheckedNoPartialArray(
    Q4_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values)
{
    I64 src_last_offset;
    I64 dst_last_base;
    I64 required_src_blocks;
    I64 required_dst_values;
    I64 block_idx;
    I64 src_index;
    I64 dst_base;
    I64 lane;
    I64 scale_q16;
    I64 q_signed;
    I64 decode_q16;
    U8 packed;

    if (!src_blocks || !dst_q16)
        return Q4_0_ERR_NULL_PTR;

    if (src_block_capacity < 0 || dst_q16_capacity < 0)
        return Q4_0_ERR_BAD_DST_LEN;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;
    if (src_block_stride <= 0 || dst_block_stride_values <= 0)
        return Q4_0_ERR_BAD_DST_LEN;
    if (dst_block_stride_values < Q4_0_VALUES_PER_BLOCK)
        return Q4_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    if (!block_count)
        return Q4_0_OK;

    if (!Q4_0TryMulI64NonNeg(block_count - 1, src_block_stride, &src_last_offset))
        return Q4_0_ERR_OVERFLOW;
    if (!Q4_0TryMulI64NonNeg(block_count - 1, dst_block_stride_values, &dst_last_base))
        return Q4_0_ERR_OVERFLOW;

    if (!Q4_0TryAddI64(src_last_offset, 1, &required_src_blocks))
        return Q4_0_ERR_OVERFLOW;
    if (!Q4_0TryAddI64(dst_last_base, Q4_0_VALUES_PER_BLOCK, &required_dst_values))
        return Q4_0_ERR_OVERFLOW;

    if (required_src_blocks > src_block_capacity)
        return Q4_0_ERR_BAD_DST_LEN;
    if (required_dst_values > dst_q16_capacity)
        return Q4_0_ERR_BAD_DST_LEN;

    // Preflight (zero-write): prove decode math is safe for every output lane.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q4_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q4_0_ERR_OVERFLOW;

        scale_q16 = Q4_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (lane = 0; lane < Q4_0_PACKED_BYTES; ++lane)
        {
            packed = src_blocks[src_index].qs[lane];

            q_signed = Q4_0NibbleToSigned(packed, FALSE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;

            q_signed = Q4_0NibbleToSigned(packed, TRUE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;
        }
    }

    // Commit pass: preflight already proved all writes/ops are valid.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q4_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q4_0_ERR_OVERFLOW;
        if (!Q4_0TryMulI64NonNeg(block_idx, dst_block_stride_values, &dst_base))
            return Q4_0_ERR_OVERFLOW;

        scale_q16 = Q4_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (lane = 0; lane < Q4_0_PACKED_BYTES; ++lane)
        {
            packed = src_blocks[src_index].qs[lane];

            q_signed = Q4_0NibbleToSigned(packed, FALSE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;
            dst_q16[dst_base + (lane << 1)] = decode_q16;

            q_signed = Q4_0NibbleToSigned(packed, TRUE);
            if (!Q4_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q4_0_ERR_OVERFLOW;
            dst_q16[dst_base + (lane << 1) + 1] = decode_q16;
        }
    }

    return Q4_0_OK;
}

// Canonical default-stride wrapper for contiguous 32-value block writes.
I32 Q4_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride(
    Q4_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity)
{
    I64 out_stride_values;

    out_stride_values = Q4_0_BLOCK_SIZE;

    return Q4_0DequantizeBlockQ16CheckedNoPartialArray(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        out_stride_values);
}
