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
#define Q8_0_BLOCK_SIZE       Q8_0_VALUES_PER_BLOCK
#define Q8_0_PACKED_BYTES     32
#define Q8_0_BLOCK_BYTES      34

#define Q8_0_OK               0
#define Q8_0_ERR_NULL_PTR     1
#define Q8_0_ERR_BAD_DST_LEN  2
#define Q8_0_ERR_OVERFLOW     3
#define Q8_0_I64_MAX          0x7FFFFFFFFFFFFFFF
#define Q8_0_I64_MIN          -0x8000000000000000

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

Bool Q8_0TryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
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

    if ((lhs == Q8_0_I64_MIN && rhs == -1) ||
        (rhs == Q8_0_I64_MIN && lhs == -1))
        return FALSE;

    abs_lhs = (lhs < 0) ? -lhs : lhs;
    abs_rhs = (rhs < 0) ? -rhs : rhs;

    if (abs_lhs > Q8_0_I64_MAX / abs_rhs)
        return FALSE;

    *out_prod = lhs * rhs;
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
    I64 q_signed;
    I64 decode_q16;

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

    // Overflow preflight pass (zero-write): validate every per-lane decode
    // multiply so commit pass cannot fault after writing a prefix.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q8_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q8_0_ERR_OVERFLOW;

        scale_q16 = Q8_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (lane = 0; lane < Q8_0_VALUES_PER_BLOCK; ++lane)
        {
            q_signed = src_blocks[src_index].qs[lane];
            if (!Q8_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q8_0_ERR_OVERFLOW;
        }
    }

    // Commit pass: decode each block only after full preflight success.
    for (block_idx = 0; block_idx < block_count; ++block_idx)
    {
        if (!Q8_0TryMulI64NonNeg(block_idx, src_block_stride, &src_index))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(block_idx, dst_block_stride_values, &dst_base))
            return Q8_0_ERR_OVERFLOW;

        scale_q16 = Q8_0F16ToQ16(src_blocks[src_index].d_fp16);
        for (lane = 0; lane < Q8_0_VALUES_PER_BLOCK; ++lane)
        {
            q_signed = src_blocks[src_index].qs[lane];
            if (!Q8_0TryMulI64(scale_q16, q_signed, &decode_q16))
                return Q8_0_ERR_OVERFLOW;
            dst_q16[dst_base + lane] = decode_q16;
        }
    }

    return Q8_0_OK;
}

// Canonical default-stride wrapper for contiguous per-block decode output.
// Keeps all checked/no-partial behavior in one audited core helper.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStride(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity)
{
    I64 out_stride_values;

    out_stride_values = Q8_0_BLOCK_SIZE;

    return Q8_0DequantizeBlockQ16CheckedNoPartialArray(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        out_stride_values);
}

// Commit-only diagnostics wrapper over the checked no-partial Q8_0 array
// dequantizer. Stages canonical geometry/byte counts, executes the canonical
// decode path, then atomically publishes staged diagnostics.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnly(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *staged_src_blocks;
    I64 *staged_dst_q16;
    I64 staged_src_block_capacity;
    I64 staged_src_block_stride;
    I64 staged_block_count;
    I64 staged_dst_q16_capacity;
    I64 staged_dst_block_stride_values;
    I64 src_last_offset;
    I64 dst_last_base;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;
    I32 status;

    if (!src_blocks || !dst_q16 ||
        !out_block_count || !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        (U8 *)out_block_count == (U8 *)src_blocks ||
        (U8 *)out_required_src_blocks == (U8 *)src_blocks ||
        (U8 *)out_required_dst_values == (U8 *)src_blocks ||
        (U8 *)out_required_src_bytes == (U8 *)src_blocks ||
        (U8 *)out_required_dst_bytes == (U8 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    // Snapshot caller geometry so staged diagnostics and decode commit use one
    // immutable view.
    staged_src_blocks = src_blocks;
    staged_dst_q16 = dst_q16;
    staged_src_block_capacity = src_block_capacity;
    staged_src_block_stride = src_block_stride;
    staged_block_count = block_count;
    staged_dst_q16_capacity = dst_q16_capacity;
    staged_dst_block_stride_values = dst_block_stride_values;

    if (staged_src_block_capacity < 0 || staged_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_src_block_stride <= 0 || staged_dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!staged_block_count)
    {
        staged_required_src_blocks = 0;
        staged_required_dst_values = 0;
        staged_required_src_bytes = 0;
        staged_required_dst_bytes = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(staged_block_count - 1,
                                 staged_src_block_stride,
                                 &src_last_offset))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(staged_block_count - 1,
                                 staged_dst_block_stride_values,
                                 &dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(src_last_offset, 1, &staged_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryAddI64(dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &staged_required_dst_values))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks,
                                 Q8_0_BLOCK_BYTES,
                                 &staged_required_src_bytes))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(staged_required_dst_values,
                                 sizeof(I64),
                                 &staged_required_dst_bytes))
            return Q8_0_ERR_OVERFLOW;
    }

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArray(
        staged_src_blocks,
        staged_src_block_capacity,
        staged_src_block_stride,
        staged_block_count,
        staged_dst_q16,
        staged_dst_q16_capacity,
        staged_dst_block_stride_values);
    if (status != Q8_0_OK)
        return status;

    // Immutable-argument revalidation before diagnostics publish.
    if (staged_src_blocks != src_blocks ||
        staged_dst_q16 != dst_q16 ||
        staged_src_block_capacity != src_block_capacity ||
        staged_src_block_stride != src_block_stride ||
        staged_block_count != block_count ||
        staged_dst_q16_capacity != dst_q16_capacity ||
        staged_dst_block_stride_values != dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_required_src_blocks > staged_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > staged_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    // Atomic diagnostics publish.
    *out_block_count = staged_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;
    return Q8_0_OK;
}

// Commit-only diagnostics wrapper over default-stride checked strided dequantizer.
// Uses canonical dst stride (Q8_0_BLOCK_SIZE), stages geometry/byte diagnostics,
// revalidates immutable snapshots, and publishes diagnostics atomically only
// after a successful no-partial decode commit.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStrideCommitOnly(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;

    I64 dst_block_stride_values;
    I64 staged_src_last_offset;
    I64 staged_dst_last_base;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;

    I32 status;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        (U8 *)out_block_count == (U8 *)src_blocks ||
        (U8 *)out_required_src_blocks == (U8 *)src_blocks ||
        (U8 *)out_required_dst_values == (U8 *)src_blocks ||
        (U8 *)out_required_src_bytes == (U8 *)src_blocks ||
        (U8 *)out_required_dst_bytes == (U8 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;

    dst_block_stride_values = Q8_0_BLOCK_SIZE;

    if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_src_block_stride <= 0)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!snapshot_block_count)
    {
        staged_required_src_blocks = 0;
        staged_required_dst_values = 0;
        staged_required_src_bytes = 0;
        staged_required_dst_bytes = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_src_block_stride,
                                 &staged_src_last_offset))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 dst_block_stride_values,
                                 &staged_dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(staged_src_last_offset, 1, &staged_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(staged_dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &staged_required_dst_values))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks,
                                 Q8_0_BLOCK_BYTES,
                                 &staged_required_src_bytes))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryMulI64NonNeg(staged_required_dst_values,
                                 sizeof(I64),
                                 &staged_required_dst_bytes))
            return Q8_0_ERR_OVERFLOW;

        if (staged_required_src_blocks > snapshot_src_block_capacity)
            return Q8_0_ERR_BAD_DST_LEN;
        if (staged_required_dst_values > snapshot_dst_q16_capacity)
            return Q8_0_ERR_BAD_DST_LEN;
    }

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArray(
        snapshot_src_blocks,
        snapshot_src_block_capacity,
        snapshot_src_block_stride,
        snapshot_block_count,
        snapshot_dst_q16,
        snapshot_dst_q16_capacity,
        dst_block_stride_values);
    if (status != Q8_0_OK)
        return status;

    if (snapshot_src_blocks != src_blocks ||
        snapshot_dst_q16 != dst_q16 ||
        snapshot_src_block_capacity != src_block_capacity ||
        snapshot_dst_q16_capacity != dst_q16_capacity ||
        snapshot_src_block_stride != src_block_stride ||
        snapshot_block_count != block_count)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    *out_block_count = snapshot_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;

    return Q8_0_OK;
}


// Diagnostics-only no-write companion over commit-only geometry staging.
//
// Computes canonical tuple
//   {block_count, required_src_blocks, required_dst_values,
//    required_src_bytes, required_dst_bytes}
// from immutable caller snapshots and atomically publishes diagnostics while
// never mutating dst_q16.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnly(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_dst_block_stride_values;

    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;
    I64 src_last_offset;
    I64 dst_last_base;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        out_block_count == (I64 *)src_blocks ||
        out_required_src_blocks == (I64 *)src_blocks ||
        out_required_dst_values == (I64 *)src_blocks ||
        out_required_src_bytes == (I64 *)src_blocks ||
        out_required_dst_bytes == (I64 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_dst_block_stride_values = dst_block_stride_values;

    if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_src_block_stride <= 0 || snapshot_dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!snapshot_block_count)
    {
        staged_required_src_blocks = 0;
        staged_required_dst_values = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_src_block_stride,
                                 &src_last_offset))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_dst_block_stride_values,
                                 &dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(src_last_offset, 1, &staged_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryAddI64(dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &staged_required_dst_values))
            return Q8_0_ERR_OVERFLOW;
    }

    if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &staged_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(staged_required_dst_values,
                             sizeof(I64),
                             &staged_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    if (src_blocks != snapshot_src_blocks ||
        dst_q16 != snapshot_dst_q16 ||
        src_block_capacity != snapshot_src_block_capacity ||
        src_block_stride != snapshot_src_block_stride ||
        block_count != snapshot_block_count ||
        dst_q16_capacity != snapshot_dst_q16_capacity ||
        dst_block_stride_values != snapshot_dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    // Atomic publish: commit all diagnostics only after full preflight success.
    *out_block_count = snapshot_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;

    return Q8_0_OK;
}


// Diagnostics-only parity gate over CommitOnlyPreflightOnly.
//
// Stages canonical geometry tuple from CommitOnlyPreflightOnly, recomputes the
// same tuple from immutable caller snapshots, and publishes only when exact
// tuple parity holds. Strict zero-write semantics: dst_q16 is never modified.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParity(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_dst_block_stride_values;

    I64 staged_block_count;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;

    I64 recomputed_required_src_blocks;
    I64 recomputed_required_dst_values;
    I64 recomputed_required_src_bytes;
    I64 recomputed_required_dst_bytes;
    I64 expected_required_src_bytes;
    I64 expected_required_dst_bytes;
    I64 src_last_offset;
    I64 dst_last_base;
    I32 status;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        out_block_count == (I64 *)src_blocks ||
        out_required_src_blocks == (I64 *)src_blocks ||
        out_required_dst_values == (I64 *)src_blocks ||
        out_required_src_bytes == (I64 *)src_blocks ||
        out_required_dst_bytes == (I64 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    // Snapshot immutable geometry/capacity arguments before staging so parity
    // recomputation uses one auditable caller view.
    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_dst_block_stride_values = dst_block_stride_values;

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnly(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        &staged_block_count,
        &staged_required_src_blocks,
        &staged_required_dst_values,
        &staged_required_src_bytes,
        &staged_required_dst_bytes);
    if (status != Q8_0_OK)
        return status;

    // Recompute canonical checked geometry tuple from immutable snapshots.
    if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_src_block_stride <= 0 || snapshot_dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!snapshot_block_count)
    {
        recomputed_required_src_blocks = 0;
        recomputed_required_dst_values = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_src_block_stride,
                                 &src_last_offset))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_dst_block_stride_values,
                                 &dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(src_last_offset, 1, &recomputed_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryAddI64(dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &recomputed_required_dst_values))
            return Q8_0_ERR_OVERFLOW;
    }

    if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &recomputed_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(recomputed_required_dst_values,
                             sizeof(I64),
                             &recomputed_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (src_blocks != snapshot_src_blocks ||
        dst_q16 != snapshot_dst_q16 ||
        src_block_capacity != snapshot_src_block_capacity ||
        src_block_stride != snapshot_src_block_stride ||
        block_count != snapshot_block_count ||
        dst_q16_capacity != snapshot_dst_q16_capacity ||
        dst_block_stride_values != snapshot_dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_block_count < 0 ||
        staged_required_src_blocks < 0 || staged_required_dst_values < 0 ||
        staged_required_src_bytes < 0 || staged_required_dst_bytes < 0)
        return Q8_0_ERR_OVERFLOW;
    if (recomputed_required_src_blocks < 0 || recomputed_required_dst_values < 0 ||
        recomputed_required_src_bytes < 0 || recomputed_required_dst_bytes < 0)
        return Q8_0_ERR_OVERFLOW;

    if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &expected_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(staged_required_dst_values,
                             sizeof(I64),
                             &expected_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_bytes != expected_required_src_bytes ||
        staged_required_dst_bytes != expected_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    if (recomputed_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (recomputed_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_block_count != snapshot_block_count ||
        staged_required_src_blocks != recomputed_required_src_blocks ||
        staged_required_dst_values != recomputed_required_dst_values ||
        staged_required_src_bytes != recomputed_required_src_bytes ||
        staged_required_dst_bytes != recomputed_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    // Atomic publish: write diagnostics only after exact tuple parity succeeds.
    *out_block_count = staged_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;

    return Q8_0_OK;
}


I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_dst_block_stride_values;

    I64 staged_block_count;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;

    I64 expected_required_src_bytes;
    I64 expected_required_dst_bytes;
    I64 recomputed_required_src_blocks;
    I64 recomputed_required_dst_values;
    I64 recomputed_required_src_bytes;
    I64 recomputed_required_dst_bytes;
    I64 src_last_offset;
    I64 dst_last_base;
    I32 status;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        out_block_count == (I64 *)src_blocks ||
        out_required_src_blocks == (I64 *)src_blocks ||
        out_required_dst_values == (I64 *)src_blocks ||
        out_required_src_bytes == (I64 *)src_blocks ||
        out_required_dst_bytes == (I64 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    // Commit-only wrapper contract (IQ-953):
    // 1) stage canonical geometry tuple from parity helper;
    // 2) revalidate immutable caller snapshots and tuple invariants;
    // 3) atomically publish diagnostics only after full success.
    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_dst_block_stride_values = dst_block_stride_values;

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParity(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        &staged_block_count,
        &staged_required_src_blocks,
        &staged_required_dst_values,
        &staged_required_src_bytes,
        &staged_required_dst_bytes);
    if (status != Q8_0_OK)
        return status;

    if (src_blocks != snapshot_src_blocks ||
        dst_q16 != snapshot_dst_q16 ||
        src_block_capacity != snapshot_src_block_capacity ||
        src_block_stride != snapshot_src_block_stride ||
        block_count != snapshot_block_count ||
        dst_q16_capacity != snapshot_dst_q16_capacity ||
        dst_block_stride_values != snapshot_dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    // Revalidate immutable snapshot geometry/capacity contract in commit path.
    if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_src_block_stride <= 0 || snapshot_dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_block_count != snapshot_block_count)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_src_blocks < 0 || staged_required_dst_values < 0 ||
        staged_required_src_bytes < 0 || staged_required_dst_bytes < 0)
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    // Recompute geometry from immutable snapshots and require exact tuple parity
    // against staged parity helper outputs before any publish.
    if (!snapshot_block_count)
    {
        recomputed_required_src_blocks = 0;
        recomputed_required_dst_values = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_src_block_stride,
                                 &src_last_offset))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_dst_block_stride_values,
                                 &dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(src_last_offset, 1, &recomputed_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryAddI64(dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &recomputed_required_dst_values))
            return Q8_0_ERR_OVERFLOW;
    }

    if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &recomputed_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(recomputed_required_dst_values,
                             sizeof(I64),
                             &recomputed_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &expected_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(staged_required_dst_values,
                             sizeof(I64),
                             &expected_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_bytes != expected_required_src_bytes)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_bytes != expected_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_required_src_blocks != recomputed_required_src_blocks ||
        staged_required_dst_values != recomputed_required_dst_values ||
        staged_required_src_bytes != recomputed_required_src_bytes ||
        staged_required_dst_bytes != recomputed_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    *out_block_count = staged_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;
    return Q8_0_OK;
}


// Diagnostics-only no-write companion over commit-only parity wrapper.
//
// Recomputes canonical tuple
//   {block_count, required_src_blocks, required_dst_values,
//    required_src_bytes, required_dst_bytes}
// from immutable caller snapshots and only publishes diagnostics after
// exact tuple parity succeeds.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_dst_block_stride_values;

    I64 staged_block_count;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;

    I64 recomputed_required_src_blocks;
    I64 recomputed_required_dst_values;
    I64 recomputed_required_src_bytes;
    I64 recomputed_required_dst_bytes;
    I64 src_last_offset;
    I64 dst_last_base;
    I32 status;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        out_block_count == (I64 *)src_blocks ||
        out_required_src_blocks == (I64 *)src_blocks ||
        out_required_dst_values == (I64 *)src_blocks ||
        out_required_src_bytes == (I64 *)src_blocks ||
        out_required_dst_bytes == (I64 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_dst_block_stride_values = dst_block_stride_values;

    // IQ-954 diagnostics-only contract:
    // 1) stage tuple from commit-only parity wrapper;
    // 2) recompute canonical tuple from immutable snapshots;
    // 3) require exact parity; publish atomically.
    status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        &staged_block_count,
        &staged_required_src_blocks,
        &staged_required_dst_values,
        &staged_required_src_bytes,
        &staged_required_dst_bytes);
    if (status != Q8_0_OK)
        return status;

    if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_src_block_stride <= 0 || snapshot_dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!snapshot_block_count)
    {
        recomputed_required_src_blocks = 0;
        recomputed_required_dst_values = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_src_block_stride,
                                 &src_last_offset))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_dst_block_stride_values,
                                 &dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(src_last_offset, 1, &recomputed_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryAddI64(dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &recomputed_required_dst_values))
            return Q8_0_ERR_OVERFLOW;
    }

    if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &recomputed_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(recomputed_required_dst_values,
                             sizeof(I64),
                             &recomputed_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (src_blocks != snapshot_src_blocks ||
        dst_q16 != snapshot_dst_q16 ||
        src_block_capacity != snapshot_src_block_capacity ||
        src_block_stride != snapshot_src_block_stride ||
        block_count != snapshot_block_count ||
        dst_q16_capacity != snapshot_dst_q16_capacity ||
        dst_block_stride_values != snapshot_dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    if (recomputed_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (recomputed_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_block_count != snapshot_block_count ||
        staged_required_src_blocks != recomputed_required_src_blocks ||
        staged_required_dst_values != recomputed_required_dst_values ||
        staged_required_src_bytes != recomputed_required_src_bytes ||
        staged_required_dst_bytes != recomputed_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    *out_block_count = staged_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;
    return Q8_0_OK;
}


// Diagnostics-only parity gate over PreflightOnly + ParityCommitOnly.
//
// Stages canonical geometry tuple from
//   ...ParityCommitOnlyPreflightOnly
// and requires exact tuple parity with
//   ...ParityCommitOnly
// under immutable caller snapshots before atomic diagnostics publish.
// Diagnostics-only parity gate over PreflightOnly and ParityCommitOnly.
//
// Stages canonical tuple from
//   Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly
// and requires exact parity with
//   Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly
// before atomic publish. Strict zero-write/no-partial semantics: dst_q16 is
// never modified.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_dst_block_stride_values;

    I64 staged_block_count;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;

    I64 parity_block_count;
    I64 parity_required_src_blocks;
    I64 parity_required_dst_values;
    I64 parity_required_src_bytes;
    I64 parity_required_dst_bytes;
    I32 status;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        out_block_count == (I64 *)src_blocks ||
        out_required_src_blocks == (I64 *)src_blocks ||
        out_required_dst_values == (I64 *)src_blocks ||
        out_required_src_bytes == (I64 *)src_blocks ||
        out_required_dst_bytes == (I64 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_dst_block_stride_values = dst_block_stride_values;

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        &staged_block_count,
        &staged_required_src_blocks,
        &staged_required_dst_values,
        &staged_required_src_bytes,
        &staged_required_dst_bytes);
    if (status != Q8_0_OK)
        return status;

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        &parity_block_count,
        &parity_required_src_blocks,
        &parity_required_dst_values,
        &parity_required_src_bytes,
        &parity_required_dst_bytes);
    if (status != Q8_0_OK)
        return status;

    if (src_blocks != snapshot_src_blocks ||
        dst_q16 != snapshot_dst_q16 ||
        src_block_capacity != snapshot_src_block_capacity ||
        src_block_stride != snapshot_src_block_stride ||
        block_count != snapshot_block_count ||
        dst_q16_capacity != snapshot_dst_q16_capacity ||
        dst_block_stride_values != snapshot_dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_required_src_blocks < 0 || staged_required_dst_values < 0 ||
        staged_required_src_bytes < 0 || staged_required_dst_bytes < 0 ||
        parity_required_src_blocks < 0 || parity_required_dst_values < 0 ||
        parity_required_src_bytes < 0 || parity_required_dst_bytes < 0)
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_block_count != snapshot_block_count ||
        staged_block_count != parity_block_count ||
        staged_required_src_blocks != parity_required_src_blocks ||
        staged_required_dst_values != parity_required_dst_values ||
        staged_required_src_bytes != parity_required_src_bytes ||
        staged_required_dst_bytes != parity_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    // Atomic publish: commit diagnostics only after dual-path parity revalidation.
    *out_block_count = staged_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;
    return Q8_0_OK;
}


// Commit-only diagnostics wrapper over PreflightOnlyParity.
//
// Stages canonical tuple from
//   Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity
// then revalidates immutable snapshot geometry/capacity and exact tuple parity
// before atomic publish. Strict no-partial semantics: dst_q16 is never written.
I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    Q8_0Block *src_blocks,
    I64 src_block_capacity,
    I64 src_block_stride,
    I64 block_count,
    I64 *dst_q16,
    I64 dst_q16_capacity,
    I64 dst_block_stride_values,
    I64 *out_block_count,
    I64 *out_required_src_blocks,
    I64 *out_required_dst_values,
    I64 *out_required_src_bytes,
    I64 *out_required_dst_bytes)
{
    Q8_0Block *snapshot_src_blocks;
    I64 *snapshot_dst_q16;
    I64 snapshot_src_block_capacity;
    I64 snapshot_src_block_stride;
    I64 snapshot_block_count;
    I64 snapshot_dst_q16_capacity;
    I64 snapshot_dst_block_stride_values;

    I64 staged_block_count;
    I64 staged_required_src_blocks;
    I64 staged_required_dst_values;
    I64 staged_required_src_bytes;
    I64 staged_required_dst_bytes;

    I64 expected_required_src_bytes;
    I64 expected_required_dst_bytes;
    I64 recomputed_required_src_blocks;
    I64 recomputed_required_dst_values;
    I64 recomputed_required_src_bytes;
    I64 recomputed_required_dst_bytes;
    I64 src_last_offset;
    I64 dst_last_base;
    I32 status;

    if (!src_blocks || !dst_q16 || !out_block_count ||
        !out_required_src_blocks || !out_required_dst_values ||
        !out_required_src_bytes || !out_required_dst_bytes)
        return Q8_0_ERR_NULL_PTR;

    if (out_block_count == out_required_src_blocks ||
        out_block_count == out_required_dst_values ||
        out_block_count == out_required_src_bytes ||
        out_block_count == out_required_dst_bytes ||
        out_required_src_blocks == out_required_dst_values ||
        out_required_src_blocks == out_required_src_bytes ||
        out_required_src_blocks == out_required_dst_bytes ||
        out_required_dst_values == out_required_src_bytes ||
        out_required_dst_values == out_required_dst_bytes ||
        out_required_src_bytes == out_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if ((I64 *)src_blocks == dst_q16 ||
        out_block_count == (I64 *)src_blocks ||
        out_required_src_blocks == (I64 *)src_blocks ||
        out_required_dst_values == (I64 *)src_blocks ||
        out_required_src_bytes == (I64 *)src_blocks ||
        out_required_dst_bytes == (I64 *)src_blocks ||
        out_block_count == dst_q16 ||
        out_required_src_blocks == dst_q16 ||
        out_required_dst_values == dst_q16 ||
        out_required_src_bytes == dst_q16 ||
        out_required_dst_bytes == dst_q16)
        return Q8_0_ERR_BAD_DST_LEN;

    // Commit-only wrapper contract (IQ-959):
    // 1) stage canonical tuple from PreflightOnlyParity diagnostics helper;
    // 2) revalidate immutable snapshot and tuple invariants;
    // 3) atomically publish only after full parity success.
    snapshot_src_blocks = src_blocks;
    snapshot_dst_q16 = dst_q16;
    snapshot_src_block_capacity = src_block_capacity;
    snapshot_src_block_stride = src_block_stride;
    snapshot_block_count = block_count;
    snapshot_dst_q16_capacity = dst_q16_capacity;
    snapshot_dst_block_stride_values = dst_block_stride_values;

    status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        &staged_block_count,
        &staged_required_src_blocks,
        &staged_required_dst_values,
        &staged_required_src_bytes,
        &staged_required_dst_bytes);
    if (status != Q8_0_OK)
        return status;

    if (src_blocks != snapshot_src_blocks ||
        dst_q16 != snapshot_dst_q16 ||
        src_block_capacity != snapshot_src_block_capacity ||
        src_block_stride != snapshot_src_block_stride ||
        block_count != snapshot_block_count ||
        dst_q16_capacity != snapshot_dst_q16_capacity ||
        dst_block_stride_values != snapshot_dst_block_stride_values)
        return Q8_0_ERR_BAD_DST_LEN;

    if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_block_count < 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_src_block_stride <= 0 || snapshot_dst_block_stride_values <= 0)
        return Q8_0_ERR_BAD_DST_LEN;
    if (snapshot_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_block_count != snapshot_block_count)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_src_blocks < 0 || staged_required_dst_values < 0 ||
        staged_required_src_bytes < 0 || staged_required_dst_bytes < 0)
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_blocks > snapshot_src_block_capacity)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_values > snapshot_dst_q16_capacity)
        return Q8_0_ERR_BAD_DST_LEN;

    if (!snapshot_block_count)
    {
        recomputed_required_src_blocks = 0;
        recomputed_required_dst_values = 0;
    }
    else
    {
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_src_block_stride,
                                 &src_last_offset))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1,
                                 snapshot_dst_block_stride_values,
                                 &dst_last_base))
            return Q8_0_ERR_OVERFLOW;

        if (!Q8_0TryAddI64(src_last_offset, 1, &recomputed_required_src_blocks))
            return Q8_0_ERR_OVERFLOW;
        if (!Q8_0TryAddI64(dst_last_base,
                           Q8_0_VALUES_PER_BLOCK,
                           &recomputed_required_dst_values))
            return Q8_0_ERR_OVERFLOW;
    }

    if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &recomputed_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(recomputed_required_dst_values,
                             sizeof(I64),
                             &recomputed_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks,
                             Q8_0_BLOCK_BYTES,
                             &expected_required_src_bytes))
        return Q8_0_ERR_OVERFLOW;
    if (!Q8_0TryMulI64NonNeg(staged_required_dst_values,
                             sizeof(I64),
                             &expected_required_dst_bytes))
        return Q8_0_ERR_OVERFLOW;

    if (staged_required_src_bytes != expected_required_src_bytes)
        return Q8_0_ERR_BAD_DST_LEN;
    if (staged_required_dst_bytes != expected_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    if (staged_required_src_blocks != recomputed_required_src_blocks ||
        staged_required_dst_values != recomputed_required_dst_values ||
        staged_required_src_bytes != recomputed_required_src_bytes ||
        staged_required_dst_bytes != recomputed_required_dst_bytes)
        return Q8_0_ERR_BAD_DST_LEN;

    *out_block_count = staged_block_count;
    *out_required_src_blocks = staged_required_src_blocks;
    *out_required_dst_values = staged_required_dst_values;
    *out_required_src_bytes = staged_required_src_bytes;
    *out_required_dst_bytes = staged_required_dst_bytes;
    return Q8_0_OK;
}




// Commit-only diagnostics wrapper over PreflightOnlyParity.
//
// Stages canonical tuple from
//   Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity
// then revalidates immutable caller snapshots and recomputed geometry tuple
// before atomic publish. Strict no-partial semantics: diagnostics are written
// only when every parity/invariant check succeeds.
