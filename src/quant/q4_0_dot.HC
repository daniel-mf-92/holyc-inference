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
#define Q4_0_ERR_OVERFLOW     3

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

I32 Q4_0DotRowsQ16MatrixVector(Q4_0Block *matrix_blocks, I64 row_count, I64 row_stride_blocks, Q4_0Block *vec_blocks, I64 vec_block_count, I64 *out_rows_q16)
{
    I64 row_index;
    I64 row_dot_q16;
    Q4_0Block *row_ptr;
    I32 status;

    if (!matrix_blocks || !vec_blocks || !out_rows_q16)
        return Q4_0_ERR_NULL_PTR;
    if (row_count < 0 || row_stride_blocks < 0 || vec_block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    // Each row starts at matrix_blocks + row_index * row_stride_blocks.
    // Only vec_block_count active blocks participate in the row dot.
    // Enforce stride >= active vector span so row slices are valid and explicit.
    if (row_count > 0 && row_stride_blocks < vec_block_count)
        return Q4_0_ERR_BAD_DST_LEN;

    for (row_index = 0; row_index < row_count; ++row_index)
    {
        row_ptr = &matrix_blocks[row_index * row_stride_blocks];
        status = Q4_0DotRowBlocksQ16(row_ptr, vec_blocks, vec_block_count, &row_dot_q16);
        if (status != Q4_0_OK)
            return status;

        out_rows_q16[row_index] = row_dot_q16;
    }

    return Q4_0_OK;
}

Bool Q4_0TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (rhs > 0 && lhs > (0x7FFFFFFFFFFFFFFF - rhs))
        return FALSE;
    if (rhs < 0 && lhs < ((-0x7FFFFFFFFFFFFFFF - 1) - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

// Checked no-partial block dot helper.
// - Accumulates per-block Q32 contributions with explicit signed overflow guard.
// - Produces deterministic Q16 output by rounding exactly once from total Q32.
// Contract: `*out_dot_q32` and `*out_dot_q16` are written only on success.
I32 Q4_0DotBlocksQ32CheckedNoPartial(Q4_0Block *lhs,
                                     Q4_0Block *rhs,
                                     I64 block_count,
                                     I64 *out_dot_q32,
                                     I64 *out_dot_q16)
{
    I64 block_index;
    I64 block_dot_q32;
    I64 staged_total_q32;
    I64 next_total_q32;
    I64 staged_total_q16;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    staged_total_q32 = 0;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        status = Q4_0DotProductBlockQ32(&lhs[block_index],
                                        &rhs[block_index],
                                        &block_dot_q32);
        if (status != Q4_0_OK)
            return status;

        if (!Q4_0TryAddI64(staged_total_q32, block_dot_q32, &next_total_q32))
            return Q4_0_ERR_OVERFLOW;

        staged_total_q32 = next_total_q32;
    }

    // Deterministic hand-off: one rounded Q32->Q16 downshift at the end.
    staged_total_q16 = Q4_0DotQ32ToQ16(staged_total_q32);

    *out_dot_q32 = staged_total_q32;
    *out_dot_q16 = staged_total_q16;
    return Q4_0_OK;
}

// Commit-only diagnostics wrapper for checked no-partial Q4_0 block dots.
//
// Contracts:
// - Delegates math/overflow behavior to `Q4_0DotBlocksQ32CheckedNoPartial`.
// - Stages `{dot_q32,dot_q16}` and publishes atomically only on success.
// - Revalidates immutable call snapshot `{lhs_ptr,rhs_ptr,block_count}` before
//   publishing diagnostics tuple so caller-observable outputs stay deterministic.
I32 Q4_0DotBlocksQ32CheckedNoPartialCommitOnly(Q4_0Block *lhs,
                                               Q4_0Block *rhs,
                                               I64 block_count,
                                               I64 *out_dot_q32,
                                               I64 *out_dot_q16)
{
    Q4_0Block *snapshot_lhs_ptr;
    Q4_0Block *snapshot_rhs_ptr;
    I64 snapshot_block_count;
    I64 staged_dot_q32;
    I64 staged_dot_q16;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    // Atomic tuple publish requires distinct destination slots.
    if (out_dot_q32 == out_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs;
    snapshot_rhs_ptr = rhs;
    snapshot_block_count = block_count;

    status = Q4_0DotBlocksQ32CheckedNoPartial(lhs,
                                              rhs,
                                              block_count,
                                              &staged_dot_q32,
                                              &staged_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (snapshot_lhs_ptr != lhs ||
        snapshot_rhs_ptr != rhs ||
        snapshot_block_count != block_count)
        return Q4_0_ERR_BAD_DST_LEN;

    *out_dot_q32 = staged_dot_q32;
    *out_dot_q16 = staged_dot_q16;
    return Q4_0_OK;
}

// Preflight-only diagnostics companion for commit-only checked Q4_0 block dots.
//
// Contracts:
// - Zero-write path: never touches caller outputs.
// - Recomputes canonical `{dot_q32,dot_q16}` from immutable
//   `{lhs_ptr,rhs_ptr,block_count}` snapshot.
// - Enforces strict tuple parity against staged commit-only outputs.
I32 Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnly(Q4_0Block *lhs,
                                                            Q4_0Block *rhs,
                                                            I64 block_count,
                                                            I64 *out_dot_q32,
                                                            I64 *out_dot_q16)
{
    Q4_0Block *snapshot_lhs_ptr;
    Q4_0Block *snapshot_rhs_ptr;
    I64 snapshot_block_count;
    I64 staged_dot_q32;
    I64 staged_dot_q16;
    I64 canonical_dot_q32;
    I64 canonical_dot_q16;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    if (out_dot_q32 == out_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs;
    snapshot_rhs_ptr = rhs;
    snapshot_block_count = block_count;

    status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnly(lhs,
                                                        rhs,
                                                        block_count,
                                                        &staged_dot_q32,
                                                        &staged_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (snapshot_lhs_ptr != lhs ||
        snapshot_rhs_ptr != rhs ||
        snapshot_block_count != block_count)
        return Q4_0_ERR_BAD_DST_LEN;

    status = Q4_0DotBlocksQ32CheckedNoPartial(lhs,
                                              rhs,
                                              block_count,
                                              &canonical_dot_q32,
                                              &canonical_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (canonical_dot_q32 != staged_dot_q32 ||
        canonical_dot_q16 != staged_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    return Q4_0_OK;
}

// Strict parity gate for preflight-only checked Q4_0 block dots.
//
// Contracts:
// - Zero-write path: never touches caller outputs.
// - Runs the preflight-only parity path and canonical commit-only path.
// - Enforces immutable call snapshot .
// - Requires strict tuple parity  between both paths.
I32 Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnlyParity(Q4_0Block *lhs,
                                                                  Q4_0Block *rhs,
                                                                  I64 block_count,
                                                                  I64 *out_dot_q32,
                                                                  I64 *out_dot_q16)
{
    Q4_0Block *snapshot_lhs_ptr;
    Q4_0Block *snapshot_rhs_ptr;
    I64 snapshot_block_count;
    I64 staged_dot_q32;
    I64 staged_dot_q16;
    I64 canonical_dot_q32;
    I64 canonical_dot_q16;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    if (out_dot_q32 == out_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs;
    snapshot_rhs_ptr = rhs;
    snapshot_block_count = block_count;

    status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnly(lhs,
                                                                     rhs,
                                                                     block_count,
                                                                     &staged_dot_q32,
                                                                     &staged_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (snapshot_lhs_ptr != lhs ||
        snapshot_rhs_ptr != rhs ||
        snapshot_block_count != block_count)
        return Q4_0_ERR_BAD_DST_LEN;

    status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnly(lhs,
                                                        rhs,
                                                        block_count,
                                                        &canonical_dot_q32,
                                                        &canonical_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (canonical_dot_q32 != staged_dot_q32 ||
        canonical_dot_q16 != staged_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    return Q4_0_OK;
}

// Commit-only parity gate for preflight-only parity checked Q4_0 block dots.
//
// Contracts:
// - Delegates tuple verification to
//   .
// - Recomputes canonical commit tuple from immutable snapshot.
// - Atomically publishes  only on strict tuple parity.
I32 Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(Q4_0Block *lhs,
                                                                             Q4_0Block *rhs,
                                                                             I64 block_count,
                                                                             I64 *out_dot_q32,
                                                                             I64 *out_dot_q16)
{
    Q4_0Block *snapshot_lhs_ptr;
    Q4_0Block *snapshot_rhs_ptr;
    I64 snapshot_block_count;
    I64 staged_dot_q32;
    I64 staged_dot_q16;
    I64 canonical_dot_q32;
    I64 canonical_dot_q16;
    I32 status;

    if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)
        return Q4_0_ERR_NULL_PTR;
    if (block_count < 0)
        return Q4_0_ERR_BAD_DST_LEN;

    if (out_dot_q32 == out_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    snapshot_lhs_ptr = lhs;
    snapshot_rhs_ptr = rhs;
    snapshot_block_count = block_count;

    status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnlyParity(lhs,
                                                                            rhs,
                                                                            block_count,
                                                                            &staged_dot_q32,
                                                                            &staged_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (snapshot_lhs_ptr != lhs ||
        snapshot_rhs_ptr != rhs ||
        snapshot_block_count != block_count)
        return Q4_0_ERR_BAD_DST_LEN;

    status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnly(lhs,
                                                        rhs,
                                                        block_count,
                                                        &canonical_dot_q32,
                                                        &canonical_dot_q16);
    if (status != Q4_0_OK)
        return status;

    if (canonical_dot_q32 != staged_dot_q32 ||
        canonical_dot_q16 != staged_dot_q16)
        return Q4_0_ERR_BAD_DST_LEN;

    *out_dot_q32 = canonical_dot_q32;
    *out_dot_q16 = canonical_dot_q16;
    return Q4_0_OK;
}

I64 Q4_0DotQ32ToQ16(I64 dot_q32)
{
    return Q4_0RoundShiftRightSigned(dot_q32, 16);
}
