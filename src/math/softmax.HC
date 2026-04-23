// softmax.HC
// Integer-only Q16.16 softmax for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_OVERFLOW  4

#define SOFTMAX_EXP_Q16_MAX_INPUT  655360
#define SOFTMAX_EXP_Q16_MIN_INPUT -655360

#define I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define I64_MIN_VALUE -0x8000000000000000
#define U64_MAX_VALUE  0xFFFFFFFFFFFFFFFF

I64 FPQ16Exp(I64 x);
I64 FPQ16Div(I64 num, I64 den);
I64 FPQ16FromInt(I64 x);
I64 FPQ16Mul(I64 a, I64 b);
I64 FPQ16Ln(I64 x_q16);
I32 FPQ16MulDivRoundedChecked(I64 a_q16,
                              I64 b_q16,
                              I64 d_q16,
                              I64 *out_q16);
I32 FPQ16ExpArrayFromClampedInputChecked(I64 *input_q16,
                                         I64 *output_q16,
                                         I64 count);
I32 FPQ16ExpArrayCheckedNoAliasChecked(I64 *input_q16,
                                       I64 *output_q16,
                                       I64 count);

// Checked Q16 temperature scaling helper for logits:
//   scaled_logits_q16[i] = round_half_up(logits_q16[i] / temperature_q16)
// which is equivalent to multiplying by reciprocal-temperature in Q16.
// Contracts:
//   - logits_q16 and scaled_logits_q16 are non-null.
//   - lane_count >= 0.
//   - temperature_q16 > 0 (rejects divide-by-zero and non-physical negative T).
//   - logits_q16 and scaled_logits_q16 each address lane_count I64 lanes.
//   - No partial writes: full lane preflight runs before output commit.
// Notes:
//   - Arithmetic routes through FPQ16MulDivRoundedChecked with b=1.0_q16 so
//     scaling uses the same one-rounding checked divider as the rest of Q16
//     softmax math.
I32 FPQ16TemperatureScaleLogitsChecked(I64 *logits_q16,
                                       I64 lane_count,
                                       I64 temperature_q16,
                                       I64 *scaled_logits_q16)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_value;
    I32 status;

    if (!logits_q16 || !scaled_logits_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (temperature_q16 <= 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)scaled_logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight all lanes first so error paths never partially write output.
    for (i = 0; i < lane_count; i++)
    {
        status = FPQ16MulDivRoundedChecked(logits_q16[i],
                                           FP_Q16_ONE,
                                           temperature_q16,
                                           &lane_value);
        if (status)
            return status;
    }

    for (i = 0; i < lane_count; i++)
    {
        status = FPQ16MulDivRoundedChecked(logits_q16[i],
                                           FP_Q16_ONE,
                                           temperature_q16,
                                           &scaled_logits_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Checked softmax split-phase helper for pre-clamped logits.
// Phase performed here:
//   1) exp_lanes_q16[i] = exp(preclamped_logits_q16[i])
//   2) out_exp_sum_q16  = sum_i exp_lanes_q16[i] (checked I64 add)
// Contracts:
//   - All pointers must be non-null
//   - lane_count must be >= 0
//   - preclamped_logits_q16 and exp_lanes_q16 must each address lane_count I64s
//   - Any I64 accumulation overflow returns FP_Q16_ERR_OVERFLOW
// Notes:
//   - Pre-clamped domain validation and lane exp math are delegated to
//     FPQ16ExpArrayFromClampedInputChecked so scalar/vector exp semantics stay
//     centralized and auditable.
I32 FPQ16SoftmaxExpPhaseFromPreclampedChecked(I64 *preclamped_logits_q16,
                                              I64 *exp_lanes_q16,
                                              I64 lane_count,
                                              I64 *out_exp_sum_q16)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 exp_sum_q16;
    I32 status;

    if (!preclamped_logits_q16 || !exp_lanes_q16 || !out_exp_sum_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Empty vectors are valid: no lanes, zero sum.
    if (!lane_count)
    {
        *out_exp_sum_q16 = 0;
        return FP_Q16_OK;
    }

    // Guard pointer-span arithmetic before any indexed access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)preclamped_logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)exp_lanes_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpArrayFromClampedInputChecked(preclamped_logits_q16,
                                                  exp_lanes_q16,
                                                  lane_count);
    if (status)
        return status;

    exp_sum_q16 = 0;
    for (i = 0; i < lane_count; i++)
    {
        if (exp_lanes_q16[i] > (I64_MAX_VALUE - exp_sum_q16))
            return FP_Q16_ERR_OVERFLOW;
        exp_sum_q16 += exp_lanes_q16[i];
    }

    *out_exp_sum_q16 = exp_sum_q16;
    return FP_Q16_OK;
}

// Checked no-alias softmax exp-phase helper for immutable-logit callers.
// Phase performed here:
//   1) exp_lanes_q16[i] = exp(preclamped_logits_q16[i])
//   2) out_exp_sum_q16  = sum_i exp_lanes_q16[i] (checked I64 add)
// Contracts:
//   - All pointers must be non-null
//   - lane_count must be >= 0
//   - preclamped_logits_q16 and exp_lanes_q16 must each address lane_count I64s
//   - preclamped_logits_q16 and exp_lanes_q16 must be disjoint buffers
// Notes:
//   - This reuses shared checked span+sum logic from
//     FPQ16SoftmaxExpPhaseFromPreclampedChecked so no-alias and alias-permitted
//     callers stay bit-identical other than explicit alias rejection.
I32 FPQ16SoftmaxExpPhaseFromPreclampedNoAliasChecked(I64 *preclamped_logits_q16,
                                                     I64 *exp_lanes_q16,
                                                     I64 lane_count,
                                                     I64 *out_exp_sum_q16)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 exp_sum_q16;
    I32 status;

    if (!preclamped_logits_q16 || !exp_lanes_q16 || !out_exp_sum_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (preclamped_logits_q16 == exp_lanes_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Empty vectors are valid: no lanes, zero sum.
    if (!lane_count)
    {
        *out_exp_sum_q16 = 0;
        return FP_Q16_OK;
    }

    // Guard pointer-span arithmetic before any indexed access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)preclamped_logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)exp_lanes_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpArrayCheckedNoAliasChecked(preclamped_logits_q16,
                                                exp_lanes_q16,
                                                lane_count);
    if (status)
        return status;

    exp_sum_q16 = 0;
    for (i = 0; i < lane_count; i++)
    {
        if (exp_lanes_q16[i] > (I64_MAX_VALUE - exp_sum_q16))
            return FP_Q16_ERR_OVERFLOW;
        exp_sum_q16 += exp_lanes_q16[i];
    }

    *out_exp_sum_q16 = exp_sum_q16;
    return FP_Q16_OK;
}

// Checked softmax split-phase normalization helper.
// Phase performed here:
//   1) probs_q16[i] = round_half_up(exp_lanes_q16[i] / exp_sum_q16)
//   2) Deterministic remainder correction to force sum(probs_q16) == 1.0_q16
// Contracts:
//   - All pointers must be non-null
//   - lane_count must be >= 0
//   - exp_sum_q16 must be strictly positive
//   - Input/output buffers must each address lane_count I64s
// Notes:
//   - Uses FPQ16MulDivRoundedChecked with b=1.0_q16 to keep one-rounding,
//     auditable fixed-point divide semantics.
//   - Remainder correction is deterministic: positive remainder goes to the
//     max-exp lane; negative remainder is removed in stable lane order
//     (max lane first, then wrap) without allowing negative probabilities.
I32 FPQ16SoftmaxNormalizePhaseChecked(I64 *exp_lanes_q16,
                                      I64 lane_count,
                                      I64 exp_sum_q16,
                                      I64 *probs_q16)
{
    I64 i;
    I64 pass;
    I64 idx;
    I64 last_index;
    I64 last_byte_offset;
    I64 prob_q16;
    I64 prob_sum_q16;
    I64 diff_q16;
    I64 remaining_q16;
    I64 take_q16;
    I64 max_exp_q16;
    I64 max_exp_index;
    I32 status;

    if (!exp_lanes_q16 || !probs_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (exp_sum_q16 <= 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)exp_lanes_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Normalize lanes and keep running probability sum for remainder fixup.
    max_exp_q16 = exp_lanes_q16[0];
    max_exp_index = 0;
    prob_sum_q16 = 0;

    for (i = 0; i < lane_count; i++)
    {
        status = FPQ16MulDivRoundedChecked(exp_lanes_q16[i],
                                           FP_Q16_ONE,
                                           exp_sum_q16,
                                           &prob_q16);
        if (status)
            return status;

        probs_q16[i] = prob_q16;

        if (prob_q16 > (I64_MAX_VALUE - prob_sum_q16))
            return FP_Q16_ERR_OVERFLOW;
        prob_sum_q16 += prob_q16;

        if (exp_lanes_q16[i] > max_exp_q16)
        {
            max_exp_q16 = exp_lanes_q16[i];
            max_exp_index = i;
        }
    }

    diff_q16 = FP_Q16_ONE - prob_sum_q16;
    if (!diff_q16)
        return FP_Q16_OK;

    // If sum is low, add full residual to argmax lane.
    if (diff_q16 > 0)
    {
        if (probs_q16[max_exp_index] > (I64_MAX_VALUE - diff_q16))
            return FP_Q16_ERR_OVERFLOW;
        probs_q16[max_exp_index] += diff_q16;
        return FP_Q16_OK;
    }

    // If sum is high, subtract excess deterministically without going < 0.
    remaining_q16 = -diff_q16;
    for (pass = 0; pass < lane_count && remaining_q16 > 0; pass++)
    {
        idx = max_exp_index + pass;
        if (idx >= lane_count)
            idx -= lane_count;

        take_q16 = probs_q16[idx];
        if (take_q16 > remaining_q16)
            take_q16 = remaining_q16;

        probs_q16[idx] -= take_q16;
        remaining_q16 -= take_q16;
    }

    if (remaining_q16)
        return FP_Q16_ERR_OVERFLOW;

    return FP_Q16_OK;
}

// Checked no-alias normalization helper for immutable exp-lane callers.
// Behavior:
//   - Rejects exp_lanes_q16 == probs_q16 with BAD_PARAM.
//   - Otherwise delegates to FPQ16SoftmaxNormalizePhaseChecked so rounding,
//     deterministic remainder correction, and overflow contracts remain
//     byte-for-byte identical with the alias-permitted path.
I32 FPQ16SoftmaxNormalizePhaseNoAliasChecked(I64 *exp_lanes_q16,
                                             I64 lane_count,
                                             I64 exp_sum_q16,
                                             I64 *probs_q16)
{
    if (!exp_lanes_q16 || !probs_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (exp_lanes_q16 == probs_q16)
        return FP_Q16_ERR_BAD_PARAM;

    return FPQ16SoftmaxNormalizePhaseChecked(exp_lanes_q16,
                                             lane_count,
                                             exp_sum_q16,
                                             probs_q16);
}

// Internal checked preflight for normalization math.
// Purpose:
//   - Re-run normalize arithmetic checks without writing output lanes.
//   - Lets composed helpers fail before mutating caller probability buffers.
// Contracts mirror FPQ16SoftmaxNormalizePhaseChecked except no output pointer.
I32 FPQ16SoftmaxNormalizePhasePreflightChecked(I64 *exp_lanes_q16,
                                               I64 lane_count,
                                               I64 exp_sum_q16)
{
    I64 i;
    I64 pass;
    I64 idx;
    I64 last_index;
    I64 last_byte_offset;
    I64 prob_q16;
    I64 prob_sum_q16;
    I64 diff_q16;
    I64 remaining_q16;
    I64 take_q16;
    I64 max_exp_q16;
    I64 max_exp_index;
    I32 status;

    if (!exp_lanes_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (exp_sum_q16 <= 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return FP_Q16_OK;

    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)exp_lanes_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    max_exp_q16 = exp_lanes_q16[0];
    max_exp_index = 0;
    prob_sum_q16 = 0;

    for (i = 0; i < lane_count; i++)
    {
        status = FPQ16MulDivRoundedChecked(exp_lanes_q16[i],
                                           FP_Q16_ONE,
                                           exp_sum_q16,
                                           &prob_q16);
        if (status)
            return status;

        if (prob_q16 > (I64_MAX_VALUE - prob_sum_q16))
            return FP_Q16_ERR_OVERFLOW;
        prob_sum_q16 += prob_q16;

        if (exp_lanes_q16[i] > max_exp_q16)
        {
            max_exp_q16 = exp_lanes_q16[i];
            max_exp_index = i;
        }
    }

    diff_q16 = FP_Q16_ONE - prob_sum_q16;
    if (!diff_q16)
        return FP_Q16_OK;

    if (diff_q16 > 0)
    {
        status = FPQ16MulDivRoundedChecked(exp_lanes_q16[max_exp_index],
                                           FP_Q16_ONE,
                                           exp_sum_q16,
                                           &prob_q16);
        if (status)
            return status;
        if (prob_q16 > (I64_MAX_VALUE - diff_q16))
            return FP_Q16_ERR_OVERFLOW;
        return FP_Q16_OK;
    }

    remaining_q16 = -diff_q16;
    for (pass = 0; pass < lane_count && remaining_q16 > 0; pass++)
    {
        idx = max_exp_index + pass;
        if (idx >= lane_count)
            idx -= lane_count;

        status = FPQ16MulDivRoundedChecked(exp_lanes_q16[idx],
                                           FP_Q16_ONE,
                                           exp_sum_q16,
                                           &take_q16);
        if (status)
            return status;

        if (take_q16 > remaining_q16)
            take_q16 = remaining_q16;

        remaining_q16 -= take_q16;
    }

    if (remaining_q16)
        return FP_Q16_ERR_OVERFLOW;

    return FP_Q16_OK;
}

// Checked prefix-sum builder for Q16 probability lanes with deterministic
// exact-one correction.
//
// Purpose:
// - Build cumulative sums for top-p scans from `probs_q16`.
// - Guarantee final cumulative total is exactly 1.0_q16.
// - Preserve no-partial-write semantics on validation/overflow failures.
//
// Correction semantics:
// - diff = 1.0_q16 - sum(probs_q16)
// - diff > 0: add `diff` to the last lane.
// - diff < 0: subtract `-diff` from lanes in reverse order (tail-first),
//             borrowing up to each lane's full value until exhausted.
I32 FPQ16SoftmaxPrefixSumChecked(I64 *probs_q16,
                                 I64 lane_count,
                                 I64 *prefix_sums_q16,
                                 I64 *out_total_q16)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 running_sum_q16;
    I64 corrected_total_q16;
    I64 diff_q16;
    I64 borrow_q16;
    I64 lane_q16;
    I64 take_q16;

    if (!probs_q16 || !prefix_sums_q16 || !out_total_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!lane_count)
    {
        *out_total_q16 = 0;
        return FP_Q16_OK;
    }

    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)prefix_sums_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight source lanes and total before any output write.
    running_sum_q16 = 0;
    for (i = 0; i < lane_count; i++)
    {
        lane_q16 = probs_q16[i];
        if (lane_q16 < 0)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_q16 > (I64_MAX_VALUE - running_sum_q16))
            return FP_Q16_ERR_OVERFLOW;
        running_sum_q16 += lane_q16;
    }

    corrected_total_q16 = running_sum_q16;
    diff_q16 = FP_Q16_ONE - running_sum_q16;

    if (diff_q16 > 0)
    {
        if (corrected_total_q16 > (I64_MAX_VALUE - diff_q16))
            return FP_Q16_ERR_OVERFLOW;
        corrected_total_q16 += diff_q16;
    }
    else if (diff_q16 < 0)
    {
        borrow_q16 = -diff_q16;
        for (i = lane_count - 1; i >= 0 && borrow_q16 > 0; i--)
        {
            lane_q16 = probs_q16[i];
            if (lane_q16 >= borrow_q16)
                borrow_q16 = 0;
            else
                borrow_q16 -= lane_q16;
        }
        if (borrow_q16)
            return FP_Q16_ERR_BAD_PARAM;

        corrected_total_q16 += diff_q16;
    }

    if (corrected_total_q16 != FP_Q16_ONE)
        return FP_Q16_ERR_OVERFLOW;

    // Stage corrected lanes into output buffer first.
    if (diff_q16 >= 0)
    {
        for (i = 0; i < lane_count; i++)
            prefix_sums_q16[i] = probs_q16[i];

        prefix_sums_q16[last_index] += diff_q16;
    }
    else
    {
        borrow_q16 = -diff_q16;
        for (i = lane_count - 1; i >= 0; i--)
        {
            lane_q16 = probs_q16[i];
            if (borrow_q16 > 0)
            {
                take_q16 = borrow_q16;
                if (take_q16 > lane_q16)
                    take_q16 = lane_q16;
                prefix_sums_q16[i] = lane_q16 - take_q16;
                borrow_q16 -= take_q16;
            }
            else
            {
                prefix_sums_q16[i] = lane_q16;
            }
        }

        if (borrow_q16)
            return FP_Q16_ERR_BAD_PARAM;
    }

    // Convert corrected lanes into cumulative sums.
    running_sum_q16 = 0;
    for (i = 0; i < lane_count; i++)
    {
        if (prefix_sums_q16[i] < 0)
            return FP_Q16_ERR_BAD_PARAM;
        if (prefix_sums_q16[i] > (I64_MAX_VALUE - running_sum_q16))
            return FP_Q16_ERR_OVERFLOW;

        running_sum_q16 += prefix_sums_q16[i];
        prefix_sums_q16[i] = running_sum_q16;
    }

    if (running_sum_q16 != FP_Q16_ONE)
        return FP_Q16_ERR_OVERFLOW;

    *out_total_q16 = running_sum_q16;
    return FP_Q16_OK;
}

// Checked no-partial wrapper over FPQ16SoftmaxPrefixSumChecked.
//
// Purpose:
// - Build exact-one cumulative distribution for top-p scans while guaranteeing
//   caller-visible outputs are all-or-nothing.
//
// Contracts:
// - probs_q16, prefix_sums_q16, out_total_q16 are non-null.
// - lane_count, probs_capacity, prefix_capacity are >= 0.
// - Immutable snapshot `{lane_count,probs_capacity,prefix_capacity}` must hold
//   from preflight through commit.
// - count/capacity checks are in I64 lanes (not bytes).
//
// No-partial guarantee:
// - Uses caller-provided `prefix_sums_q16` as staging for canonical prefix sums,
//   then commits to `out_total_q16` only after successful revalidation.
// - On any failure path, `out_total_q16` is not modified.
I32 FPQ16SoftmaxPrefixSumCheckedNoPartial(I64 *probs_q16,
                                          I64 lane_count,
                                          I64 probs_capacity,
                                          I64 *prefix_sums_q16,
                                          I64 prefix_capacity,
                                          I64 *out_total_q16)
{
    I64 snapshot_lane_count;
    I64 snapshot_probs_capacity;
    I64 snapshot_prefix_capacity;
    I64 required_probs;
    I64 required_prefix;
    I64 staged_total_q16;
    I32 status;

    if (!probs_q16 || !prefix_sums_q16 || !out_total_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (probs_capacity < 0 || prefix_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_lane_count = lane_count;
    snapshot_probs_capacity = probs_capacity;
    snapshot_prefix_capacity = prefix_capacity;

    required_probs = snapshot_lane_count;
    required_prefix = snapshot_lane_count;

    if (required_probs > snapshot_probs_capacity)
        return FP_Q16_ERR_BAD_PARAM;
    if (required_prefix > snapshot_prefix_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    // Build canonical prefix sums into staged output lanes.
    status = FPQ16SoftmaxPrefixSumChecked(probs_q16,
                                          snapshot_lane_count,
                                          prefix_sums_q16,
                                          &staged_total_q16);
    if (status)
        return status;

    // Revalidate immutable geometry snapshot before publishing total.
    if (snapshot_lane_count != lane_count ||
        snapshot_probs_capacity != probs_capacity ||
        snapshot_prefix_capacity != prefix_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    // Commit: publish staged exact-one total only after full success.
    *out_total_q16 = staged_total_q16;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over FPQ16SoftmaxPrefixSumCheckedNoPartial.
//
// Purpose:
// - Reuse no-partial prefix construction for the committed caller prefix lanes.
// - Recompute canonical prefix totals independently and require strict parity
//   before publishing summary tuple outputs.
// - Publish `{total_q16,last_prefix_q16}` atomically only on full success.
//
// Contracts:
// - Enforces immutable snapshot `{lane_count,probs_capacity,prefix_capacity}`.
// - Enforces pointer snapshot stability for all input/output pointers.
I32 FPQ16SoftmaxPrefixSumCheckedNoPartialCommitOnly(I64 *probs_q16,
                                                    I64 lane_count,
                                                    I64 probs_capacity,
                                                    I64 *prefix_sums_q16,
                                                    I64 prefix_capacity,
                                                    I64 *out_total_q16,
                                                    I64 *out_last_prefix_q16)
{
    I64 *snapshot_probs_q16;
    I64 snapshot_lane_count;
    I64 snapshot_probs_capacity;
    I64 *snapshot_prefix_sums_q16;
    I64 snapshot_prefix_capacity;
    I64 *snapshot_out_total_q16;
    I64 *snapshot_out_last_prefix_q16;
    I64 required_prefix_bytes;
    I64 *recomputed_prefix_q16;
    I64 staged_total_q16;
    I64 staged_last_prefix_q16;
    I64 recomputed_total_q16;
    I64 recomputed_last_prefix_q16;
    I32 status;

    if (!probs_q16 || !prefix_sums_q16 || !out_total_q16 || !out_last_prefix_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (probs_capacity < 0 || prefix_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_probs_q16 = probs_q16;
    snapshot_lane_count = lane_count;
    snapshot_probs_capacity = probs_capacity;
    snapshot_prefix_sums_q16 = prefix_sums_q16;
    snapshot_prefix_capacity = prefix_capacity;
    snapshot_out_total_q16 = out_total_q16;
    snapshot_out_last_prefix_q16 = out_last_prefix_q16;

    status = FPQ16SoftmaxPrefixSumCheckedNoPartial(snapshot_probs_q16,
                                                   snapshot_lane_count,
                                                   snapshot_probs_capacity,
                                                   snapshot_prefix_sums_q16,
                                                   snapshot_prefix_capacity,
                                                   &staged_total_q16);
    if (status)
        return status;

    if (snapshot_lane_count)
        staged_last_prefix_q16 = snapshot_prefix_sums_q16[snapshot_lane_count - 1];
    else
        staged_last_prefix_q16 = 0;

    recomputed_total_q16 = 0;
    recomputed_last_prefix_q16 = 0;

    if (snapshot_lane_count)
    {
        if (snapshot_lane_count > (I64_MAX_VALUE >> 3))
            return FP_Q16_ERR_OVERFLOW;

        required_prefix_bytes = snapshot_lane_count << 3;
        recomputed_prefix_q16 = MAlloc(required_prefix_bytes);
        if (!recomputed_prefix_q16)
            return FP_Q16_ERR_OVERFLOW;

        status = FPQ16SoftmaxPrefixSumChecked(snapshot_probs_q16,
                                              snapshot_lane_count,
                                              recomputed_prefix_q16,
                                              &recomputed_total_q16);
        if (status)
        {
            Free(recomputed_prefix_q16);
            return status;
        }

        recomputed_last_prefix_q16 = recomputed_prefix_q16[snapshot_lane_count - 1];
        Free(recomputed_prefix_q16);
    }

    if (snapshot_probs_q16 != probs_q16 ||
        snapshot_lane_count != lane_count ||
        snapshot_probs_capacity != probs_capacity ||
        snapshot_prefix_sums_q16 != prefix_sums_q16 ||
        snapshot_prefix_capacity != prefix_capacity ||
        snapshot_out_total_q16 != out_total_q16 ||
        snapshot_out_last_prefix_q16 != out_last_prefix_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_total_q16 != recomputed_total_q16 ||
        staged_last_prefix_q16 != recomputed_last_prefix_q16)
        return FP_Q16_ERR_OVERFLOW;

    *out_total_q16 = staged_total_q16;
    *out_last_prefix_q16 = staged_last_prefix_q16;
    return FP_Q16_OK;
}

// Commit-only wrapper over FPQ16SoftmaxPrefixSumChecked.
//
// Purpose:
// - Build canonical corrected prefix sums on staged scratch.
// - Publish caller prefix lanes and summary tuple atomically only after
//   canonical kernel success and immutable snapshot revalidation.
//
// Summary tuple:
// - total_q16      := final corrected cumulative total.
// - last_prefix_q16:= final lane in cumulative prefix output.
I32 FPQ16SoftmaxPrefixSumCheckedCommitOnly(I64 *probs_q16,
                                           I64 lane_count,
                                           I64 *prefix_sums_q16,
                                           I64 *out_total_q16,
                                           I64 *out_last_prefix_q16)
{
    I64 *snapshot_probs_q16;
    I64 snapshot_lane_count;
    I64 *snapshot_prefix_sums_q16;
    I64 *snapshot_out_total_q16;
    I64 *snapshot_out_last_prefix_q16;
    I64 required_prefix_bytes;
    I64 *staged_prefix_q16;
    I64 staged_total_q16;
    I64 staged_last_prefix_q16;
    I64 i;
    I32 status;

    if (!probs_q16 || !prefix_sums_q16 || !out_total_q16 || !out_last_prefix_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_probs_q16 = probs_q16;
    snapshot_lane_count = lane_count;
    snapshot_prefix_sums_q16 = prefix_sums_q16;
    snapshot_out_total_q16 = out_total_q16;
    snapshot_out_last_prefix_q16 = out_last_prefix_q16;

    if (!snapshot_lane_count)
    {
        *out_total_q16 = 0;
        *out_last_prefix_q16 = 0;
        return FP_Q16_OK;
    }

    if (snapshot_lane_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    required_prefix_bytes = snapshot_lane_count << 3;
    staged_prefix_q16 = MAlloc(required_prefix_bytes);
    if (!staged_prefix_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16SoftmaxPrefixSumChecked(snapshot_probs_q16,
                                          snapshot_lane_count,
                                          staged_prefix_q16,
                                          &staged_total_q16);
    if (status)
    {
        Free(staged_prefix_q16);
        return status;
    }

    staged_last_prefix_q16 = staged_prefix_q16[snapshot_lane_count - 1];

    if (snapshot_probs_q16 != probs_q16 ||
        snapshot_lane_count != lane_count ||
        snapshot_prefix_sums_q16 != prefix_sums_q16 ||
        snapshot_out_total_q16 != out_total_q16 ||
        snapshot_out_last_prefix_q16 != out_last_prefix_q16)
    {
        Free(staged_prefix_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    for (i = 0; i < snapshot_lane_count; i++)
        prefix_sums_q16[i] = staged_prefix_q16[i];

    *out_total_q16 = staged_total_q16;
    *out_last_prefix_q16 = staged_last_prefix_q16;

    Free(staged_prefix_q16);
    return FP_Q16_OK;
}

// Diagnostics zero-write companion over commit-only and canonical checked
// prefix-sum builders.
//
// Purpose:
// - Leave caller `prefix_sums_q16` untouched on every path.
// - Recompute summary tuple through two independent paths:
//     1) FPQ16SoftmaxPrefixSumCheckedCommitOnly
//     2) FPQ16SoftmaxPrefixSumChecked
// - Publish `{total_q16,last_prefix_q16}` only when both paths agree exactly.
//
// Contracts:
// - Enforces immutable snapshot `{lane_count}`.
// - Enforces output-pointer snapshot stability.
I32 FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnly(I64 *probs_q16,
                                                        I64 lane_count,
                                                        I64 *prefix_sums_q16,
                                                        I64 *out_total_q16,
                                                        I64 *out_last_prefix_q16)
{
    I64 snapshot_lane_count;
    I64 *snapshot_out_total_q16;
    I64 *snapshot_out_last_prefix_q16;
    I64 required_prefix_bytes;
    I64 dummy_prefix_q16;
    I64 *commit_prefix_q16;
    I64 *checked_prefix_q16;
    I64 staged_total_q16;
    I64 staged_last_prefix_q16;
    I64 canonical_total_q16;
    I64 canonical_last_prefix_q16;
    I32 status;

    if (!probs_q16 || !prefix_sums_q16 || !out_total_q16 || !out_last_prefix_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    *out_total_q16 = 0;
    *out_last_prefix_q16 = 0;

    snapshot_lane_count = lane_count;
    snapshot_out_total_q16 = out_total_q16;
    snapshot_out_last_prefix_q16 = out_last_prefix_q16;

    if (!snapshot_lane_count)
    {
        *out_total_q16 = 0;
        *out_last_prefix_q16 = 0;
        return FP_Q16_OK;
    }

    if (snapshot_lane_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    required_prefix_bytes = snapshot_lane_count << 3;
    commit_prefix_q16 = MAlloc(required_prefix_bytes);
    if (!commit_prefix_q16)
        return FP_Q16_ERR_OVERFLOW;

    checked_prefix_q16 = MAlloc(required_prefix_bytes);
    if (!checked_prefix_q16)
    {
        Free(commit_prefix_q16);
        return FP_Q16_ERR_OVERFLOW;
    }

    status = FPQ16SoftmaxPrefixSumCheckedCommitOnly(probs_q16,
                                                    snapshot_lane_count,
                                                    commit_prefix_q16,
                                                    &staged_total_q16,
                                                    &staged_last_prefix_q16);
    if (status)
    {
        Free(checked_prefix_q16);
        Free(commit_prefix_q16);
        return status;
    }

    status = FPQ16SoftmaxPrefixSumChecked(probs_q16,
                                          snapshot_lane_count,
                                          checked_prefix_q16,
                                          &canonical_total_q16);
    if (status)
    {
        Free(checked_prefix_q16);
        Free(commit_prefix_q16);
        return status;
    }

    canonical_last_prefix_q16 = checked_prefix_q16[snapshot_lane_count - 1];

    if (snapshot_lane_count != lane_count ||
        snapshot_out_total_q16 != out_total_q16 ||
        snapshot_out_last_prefix_q16 != out_last_prefix_q16)
    {
        Free(checked_prefix_q16);
        Free(commit_prefix_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    if (staged_total_q16 != canonical_total_q16 ||
        staged_last_prefix_q16 != canonical_last_prefix_q16)
    {
        Free(checked_prefix_q16);
        Free(commit_prefix_q16);
        return FP_Q16_ERR_OVERFLOW;
    }

    *out_total_q16 = staged_total_q16;
    *out_last_prefix_q16 = staged_last_prefix_q16;

    Free(checked_prefix_q16);
    Free(commit_prefix_q16);
    return FP_Q16_OK;
}

// Checked composed helper for alias-permitted split-phase softmax.
// Pipeline:
//   1) exp_lanes_q16 <- exp(preclamped_logits_q16)
//   2) exp_sum_q16   <- checked sum(exp_lanes_q16)
//   3) probs_q16     <- normalize(exp_lanes_q16 / exp_sum_q16)
// Contracts:
//   - All pointers non-null
//   - lane_count >= 0
//   - Alias is allowed across {preclamped_logits_q16, exp_lanes_q16, probs_q16}
//   - No partial writes to probs_q16 on normalization failure paths
// Notes:
//   - Composes alias-permitted split-phase helpers so checked arithmetic,
//     rounding, and overflow behavior stay centralized and auditable.
I32 FPQ16SoftmaxFromPreclampedChecked(I64 *preclamped_logits_q16,
                                      I64 *exp_lanes_q16,
                                      I64 lane_count,
                                      I64 *probs_q16)
{
    I64 exp_sum_q16;
    I32 status;

    if (!preclamped_logits_q16 || !exp_lanes_q16 || !probs_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SoftmaxExpPhaseFromPreclampedChecked(preclamped_logits_q16,
                                                        exp_lanes_q16,
                                                        lane_count,
                                                        &exp_sum_q16);
    if (status)
        return status;

    if (!lane_count)
        return FP_Q16_OK;

    status = FPQ16SoftmaxNormalizePhasePreflightChecked(exp_lanes_q16,
                                                        lane_count,
                                                        exp_sum_q16);
    if (status)
        return status;

    return FPQ16SoftmaxNormalizePhaseChecked(exp_lanes_q16,
                                             lane_count,
                                             exp_sum_q16,
                                             probs_q16);
}

// Checked composed helper for immutable-logit, no-alias split-phase softmax.
// Pipeline:
//   1) exp_lanes_q16 <- exp(preclamped_logits_q16)
//   2) exp_sum_q16   <- checked sum(exp_lanes_q16)
//   3) probs_q16     <- normalize(exp_lanes_q16 / exp_sum_q16)
// Contracts:
//   - All pointers non-null
//   - lane_count >= 0
//   - preclamped_logits_q16, exp_lanes_q16, probs_q16 are pairwise disjoint
//   - No partial writes to probs_q16 on any failure path
// Notes:
//   - Uses no-alias split-phase helpers so arithmetic stays centralized.
//   - Runs a normalization preflight before writing probs_q16.
I32 FPQ16SoftmaxFromPreclampedNoAliasChecked(I64 *preclamped_logits_q16,
                                             I64 *exp_lanes_q16,
                                             I64 lane_count,
                                             I64 *probs_q16)
{
    I64 exp_sum_q16;
    I32 status;

    if (!preclamped_logits_q16 || !exp_lanes_q16 || !probs_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Immutable-buffer contract: reject all pairwise alias combinations.
    if (preclamped_logits_q16 == exp_lanes_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (preclamped_logits_q16 == probs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (exp_lanes_q16 == probs_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Split phase 1: exp lanes + checked exp-sum accumulation.
    status = FPQ16SoftmaxExpPhaseFromPreclampedNoAliasChecked(preclamped_logits_q16,
                                                               exp_lanes_q16,
                                                               lane_count,
                                                               &exp_sum_q16);
    if (status)
        return status;

    if (!lane_count)
        return FP_Q16_OK;

    // Preflight phase 2 math before writing output probabilities.
    status = FPQ16SoftmaxNormalizePhasePreflightChecked(exp_lanes_q16,
                                                        lane_count,
                                                        exp_sum_q16);
    if (status)
        return status;

    // Split phase 2: no-alias normalization write.
    return FPQ16SoftmaxNormalizePhaseNoAliasChecked(exp_lanes_q16,
                                                    lane_count,
                                                    exp_sum_q16,
                                                    probs_q16);
}

// Checked composed helper for immutable-logit, no-alias split-phase softmax
// with explicit exp-sum output for diagnostics/logging.
// Pipeline:
//   1) exp_lanes_q16 <- exp(preclamped_logits_q16)
//   2) exp_sum_q16   <- checked sum(exp_lanes_q16)
//   3) probs_q16     <- normalize(exp_lanes_q16 / exp_sum_q16)
//   4) out_exp_sum_q16 <- exp_sum_q16
// Contracts:
//   - All pointers non-null
//   - lane_count >= 0
//   - preclamped_logits_q16, exp_lanes_q16, probs_q16 are pairwise disjoint
//   - No partial writes to probs_q16 on any failure path
// Notes:
//   - out_exp_sum_q16 is written only after exp-phase success.
//   - For lane_count == 0, writes out_exp_sum_q16 = 0 and returns OK.
I32 FPQ16SoftmaxFromPreclampedNoAliasCheckedWithSumOut(I64 *preclamped_logits_q16,
                                                       I64 *exp_lanes_q16,
                                                       I64 lane_count,
                                                       I64 *probs_q16,
                                                       I64 *out_exp_sum_q16)
{
    I64 exp_sum_q16;
    I32 status;

    if (!preclamped_logits_q16 || !exp_lanes_q16 || !probs_q16 || !out_exp_sum_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (preclamped_logits_q16 == exp_lanes_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (preclamped_logits_q16 == probs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (exp_lanes_q16 == probs_q16)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SoftmaxExpPhaseFromPreclampedNoAliasChecked(preclamped_logits_q16,
                                                               exp_lanes_q16,
                                                               lane_count,
                                                               &exp_sum_q16);
    if (status)
        return status;

    *out_exp_sum_q16 = exp_sum_q16;

    if (!lane_count)
        return FP_Q16_OK;

    status = FPQ16SoftmaxNormalizePhasePreflightChecked(exp_lanes_q16,
                                                        lane_count,
                                                        exp_sum_q16);
    if (status)
        return status;

    return FPQ16SoftmaxNormalizePhaseNoAliasChecked(exp_lanes_q16,
                                                    lane_count,
                                                    exp_sum_q16,
                                                    probs_q16);
}

// Checked composed helper for alias-permitted split-phase softmax
// with explicit exp-sum output for diagnostics/logging.
// Pipeline:
//   1) exp_lanes_q16 <- exp(preclamped_logits_q16)
//   2) exp_sum_q16   <- checked sum(exp_lanes_q16)
//   3) probs_q16     <- normalize(exp_lanes_q16 / exp_sum_q16)
//   4) out_exp_sum_q16 <- exp_sum_q16
// Contracts:
//   - All pointers non-null
//   - lane_count >= 0
//   - Alias is allowed across {preclamped_logits_q16, exp_lanes_q16, probs_q16}
//   - No partial writes to probs_q16 on normalization failure paths
// Notes:
//   - out_exp_sum_q16 is written only after exp-phase success.
//   - For lane_count == 0, writes out_exp_sum_q16 = 0 and returns OK.
I32 FPQ16SoftmaxFromPreclampedCheckedWithSumOut(I64 *preclamped_logits_q16,
                                                I64 *exp_lanes_q16,
                                                I64 lane_count,
                                                I64 *probs_q16,
                                                I64 *out_exp_sum_q16)
{
    I64 exp_sum_q16;
    I32 status;

    if (!preclamped_logits_q16 || !exp_lanes_q16 || !probs_q16 || !out_exp_sum_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16SoftmaxExpPhaseFromPreclampedChecked(preclamped_logits_q16,
                                                        exp_lanes_q16,
                                                        lane_count,
                                                        &exp_sum_q16);
    if (status)
        return status;

    *out_exp_sum_q16 = exp_sum_q16;

    if (!lane_count)
        return FP_Q16_OK;

    status = FPQ16SoftmaxNormalizePhasePreflightChecked(exp_lanes_q16,
                                                        lane_count,
                                                        exp_sum_q16);
    if (status)
        return status;

    return FPQ16SoftmaxNormalizePhaseChecked(exp_lanes_q16,
                                             lane_count,
                                             exp_sum_q16,
                                             probs_q16);
}

// Checked deterministic top-k selector over Q16 logits.
// Output contract:
//   - Writes exactly k indices to out_indices[0..k-1].
//   - Ordering is descending logit value with deterministic tie-break:
//     lower token index wins when logits are equal.
//   - No partial writes on parameter/preflight failures.
// Checked guards:
//   - logits_q16 and out_indices are non-null.
//   - lane_count >= 0, k >= 0, out_index_capacity >= 0.
//   - k <= lane_count and k <= out_index_capacity.
//   - Pointer-span arithmetic for indexed accesses cannot overflow U64.
// Notes:
//   - Implementation is intentionally direct O(k*lane_count*k) scanning.
//     TempleOS purity goal favors transparent, auditable math/selection flow.
I32 FPQ16TopKSelectIndicesChecked(I64 *logits_q16,
                                  I64 lane_count,
                                  I64 k,
                                  I64 *out_indices,
                                  I64 out_index_capacity)
{
    I64 i;
    I64 rank;
    I64 prior;
    I64 last_index;
    I64 last_byte_offset;
    I64 best_index;
    I64 best_logit_q16;
    I64 candidate_logit_q16;
    Bool already_selected;

    if (!logits_q16 || !out_indices)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0 || k < 0 || out_index_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (k > lane_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (k > out_index_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (!k)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed logits/out writes.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    last_index = k - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)out_indices > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Deterministic repeated max-select with stable lower-index tie-break.
    // All error paths are preflighted above; selection/write loop is total.
    for (rank = 0; rank < k; rank++)
    {
        best_index = -1;
        best_logit_q16 = 0;

        for (i = 0; i < lane_count; i++)
        {
            already_selected = FALSE;
            for (prior = 0; prior < rank; prior++)
            {
                if (out_indices[prior] == i)
                {
                    already_selected = TRUE;
                    break;
                }
            }

            if (already_selected)
                continue;

            candidate_logit_q16 = logits_q16[i];
            if (best_index < 0)
            {
                best_index = i;
                best_logit_q16 = candidate_logit_q16;
            }
            else if (candidate_logit_q16 > best_logit_q16)
            {
                best_index = i;
                best_logit_q16 = candidate_logit_q16;
            }
            else if (candidate_logit_q16 == best_logit_q16 && i < best_index)
            {
                best_index = i;
            }
        }

        if (best_index < 0)
            return FP_Q16_ERR_OVERFLOW;

        out_indices[rank] = best_index;
    }

    return FP_Q16_OK;
}

// IQ-1083 diagnostics-only stable softmax/top-k preflight helper.
//
// This helper computes the canonical diagnostics tuple used by higher-level
// sampling paths before any probability or token writes occur:
//   - required_workspace_cells: scratch cells needed for max-subtracted
//     softmax staging plus top-k staging.
//   - selected_count: effective top-k lane count to process.
//   - max_logit_q16: maximum lane used as the stable subtraction anchor.
//
// Contracts:
//   - logits_q16 and all outputs are non-null.
//   - logits_count, top_k, workspace_capacity are all >= 0.
//   - top_k <= logits_count.
//   - workspace_capacity must cover required_workspace_cells.
//   - No partial writes: all outputs are staged then atomically committed.
I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly(
    I64 *logits_q16,
    I64 logits_count,
    I64 top_k,
    I64 workspace_capacity,
    I64 *out_required_workspace_cells,
    I64 *out_selected_count,
    I64 *out_max_logit_q16)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staged_required_workspace_cells;
    I64 staged_selected_count;
    I64 staged_max_logit_q16;
    I64 window_delta_q16;

    if (!logits_q16 || !out_required_workspace_cells ||
        !out_selected_count || !out_max_logit_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (out_required_workspace_cells == out_selected_count ||
        out_required_workspace_cells == out_max_logit_q16 ||
        out_selected_count == out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (logits_count < 0 || top_k < 0 || workspace_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_k > logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    staged_required_workspace_cells = 0;
    staged_selected_count = 0;
    staged_max_logit_q16 = 0;

    if (!logits_count)
    {
        *out_required_workspace_cells = 0;
        *out_selected_count = 0;
        *out_max_logit_q16 = 0;
        return FP_Q16_OK;
    }

    // Guard pointer-span arithmetic before indexed reads.
    last_index = logits_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Scratch model: one lane for stable max-subtracted logits and one lane
    // for top-k staged logits/indices diagnostics.
    if (logits_count > (I64_MAX_VALUE - top_k))
        return FP_Q16_ERR_OVERFLOW;
    staged_required_workspace_cells = logits_count + top_k;

    if (staged_required_workspace_cells > workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    staged_selected_count = top_k;
    staged_max_logit_q16 = logits_q16[0];

    for (i = 1; i < logits_count; i++)
    {
        if (logits_q16[i] > staged_max_logit_q16)
            staged_max_logit_q16 = logits_q16[i];
    }

    // Validate max-subtracted softmax window arithmetic in pure integer domain:
    // delta = max_logit_q16 - lane_logit_q16 must be non-negative and
    // representable in signed I64 for every lane.
    for (i = 0; i < logits_count; i++)
    {
        if (logits_q16[i] < 0 &&
            staged_max_logit_q16 > (I64_MAX_VALUE + logits_q16[i]))
            return FP_Q16_ERR_OVERFLOW;

        window_delta_q16 = staged_max_logit_q16 - logits_q16[i];
        if (window_delta_q16 < 0)
            return FP_Q16_ERR_BAD_PARAM;
    }

    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_selected_count = staged_selected_count;
    *out_max_logit_q16 = staged_max_logit_q16;
    return FP_Q16_OK;
}

// IQ-1084 diagnostics-only parity gate for top-k softmax preflight geometry.
//
// Purpose:
//   - Re-run canonical top-k preflight geometry in-line and require exact tuple
//     parity with SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly.
//   - Detect stale or diverged staging logic before any caller-visible publish.
//
// Canonical tuple:
//   {required_workspace_cells, selected_count, max_logit_q16}
//
// Contracts:
//   - No partial writes: outputs are committed only after both staged and
//     canonical paths succeed and parity is exact.
//   - Immutable snapshots for pointers/geometry are revalidated before commit.
I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParity(
    I64 *logits_q16,
    I64 logits_count,
    I64 top_k,
    I64 workspace_capacity,
    I64 *out_required_workspace_cells,
    I64 *out_selected_count,
    I64 *out_max_logit_q16)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 window_delta_q16;

    I64 staged_required_workspace_cells;
    I64 staged_selected_count;
    I64 staged_max_logit_q16;

    I64 canonical_required_workspace_cells;
    I64 canonical_selected_count;
    I64 canonical_max_logit_q16;

    I64 snapshot_logits_count;
    I64 snapshot_top_k;
    I64 snapshot_workspace_capacity;
    I64 *snapshot_logits_q16;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_selected_count;
    I64 *snapshot_out_max_logit_q16;

    I32 status;

    if (!logits_q16 || !out_required_workspace_cells ||
        !out_selected_count || !out_max_logit_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (out_required_workspace_cells == logits_q16 ||
        out_selected_count == logits_q16 ||
        out_max_logit_q16 == logits_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_workspace_cells == out_selected_count ||
        out_required_workspace_cells == out_max_logit_q16 ||
        out_selected_count == out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (logits_count < 0 || top_k < 0 || workspace_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_k > logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_logits_q16 = logits_q16;
    snapshot_logits_count = logits_count;
    snapshot_top_k = top_k;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_selected_count = out_selected_count;
    snapshot_out_max_logit_q16 = out_max_logit_q16;

    status = SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &staged_required_workspace_cells,
        &staged_selected_count,
        &staged_max_logit_q16);
    if (status)
        return status;

    // Harden staged diagnostics before canonical recompute: these fields are
    // geometry counters and must never become negative or internally inverted.
    if (staged_required_workspace_cells < 0 || staged_selected_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (staged_selected_count > staged_required_workspace_cells)
        return FP_Q16_ERR_BAD_PARAM;

    canonical_required_workspace_cells = 0;
    canonical_selected_count = 0;
    canonical_max_logit_q16 = 0;

    if (!logits_count)
    {
        canonical_required_workspace_cells = 0;
        canonical_selected_count = 0;
        canonical_max_logit_q16 = 0;
    }
    else
    {
        last_index = logits_count - 1;
        if (last_index > (I64_MAX_VALUE >> 3))
            return FP_Q16_ERR_OVERFLOW;
        last_byte_offset = last_index << 3;
        if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
            return FP_Q16_ERR_OVERFLOW;

        if (logits_count > (I64_MAX_VALUE - top_k))
            return FP_Q16_ERR_OVERFLOW;
        canonical_required_workspace_cells = logits_count + top_k;
        if (canonical_required_workspace_cells > workspace_capacity)
            return FP_Q16_ERR_BAD_PARAM;

        canonical_selected_count = top_k;
        canonical_max_logit_q16 = logits_q16[0];
        for (i = 1; i < logits_count; i++)
        {
            if (logits_q16[i] > canonical_max_logit_q16)
                canonical_max_logit_q16 = logits_q16[i];
        }

        for (i = 0; i < logits_count; i++)
        {
            if (logits_q16[i] < 0 &&
                canonical_max_logit_q16 > (I64_MAX_VALUE + logits_q16[i]))
                return FP_Q16_ERR_OVERFLOW;

            window_delta_q16 = canonical_max_logit_q16 - logits_q16[i];
            if (window_delta_q16 < 0)
                return FP_Q16_ERR_BAD_PARAM;
        }
    }

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_logits_count != logits_count ||
        snapshot_top_k != top_k ||
        snapshot_workspace_capacity != workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_selected_count != out_selected_count ||
        snapshot_out_max_logit_q16 != out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells != canonical_required_workspace_cells ||
        staged_selected_count != canonical_selected_count ||
        staged_max_logit_q16 != canonical_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_selected_count = staged_selected_count;
    *out_max_logit_q16 = staged_max_logit_q16;
    return FP_Q16_OK;
}

// IQ-1086 diagnostics-only zero-write companion for parity/geometry preflight.
//
// Compose:
//   1) `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParity`
//      as staged diagnostics tuple producer.
//   2) Canonical geometry recomputation through
//      `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly`.
//
// Publish contract:
//   - Output tuple `{required_workspace_cells,selected_count,max_logit_q16}`
//     is committed only when staged and canonical tuples match exactly.
//   - All error paths preserve caller outputs (no partial writes).
I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 *logits_q16,
    I64 logits_count,
    I64 top_k,
    I64 workspace_capacity,
    I64 *out_required_workspace_cells,
    I64 *out_selected_count,
    I64 *out_max_logit_q16)
{
    I64 staged_required_workspace_cells;
    I64 staged_selected_count;
    I64 staged_max_logit_q16;

    I64 canonical_required_workspace_cells;
    I64 canonical_selected_count;
    I64 canonical_max_logit_q16;

    I64 *snapshot_logits_q16;
    I64 snapshot_logits_count;
    I64 snapshot_top_k;
    I64 snapshot_workspace_capacity;

    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_selected_count;
    I64 *snapshot_out_max_logit_q16;

    I32 status;

    if (!logits_q16 || !out_required_workspace_cells ||
        !out_selected_count || !out_max_logit_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (out_required_workspace_cells == out_selected_count ||
        out_required_workspace_cells == out_max_logit_q16 ||
        out_selected_count == out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (logits_count < 0 || top_k < 0 || workspace_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_k > logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_logits_q16 = logits_q16;
    snapshot_logits_count = logits_count;
    snapshot_top_k = top_k;
    snapshot_workspace_capacity = workspace_capacity;

    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_selected_count = out_selected_count;
    snapshot_out_max_logit_q16 = out_max_logit_q16;

    status = SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParity(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &staged_required_workspace_cells,
        &staged_selected_count,
        &staged_max_logit_q16);
    if (status)
        return status;

    status = SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &canonical_required_workspace_cells,
        &canonical_selected_count,
        &canonical_max_logit_q16);
    if (status)
        return status;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_logits_count != logits_count ||
        snapshot_top_k != top_k ||
        snapshot_workspace_capacity != workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_selected_count != out_selected_count ||
        snapshot_out_max_logit_q16 != out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Geometry tuple fields must stay non-negative for both staged and
    // canonical paths. max_logit_q16 may be negative and is parity-checked
    // below, but required workspace and selected count are cardinalities.
    if (staged_required_workspace_cells < 0 ||
        staged_selected_count < 0 ||
        canonical_required_workspace_cells < 0 ||
        canonical_selected_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Enforce cardinality ordering before parity publication.
    if (staged_selected_count > staged_required_workspace_cells ||
        canonical_selected_count > canonical_required_workspace_cells)
        return FP_Q16_ERR_BAD_PARAM;

    // Both staged and canonical requirements must remain within the immutable
    // caller workspace budget snapshot.
    if (staged_required_workspace_cells > snapshot_workspace_capacity ||
        canonical_required_workspace_cells > snapshot_workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells != canonical_required_workspace_cells ||
        staged_selected_count != canonical_selected_count ||
        staged_max_logit_q16 != canonical_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_selected_count = staged_selected_count;
    *out_max_logit_q16 = staged_max_logit_q16;
    return FP_Q16_OK;
}

// IQ-1087 diagnostics-only parity gate over composed commit/preflight paths.
//
// Compose:
//   1) `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly`
//      as staged commit-only tuple producer.
//   2) `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParity`
//      as canonical parity tuple producer.
//
// Publish contract:
//   - Output tuple `{required_workspace_cells,selected_count,max_logit_q16}`
//     is committed only when staged and canonical tuples match exactly.
//   - All error paths preserve caller outputs (no partial writes).
I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 *logits_q16,
    I64 logits_count,
    I64 top_k,
    I64 workspace_capacity,
    I64 *out_required_workspace_cells,
    I64 *out_selected_count,
    I64 *out_max_logit_q16)
{
    I64 staged_required_workspace_cells;
    I64 staged_selected_count;
    I64 staged_max_logit_q16;

    I64 canonical_required_workspace_cells;
    I64 canonical_selected_count;
    I64 canonical_max_logit_q16;

    I64 *snapshot_logits_q16;
    I64 snapshot_logits_count;
    I64 snapshot_top_k;
    I64 snapshot_workspace_capacity;

    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_selected_count;
    I64 *snapshot_out_max_logit_q16;

    I32 status;

    if (!logits_q16 || !out_required_workspace_cells ||
        !out_selected_count || !out_max_logit_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (out_required_workspace_cells == logits_q16 ||
        out_selected_count == logits_q16 ||
        out_max_logit_q16 == logits_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_workspace_cells == out_selected_count ||
        out_required_workspace_cells == out_max_logit_q16 ||
        out_selected_count == out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (logits_count < 0 || top_k < 0 || workspace_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_k > logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_logits_q16 = logits_q16;
    snapshot_logits_count = logits_count;
    snapshot_top_k = top_k;
    snapshot_workspace_capacity = workspace_capacity;

    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_selected_count = out_selected_count;
    snapshot_out_max_logit_q16 = out_max_logit_q16;

    status =
        SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
            logits_q16,
            logits_count,
            top_k,
            workspace_capacity,
            &staged_required_workspace_cells,
            &staged_selected_count,
            &staged_max_logit_q16);
    if (status)
        return status;

    status = SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParity(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &canonical_required_workspace_cells,
        &canonical_selected_count,
        &canonical_max_logit_q16);
    if (status)
        return status;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_logits_count != logits_count ||
        snapshot_top_k != top_k ||
        snapshot_workspace_capacity != workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_selected_count != out_selected_count ||
        snapshot_out_max_logit_q16 != out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells != canonical_required_workspace_cells ||
        staged_selected_count != canonical_selected_count ||
        staged_max_logit_q16 != canonical_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_selected_count = staged_selected_count;
    *out_max_logit_q16 = staged_max_logit_q16;
    return FP_Q16_OK;
}

// IQ-1104 commit-only diagnostics wrapper over deep parity-preflight chain.
//
// Compose:
//   1) staged diagnostics tuple from
//      `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity`
//   2) canonical diagnostics tuple from
//      `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly`
// and atomically publish only after immutable snapshot + tuple parity checks.
I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    I64 *logits_q16,
    I64 logits_count,
    I64 top_k,
    I64 workspace_capacity,
    I64 *out_required_workspace_cells,
    I64 *out_selected_count,
    I64 *out_max_logit_q16)
{
    I64 staged_required_workspace_cells;
    I64 staged_selected_count;
    I64 staged_max_logit_q16;

    I64 canonical_required_workspace_cells;
    I64 canonical_selected_count;
    I64 canonical_max_logit_q16;

    I64 *snapshot_logits_q16;
    I64 snapshot_logits_count;
    I64 snapshot_top_k;
    I64 snapshot_workspace_capacity;

    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_selected_count;
    I64 *snapshot_out_max_logit_q16;

    I32 status;

    if (!logits_q16 || !out_required_workspace_cells ||
        !out_selected_count || !out_max_logit_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (out_required_workspace_cells == logits_q16 ||
        out_selected_count == logits_q16 ||
        out_max_logit_q16 == logits_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_workspace_cells == out_selected_count ||
        out_required_workspace_cells == out_max_logit_q16 ||
        out_selected_count == out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (logits_count < 0 || top_k < 0 || workspace_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_k > logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_logits_q16 = logits_q16;
    snapshot_logits_count = logits_count;
    snapshot_top_k = top_k;
    snapshot_workspace_capacity = workspace_capacity;

    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_selected_count = out_selected_count;
    snapshot_out_max_logit_q16 = out_max_logit_q16;

    status =
        SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &staged_required_workspace_cells,
        &staged_selected_count,
        &staged_max_logit_q16);
    if (status)
        return status;

    status = SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &canonical_required_workspace_cells,
        &canonical_selected_count,
        &canonical_max_logit_q16);
    if (status)
        return status;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_logits_count != logits_count ||
        snapshot_top_k != top_k ||
        snapshot_workspace_capacity != workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_selected_count != out_selected_count ||
        snapshot_out_max_logit_q16 != out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Cardinality fields are counts and must never go negative.
    if (staged_required_workspace_cells < 0 ||
        staged_selected_count < 0 ||
        canonical_required_workspace_cells < 0 ||
        canonical_selected_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // top-k selected lanes cannot exceed required scratch lanes.
    if (staged_selected_count > staged_required_workspace_cells ||
        canonical_selected_count > canonical_required_workspace_cells)
        return FP_Q16_ERR_BAD_PARAM;

    // Immutable top-k snapshot is the publication ceiling for selected lanes.
    if (staged_selected_count > snapshot_top_k ||
        canonical_selected_count > snapshot_top_k)
        return FP_Q16_ERR_BAD_PARAM;

    // Required scratch cells must always include all logits lanes.
    if (staged_required_workspace_cells < snapshot_logits_count ||
        canonical_required_workspace_cells < snapshot_logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    // Immutable workspace snapshot remains the hard publication ceiling.
    if (staged_required_workspace_cells > snapshot_workspace_capacity ||
        canonical_required_workspace_cells > snapshot_workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells != canonical_required_workspace_cells ||
        staged_selected_count != canonical_selected_count ||
        staged_max_logit_q16 != canonical_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_selected_count = staged_selected_count;
    *out_max_logit_q16 = staged_max_logit_q16;
    return FP_Q16_OK;
}

// Checked deterministic top-k selector that emits both logits and indices into
// disjoint caller-owned output buffers.
// Output contract:
//   - Writes exactly k lanes to out_topk_logits_q16[0..k-1] and
//     out_topk_indices[0..k-1].
//   - Emitted order is descending logit value with deterministic tie-break:
//     lower token index wins when logits are equal.
//   - No partial writes on all parameter/preflight failure paths.
// Checked guards:
//   - logits_q16, out_topk_logits_q16, out_topk_indices are non-null.
//   - lane_count >= 0, k >= 0, out_lane_capacity >= 0.
//   - k <= lane_count and k <= out_lane_capacity.
//   - Pointer-span arithmetic for every indexed read/write cannot overflow U64.
//   - Strict no-alias: logits/input span, top-k logit output span, and top-k
//     index output span must be pairwise disjoint.
// Notes:
//   - Selection uses explicit rank counting (stable descending order) so every
//     ordering decision is visible and auditable.

// IQ-1107 commit-only diagnostics wrapper over deep parity-preflight chain.
//
// Compose:
//   1) staged diagnostics tuple from
//      `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity`
//   2) canonical diagnostics tuple from
//      `SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly`
// and atomically publish only after immutable snapshot + tuple parity checks.
I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    I64 *logits_q16,
    I64 logits_count,
    I64 top_k,
    I64 workspace_capacity,
    I64 *out_required_workspace_cells,
    I64 *out_selected_count,
    I64 *out_max_logit_q16)
{
    I64 staged_required_workspace_cells;
    I64 staged_selected_count;
    I64 staged_max_logit_q16;

    I64 canonical_required_workspace_cells;
    I64 canonical_selected_count;
    I64 canonical_max_logit_q16;

    I64 *snapshot_logits_q16;
    I64 snapshot_logits_count;
    I64 snapshot_top_k;
    I64 snapshot_workspace_capacity;

    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_selected_count;
    I64 *snapshot_out_max_logit_q16;

    I32 status;

    if (!logits_q16 || !out_required_workspace_cells ||
        !out_selected_count || !out_max_logit_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (out_required_workspace_cells == logits_q16 ||
        out_selected_count == logits_q16 ||
        out_max_logit_q16 == logits_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_workspace_cells == out_selected_count ||
        out_required_workspace_cells == out_max_logit_q16 ||
        out_selected_count == out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (logits_count < 0 || top_k < 0 || workspace_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_k > logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_logits_q16 = logits_q16;
    snapshot_logits_count = logits_count;
    snapshot_top_k = top_k;
    snapshot_workspace_capacity = workspace_capacity;

    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_selected_count = out_selected_count;
    snapshot_out_max_logit_q16 = out_max_logit_q16;

    status =
        SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &staged_required_workspace_cells,
        &staged_selected_count,
        &staged_max_logit_q16);
    if (status)
        return status;

    status = SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        &canonical_required_workspace_cells,
        &canonical_selected_count,
        &canonical_max_logit_q16);
    if (status)
        return status;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_logits_count != logits_count ||
        snapshot_top_k != top_k ||
        snapshot_workspace_capacity != workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_selected_count != out_selected_count ||
        snapshot_out_max_logit_q16 != out_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells < 0 ||
        staged_selected_count < 0 ||
        canonical_required_workspace_cells < 0 ||
        canonical_selected_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_selected_count > staged_required_workspace_cells ||
        canonical_selected_count > canonical_required_workspace_cells)
        return FP_Q16_ERR_BAD_PARAM;

    // Effective selected lanes must match immutable caller top-k geometry.
    if (staged_selected_count != snapshot_top_k ||
        canonical_selected_count != snapshot_top_k)
        return FP_Q16_ERR_BAD_PARAM;

    // Required workspace must at least cover the logits lane count.
    if (staged_required_workspace_cells < snapshot_logits_count ||
        canonical_required_workspace_cells < snapshot_logits_count)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells > snapshot_workspace_capacity ||
        canonical_required_workspace_cells > snapshot_workspace_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_workspace_cells != canonical_required_workspace_cells ||
        staged_selected_count != canonical_selected_count ||
        staged_max_logit_q16 != canonical_max_logit_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_selected_count = staged_selected_count;
    *out_max_logit_q16 = staged_max_logit_q16;
    return FP_Q16_OK;
}

I32 FPQ16TopKSelectLogitsCheckedNoAlias(I64 *logits_q16,
                                        I64 lane_count,
                                        I64 k,
                                        I64 *out_topk_logits_q16,
                                        I64 *out_topk_indices,
                                        I64 out_lane_capacity)
{
    I64 i;
    I64 j;
    I64 rank;
    I64 lane_order_rank;
    I64 selected_index;
    I64 selected_logit_q16;
    I64 last_index;
    I64 last_byte_offset;
    U64 logits_start;
    U64 logits_end;
    U64 out_logits_start;
    U64 out_logits_end;
    U64 out_indices_start;
    U64 out_indices_end;

    if (!logits_q16 || !out_topk_logits_q16 || !out_topk_indices)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0 || k < 0 || out_lane_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (k > lane_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (k > out_lane_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (!k)
        return FP_Q16_OK;

    // Guard full input span for indexed logit reads.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    logits_start = (U64)logits_q16;
    logits_end = logits_start + (U64)last_byte_offset;

    // Guard full output-capacity spans so callers can safely reuse the same
    // checked capacity contract across all top-k emitters.
    last_index = out_lane_capacity - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;

    if ((U64)out_topk_logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_topk_indices > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    out_logits_start = (U64)out_topk_logits_q16;
    out_logits_end = out_logits_start + (U64)last_byte_offset;
    out_indices_start = (U64)out_topk_indices;
    out_indices_end = out_indices_start + (U64)last_byte_offset;

    // Strict no-alias policy: every participating span must be disjoint.
    if (!(logits_end < out_logits_start || out_logits_end < logits_start))
        return FP_Q16_ERR_BAD_PARAM;
    if (!(logits_end < out_indices_start || out_indices_end < logits_start))
        return FP_Q16_ERR_BAD_PARAM;
    if (!(out_logits_end < out_indices_start || out_indices_end < out_logits_start))
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight full rank-selection to guarantee no-partial-write semantics.
    for (rank = 0; rank < k; rank++)
    {
        selected_index = -1;
        selected_logit_q16 = 0;

        for (i = 0; i < lane_count; i++)
        {
            lane_order_rank = 0;
            for (j = 0; j < lane_count; j++)
            {
                if (logits_q16[j] > logits_q16[i])
                    lane_order_rank++;
                else if (logits_q16[j] == logits_q16[i] && j < i)
                    lane_order_rank++;
            }

            if (lane_order_rank == rank)
            {
                selected_index = i;
                selected_logit_q16 = logits_q16[i];
                break;
            }
        }

        if (selected_index < 0)
            return FP_Q16_ERR_OVERFLOW;
    }

    // Commit pass writes both output vectors in rank order.
    for (rank = 0; rank < k; rank++)
    {
        selected_index = -1;
        selected_logit_q16 = 0;

        for (i = 0; i < lane_count; i++)
        {
            lane_order_rank = 0;
            for (j = 0; j < lane_count; j++)
            {
                if (logits_q16[j] > logits_q16[i])
                    lane_order_rank++;
                else if (logits_q16[j] == logits_q16[i] && j < i)
                    lane_order_rank++;
            }

            if (lane_order_rank == rank)
            {
                selected_index = i;
                selected_logit_q16 = logits_q16[i];
                break;
            }
        }

        if (selected_index < 0)
            return FP_Q16_ERR_OVERFLOW;

        out_topk_indices[rank] = selected_index;
        out_topk_logits_q16[rank] = selected_logit_q16;
    }

    return FP_Q16_OK;
}

// Canonical default-capacity wrapper for checked no-alias top-k selection.
// Default contract:
//   - out_lane_capacity is derived as k, so callers only pass the exact
//     emission length and this wrapper forwards all validation/selection logic
//     to the checked core helper.
//   - Success/error/no-partial-write semantics are intentionally identical to
//     FPQ16TopKSelectLogitsCheckedNoAlias because this wrapper only fixes the
//     capacity parameter.
I32 FPQ16TopKSelectLogitsCheckedNoAliasDefault(I64 *logits_q16,
                                               I64 lane_count,
                                               I64 k,
                                               I64 *out_topk_logits_q16,
                                               I64 *out_topk_indices)
{
    return FPQ16TopKSelectLogitsCheckedNoAlias(logits_q16,
                                               lane_count,
                                               k,
                                               out_topk_logits_q16,
                                               out_topk_indices,
                                               k);
}

// Checked deterministic cumulative-mass helper for a caller-supplied top-k
// index set over normalized Q16 probabilities.
// Output contract:
//   - Writes exactly k cumulative prefix masses to
//     out_prefix_mass_q16[0..k-1].
//   - Prefix mass i equals sum(probs_q16[topk_indices[0..i]]).
//   - No partial writes on all parameter/preflight failure paths.
// Checked guards:
//   - probs_q16, topk_indices, out_prefix_mass_q16 are non-null.
//   - lane_count >= 0, k >= 0, out_mass_capacity >= 0.
//   - k <= lane_count and k <= out_mass_capacity.
//   - Pointer-span arithmetic for indexed reads/writes cannot overflow U64.
//   - Every selected index is in [0, lane_count).
//   - Selected indices are unique (top-k set semantics).
//   - Every selected probability lane is in [0, 1.0_q16].
//   - Checked cumulative mass cannot overflow I64 and cannot exceed 1.0_q16.
I32 FPQ16TopKPrefixMassQ16Checked(I64 *probs_q16,
                                  I64 lane_count,
                                  I64 *topk_indices,
                                  I64 k,
                                  I64 *out_prefix_mass_q16,
                                  I64 out_mass_capacity)
{
    I64 rank;
    I64 prior;
    I64 selected_index;
    I64 lane_prob_q16;
    I64 cumulative_q16;
    I64 prev_cumulative_q16;
    I64 last_index;
    I64 last_byte_offset;

    if (!probs_q16 || !topk_indices || !out_prefix_mass_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0 || k < 0 || out_mass_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (k > lane_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (k > out_mass_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (!k)
        return FP_Q16_OK;

    // Guard full probability input span before indexed reads.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Guard selected-index and output spans for checked top-k traversal.
    last_index = k - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)topk_indices > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    last_index = out_mass_capacity - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)out_prefix_mass_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight uniqueness/domain/overflow over the whole selected set.
    cumulative_q16 = 0;
    prev_cumulative_q16 = 0;
    for (rank = 0; rank < k; rank++)
    {
        selected_index = topk_indices[rank];
        if (selected_index < 0 || selected_index >= lane_count)
            return FP_Q16_ERR_BAD_PARAM;

        for (prior = 0; prior < rank; prior++)
        {
            if (topk_indices[prior] == selected_index)
                return FP_Q16_ERR_BAD_PARAM;
        }

        lane_prob_q16 = probs_q16[selected_index];
        if (lane_prob_q16 < 0 || lane_prob_q16 > FP_Q16_ONE)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > (I64_MAX_VALUE - cumulative_q16))
            return FP_Q16_ERR_OVERFLOW;

        cumulative_q16 += lane_prob_q16;
        if (cumulative_q16 < prev_cumulative_q16)
            return FP_Q16_ERR_OVERFLOW;
        if (cumulative_q16 > FP_Q16_ONE)
            return FP_Q16_ERR_BAD_PARAM;

        prev_cumulative_q16 = cumulative_q16;
    }

    // Commit pass writes deterministic prefix masses in caller-selected order.
    cumulative_q16 = 0;
    for (rank = 0; rank < k; rank++)
    {
        selected_index = topk_indices[rank];
        lane_prob_q16 = probs_q16[selected_index];
        cumulative_q16 += lane_prob_q16;
        out_prefix_mass_q16[rank] = cumulative_q16;
    }

    return FP_Q16_OK;
}

// Canonical default-capacity wrapper for checked top-k prefix-mass emission.
// Default contract:
//   - out_mass_capacity is derived as k so callers pass only the target top-k
//     length, while this wrapper forwards all checked validation/mass logic to
//     the canonical core helper.
//   - Success/error/no-partial-write semantics intentionally match
//     FPQ16TopKPrefixMassQ16Checked exactly because this wrapper only fixes the
//     output-capacity parameter.
I32 FPQ16TopKPrefixMassQ16CheckedDefault(I64 *probs_q16,
                                         I64 lane_count,
                                         I64 *topk_indices,
                                         I64 k,
                                         I64 *out_prefix_mass_q16)
{
    return FPQ16TopKPrefixMassQ16Checked(probs_q16,
                                         lane_count,
                                         topk_indices,
                                         k,
                                         out_prefix_mass_q16,
                                         k);
}

// Checked deterministic stable descending argsort over Q16 logits.
// Output contract:
//   - Writes exactly lane_count indices into out_indices[0..lane_count-1].
//   - Order is descending logit value.
//   - Tie-break is deterministic and stable: lower token index wins.
//   - No partial writes on all parameter/preflight failure paths.
// Checked guards:
//   - logits_q16 and out_indices are non-null.
//   - lane_count >= 0 and out_index_capacity >= 0.
//   - out_index_capacity >= lane_count.
//   - Pointer-span arithmetic for indexed logits/out accesses cannot overflow U64.
I32 FPQ16ArgsortDescStableChecked(I64 *logits_q16,
                                  I64 lane_count,
                                  I64 *out_indices,
                                  I64 out_index_capacity)
{
    I64 i;
    I64 j;
    I64 last_index;
    I64 last_byte_offset;
    I64 key_index;
    I64 key_logit_q16;
    I64 scan_index;
    I64 scan_logit_q16;

    if (!logits_q16 || !out_indices)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0 || out_index_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_index_capacity < lane_count)
        return FP_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed logits/out writes.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_indices > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Seed index identity mapping before stable insertion sort.
    for (i = 0; i < lane_count; i++)
        out_indices[i] = i;

    // Stable insertion sort:
    //   - Higher logit lanes move earlier.
    //   - Equal logits resolve by lower token index first.
    for (i = 1; i < lane_count; i++)
    {
        key_index = out_indices[i];
        key_logit_q16 = logits_q16[key_index];
        j = i - 1;

        while (j >= 0)
        {
            scan_index = out_indices[j];
            scan_logit_q16 = logits_q16[scan_index];

            if (scan_logit_q16 < key_logit_q16)
            {
                out_indices[j + 1] = scan_index;
                j--;
                continue;
            }

            if (scan_logit_q16 == key_logit_q16 && scan_index > key_index)
            {
                out_indices[j + 1] = scan_index;
                j--;
                continue;
            }

            break;
        }

        out_indices[j + 1] = key_index;
    }

    return FP_Q16_OK;
}

// Checked deterministic top-p nucleus-prefix selector over descending Q16
// probabilities.
// Output contract:
//   - Writes exactly one prefix length to *out_prefix_len on success.
//   - Returned prefix length is the minimal k in [1, lane_count] such that
//     cumulative_sum(probs_q16[0..k-1]) >= top_p_q16.
//   - If cumulative mass never reaches top_p_q16 due to rounding residue,
//     returns lane_count.
//   - No partial writes on all failure paths.
// Checked guards:
//   - probs_q16 and out_prefix_len are non-null.
//   - lane_count > 0.
//   - top_p_q16 in (0, 1.0_q16].
//   - Pointer-span arithmetic for indexed probs access cannot overflow U64.
//   - Each probability lane is in [0, 1.0_q16].
//   - Probabilities are monotone non-increasing (descending with ties).
//   - Checked cumulative-sum monotonicity + I64 overflow guards.
//   - Total probability mass equals 1.0_q16 exactly.
I32 FPQ16TopPSelectPrefixLenChecked(I64 *probs_q16,
                                    I64 lane_count,
                                    I64 top_p_q16,
                                    I64 *out_prefix_len)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_prob_q16;
    I64 prev_prob_q16;
    I64 prev_cumulative_q16;
    I64 cumulative_q16;
    I64 prefix_len;

    if (!probs_q16 || !out_prefix_len)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // Guard pointer-span arithmetic before indexed probability access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Scan descending probabilities, track checked cumulative mass, and stop
    // at the first prefix crossing top_p_q16.
    cumulative_q16 = 0;
    prev_cumulative_q16 = 0;
    prefix_len = -1;
    prev_prob_q16 = FP_Q16_ONE;

    for (i = 0; i < lane_count; i++)
    {
        lane_prob_q16 = probs_q16[i];
        if (lane_prob_q16 < 0 || lane_prob_q16 > FP_Q16_ONE)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > prev_prob_q16)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > (I64_MAX_VALUE - cumulative_q16))
            return FP_Q16_ERR_OVERFLOW;

        cumulative_q16 += lane_prob_q16;
        if (cumulative_q16 < prev_cumulative_q16)
            return FP_Q16_ERR_OVERFLOW;
        if (cumulative_q16 > FP_Q16_ONE)
            return FP_Q16_ERR_BAD_PARAM;

        if (prefix_len < 0 && cumulative_q16 >= top_p_q16)
            prefix_len = i + 1;

        prev_prob_q16 = lane_prob_q16;
        prev_cumulative_q16 = cumulative_q16;
    }

    if (cumulative_q16 != FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve deterministic progress even when finite Q16 rounding leaves
    // top_p_q16 just above the final prefix mass.
    if (prefix_len < 0)
        prefix_len = lane_count;

    *out_prefix_len = prefix_len;
    return FP_Q16_OK;
}

// Checked no-alias top-p nucleus-prefix selector wrapper.
// Contract:
//   - Matches FPQ16TopPSelectPrefixLenChecked semantics exactly on all
//     non-alias input domains.
//   - Rejects any overlap between probs_q16[lane_count] storage and
//     out_prefix_len scalar storage.
//   - Preserves no-partial-write behavior from the checked core helper.
// Checked guards:
//   - All core input guards are inherited from
//     FPQ16TopPSelectPrefixLenChecked.
//   - For lane_count > 0, validates span arithmetic for both probs and the
//     output scalar, then rejects any byte-range overlap.
I32 FPQ16TopPSelectPrefixLenCheckedNoAlias(I64 *probs_q16,
                                           I64 lane_count,
                                           I64 top_p_q16,
                                           I64 *out_prefix_len)
{
    I64 probs_last_index;
    I64 probs_last_byte_offset;
    U64 probs_start;
    U64 probs_end;
    U64 out_start;
    U64 out_end;

    if (!probs_q16 || !out_prefix_len)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count <= 0)
        return FP_Q16_ERR_BAD_PARAM;

    probs_last_index = lane_count - 1;
    if (probs_last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    probs_last_byte_offset = probs_last_index << 3;

    probs_start = (U64)probs_q16;
    if (probs_start > (U64_MAX_VALUE - (U64)probs_last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    probs_end = probs_start + (U64)probs_last_byte_offset;

    out_start = (U64)out_prefix_len;
    if (out_start > (U64_MAX_VALUE - 7))
        return FP_Q16_ERR_OVERFLOW;
    out_end = out_start + 7;

    if (!(probs_end < out_start || out_end < probs_start))
        return FP_Q16_ERR_BAD_PARAM;

    return FPQ16TopPSelectPrefixLenChecked(probs_q16,
                                           lane_count,
                                           top_p_q16,
                                           out_prefix_len);
}

// Checked deterministic top-p nucleus logit-cutoff selector.
// Output contract:
//   - Writes cutoff logit to *out_cutoff_logit_q16 and final nucleus length to
//     *out_nucleus_len on success.
//   - First chooses the minimal top-p prefix via
//     FPQ16TopPSelectPrefixLenChecked(probs_q16,...).
//   - Then performs deterministic tie-extension over descending logits:
//     include every trailing lane with logit == boundary logit.
//   - No partial writes on all failure paths.
// Checked guards:
//   - logits_q16, probs_q16, out_cutoff_logit_q16, out_nucleus_len are
//     non-null.
//   - lane_count > 0.
//   - top_p_q16 in (0, 1.0_q16].
//   - Pointer-span arithmetic for indexed logits/probabilities cannot overflow
//     U64.
//   - logits_q16 is monotone non-increasing (descending with ties).
I32 FPQ16TopPSelectLogitCutoffChecked(I64 *logits_q16,
                                      I64 *probs_q16,
                                      I64 lane_count,
                                      I64 top_p_q16,
                                      I64 *out_cutoff_logit_q16,
                                      I64 *out_nucleus_len)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 prev_logit_q16;
    I64 current_logit_q16;
    I64 prefix_len;
    I64 cutoff_index;
    I64 cutoff_logit_q16;
    I64 extended_len;
    I32 status;

    if (!logits_q16 || !probs_q16 || !out_cutoff_logit_q16 || !out_nucleus_len)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // Guard pointer-span arithmetic before indexed logit/probability access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight descending-logit contract over the full lane span.
    prev_logit_q16 = logits_q16[0];
    for (i = 1; i < lane_count; i++)
    {
        current_logit_q16 = logits_q16[i];
        if (current_logit_q16 > prev_logit_q16)
            return FP_Q16_ERR_BAD_PARAM;
        prev_logit_q16 = current_logit_q16;
    }

    // Compose canonical top-p prefix selector over normalized probabilities.
    status = FPQ16TopPSelectPrefixLenChecked(probs_q16,
                                             lane_count,
                                             top_p_q16,
                                             &prefix_len);
    if (status)
        return status;
    if (prefix_len <= 0 || prefix_len > lane_count)
        return FP_Q16_ERR_OVERFLOW;

    // Boundary logit at first top-p crossing.
    cutoff_index = prefix_len - 1;
    cutoff_logit_q16 = logits_q16[cutoff_index];

    // Deterministic tie-extension: include all trailing equal-logit lanes.
    extended_len = prefix_len;
    while (extended_len < lane_count)
    {
        if (logits_q16[extended_len] != cutoff_logit_q16)
            break;
        extended_len++;
    }

    *out_cutoff_logit_q16 = cutoff_logit_q16;
    *out_nucleus_len = extended_len;
    return FP_Q16_OK;
}

// Checked deterministic top-p sampler from a descending nucleus window.
// Output contract:
//   - Writes exactly one sampled lane index to *out_index on success.
//   - Sampling domain is probs_q16[0..prefix_len-1] only.
//   - Selection walk is deterministic: first cumulative bucket strictly above
//     threshold wins. For ties/equal probabilities this preserves lower-index
//     priority because the walk is in ascending index order.
//   - No partial writes on all failure paths.
// Checked guards:
//   - probs_q16 and out_index are non-null.
//   - lane_count > 0.
//   - prefix_len in [1, lane_count].
//   - random_q16 in [0, 1.0_q16).
//   - Pointer-span arithmetic for indexed probs access cannot overflow U64.
//   - probs_q16[0..prefix_len-1] are descending and each lane in [0, 1.0_q16].
//   - Prefix cumulative mass is strictly positive and checked for overflow.
// Notes:
//   - random_q16 is mapped into prefix mass domain via integer-only
//     threshold_q16 = floor(random_q16 * prefix_mass_q16 / 1.0_q16).
//   - If Q16 truncation places threshold on the upper edge, result clamps
//     deterministically to the last lane in the nucleus window.
I32 FPQ16TopPSampleIndexChecked(I64 *probs_q16,
                                I64 lane_count,
                                I64 prefix_len,
                                I64 random_q16,
                                I64 *out_index)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_prob_q16;
    I64 prev_prob_q16;
    I64 prefix_mass_q16;
    I64 cumulative_q16;
    I64 threshold_q16;

    if (!probs_q16 || !out_index)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (prefix_len <= 0 || prefix_len > lane_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (random_q16 < 0 || random_q16 >= FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // Guard pointer-span arithmetic before indexed probability access.
    last_index = prefix_len - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight nucleus window probability domain, descending order, and
    // checked cumulative mass before any output write.
    prefix_mass_q16 = 0;
    prev_prob_q16 = FP_Q16_ONE;
    for (i = 0; i < prefix_len; i++)
    {
        lane_prob_q16 = probs_q16[i];
        if (lane_prob_q16 < 0 || lane_prob_q16 > FP_Q16_ONE)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > prev_prob_q16)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > (I64_MAX_VALUE - prefix_mass_q16))
            return FP_Q16_ERR_OVERFLOW;

        prefix_mass_q16 += lane_prob_q16;
        prev_prob_q16 = lane_prob_q16;
    }

    if (prefix_mass_q16 <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (prefix_mass_q16 > FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // Map uniform random [0,1) into nucleus cumulative mass [0,prefix_mass).
    threshold_q16 = (random_q16 * prefix_mass_q16) >> FP_Q16_SHIFT;

    cumulative_q16 = 0;
    for (i = 0; i < prefix_len; i++)
    {
        lane_prob_q16 = probs_q16[i];
        if (lane_prob_q16 > (I64_MAX_VALUE - cumulative_q16))
            return FP_Q16_ERR_OVERFLOW;

        cumulative_q16 += lane_prob_q16;
        if (threshold_q16 < cumulative_q16)
        {
            *out_index = i;
            return FP_Q16_OK;
        }
    }

    // Deterministic clamp when threshold lands exactly on upper boundary.
    *out_index = prefix_len - 1;
    return FP_Q16_OK;
}

// Checked deterministic top-k sampler over normalized Q16 probabilities.
// Output contract:
//   - Writes exactly one sampled token index to *out_index on success.
//   - Sampling domain is restricted to the top-k highest probabilities.
//   - Ordering is descending probability with deterministic tie-break:
//     lower token index wins when probabilities are equal.
//   - No partial writes on all failure paths.
// Checked guards:
//   - probs_q16 and out_index are non-null.
//   - lane_count >= 0, k >= 0.
//   - k > 0 and k <= lane_count.
//   - random_q16 is in [0, 1.0_q16).
//   - Pointer-span arithmetic for indexed probs access cannot overflow U64.
//   - Each probability lane is in [0, 1.0_q16].
//   - Sum(probs_q16[0..lane_count-1]) equals 1.0_q16 exactly.
//   - Sum(top-k probabilities) is strictly positive.
// Notes:
//   - Implementation intentionally avoids hidden containers. Rank selection
//     uses direct pairwise ordering tests so every comparison remains auditable.
I32 FPQ16TopKSampleFromProbabilitiesChecked(I64 *probs_q16,
                                            I64 lane_count,
                                            I64 k,
                                            I64 random_q16,
                                            I64 *out_index)
{
    I64 i;
    I64 j;
    I64 rank;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_prob_q16;
    I64 lane_sum_q16;
    I64 lane_order_rank;
    I64 selected_index;
    I64 selected_prob_q16;
    I64 topk_sum_q16;
    I64 topk_cumulative_q16;
    I64 threshold_q16;

    if (!probs_q16 || !out_index)
        return FP_Q16_ERR_NULL_PTR;
    if (lane_count < 0 || k < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!k || k > lane_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (random_q16 < 0 || random_q16 >= FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // Guard pointer-span arithmetic before indexed probability access.
    last_index = lane_count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight probability domain and exact normalized sum.
    lane_sum_q16 = 0;
    for (i = 0; i < lane_count; i++)
    {
        lane_prob_q16 = probs_q16[i];
        if (lane_prob_q16 < 0 || lane_prob_q16 > FP_Q16_ONE)
            return FP_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > (I64_MAX_VALUE - lane_sum_q16))
            return FP_Q16_ERR_OVERFLOW;
        lane_sum_q16 += lane_prob_q16;
    }
    if (lane_sum_q16 != FP_Q16_ONE)
        return FP_Q16_ERR_BAD_PARAM;

    // First pass: compute top-k probability mass via explicit rank lookup.
    // Rank definition: number of lanes strictly ahead in descending order.
    topk_sum_q16 = 0;
    for (rank = 0; rank < k; rank++)
    {
        selected_index = -1;
        selected_prob_q16 = 0;

        for (i = 0; i < lane_count; i++)
        {
            lane_order_rank = 0;
            for (j = 0; j < lane_count; j++)
            {
                if (probs_q16[j] > probs_q16[i])
                    lane_order_rank++;
                else if (probs_q16[j] == probs_q16[i] && j < i)
                    lane_order_rank++;
            }

            if (lane_order_rank == rank)
            {
                selected_index = i;
                selected_prob_q16 = probs_q16[i];
                break;
            }
        }

        if (selected_index < 0)
            return FP_Q16_ERR_OVERFLOW;
        if (selected_prob_q16 > (I64_MAX_VALUE - topk_sum_q16))
            return FP_Q16_ERR_OVERFLOW;

        topk_sum_q16 += selected_prob_q16;
    }

    if (topk_sum_q16 <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (topk_sum_q16 > FP_Q16_ONE)
        return FP_Q16_ERR_OVERFLOW;

    // Convert uniform random_q16 in [0,1) into top-k mass domain [0,topk_sum).
    threshold_q16 = (random_q16 * topk_sum_q16) >> FP_Q16_SHIFT;

    // Second pass: walk top-k in stable descending order and select first
    // cumulative bucket that exceeds threshold.
    topk_cumulative_q16 = 0;
    selected_index = -1;
    for (rank = 0; rank < k; rank++)
    {
        I64 rank_index;
        I64 rank_prob_q16;

        rank_index = -1;
        rank_prob_q16 = 0;
        for (i = 0; i < lane_count; i++)
        {
            lane_order_rank = 0;
            for (j = 0; j < lane_count; j++)
            {
                if (probs_q16[j] > probs_q16[i])
                    lane_order_rank++;
                else if (probs_q16[j] == probs_q16[i] && j < i)
                    lane_order_rank++;
            }

            if (lane_order_rank == rank)
            {
                rank_index = i;
                rank_prob_q16 = probs_q16[i];
                break;
            }
        }

        if (rank_index < 0)
            return FP_Q16_ERR_OVERFLOW;
        if (rank_prob_q16 > (I64_MAX_VALUE - topk_cumulative_q16))
            return FP_Q16_ERR_OVERFLOW;

        topk_cumulative_q16 += rank_prob_q16;
        if (threshold_q16 < topk_cumulative_q16)
        {
            selected_index = rank_index;
            break;
        }
    }

    // If truncation to Q16 bucket boundary left threshold on the upper edge,
    // deterministically clamp to the lowest-priority lane in top-k.
    if (selected_index < 0)
    {
        for (i = 0; i < lane_count; i++)
        {
            lane_order_rank = 0;
            for (j = 0; j < lane_count; j++)
            {
                if (probs_q16[j] > probs_q16[i])
                    lane_order_rank++;
                else if (probs_q16[j] == probs_q16[i] && j < i)
                    lane_order_rank++;
            }

            if (lane_order_rank == (k - 1))
            {
                selected_index = i;
                break;
            }
        }
        if (selected_index < 0)
            return FP_Q16_ERR_OVERFLOW;
    }

    *out_index = selected_index;
    return FP_Q16_OK;
}

// Checked row-wise stable softmax over Q16 logits.
//
// For each logical row r and lane i:
//   shifted[r,i] = clamp(logits[r,i] - max_j(logits[r,j]), EXP_MIN, EXP_MAX)
//   probs[r,*]   = softmax(shifted[r,*])
//
// Contracts:
//   - logits_q16, probs_q16, scratch_q16 are non-null.
//   - row_count and lane_count are >= 0.
//   - When lane_count > 0, logits_row_stride and probs_row_stride are >= lane_count.
//   - logits_capacity/probs_capacity are in I64 cells, not bytes.
//   - scratch_capacity must cover:
//       stage_cells = row_count * lane_count
//       temp_cells  = 2 * lane_count (shifted row + exp row scratch)
//       total_cells = stage_cells + temp_cells
//
// No-partial write guarantee:
//   - Output rows in probs_q16 are published only after all rows succeed.
I32 FPQ16SoftmaxRowsStableCheckedNoPartial(I64 *logits_q16,
                                           I64 logits_capacity,
                                           I64 row_count,
                                           I64 lane_count,
                                           I64 logits_row_stride,
                                           I64 *probs_q16,
                                           I64 probs_capacity,
                                           I64 probs_row_stride,
                                           I64 *scratch_q16,
                                           I64 scratch_capacity)
{
    I64 required_logits_cells;
    I64 required_probs_cells;
    I64 stage_cells;
    I64 temp_cells;
    I64 required_scratch_cells;
    I64 row;
    I64 lane;
    I64 row_max_q16;
    I64 delta_q16;
    I64 *shifted_row_q16;
    I64 *exp_row_q16;
    I64 *staged_probs_q16;
    I64 *logits_row_q16;
    I64 *probs_row_q16;
    I32 status;

    if (!logits_q16 || !probs_q16 || !scratch_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (row_count < 0 || lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (logits_capacity < 0 || probs_capacity < 0 || scratch_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (lane_count > 0)
    {
        if (logits_row_stride < lane_count || probs_row_stride < lane_count)
            return FP_Q16_ERR_BAD_PARAM;
    }

    required_logits_cells = 0;
    required_probs_cells = 0;
    if (row_count > 0 && lane_count > 0)
    {
        I64 last_row;
        I64 last_logits_base;
        I64 last_probs_base;

        last_row = row_count - 1;

        if (last_row > 0 && logits_row_stride > (I64_MAX_VALUE / last_row))
            return FP_Q16_ERR_OVERFLOW;
        if (last_row > 0 && probs_row_stride > (I64_MAX_VALUE / last_row))
            return FP_Q16_ERR_OVERFLOW;

        last_logits_base = last_row * logits_row_stride;
        last_probs_base = last_row * probs_row_stride;

        if (last_logits_base > (I64_MAX_VALUE - (lane_count - 1)))
            return FP_Q16_ERR_OVERFLOW;
        if (last_probs_base > (I64_MAX_VALUE - (lane_count - 1)))
            return FP_Q16_ERR_OVERFLOW;

        required_logits_cells = last_logits_base + lane_count;
        required_probs_cells = last_probs_base + lane_count;
    }

    if (logits_capacity < required_logits_cells)
        return FP_Q16_ERR_BAD_PARAM;
    if (probs_capacity < required_probs_cells)
        return FP_Q16_ERR_BAD_PARAM;

    stage_cells = 0;
    if (row_count > 0 && lane_count > 0)
    {
        if (row_count > (I64_MAX_VALUE / lane_count))
            return FP_Q16_ERR_OVERFLOW;
        stage_cells = row_count * lane_count;
    }

    if (lane_count > (I64_MAX_VALUE / 2))
        return FP_Q16_ERR_OVERFLOW;
    temp_cells = lane_count * 2;

    if (stage_cells > (I64_MAX_VALUE - temp_cells))
        return FP_Q16_ERR_OVERFLOW;
    required_scratch_cells = stage_cells + temp_cells;

    if (scratch_capacity < required_scratch_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (row_count == 0 || lane_count == 0)
        return FP_Q16_OK;

    staged_probs_q16 = scratch_q16;
    shifted_row_q16 = scratch_q16 + stage_cells;
    exp_row_q16 = shifted_row_q16 + lane_count;

    // Stage every row into scratch first; caller output stays untouched on error.
    for (row = 0; row < row_count; row++)
    {
        logits_row_q16 = logits_q16 + (row * logits_row_stride);
        row_max_q16 = logits_row_q16[0];

        for (lane = 1; lane < lane_count; lane++)
        {
            if (logits_row_q16[lane] > row_max_q16)
                row_max_q16 = logits_row_q16[lane];
        }

        for (lane = 0; lane < lane_count; lane++)
        {
            // Checked subtraction for delta = logits - row_max.
            if (row_max_q16 < 0 && logits_row_q16[lane] > (I64_MAX_VALUE + row_max_q16))
                return FP_Q16_ERR_OVERFLOW;
            if (row_max_q16 > 0 && logits_row_q16[lane] < (I64_MIN_VALUE + row_max_q16))
                return FP_Q16_ERR_OVERFLOW;

            delta_q16 = logits_row_q16[lane] - row_max_q16;

            // Clamp to checked exp-domain required by FPQ16SoftmaxFromPreclampedChecked.
            if (delta_q16 < SOFTMAX_EXP_Q16_MIN_INPUT)
                delta_q16 = SOFTMAX_EXP_Q16_MIN_INPUT;
            else if (delta_q16 > SOFTMAX_EXP_Q16_MAX_INPUT)
                delta_q16 = SOFTMAX_EXP_Q16_MAX_INPUT;

            shifted_row_q16[lane] = delta_q16;
        }

        status = FPQ16SoftmaxFromPreclampedChecked(shifted_row_q16,
                                                   exp_row_q16,
                                                   lane_count,
                                                   staged_probs_q16 + (row * lane_count));
        if (status)
            return status;
    }

    // Commit phase: publish staged rows once all rows are valid.
    for (row = 0; row < row_count; row++)
    {
        probs_row_q16 = probs_q16 + (row * probs_row_stride);
        for (lane = 0; lane < lane_count; lane++)
            probs_row_q16[lane] = staged_probs_q16[(row * lane_count) + lane];
    }

    return FP_Q16_OK;
}

U0 FPQ16Softmax(I64 *logits, I64 *probs, I64 count)
{
    I64 i;
    I64 max_logit;
    I64 shifted;
    I64 exp_val;
    I64 exp_sum;
    I64 prob_sum;
    I64 diff;
    I64 max_idx;

    if (!logits || !probs || count <= 0)
        return;

    max_logit = logits[0];
    max_idx = 0;
    for (i = 1; i < count; i++)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
            max_idx = i;
        }
    }

    exp_sum = 0;
    for (i = 0; i < count; i++)
    {
        shifted = logits[i] - max_logit;
        exp_val = FPQ16Exp(shifted);
        probs[i] = exp_val;

        if (exp_sum > 0x7FFFFFFFFFFFFFFF - exp_val)
            exp_sum = 0x7FFFFFFFFFFFFFFF;
        else
            exp_sum += exp_val;
    }

    if (exp_sum <= 0)
    {
        exp_val = FPQ16Div(FP_Q16_ONE, FPQ16FromInt(count));
        for (i = 0; i < count; i++)
            probs[i] = exp_val;
        return;
    }

    prob_sum = 0;
    for (i = 0; i < count; i++)
    {
        probs[i] = FPQ16Div(probs[i], exp_sum);
        prob_sum += probs[i];
    }

    diff = FP_Q16_ONE - prob_sum;
    probs[max_idx] += diff;
}

I64 FPQ16EntropyFromProbs(I64 *probs, I64 count)
{
    I64 i;
    I64 p_q16;
    I64 ln_p_q16;
    I64 plnp_q16;
    I64 sum_plnp_q16;

    // Shannon entropy in natural-log units:
    // H(p) = -sum_i p_i * ln(p_i)
    // Input probabilities are Q16.16 fixed-point values.
    if (!probs || count <= 0)
        return 0;

    sum_plnp_q16 = 0;

    for (i = 0; i < count; i++)
    {
        p_q16 = probs[i];

        // p=0 contributes exactly 0 in the limit p*ln(p)->0.
        if (p_q16 <= 0)
            continue;

        // Keep diagnostics stable if callers pass slightly out-of-range values.
        if (p_q16 > FP_Q16_ONE)
            p_q16 = FP_Q16_ONE;

        ln_p_q16 = FPQ16Ln(p_q16);
        if (ln_p_q16 == I64_MIN_VALUE)
            continue;

        plnp_q16 = FPQ16Mul(p_q16, ln_p_q16);

        // For p in (0,1], p*ln(p) <= 0. Clamp tiny positive approximation noise.
        if (plnp_q16 > 0)
            plnp_q16 = 0;

        if (sum_plnp_q16 < I64_MIN_VALUE - plnp_q16)
            sum_plnp_q16 = I64_MIN_VALUE;
        else
            sum_plnp_q16 += plnp_q16;
    }

    if (sum_plnp_q16 == I64_MIN_VALUE)
        return I64_MAX_VALUE;

    return -sum_plnp_q16;
}
