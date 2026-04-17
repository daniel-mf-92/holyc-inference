// softmax.HC
// Integer-only Q16.16 softmax for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_OVERFLOW  4

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
