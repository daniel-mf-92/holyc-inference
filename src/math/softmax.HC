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
I32 FPQ16ExpArrayFromClampedInputChecked(I64 *input_q16,
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
