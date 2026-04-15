// softmax.HC
// Integer-only Q16.16 softmax for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define I64_MIN_VALUE -0x8000000000000000

I64 FPQ16Exp(I64 x);
I64 FPQ16Div(I64 num, I64 den);
I64 FPQ16FromInt(I64 x);
I64 FPQ16Mul(I64 a, I64 b);
I64 FPQ16Ln(I64 x_q16);

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
