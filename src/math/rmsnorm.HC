// rmsnorm.HC
// Integer-only Q16.16 RMSNorm for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)
#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_DOMAIN    3
#define FP_Q16_ERR_OVERFLOW  4

I64 FPQ16Mul(I64 a, I64 b);
I64 FPQ16Div(I64 num, I64 den);
I64 FPQ16FromInt(I64 x);
I64 FPQ16Sqrt(I64 x);
I32 FPQ16MulDivRoundedChecked(I64 a_q16,
                              I64 b_q16,
                              I64 d_q16,
                              I64 *out_q16);
I32 FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(I64 a_q16,
                                                     I64 b_q16,
                                                     I64 d_q16,
                                                     I64 *out_q16);

// Apply precomputed RMSNorm inverse denominator to a Q16 vector with explicit
// checked error surfaces. This uses one-rounding math:
//   out[i] = round_half_up((input[i] * inv_denom_q16) / den_q16)
// where den_q16 is a positive Q16 whole integer (d_int << 16).
I32 FPQ16RMSNormApplyInvDenomChecked(I64 *input_q16,
                                     I64 *output_q16,
                                     I64 count,
                                     I64 inv_denom_q16,
                                     I64 den_q16)
{
    I64 i;
    I64 scaled_q16;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    // Delegate denominator contract + checked one-rounding to fixedpoint core.
    // That keeps RMSNorm scaling error behavior identical to shared math helpers.
    for (i = 0; i < count; i++)
    {
        status = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(input_q16[i],
                                                                   inv_denom_q16,
                                                                   den_q16,
                                                                   &scaled_q16);
        if (status)
            return status;

        output_q16[i] = scaled_q16;
    }

    return 0;
}

// Apply precomputed RMSNorm inverse denominator and per-channel gamma to a
// Q16 vector with explicit checked error surfaces:
//   out[i] = round_half_up(round_half_up((input[i] * inv_denom_q16) / den_q16)
//                          * gamma[i] / FP_Q16_ONE)
// The first stage enforces the positive whole-Q16 denominator contract used by
// WS5 norm call chains. The second stage is checked Q16*Q16 scaling.
I32 FPQ16RMSNormApplyInvDenomWeightedChecked(I64 *input_q16,
                                             I64 *gamma_q16,
                                             I64 *output_q16,
                                             I64 count,
                                             I64 inv_denom_q16,
                                             I64 den_q16)
{
    I64 i;
    I64 scaled_q16;
    I64 weighted_q16;
    I32 status;

    if (!input_q16 || !gamma_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    for (i = 0; i < count; i++)
    {
        status = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(input_q16[i],
                                                                   inv_denom_q16,
                                                                   den_q16,
                                                                   &scaled_q16);
        if (status)
            return status;

        status = FPQ16MulDivRoundedChecked(scaled_q16,
                                           gamma_q16[i],
                                           FP_Q16_ONE,
                                           &weighted_q16);
        if (status)
            return status;

        output_q16[i] = weighted_q16;
    }

    return 0;
}

U0 FPQ16RMSNorm(I64 *input, I64 *weight, I64 *output, I64 count, I64 eps_q16)
{
    I64 i;
    I64 sq;
    I64 sum_sq;
    I64 mean_sq;
    I64 denom;
    I64 inv_denom;
    I64 norm;

    if (!input || !weight || !output || count <= 0)
        return;

    if (eps_q16 < 0)
        eps_q16 = 0;

    sum_sq = 0;
    for (i = 0; i < count; i++)
    {
        sq = FPQ16Mul(input[i], input[i]);
        if (sum_sq > I64_MAX_VALUE - sq)
            sum_sq = I64_MAX_VALUE;
        else
            sum_sq += sq;
    }

    mean_sq = FPQ16Div(sum_sq, FPQ16FromInt(count));

    if (mean_sq > I64_MAX_VALUE - eps_q16)
        mean_sq = I64_MAX_VALUE;
    else
        mean_sq += eps_q16;

    denom = FPQ16Sqrt(mean_sq);
    if (denom <= 0)
    {
        for (i = 0; i < count; i++)
            output[i] = 0;
        return;
    }

    inv_denom = FPQ16Div(FP_Q16_ONE, denom);

    for (i = 0; i < count; i++)
    {
        norm = FPQ16Mul(input[i], inv_denom);
        output[i] = FPQ16Mul(norm, weight[i]);
    }
}
