// rmsnorm.HC
// Integer-only Q16.16 RMSNorm for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)
#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF

I64 FPQ16Mul(I64 a, I64 b);
I64 FPQ16Div(I64 num, I64 den);
I64 FPQ16FromInt(I64 x);
I64 FPQ16Sqrt(I64 x);

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
