// intsqrt.HC
// Integer square-root helpers for Q16.16 fixed-point inference math.

#define FP_Q16_SHIFT 16
#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_DOMAIN    3
#define FP_Q16_ERR_OVERFLOW  4

U64 IntSqrtU64(U64 x)
{
    U64 res;
    U64 bit;

    res = 0;
    bit = 1 << 62;

    while (bit > x)
        bit >>= 2;

    while (bit)
    {
        if (x >= res + bit)
        {
            x -= res + bit;
            res = (res >> 1) + bit;
        }
        else
            res >>= 1;

        bit >>= 2;
    }

    return res;
}

I64 FPQ16Sqrt(I64 x)
{
    U64 shifted;

    if (x <= 0)
        return 0;

    if (x > (I64_MAX_VALUE >> FP_Q16_SHIFT))
        shifted = I64_MAX_VALUE;
    else
        shifted = ((U64)x) << FP_Q16_SHIFT;

    return IntSqrtU64(shifted);
}

// Commit-only checked Q16 square-root wrapper over FPQ16Sqrt.
//
// Contract:
// - `out_q16` must be non-null.
// - `x_q16` must be non-negative and in range where `(x_q16 << 16)` does not
//   overflow signed I64.
// - Uses a staged scalar and publishes to caller only after all checks pass.
// - Revalidates immutable input snapshot parity before publish.
I32 FPQ16SqrtCheckedNoPartialCommitOnly(I64 x_q16,
                                        I64 *out_q16)
{
    I64 snapshot_x_q16;
    I64 staged_out_q16;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (x_q16 < 0)
        return FP_Q16_ERR_DOMAIN;

    if (x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT))
        return FP_Q16_ERR_OVERFLOW;

    snapshot_x_q16 = x_q16;
    staged_out_q16 = FPQ16Sqrt(x_q16);

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_q16 = staged_out_q16;
    return FP_Q16_OK;
}
