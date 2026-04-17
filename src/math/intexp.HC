// intexp.HC
// Integer-only Q16.16 exponential approximation for TempleOS HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2

#define EXP_Q16_LN2       45426   // round(ln(2) * 65536)
#define EXP_Q16_MAX_INPUT 655360  // +10.0 in Q16
#define EXP_Q16_MIN_INPUT -655360 // -10.0 in Q16

#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF

I64 FPQ16Mul(I64 a, I64 b);

// Checked clamp helper used by scalar exp and vector exp callers.
// Clamps to the saturating exp input domain [-10.0, +10.0] in Q16.
I32 FPQ16ExpClampToInputDomainChecked(I64 input_q16,
                                      I64 *out_clamped_q16)
{
    if (!out_clamped_q16)
        return FP_Q16_ERR_NULL_PTR;

    // Protect against accidental constant inversion during refactors.
    if (EXP_Q16_MIN_INPUT > EXP_Q16_MAX_INPUT)
        return FP_Q16_ERR_BAD_PARAM;

    if (input_q16 < EXP_Q16_MIN_INPUT)
    {
        *out_clamped_q16 = EXP_Q16_MIN_INPUT;
        return FP_Q16_OK;
    }

    if (input_q16 > EXP_Q16_MAX_INPUT)
    {
        *out_clamped_q16 = EXP_Q16_MAX_INPUT;
        return FP_Q16_OK;
    }

    *out_clamped_q16 = input_q16;
    return FP_Q16_OK;
}

I64 FPQ16Exp(I64 x)
{
    I32 clamp_status;
    I64 clamped_x;
    I64 k;
    I64 r;
    I64 r2;
    I64 r3;
    I64 r4;
    I64 t1;
    I64 t2;
    I64 t3;
    I64 t4;
    I64 poly;

    clamp_status = FPQ16ExpClampToInputDomainChecked(x,
                                                     &clamped_x);
    if (clamp_status)
        return 0;

    // Keep scalar exp saturation behavior exactly explicit at boundaries.
    if (clamped_x >= EXP_Q16_MAX_INPUT)
        return I64_MAX_VALUE;
    if (clamped_x <= EXP_Q16_MIN_INPUT)
        return 0;

    // Range reduction: x = k*ln(2) + r, where r in [0, ln(2))
    k = clamped_x / EXP_Q16_LN2;
    if (clamped_x < 0 && (clamped_x % EXP_Q16_LN2))
        k--;

    r = clamped_x - (k * EXP_Q16_LN2);

    // exp(r) ~= 1 + r + r^2/2 + r^3/6 + r^4/24
    r2 = FPQ16Mul(r, r);
    r3 = FPQ16Mul(r2, r);
    r4 = FPQ16Mul(r3, r);

    t1 = r;
    t2 = r2 / 2;
    t3 = r3 / 6;
    t4 = r4 / 24;

    poly = FP_Q16_ONE + t1 + t2 + t3 + t4;

    if (k >= 0)
    {
        if (k >= 30)
            return I64_MAX_VALUE;

        if (poly > (I64_MAX_VALUE >> k))
            return I64_MAX_VALUE;

        return poly << k;
    }

    if (k <= -63)
        return 0;

    return poly >> (-k);
}
