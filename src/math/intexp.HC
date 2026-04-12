// intexp.HC
// Integer-only Q16.16 exponential approximation for TempleOS HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define EXP_Q16_LN2       45426   // round(ln(2) * 65536)
#define EXP_Q16_MAX_INPUT 655360  // +10.0 in Q16
#define EXP_Q16_MIN_INPUT -655360 // -10.0 in Q16

#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF

I64 FPQ16Mul(I64 a, I64 b);

I64 FPQ16Exp(I64 x)
{
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

    if (x >= EXP_Q16_MAX_INPUT)
        return I64_MAX_VALUE;
    if (x <= EXP_Q16_MIN_INPUT)
        return 0;

    // Range reduction: x = k*ln(2) + r, where r in [0, ln(2))
    k = x / EXP_Q16_LN2;
    if (x < 0 && (x % EXP_Q16_LN2))
        k--;

    r = x - (k * EXP_Q16_LN2);

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
