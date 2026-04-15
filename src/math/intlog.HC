// intlog.HC
// Integer-only Q16.16 natural-log approximation for TempleOS HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define LOG_Q16_LN2  45426 // round(ln(2) * 65536)
#define LOG_Q16_INV_LN2 94549 // round((1/ln(2)) * 65536)
#define LOG_Q16_HALF (1 << (FP_Q16_SHIFT - 1))
#define LOG_Q16_SPLIT ((FP_Q16_ONE * 3) / 2) // 1.5 in Q16

#define I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define I64_MIN_VALUE -0x8000000000000000

I64 FPQ16Mul(I64 a, I64 b);

I64 FPQ16Ln1pPoly(I64 y_q16)
{
    I64 y2;
    I64 y3;
    I64 y4;
    I64 y5;

    // 5th-order alternating series:
    // ln(1+y) ~= y - y^2/2 + y^3/3 - y^4/4 + y^5/5
    // Accurate and monotonic on y in [-0.25, 0.5].
    y2 = FPQ16Mul(y_q16, y_q16);
    y3 = FPQ16Mul(y2, y_q16);
    y4 = FPQ16Mul(y3, y_q16);
    y5 = FPQ16Mul(y4, y_q16);

    return y_q16 - (y2 / 2) + (y3 / 3) - (y4 / 4) + (y5 / 5);
}

I64 FPQ16LnReduce(I64 x_q16, I64 *k_out)
{
    U64 m_q16;
    I64 k;

    if (!k_out)
        return 0;
    if (x_q16 <= 0)
    {
        *k_out = 0;
        return 0;
    }

    // Normalize x = m * 2^k where m in [1, 2) in Q16.
    m_q16 = (U64)x_q16;
    k = 0;

    while (m_q16 >= (((U64)FP_Q16_ONE) << 1))
    {
        m_q16 >>= 1;
        k++;
    }
    while (m_q16 < (U64)FP_Q16_ONE)
    {
        m_q16 <<= 1;
        k--;
    }

    *k_out = k;
    return (I64)m_q16;
}

I64 FPQ16Ln(I64 x_q16)
{
    I64 k;
    I64 m_q16;
    I64 y_q16;
    I64 poly_q16;
    I64 base_q16;
    I64 ln_q16;

    // ln(x) is undefined for x <= 0. Saturate to a deterministic floor.
    if (x_q16 <= 0)
        return I64_MIN_VALUE;

    m_q16 = FPQ16LnReduce(x_q16, &k);

    // Piecewise range split for monotonic polynomial accuracy:
    // branch A: y in [0, 0.5), branch B: y in [-0.25, 0).
    if (m_q16 >= LOG_Q16_SPLIT)
    {
        m_q16 = (m_q16 + 1) >> 1;
        k++;
    }

    y_q16 = m_q16 - FP_Q16_ONE;
    poly_q16 = FPQ16Ln1pPoly(y_q16);

    base_q16 = k * LOG_Q16_LN2;

    if (base_q16 > 0 && poly_q16 > I64_MAX_VALUE - base_q16)
        return I64_MAX_VALUE;
    if (base_q16 < 0 && poly_q16 < I64_MIN_VALUE - base_q16)
        return I64_MIN_VALUE;

    ln_q16 = base_q16 + poly_q16;
    return ln_q16;
}

I64 FPQ16Log2(I64 x_q16)
{
    I64 ln_q16;
    I64 abs_ln;
    I64 abs_inv;
    I64 max_abs_ln;
    I64 prod;

    ln_q16 = FPQ16Ln(x_q16);

    // Preserve ln(x<=0) deterministic domain floor.
    if (ln_q16 == I64_MIN_VALUE)
        return I64_MIN_VALUE;

    // log2(x) = ln(x) * (1/ln(2)).
    // Keep overflow checks explicit before multiply.
    abs_ln = ln_q16;
    if (abs_ln < 0)
        abs_ln = -abs_ln;

    abs_inv = LOG_Q16_INV_LN2;
    max_abs_ln = I64_MAX_VALUE / abs_inv;
    if (abs_ln > max_abs_ln)
    {
        if (ln_q16 < 0)
            return I64_MIN_VALUE;
        return I64_MAX_VALUE;
    }

    prod = ln_q16 * LOG_Q16_INV_LN2;

    // Q16*Q16 -> Q32. Round to Q16 with sign-aware half-up policy.
    if (prod >= 0)
        return (prod + LOG_Q16_HALF) >> FP_Q16_SHIFT;
    return -(((-prod) + LOG_Q16_HALF) >> FP_Q16_SHIFT);
}
