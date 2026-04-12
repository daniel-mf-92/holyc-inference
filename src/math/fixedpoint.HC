// fixedpoint.HC
// Q16.16 fixed-point helpers for integer-only inference math.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)
#define FP_Q16_HALF  (1 << (FP_Q16_SHIFT - 1))

I64 FPQ16Abs(I64 x)
{
    if (x < 0)
        return -x;
    return x;
}

I64 FPQ16FromInt(I64 x)
{
    return x << FP_Q16_SHIFT;
}

I64 FPQ16ToInt(I64 x)
{
    if (x >= 0)
        return (x + FP_Q16_HALF) >> FP_Q16_SHIFT;
    return -(((-x) + FP_Q16_HALF) >> FP_Q16_SHIFT);
}

I64 FPQ16Mul(I64 a, I64 b)
{
    I64 prod;

    prod = a * b;
    if (prod >= 0)
        return (prod + FP_Q16_HALF) >> FP_Q16_SHIFT;
    return -(((-prod) + FP_Q16_HALF) >> FP_Q16_SHIFT);
}

I64 FPQ16Div(I64 num, I64 den)
{
    I64 sign;
    I64 abs_num;
    I64 abs_den;
    I64 scaled;
    I64 q;

    if (!den)
        return 0;

    sign = 1;
    if (num < 0)
    {
        sign = -sign;
        num = -num;
    }
    if (den < 0)
    {
        sign = -sign;
        den = -den;
    }

    abs_num = num;
    abs_den = den;

    scaled = abs_num << FP_Q16_SHIFT;
    q = (scaled + (abs_den >> 1)) / abs_den;

    if (sign < 0)
        return -q;
    return q;
}
