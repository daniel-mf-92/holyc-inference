// intsqrt.HC
// Integer square-root helpers for Q16.16 fixed-point inference math.

#define FP_Q16_SHIFT 16
#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF

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
