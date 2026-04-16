// fixedpoint.HC
// Q16.16 fixed-point helpers for integer-only inference math.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)
#define FP_Q16_HALF  (1 << (FP_Q16_SHIFT - 1))

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_DOMAIN    3
#define FP_Q16_ERR_OVERFLOW  4

#define I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define I64_MIN_VALUE -0x8000000000000000
#define U64_MAX_VALUE  0xFFFFFFFFFFFFFFFF

U64 FPAbsToU64(I64 x)
{
    if (x >= 0)
        return (U64)x;

    // Avoid overflow for I64_MIN by using two's-complement identity.
    return ((U64)(-(x + 1))) + 1;
}

I64 FPSignApplyFromU64(U64 mag, I64 is_negative)
{
    if (is_negative)
    {
        if (mag >= 0x8000000000000000)
            return I64_MIN_VALUE;
        return -(I64)mag;
    }

    if (mag > (U64)I64_MAX_VALUE)
        return I64_MAX_VALUE;
    return (I64)mag;
}

I32 FPTryApplySignFromU64Checked(U64 mag, I64 is_negative, I64 *out_value)
{
    U64 limit;

    if (!out_value)
        return FP_Q16_ERR_NULL_PTR;

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    if (mag > limit)
        return FP_Q16_ERR_OVERFLOW;

    *out_value = FPSignApplyFromU64(mag, is_negative);
    return FP_Q16_OK;
}

I64 FPQ16RoundShiftSigned(I64 value, I64 shift)
{
    U64 abs_value;
    U64 round_bias;
    U64 rounded;
    I64 is_negative;

    if (shift <= 0)
        return value;
    if (shift >= 63)
        return 0;

    is_negative = value < 0;
    abs_value = FPAbsToU64(value);
    round_bias = (U64)1 << (shift - 1);

    if (abs_value > U64_MAX_VALUE - round_bias)
        rounded = U64_MAX_VALUE >> shift;
    else
        rounded = (abs_value + round_bias) >> shift;

    return FPSignApplyFromU64(rounded, is_negative);
}

I64 FPQ16MulSat(I64 a, I64 b)
{
    U64 abs_a;
    U64 abs_b;
    U64 limit;
    I64 is_negative;
    I64 prod;

    if (!a || !b)
        return 0;

    abs_a = FPAbsToU64(a);
    abs_b = FPAbsToU64(b);
    is_negative = ((a < 0) ^ (b < 0));

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    if (abs_a > (limit / abs_b))
    {
        if (is_negative)
            return I64_MIN_VALUE;
        return I64_MAX_VALUE;
    }

    prod = a * b;
    return FPQ16RoundShiftSigned(prod, FP_Q16_SHIFT);
}

I64 FPQ16Abs(I64 x)
{
    if (x == I64_MIN_VALUE)
        return I64_MAX_VALUE;

    if (x < 0)
        return -x;
    return x;
}

I64 FPQ16FromInt(I64 x)
{
    if (x > (I64_MAX_VALUE >> FP_Q16_SHIFT))
        return I64_MAX_VALUE;
    if (x < (I64_MIN_VALUE >> FP_Q16_SHIFT))
        return I64_MIN_VALUE;

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
    return FPQ16MulSat(a, b);
}

I64 FPQ16Div(I64 num, I64 den)
{
    U64 abs_num;
    U64 abs_den;
    U64 limit;
    U64 int_part;
    U64 rem;
    U64 result_mag;
    I64 is_negative;
    I64 bit;

    if (!den)
        return 0;

    abs_num = FPAbsToU64(num);
    abs_den = FPAbsToU64(den);
    is_negative = ((num < 0) ^ (den < 0));

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    int_part = abs_num / abs_den;
    if (int_part > (limit >> FP_Q16_SHIFT))
    {
        if (is_negative)
            return I64_MIN_VALUE;
        return I64_MAX_VALUE;
    }

    result_mag = int_part << FP_Q16_SHIFT;
    rem = abs_num % abs_den;

    // Emit 16 fractional bits using integer long division.
    for (bit = FP_Q16_SHIFT - 1; bit >= 0; bit--)
    {
        rem <<= 1;
        if (rem >= abs_den)
        {
            rem -= abs_den;
            if (result_mag <= limit - (((U64)1) << bit))
                result_mag |= ((U64)1) << bit;
            else
                result_mag = limit;
        }
    }

    // Half-up rounding by checking whether residual remainder is >= 0.5.
    if (rem >= ((abs_den + 1) >> 1))
    {
        if (result_mag < limit)
            result_mag++;
        else
            result_mag = limit;
    }

    return FPSignApplyFromU64(result_mag, is_negative);
}

// Checked one-rounding Q16 helper for:
//   out_q16 = round_half_up((a_q16 * b_q16) / d_q16)
//
// Inputs/outputs are Q16.16 integers, so this computes the exact Q16 result
// of (a*b)/d without an intermediate Q16 downshift between mul and div.
// That avoids double-rounding drift in RMSNorm/softmax scale paths.
I32 FPQ16MulDivRoundedChecked(I64 a_q16,
                              I64 b_q16,
                              I64 d_q16,
                              I64 *out_q16)
{
    U64 abs_a;
    U64 abs_b;
    U64 abs_d;
    U64 abs_num;
    U64 q;
    U64 r;
    U64 limit;
    I64 is_negative;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (!d_q16)
        return FP_Q16_ERR_DOMAIN;

    abs_a = FPAbsToU64(a_q16);
    abs_b = FPAbsToU64(b_q16);
    abs_d = FPAbsToU64(d_q16);
    is_negative = ((a_q16 < 0) ^ (b_q16 < 0) ^ (d_q16 < 0));

    if (abs_a && abs_b && abs_a > (U64_MAX_VALUE / abs_b))
        return FP_Q16_ERR_OVERFLOW;

    abs_num = abs_a * abs_b;
    q = abs_num / abs_d;
    r = abs_num % abs_d;

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    if (q > limit)
        return FP_Q16_ERR_OVERFLOW;

    // Half-up rounding: bump quotient when residual >= 0.5 denominator.
    if (r >= ((abs_d + 1) >> 1))
    {
        if (q == limit)
            return FP_Q16_ERR_OVERFLOW;
        q++;
    }

    return FPTryApplySignFromU64Checked(q,
                                        is_negative,
                                        out_q16);
}

// Checked one-rounding Q16 helper for positive integer divisors:
//   out_q16 = round_half_up((a_q16 * b_q16) / d_int)
// where d_int is an integer (not Q16) with d_int > 0.
//
// Since a_q16*b_q16 is Q32, dividing by d_int keeps Q32. Returning Q16 means
// one final divide by 2^16. We combine both denominators into:
//   denom = d_int * 2^16
// and perform a single rounded integer divide to avoid double-rounding drift.
I32 FPQ16MulDivRoundedByPositiveIntChecked(I64 a_q16,
                                           I64 b_q16,
                                           I64 d_int,
                                           I64 *out_q16)
{
    U64 abs_a;
    U64 abs_b;
    U64 abs_num;
    U64 abs_den_q16;
    U64 q;
    U64 r;
    U64 limit;
    I64 is_negative;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (d_int <= 0)
        return FP_Q16_ERR_DOMAIN;

    if ((U64)d_int > (U64_MAX_VALUE >> FP_Q16_SHIFT))
        return FP_Q16_ERR_OVERFLOW;

    abs_a = FPAbsToU64(a_q16);
    abs_b = FPAbsToU64(b_q16);
    is_negative = ((a_q16 < 0) ^ (b_q16 < 0));

    if (abs_a && abs_b && abs_a > (U64_MAX_VALUE / abs_b))
        return FP_Q16_ERR_OVERFLOW;

    abs_num = abs_a * abs_b;
    abs_den_q16 = ((U64)d_int) << FP_Q16_SHIFT;

    q = abs_num / abs_den_q16;
    r = abs_num % abs_den_q16;

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    if (q > limit)
        return FP_Q16_ERR_OVERFLOW;

    // Half-up rounding: bump quotient when residual >= 0.5 denominator.
    if (r >= ((abs_den_q16 + 1) >> 1))
    {
        if (q == limit)
            return FP_Q16_ERR_OVERFLOW;
        q++;
    }

    return FPTryApplySignFromU64Checked(q,
                                        is_negative,
                                        out_q16);
}

// Checked wrapper for positive integer divisors encoded as Q16 whole values:
//   d_q16 = d_int << 16, d_int > 0
//
// This enforces the encoded-denominator contract used by RMSNorm call sites:
// - denominator must be positive
// - denominator must be an exact Q16 whole integer (no fractional bits)
// - then route to FPQ16MulDivRoundedByPositiveIntChecked for single-rounding
//   (a_q16*b_q16)/(d_int*2^16) semantics.
I32 FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(I64 a_q16,
                                                     I64 b_q16,
                                                     I64 d_q16,
                                                     I64 *out_q16)
{
    I64 d_int;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (d_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    // Strict encoding contract: denominator is an integer represented in Q16.
    if (d_q16 & (FP_Q16_ONE - 1))
        return FP_Q16_ERR_BAD_PARAM;

    d_int = d_q16 >> FP_Q16_SHIFT;
    if (d_int <= 0)
        return FP_Q16_ERR_DOMAIN;

    return FPQ16MulDivRoundedByPositiveIntChecked(a_q16,
                                                  b_q16,
                                                  d_int,
                                                  out_q16);
}
