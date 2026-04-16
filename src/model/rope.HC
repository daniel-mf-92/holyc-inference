// rope.HC
// Integer-only RoPE helpers for HolyC inference.
//
// RoPE frequency step math (for one rotary pair index i):
//   theta_step(i) = base^(-(2*i)/head_dim)
//                 = exp(-(2*i/head_dim) * ln(base))
//
// This file computes theta_step in Q16.16 with explicit checked arithmetic.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define ROPE_Q16_OK            0
#define ROPE_Q16_ERR_NULL_PTR  1
#define ROPE_Q16_ERR_BAD_PARAM 2
#define ROPE_Q16_ERR_DOMAIN    3
#define ROPE_Q16_ERR_OVERFLOW  4

#define ROPE_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define ROPE_I64_MIN_VALUE -0x8000000000000000
#define ROPE_U64_MAX_VALUE  0xFFFFFFFFFFFFFFFF

#define ROPE_Q16_PI       205887
#define ROPE_Q16_TWO_PI   411775
#define ROPE_Q16_HALF_PI  102944

// Declared in src/math/intlog.HC and src/math/intexp.HC.
I64 FPQ16Ln(I64 x_q16);
I64 FPQ16Exp(I64 x_q16);

U64 RoPEAbsToU64(I64 x)
{
    if (x >= 0)
        return (U64)x;

    // Keep I64_MIN representable using two's-complement identity.
    return ((U64)(-(x + 1))) + 1;
}

I32 RoPEApplySignFromMagnitudeChecked(U64 mag, Bool is_negative, I64 *out_value)
{
    if (!out_value)
        return ROPE_Q16_ERR_NULL_PTR;

    if (is_negative)
    {
        if (mag > 0x8000000000000000)
            return ROPE_Q16_ERR_OVERFLOW;

        if (mag == 0x8000000000000000)
        {
            *out_value = ROPE_I64_MIN_VALUE;
            return ROPE_Q16_OK;
        }

        *out_value = -(I64)mag;
        return ROPE_Q16_OK;
    }

    if (mag > (U64)ROPE_I64_MAX_VALUE)
        return ROPE_Q16_ERR_OVERFLOW;

    *out_value = (I64)mag;
    return ROPE_Q16_OK;
}

I32 RoPETryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return ROPE_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > ROPE_I64_MAX_VALUE - rhs)
        return ROPE_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < ROPE_I64_MIN_VALUE - rhs)
        return ROPE_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return ROPE_Q16_OK;
}

I32 RoPETryNegateI64Checked(I64 value, I64 *out_neg)
{
    if (!out_neg)
        return ROPE_Q16_ERR_NULL_PTR;

    if (value == ROPE_I64_MIN_VALUE)
        return ROPE_Q16_ERR_OVERFLOW;

    *out_neg = -value;
    return ROPE_Q16_OK;
}

I32 RoPETrySubI64Checked(I64 lhs, I64 rhs, I64 *out_diff)
{
    I64 neg_rhs;
    I32 status;

    if (!out_diff)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPETryNegateI64Checked(rhs,
                                     &neg_rhs);
    if (status != ROPE_Q16_OK)
        return status;

    return RoPETryAddI64Checked(lhs,
                                neg_rhs,
                                out_diff);
}

I32 RoPEQ16FromIntChecked(I64 x, I64 *out_q16)
{
    if (!out_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    if (x > (ROPE_I64_MAX_VALUE >> FP_Q16_SHIFT) || x < (ROPE_I64_MIN_VALUE >> FP_Q16_SHIFT))
        return ROPE_Q16_ERR_OVERFLOW;

    *out_q16 = x << FP_Q16_SHIFT;
    return ROPE_Q16_OK;
}

I32 RoPEQ16MulChecked(I64 lhs_q16, I64 rhs_q16, I64 *out_q16)
{
    U64 abs_lhs;
    U64 abs_rhs;
    U64 abs_prod;
    U64 round_bias;
    U64 rounded;
    Bool is_negative;

    if (!out_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    abs_lhs = RoPEAbsToU64(lhs_q16);
    abs_rhs = RoPEAbsToU64(rhs_q16);
    is_negative = ((lhs_q16 < 0) ^ (rhs_q16 < 0));

    if (abs_lhs && abs_rhs && abs_lhs > (ROPE_U64_MAX_VALUE / abs_rhs))
        return ROPE_Q16_ERR_OVERFLOW;

    abs_prod = abs_lhs * abs_rhs;
    round_bias = (U64)1 << (FP_Q16_SHIFT - 1);

    if (abs_prod > ROPE_U64_MAX_VALUE - round_bias)
        rounded = ROPE_U64_MAX_VALUE >> FP_Q16_SHIFT;
    else
        rounded = (abs_prod + round_bias) >> FP_Q16_SHIFT;

    return RoPEApplySignFromMagnitudeChecked(rounded, is_negative, out_q16);
}

I32 RoPEQ16DivChecked(I64 num_q16, I64 den_q16, I64 *out_q16)
{
    U64 abs_num;
    U64 abs_den;
    U64 limit;
    U64 int_part;
    U64 rem;
    U64 result_mag;
    Bool is_negative;
    I64 bit;

    if (!out_q16)
        return ROPE_Q16_ERR_NULL_PTR;
    if (!den_q16)
        return ROPE_Q16_ERR_DOMAIN;

    abs_num = RoPEAbsToU64(num_q16);
    abs_den = RoPEAbsToU64(den_q16);
    is_negative = ((num_q16 < 0) ^ (den_q16 < 0));

    limit = (U64)ROPE_I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    int_part = abs_num / abs_den;
    if (int_part > (limit >> FP_Q16_SHIFT))
        return ROPE_Q16_ERR_OVERFLOW;

    result_mag = int_part << FP_Q16_SHIFT;
    rem = abs_num % abs_den;

    // Emit 16 fractional bits by long division.
    for (bit = FP_Q16_SHIFT - 1; bit >= 0; bit--)
    {
        rem <<= 1;
        if (rem >= abs_den)
        {
            rem -= abs_den;
            if (result_mag > limit - (((U64)1) << bit))
                return ROPE_Q16_ERR_OVERFLOW;
            result_mag |= ((U64)1) << bit;
        }
    }

    // Half-up rounding when residual remainder is >= 0.5.
    if (rem >= ((abs_den + 1) >> 1))
    {
        if (result_mag == limit)
            return ROPE_Q16_ERR_OVERFLOW;
        result_mag++;
    }

    return RoPEApplySignFromMagnitudeChecked(result_mag, is_negative, out_q16);
}

I32 RoPEQ16DivByPositiveIntRoundedChecked(I64 num_q16, I64 den, I64 *out_q16)
{
    U64 abs_num;
    U64 den_u64;
    U64 q;
    U64 r;
    U64 limit;
    Bool is_negative;

    if (!out_q16)
        return ROPE_Q16_ERR_NULL_PTR;
    if (den <= 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    abs_num = RoPEAbsToU64(num_q16);
    den_u64 = (U64)den;
    is_negative = (num_q16 < 0);

    q = abs_num / den_u64;
    r = abs_num % den_u64;

    limit = (U64)ROPE_I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    if (q > limit)
        return ROPE_Q16_ERR_OVERFLOW;

    if (r >= ((den_u64 + 1) >> 1))
    {
        if (q == limit)
            return ROPE_Q16_ERR_OVERFLOW;
        q++;
    }

    return RoPEApplySignFromMagnitudeChecked(q,
                                             is_negative,
                                             out_q16);
}

I32 RoPEQ16NormalizeAnglePiChecked(I64 angle_q16,
                                   I64 *out_norm_q16)
{
    I64 reduced;
    I32 status;

    if (!out_norm_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    reduced = angle_q16 % ROPE_Q16_TWO_PI;

    if (reduced > ROPE_Q16_PI)
    {
        status = RoPETrySubI64Checked(reduced,
                                      ROPE_Q16_TWO_PI,
                                      &reduced);
        if (status != ROPE_Q16_OK)
            return status;
    }
    else if (reduced < -ROPE_Q16_PI)
    {
        status = RoPETryAddI64Checked(reduced,
                                      ROPE_Q16_TWO_PI,
                                      &reduced);
        if (status != ROPE_Q16_OK)
            return status;
    }

    *out_norm_q16 = reduced;
    return ROPE_Q16_OK;
}

// Integer-only sine approximation in Q16:
//   sin(x) ~= x - x^3/6 + x^5/120 - x^7/5040
// after reducing x to [-pi, pi] and reflecting into [-pi/2, pi/2].
I32 RoPEQ16SinApproxChecked(I64 angle_q16,
                            I64 *out_sin_q16)
{
    I64 x;
    I64 x2;
    I64 x3;
    I64 x5;
    I64 x7;
    I64 t1;
    I64 t2;
    I64 t3;
    I64 sum;
    I64 tmp;
    I32 status;

    if (!out_sin_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16NormalizeAnglePiChecked(angle_q16,
                                            &x);
    if (status != ROPE_Q16_OK)
        return status;

    if (x > ROPE_Q16_HALF_PI)
    {
        status = RoPETrySubI64Checked(ROPE_Q16_PI,
                                      x,
                                      &x);
        if (status != ROPE_Q16_OK)
            return status;
    }
    else if (x < -ROPE_Q16_HALF_PI)
    {
        status = RoPETrySubI64Checked(-ROPE_Q16_PI,
                                      x,
                                      &x);
        if (status != ROPE_Q16_OK)
            return status;
    }

    status = RoPEQ16MulChecked(x,
                               x,
                               &x2);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x2,
                               x,
                               &x3);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x3,
                               x2,
                               &x5);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x5,
                               x2,
                               &x7);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16DivByPositiveIntRoundedChecked(x3,
                                                   6,
                                                   &t1);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16DivByPositiveIntRoundedChecked(x5,
                                                   120,
                                                   &t2);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16DivByPositiveIntRoundedChecked(x7,
                                                   5040,
                                                   &t3);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETrySubI64Checked(x,
                                  t1,
                                  &sum);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(sum,
                                  t2,
                                  &tmp);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETrySubI64Checked(tmp,
                                  t3,
                                  out_sin_q16);
    return status;
}

I32 RoPEQ16CosApproxChecked(I64 angle_q16,
                            I64 *out_cos_q16)
{
    I64 shifted;
    I32 status;

    if (!out_cos_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPETryAddI64Checked(angle_q16,
                                  ROPE_Q16_HALF_PI,
                                  &shifted);
    if (status != ROPE_Q16_OK)
        return status;

    return RoPEQ16SinApproxChecked(shifted,
                                   out_cos_q16);
}

// Compute RoPE per-position angle increment for one rotary pair index:
//   step_q16 = exp(-(2*pair_index/head_dim) * ln(freq_base_q16))
//
// Contracts:
//   - freq_base_q16 must be > 0.
//   - head_dim must be positive and even.
//   - pair_index must satisfy 0 <= pair_index < head_dim/2.
I32 RoPEQ16AngleStepChecked(I64 freq_base_q16,
                            I64 head_dim,
                            I64 pair_index,
                            I64 *out_step_q16)
{
    I64 pair_count;
    I64 two_i;
    I64 two_i_q16;
    I64 head_dim_q16;
    I64 ratio_q16;
    I64 ln_base_q16;
    I64 exponent_mag_q16;
    I64 exponent_q16;
    I32 status;

    if (!out_step_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    if (freq_base_q16 <= 0)
        return ROPE_Q16_ERR_DOMAIN;
    if (head_dim <= 0 || (head_dim & 1))
        return ROPE_Q16_ERR_BAD_PARAM;
    if (pair_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    pair_count = head_dim >> 1;
    if (pair_index >= pair_count)
        return ROPE_Q16_ERR_BAD_PARAM;

    // two_i = 2 * pair_index (checked add avoids signed-mul overflow).
    status = RoPETryAddI64Checked(pair_index,
                                  pair_index,
                                  &two_i);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16FromIntChecked(two_i,
                                   &two_i_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16FromIntChecked(head_dim,
                                   &head_dim_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16DivChecked(two_i_q16,
                               head_dim_q16,
                               &ratio_q16);
    if (status != ROPE_Q16_OK)
        return status;

    ln_base_q16 = FPQ16Ln(freq_base_q16);
    if (ln_base_q16 == ROPE_I64_MIN_VALUE)
        return ROPE_Q16_ERR_DOMAIN;

    status = RoPEQ16MulChecked(ratio_q16,
                               ln_base_q16,
                               &exponent_mag_q16);
    if (status != ROPE_Q16_OK)
        return status;

    if (exponent_mag_q16 == ROPE_I64_MIN_VALUE)
        return ROPE_Q16_ERR_OVERFLOW;
    exponent_q16 = -exponent_mag_q16;

    *out_step_q16 = FPQ16Exp(exponent_q16);
    return ROPE_Q16_OK;
}

// Compute absolute rotary angle in radians (Q16) for one token position:
//   angle_q16 = position * angle_step_q16
//
// Contracts:
//   - angle_step_q16 must be > 0 (radians per position, Q16).
//   - position must be >= 0.
//   - checked Q16 multiply is used so overflow surfaces as ERR_OVERFLOW.
I32 RoPEQ16AngleForPositionChecked(I64 angle_step_q16,
                                   I64 position,
                                   I64 *out_angle_q16)
{
    I64 position_q16;
    I32 status;

    if (!out_angle_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    if (angle_step_q16 <= 0)
        return ROPE_Q16_ERR_DOMAIN;
    if (position < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    // Convert integer token position to Q16 once, then reuse checked Q16 mul.
    status = RoPEQ16FromIntChecked(position,
                                   &position_q16);
    if (status != ROPE_Q16_OK)
        return status;

    return RoPEQ16MulChecked(position_q16,
                             angle_step_q16,
                             out_angle_q16);
}

// Rotate one Q16 vector pair by a RoPE angle using integer sin/cos approximations:
//   x' = x*cos(theta) - y*sin(theta)
//   y' = x*sin(theta) + y*cos(theta)
I32 RoPEQ16RotatePairChecked(I64 x_q16,
                             I64 y_q16,
                             I64 angle_q16,
                             I64 *out_x_rot_q16,
                             I64 *out_y_rot_q16)
{
    I64 sin_q16;
    I64 cos_q16;
    I64 x_cos;
    I64 y_sin;
    I64 x_sin;
    I64 y_cos;
    I32 status;

    if (!out_x_rot_q16 || !out_y_rot_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16SinApproxChecked(angle_q16,
                                     &sin_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16CosApproxChecked(angle_q16,
                                     &cos_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x_q16,
                               cos_q16,
                               &x_cos);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(y_q16,
                               sin_q16,
                               &y_sin);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x_q16,
                               sin_q16,
                               &x_sin);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(y_q16,
                               cos_q16,
                               &y_cos);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETrySubI64Checked(x_cos,
                                  y_sin,
                                  out_x_rot_q16);
    if (status != ROPE_Q16_OK)
        return status;

    return RoPETryAddI64Checked(x_sin,
                                y_cos,
                                out_y_rot_q16);
}

// Compose full one-step rotary application for one pair at one token position:
//   step  = base^(-(2*pair_index)/head_dim)
//   angle = position * step
//   (x',y') = rotate((x,y), angle)
//
// Contracts:
//   - freq_base_q16 must be > 0.
//   - head_dim must be positive and even.
//   - pair_index must be in [0, head_dim/2).
//   - position must be >= 0.
I32 RoPEQ16RotatePairByPositionChecked(I64 x_q16,
                                       I64 y_q16,
                                       I64 freq_base_q16,
                                       I64 head_dim,
                                       I64 pair_index,
                                       I64 position,
                                       I64 *out_x_rot_q16,
                                       I64 *out_y_rot_q16)
{
    I64 angle_step_q16;
    I64 angle_q16;
    I32 status;

    if (!out_x_rot_q16 || !out_y_rot_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16AngleStepChecked(freq_base_q16,
                                     head_dim,
                                     pair_index,
                                     &angle_step_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16AngleForPositionChecked(angle_step_q16,
                                            position,
                                            &angle_q16);
    if (status != ROPE_Q16_OK)
        return status;

    return RoPEQ16RotatePairChecked(x_q16,
                                    y_q16,
                                    angle_q16,
                                    out_x_rot_q16,
                                    out_y_rot_q16);
}
