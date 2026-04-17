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
I32 RoPEQ16RotateHeadRangeByTokenWindowPreflightedHeadBasesChecked(I64 *head_cells_q16,
                                                                   I64 head_cell_capacity,
                                                                   I64 range_base_index,
                                                                   I64 head_count,
                                                                   I64 head_stride_cells,
                                                                   I64 head_dim,
                                                                   I64 pair_stride_cells,
                                                                   I64 freq_base_q16,
                                                                   I64 position_start,
                                                                   I64 token_count,
                                                                   I64 position_step,
                                                                   I64 *head_base_cache,
                                                                   I64 head_base_cache_capacity);

I32 RoPEQ16RotateHeadRangeByPositionCheckedStrided(I64 *head_cells_q16,
                                                   I64 head_cell_capacity,
                                                   I64 token_base_index,
                                                   I64 token_stride_q16,
                                                   I64 token_index,
                                                   I64 range_base_index,
                                                   I64 head_count,
                                                   I64 head_stride_cells,
                                                   I64 head_dim,
                                                   I64 pair_stride_cells,
                                                   I64 freq_base_q16,
                                                   I64 position);
I32 RoPEQ16RotateHeadRangeByTokenWindowCheckedStrided(I64 *head_cells_q16,
                                                      I64 head_cell_capacity,
                                                      I64 token_base_index,
                                                      I64 token_stride_q16,
                                                      I64 range_base_index,
                                                      I64 head_count,
                                                      I64 head_stride_cells,
                                                      I64 head_dim,
                                                      I64 pair_stride_cells,
                                                      I64 freq_base_q16,
                                                      I64 position_start,
                                                      I64 token_count,
                                                      I64 position_step);

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

I32 RoPETryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    U64 abs_lhs;
    U64 abs_rhs;
    U64 abs_prod;
    Bool is_negative;

    if (!out_prod)
        return ROPE_Q16_ERR_NULL_PTR;

    abs_lhs = RoPEAbsToU64(lhs);
    abs_rhs = RoPEAbsToU64(rhs);
    is_negative = ((lhs < 0) ^ (rhs < 0));

    if (abs_lhs && abs_rhs && abs_lhs > (ROPE_U64_MAX_VALUE / abs_rhs))
        return ROPE_Q16_ERR_OVERFLOW;

    abs_prod = abs_lhs * abs_rhs;
    return RoPEApplySignFromMagnitudeChecked(abs_prod,
                                             is_negative,
                                             out_prod);
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
    if (!out_angle_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    if (angle_step_q16 <= 0)
        return ROPE_Q16_ERR_DOMAIN;
    if (position < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    // angle_step_q16 is already radians-per-position in Q16, so multiplying
    // by an integer position is exact in Q16 space (no extra <<16 conversion).
    // This also avoids rejecting large-but-valid positions that only overflow
    // in an intermediate `position<<16`, not in the final `position*step`.
    return RoPETryMulI64Checked(angle_step_q16,
                                position,
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

// Rotate all Q16 `(x,y)` pairs for one attention head slice in-place.
// Layout contract:
//   - For pair index `i`, `x` is at `head_base_index + i*pair_stride_cells`.
//   - The corresponding `y` is at `x_index + 1`.
// This mirrors interleaved RoPE layouts like `[x0,y0,x1,y1,...]` when stride=2,
// and also supports wider-stride packed layouts used by caller-owned buffers.
I32 RoPEQ16ValidateHeadSpanForDimChecked(I64 head_cell_capacity,
                                         I64 head_base_index,
                                         I64 head_dim,
                                         I64 pair_stride_cells,
                                         I64 *out_last_pair_y_index)
{
    I64 pair_count;
    I64 last_pair_index;
    I64 last_pair_offset;
    I64 last_pair_x_index;
    I64 last_pair_y_index;
    I32 status;

    if (!out_last_pair_y_index)
        return ROPE_Q16_ERR_NULL_PTR;

    if (head_cell_capacity < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_base_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_dim <= 0 || (head_dim & 1))
        return ROPE_Q16_ERR_BAD_PARAM;
    if (pair_stride_cells < 2)
        return ROPE_Q16_ERR_BAD_PARAM;

    if (head_base_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    pair_count = head_dim >> 1;

    // Checked end-span math for one head:
    //   last_pair_index = (head_dim/2)-1
    //   last_pair_x = head_base + last_pair_index*pair_stride
    //   last_pair_y = last_pair_x + 1
    // Require last_pair_y < head_cell_capacity.
    status = RoPETrySubI64Checked(pair_count,
                                  1,
                                  &last_pair_index);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryMulI64Checked(last_pair_index,
                                  pair_stride_cells,
                                  &last_pair_offset);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(head_base_index,
                                  last_pair_offset,
                                  &last_pair_x_index);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(last_pair_x_index,
                                  1,
                                  &last_pair_y_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (last_pair_x_index < 0 || last_pair_y_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (last_pair_y_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    *out_last_pair_y_index = last_pair_y_index;
    return ROPE_Q16_OK;
}

I32 RoPEQ16RotateHeadByPositionChecked(I64 *head_cells_q16,
                                       I64 head_cell_capacity,
                                       I64 head_base_index,
                                       I64 head_dim,
                                       I64 pair_stride_cells,
                                       I64 freq_base_q16,
                                       I64 position)
{
    I64 pair_count;
    I64 pair_index;
    I64 last_pair_y_index;
    I64 pair_offset;
    I64 x_index;
    I64 y_index;
    I64 x_q16;
    I64 y_q16;
    I64 x_rot_q16;
    I64 y_rot_q16;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16ValidateHeadSpanForDimChecked(head_cell_capacity,
                                                  head_base_index,
                                                  head_dim,
                                                  pair_stride_cells,
                                                  &last_pair_y_index);
    if (status != ROPE_Q16_OK)
        return status;

    pair_count = head_dim >> 1;

    for (pair_index = 0; pair_index < pair_count; pair_index++)
    {
        // Checked index math keeps every address derivation auditable:
        //   pair_offset = pair_index * pair_stride
        //   x_index = head_base + pair_offset
        //   y_index = x_index + 1
        status = RoPETryMulI64Checked(pair_index,
                                      pair_stride_cells,
                                      &pair_offset);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryAddI64Checked(head_base_index,
                                      pair_offset,
                                      &x_index);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryAddI64Checked(x_index,
                                      1,
                                      &y_index);
        if (status != ROPE_Q16_OK)
            return status;

        if (x_index < 0 || y_index < 0)
            return ROPE_Q16_ERR_BAD_PARAM;
        if (x_index > last_pair_y_index || y_index > last_pair_y_index)
            return ROPE_Q16_ERR_BAD_PARAM;

        x_q16 = head_cells_q16[x_index];
        y_q16 = head_cells_q16[y_index];

        status = RoPEQ16RotatePairByPositionChecked(x_q16,
                                                    y_q16,
                                                    freq_base_q16,
                                                    head_dim,
                                                    pair_index,
                                                    position,
                                                    &x_rot_q16,
                                                    &y_rot_q16);
        if (status != ROPE_Q16_OK)
            return status;

        head_cells_q16[x_index] = x_rot_q16;
        head_cells_q16[y_index] = y_rot_q16;
    }

    return ROPE_Q16_OK;
}

// Compute one head base index inside a contiguous head range:
//   head_base = range_base_index + head_index * head_stride_cells
//
// This isolates checked multiply+add arithmetic so multi-head kernels can
// share one auditable contract for base-address derivation.
I32 RoPEQ16ComputeHeadBaseIndexChecked(I64 range_base_index,
                                       I64 head_index,
                                       I64 head_stride_cells,
                                       I64 *out_head_base_index)
{
    I64 head_offset;
    I32 status;

    if (!out_head_base_index)
        return ROPE_Q16_ERR_NULL_PTR;

    if (range_base_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_stride_cells <= 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    status = RoPETryMulI64Checked(head_index,
                                  head_stride_cells,
                                  &head_offset);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(range_base_index,
                                  head_offset,
                                  out_head_base_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (*out_head_base_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    return ROPE_Q16_OK;
}

// Validate a contiguous multi-head range before any in-place rotation.
//
// Range contract:
//   head_base(h) = range_base_index + h * head_stride_cells, for h in [0, head_count)
//
// This helper centralizes all checked preflight math for head-range traversal so
// callers can fail fast on malformed capacities and arithmetic overflow before
// touching tensor cells.
I32 RoPEQ16ValidateHeadRangeCapacityChecked(I64 head_cell_capacity,
                                            I64 range_base_index,
                                            I64 head_count,
                                            I64 head_stride_cells,
                                            I64 *out_last_head_base_index)
{
    I64 last_head_index;
    I64 last_head_base_index;
    I32 status;

    if (!out_last_head_base_index)
        return ROPE_Q16_ERR_NULL_PTR;

    if (head_cell_capacity < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (range_base_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_count < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_stride_cells <= 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    if (!head_count)
    {
        // Empty range is a no-op. No capacity touch occurs.
        *out_last_head_base_index = range_base_index;
        return ROPE_Q16_OK;
    }

    if (range_base_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    // Tail-base validation catches overflow and far-end capacity faults once,
    // before the rotation loop starts.
    status = RoPETrySubI64Checked(head_count,
                                  1,
                                  &last_head_index);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16ComputeHeadBaseIndexChecked(range_base_index,
                                                last_head_index,
                                                head_stride_cells,
                                                &last_head_base_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (last_head_base_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    *out_last_head_base_index = last_head_base_index;
    return ROPE_Q16_OK;
}

// Preflight full per-head RoPE span bounds across a contiguous head range.
// For each head `h` in `[0, head_count)` this validates:
//   head_base(h) = range_base_index + h*head_stride_cells
//   last_pair_y(h) = head_base(h) + ((head_dim/2)-1)*pair_stride_cells + 1
// with checked arithmetic and `last_pair_y(h) < head_cell_capacity`.
//
// This runs before multi-head rotation loops so callers can fail fast on
// malformed head ranges/dimensions without mutating tensor cells.
I32 RoPEQ16ValidateHeadRangeSpanForDimChecked(I64 head_cell_capacity,
                                              I64 range_base_index,
                                              I64 head_count,
                                              I64 head_stride_cells,
                                              I64 head_dim,
                                              I64 pair_stride_cells,
                                              I64 *out_last_head_last_pair_y_index)
{
    I64 head_index;
    I64 head_base_index;
    I64 last_head_base_index;
    I64 last_pair_y_index;
    I32 status;

    if (!out_last_head_last_pair_y_index)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16ValidateHeadRangeCapacityChecked(head_cell_capacity,
                                                     range_base_index,
                                                     head_count,
                                                     head_stride_cells,
                                                     &last_head_base_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (!head_count)
    {
        // Empty range is a no-op and has no per-head pair span.
        *out_last_head_last_pair_y_index = range_base_index;
        return ROPE_Q16_OK;
    }

    for (head_index = 0; head_index < head_count; head_index++)
    {
        status = RoPEQ16ComputeHeadBaseIndexChecked(range_base_index,
                                                    head_index,
                                                    head_stride_cells,
                                                    &head_base_index);
        if (status != ROPE_Q16_OK)
            return status;

        if (head_base_index > last_head_base_index)
            return ROPE_Q16_ERR_BAD_PARAM;

        status = RoPEQ16ValidateHeadSpanForDimChecked(head_cell_capacity,
                                                      head_base_index,
                                                      head_dim,
                                                      pair_stride_cells,
                                                      &last_pair_y_index);
        if (status != ROPE_Q16_OK)
            return status;
    }

    *out_last_head_last_pair_y_index = last_pair_y_index;
    return ROPE_Q16_OK;
}

// Rotate a contiguous range of attention heads in-place at one token position.
// For head index `h` in `[0, head_count)`, this derives:
//   head_base = range_base_index + h*head_stride_cells
// and applies RoPE pair rotation across that head slice via
// `RoPEQ16RotateHeadByPositionChecked`.
I32 RoPEQ16RotateHeadRangeByPositionChecked(I64 *head_cells_q16,
                                            I64 head_cell_capacity,
                                            I64 range_base_index,
                                            I64 head_count,
                                            I64 head_stride_cells,
                                            I64 head_dim,
                                            I64 pair_stride_cells,
                                            I64 freq_base_q16,
                                            I64 position)
{
    I64 head_index;
    I64 head_base_index;
    I64 last_head_base_index;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16ValidateHeadRangeCapacityChecked(head_cell_capacity,
                                                     range_base_index,
                                                     head_count,
                                                     head_stride_cells,
                                                     &last_head_base_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (!head_count)
        return ROPE_Q16_OK;

    for (head_index = 0; head_index < head_count; head_index++)
    {
        status = RoPEQ16ComputeHeadBaseIndexChecked(range_base_index,
                                                    head_index,
                                                    head_stride_cells,
                                                    &head_base_index);
        if (status != ROPE_Q16_OK)
            return status;

        if (head_base_index > last_head_base_index)
            return ROPE_Q16_ERR_BAD_PARAM;

        if (head_base_index < 0 || head_base_index >= head_cell_capacity)
            return ROPE_Q16_ERR_BAD_PARAM;

        status = RoPEQ16RotateHeadByPositionChecked(head_cells_q16,
                                                    head_cell_capacity,
                                                    head_base_index,
                                                    head_dim,
                                                    pair_stride_cells,
                                                    freq_base_q16,
                                                    position);
        if (status != ROPE_Q16_OK)
            return status;
    }

    return ROPE_Q16_OK;
}

// Rotate one head-range for a token stored in a strided KV-cache slab layout.
//
// Absolute range base derivation (all checked integer math):
//   token_base = token_base_index + token_index * token_stride_q16
//   abs_range_base = token_base + range_base_index
//
// After deriving `abs_range_base`, this helper composes the same preflighted
// contiguous-head path used elsewhere, so range/span validation and no-partial-
// write behavior stay identical to `RoPEQ16RotateHeadRangeByPositionPreflightedChecked`.
I32 RoPEQ16RotateHeadRangeByPositionCheckedStrided(I64 *head_cells_q16,
                                                   I64 head_cell_capacity,
                                                   I64 token_base_index,
                                                   I64 token_stride_q16,
                                                   I64 token_index,
                                                   I64 range_base_index,
                                                   I64 head_count,
                                                   I64 head_stride_cells,
                                                   I64 head_dim,
                                                   I64 pair_stride_cells,
                                                   I64 freq_base_q16,
                                                   I64 position)
{
    I64 token_offset;
    I64 token_base;
    I64 abs_range_base;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;
    if (token_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (token_stride_q16 < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    status = RoPETryMulI64Checked(token_index,
                                  token_stride_q16,
                                  &token_offset);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(token_base_index,
                                  token_offset,
                                  &token_base);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(token_base,
                                  range_base_index,
                                  &abs_range_base);
    if (status != ROPE_Q16_OK)
        return status;

    return RoPEQ16RotateHeadRangeByPositionPreflightedChecked(head_cells_q16,
                                                               head_cell_capacity,
                                                               abs_range_base,
                                                               head_count,
                                                               head_stride_cells,
                                                               head_dim,
                                                               pair_stride_cells,
                                                               freq_base_q16,
                                                               position);
}

// Rotate a contiguous range of heads after full range preflight.
//
// Preflight contract:
//   1) Validate range base/stride/capacity across all heads.
//   2) Validate per-head RoPE pair span (`head_dim`, `pair_stride_cells`) across
//      the same range.
//
// Only after both checks pass does this helper apply per-head rotation. This
// gives callers a fail-fast path that guarantees no partial mutation from range
// shape/capacity faults discovered mid-loop.
I32 RoPEQ16RotateHeadRangeByPositionPreflightedChecked(I64 *head_cells_q16,
                                                       I64 head_cell_capacity,
                                                       I64 range_base_index,
                                                       I64 head_count,
                                                       I64 head_stride_cells,
                                                       I64 head_dim,
                                                       I64 pair_stride_cells,
                                                       I64 freq_base_q16,
                                                       I64 position)
{
    I64 head_index;
    I64 head_base_index;
    I64 last_head_base_index;
    I64 last_head_last_pair_y_index;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    status = RoPEQ16ValidateHeadRangeCapacityChecked(head_cell_capacity,
                                                     range_base_index,
                                                     head_count,
                                                     head_stride_cells,
                                                     &last_head_base_index);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16ValidateHeadRangeSpanForDimChecked(head_cell_capacity,
                                                       range_base_index,
                                                       head_count,
                                                       head_stride_cells,
                                                       head_dim,
                                                       pair_stride_cells,
                                                       &last_head_last_pair_y_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (!head_count)
        return ROPE_Q16_OK;

    for (head_index = 0; head_index < head_count; head_index++)
    {
        status = RoPEQ16ComputeHeadBaseIndexChecked(range_base_index,
                                                    head_index,
                                                    head_stride_cells,
                                                    &head_base_index);
        if (status != ROPE_Q16_OK)
            return status;

        if (head_base_index > last_head_base_index)
            return ROPE_Q16_ERR_BAD_PARAM;

        status = RoPEQ16RotateHeadByPositionChecked(head_cells_q16,
                                                    head_cell_capacity,
                                                    head_base_index,
                                                    head_dim,
                                                    pair_stride_cells,
                                                    freq_base_q16,
                                                    position);
        if (status != ROPE_Q16_OK)
            return status;
    }

    // Postcondition echoes preflight ceiling to keep behavior auditable.
    if (last_head_last_pair_y_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    return ROPE_Q16_OK;
}

// Rotate one head-range across a window of token positions.
//
// The same head-range shape/capacity contract is preflighted once, then reused
// for every token position in the window. Position math is fully checked:
//
//   position(token_index) = position_start + token_index * position_step
//
// This keeps all token-position stepping in explicit integer arithmetic and
// returns overflow/domain errors before mutating the failing token iteration.
I32 RoPEQ16RotateHeadRangeByTokenWindowPreflightedChecked(I64 *head_cells_q16,
                                                          I64 head_cell_capacity,
                                                          I64 range_base_index,
                                                          I64 head_count,
                                                          I64 head_stride_cells,
                                                          I64 head_dim,
                                                          I64 pair_stride_cells,
                                                          I64 freq_base_q16,
                                                          I64 position_start,
                                                          I64 token_count,
                                                          I64 position_step)
{
    I64 head_base_cache_bytes;
    I64 *head_base_cache;
    I64 head_base_dummy;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;
    if (token_count < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    // No work path still uses the same preflighted execution helper with
    // an always-valid scratch pointer so return/error surfaces stay shared.
    if (!token_count || !head_count)
        return RoPEQ16RotateHeadRangeByTokenWindowPreflightedHeadBasesChecked(head_cells_q16,
                                                                               head_cell_capacity,
                                                                               range_base_index,
                                                                               head_count,
                                                                               head_stride_cells,
                                                                               head_dim,
                                                                               pair_stride_cells,
                                                                               freq_base_q16,
                                                                               position_start,
                                                                               token_count,
                                                                               position_step,
                                                                               &head_base_dummy,
                                                                               0);

    status = RoPETryMulI64Checked(head_count,
                                  sizeof(I64),
                                  &head_base_cache_bytes);
    if (status != ROPE_Q16_OK)
        return status;

    head_base_cache = MAlloc(head_base_cache_bytes);
    if (!head_base_cache)
        return ROPE_Q16_ERR_OVERFLOW;

    status = RoPEQ16RotateHeadRangeByTokenWindowPreflightedHeadBasesChecked(head_cells_q16,
                                                                             head_cell_capacity,
                                                                             range_base_index,
                                                                             head_count,
                                                                             head_stride_cells,
                                                                             head_dim,
                                                                             pair_stride_cells,
                                                                             freq_base_q16,
                                                                             position_start,
                                                                             token_count,
                                                                             position_step,
                                                                             head_base_cache,
                                                                             head_count);
    Free(head_base_cache);
    return status;
}

// Rotate one head-range across a token window with precomputed head bases.
//
// This variant is built for caller-owned scratch buffers so head-base index
// math is done once:
//   head_base_cache[h] = range_base_index + h*head_stride_cells
// Then every token iteration only performs checked position stepping plus
// per-head pair rotation using cached base indices.
I32 RoPEQ16RotateHeadRangeByTokenWindowPreflightedHeadBasesChecked(I64 *head_cells_q16,
                                                                   I64 head_cell_capacity,
                                                                   I64 range_base_index,
                                                                   I64 head_count,
                                                                   I64 head_stride_cells,
                                                                   I64 head_dim,
                                                                   I64 pair_stride_cells,
                                                                   I64 freq_base_q16,
                                                                   I64 position_start,
                                                                   I64 token_count,
                                                                   I64 position_step,
                                                                   I64 *head_base_cache,
                                                                   I64 head_base_cache_capacity)
{
    I64 token_index;
    I64 head_index;
    I64 token_position_offset;
    I64 token_position;
    I64 head_base_index;
    I64 last_head_base_index;
    I64 last_head_last_pair_y_index;
    I32 status;

    if (!head_cells_q16 || !head_base_cache)
        return ROPE_Q16_ERR_NULL_PTR;
    if (token_count < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (head_base_cache_capacity < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    status = RoPEQ16ValidateHeadRangeCapacityChecked(head_cell_capacity,
                                                     range_base_index,
                                                     head_count,
                                                     head_stride_cells,
                                                     &last_head_base_index);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16ValidateHeadRangeSpanForDimChecked(head_cell_capacity,
                                                       range_base_index,
                                                       head_count,
                                                       head_stride_cells,
                                                       head_dim,
                                                       pair_stride_cells,
                                                       &last_head_last_pair_y_index);
    if (status != ROPE_Q16_OK)
        return status;

    if (!token_count)
    {
        status = RoPETryAddI64Checked(token_base_index,
                                      range_base_index,
                                      &abs_range_base);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPEQ16ValidateHeadRangeCapacityChecked(head_cell_capacity,
                                                         abs_range_base,
                                                         head_count,
                                                         head_stride_cells,
                                                         &last_head_base_index);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPEQ16ValidateHeadRangeSpanForDimChecked(head_cell_capacity,
                                                           abs_range_base,
                                                           head_count,
                                                           head_stride_cells,
                                                           head_dim,
                                                           pair_stride_cells,
                                                           &last_head_last_pair_y_index);
        if (status != ROPE_Q16_OK)
            return status;

        return ROPE_Q16_OK;
    }

    if (head_count > head_base_cache_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    for (head_index = 0; head_index < head_count; head_index++)
    {
        status = RoPEQ16ComputeHeadBaseIndexChecked(range_base_index,
                                                    head_index,
                                                    head_stride_cells,
                                                    &head_base_index);
        if (status != ROPE_Q16_OK)
            return status;

        if (head_base_index > last_head_base_index)
            return ROPE_Q16_ERR_BAD_PARAM;

        head_base_cache[head_index] = head_base_index;
    }

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = RoPETryMulI64Checked(token_index,
                                      position_step,
                                      &token_position_offset);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryAddI64Checked(position_start,
                                      token_position_offset,
                                      &token_position);
        if (status != ROPE_Q16_OK)
            return status;

        for (head_index = 0; head_index < head_count; head_index++)
        {
            head_base_index = head_base_cache[head_index];
            if (head_base_index < 0 || head_base_index > last_head_base_index)
                return ROPE_Q16_ERR_BAD_PARAM;

            status = RoPEQ16RotateHeadByPositionChecked(head_cells_q16,
                                                        head_cell_capacity,
                                                        head_base_index,
                                                        head_dim,
                                                        pair_stride_cells,
                                                        freq_base_q16,
                                                        token_position);
            if (status != ROPE_Q16_OK)
                return status;
        }
    }

    if (last_head_last_pair_y_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    return ROPE_Q16_OK;
}

// Compatibility wrapper preserving existing API name while routing all callers
// through the explicit preflighted implementation.
I32 RoPEQ16RotateHeadRangeByTokenWindowChecked(I64 *head_cells_q16,
                                               I64 head_cell_capacity,
                                               I64 range_base_index,
                                               I64 head_count,
                                               I64 head_stride_cells,
                                               I64 head_dim,
                                               I64 pair_stride_cells,
                                               I64 freq_base_q16,
                                               I64 position_start,
                                               I64 token_count,
                                               I64 position_step)
{
    return RoPEQ16RotateHeadRangeByTokenWindowPreflightedChecked(head_cells_q16,
                                                                  head_cell_capacity,
                                                                  range_base_index,
                                                                  head_count,
                                                                  head_stride_cells,
                                                                  head_dim,
                                                                  pair_stride_cells,
                                                                  freq_base_q16,
                                                                  position_start,
                                                                  token_count,
                                                                  position_step);
}

// Rotate one head-range across a token window stored in a strided KV-cache slab.
//
// Absolute base for each token is derived with checked integer arithmetic:
//   token_base(token_index) = token_base_index + token_index * token_stride_q16
//   abs_range_base          = token_base(token_index) + range_base_index
//   token_position          = position_start + token_index * position_step
//
// Contracts:
//   - Uses shared preflight validators for each token slab before any mutation.
//   - Executes all writes on a scratch copy, then commits only on full success.
//   - Guarantees no-partial-write behavior for caller-visible `head_cells_q16`.
I32 RoPEQ16RotateHeadRangeByTokenWindowCheckedStrided(I64 *head_cells_q16,
                                                      I64 head_cell_capacity,
                                                      I64 token_base_index,
                                                      I64 token_stride_q16,
                                                      I64 range_base_index,
                                                      I64 head_count,
                                                      I64 head_stride_cells,
                                                      I64 head_dim,
                                                      I64 pair_stride_cells,
                                                      I64 freq_base_q16,
                                                      I64 position_start,
                                                      I64 token_count,
                                                      I64 position_step)
{
    I64 token_index;
    I64 token_base_offset;
    I64 token_base;
    I64 abs_range_base;
    I64 token_position_offset;
    I64 token_position;
    I64 last_head_base_index;
    I64 last_head_last_pair_y_index;
    I64 scratch_bytes;
    I64 *scratch_cells_q16;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;
    if (head_cell_capacity < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (token_stride_q16 < 0)
        return ROPE_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    if (!token_count || !head_count)
        return ROPE_Q16_OK;

    // Phase 1: preflight each token slab and token-position tuple without writes.
    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = RoPETryMulI64Checked(token_index,
                                      token_stride_q16,
                                      &token_base_offset);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryAddI64Checked(token_base_index,
                                      token_base_offset,
                                      &token_base);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryAddI64Checked(token_base,
                                      range_base_index,
                                      &abs_range_base);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryMulI64Checked(token_index,
                                      position_step,
                                      &token_position_offset);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPETryAddI64Checked(position_start,
                                      token_position_offset,
                                      &token_position);
        if (status != ROPE_Q16_OK)
            return status;

        if (token_position < 0)
            return ROPE_Q16_ERR_BAD_PARAM;

        status = RoPEQ16ValidateHeadRangeCapacityChecked(head_cell_capacity,
                                                         abs_range_base,
                                                         head_count,
                                                         head_stride_cells,
                                                         &last_head_base_index);
        if (status != ROPE_Q16_OK)
            return status;

        status = RoPEQ16ValidateHeadRangeSpanForDimChecked(head_cell_capacity,
                                                           abs_range_base,
                                                           head_count,
                                                           head_stride_cells,
                                                           head_dim,
                                                           pair_stride_cells,
                                                           &last_head_last_pair_y_index);
        if (status != ROPE_Q16_OK)
            return status;
    }

    // No work after preflight for empty head ranges.
    if (!head_count)
        return ROPE_Q16_OK;

    // Phase 2: run full token-window rotation on a scratch copy then commit.
    status = RoPETryMulI64Checked(head_cell_capacity,
                                  sizeof(I64),
                                  &scratch_bytes);
    if (status != ROPE_Q16_OK)
        return status;

    scratch_cells_q16 = MAlloc(scratch_bytes);
    if (!scratch_cells_q16)
        return ROPE_Q16_ERR_OVERFLOW;

    for (token_index = 0; token_index < head_cell_capacity; token_index++)
        scratch_cells_q16[token_index] = head_cells_q16[token_index];

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = RoPETryMulI64Checked(token_index,
                                      token_stride_q16,
                                      &token_base_offset);
        if (status != ROPE_Q16_OK)
        {
            Free(scratch_cells_q16);
            return status;
        }

        status = RoPETryAddI64Checked(token_base_index,
                                      token_base_offset,
                                      &token_base);
        if (status != ROPE_Q16_OK)
        {
            Free(scratch_cells_q16);
            return status;
        }

        status = RoPETryAddI64Checked(token_base,
                                      range_base_index,
                                      &abs_range_base);
        if (status != ROPE_Q16_OK)
        {
            Free(scratch_cells_q16);
            return status;
        }

        status = RoPETryMulI64Checked(token_index,
                                      position_step,
                                      &token_position_offset);
        if (status != ROPE_Q16_OK)
        {
            Free(scratch_cells_q16);
            return status;
        }

        status = RoPETryAddI64Checked(position_start,
                                      token_position_offset,
                                      &token_position);
        if (status != ROPE_Q16_OK)
        {
            Free(scratch_cells_q16);
            return status;
        }

        status = RoPEQ16RotateHeadRangeByPositionPreflightedChecked(scratch_cells_q16,
                                                                     head_cell_capacity,
                                                                     abs_range_base,
                                                                     head_count,
                                                                     head_stride_cells,
                                                                     head_dim,
                                                                     pair_stride_cells,
                                                                     freq_base_q16,
                                                                     token_position);
        if (status != ROPE_Q16_OK)
        {
            Free(scratch_cells_q16);
            return status;
        }
    }

    for (token_index = 0; token_index < head_cell_capacity; token_index++)
        head_cells_q16[token_index] = scratch_cells_q16[token_index];

    Free(scratch_cells_q16);
    return ROPE_Q16_OK;
}
