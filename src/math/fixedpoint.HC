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

// Core Q16.16 aliases used by inference math call sites.
typedef I64 FPQ16;
typedef I64 FPStatus;

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

// Checked Q16 multiply:
//   out_q16 = round_half_up((a_q16 * b_q16) / 2^16)
// Returns OVERFLOW instead of saturating and never writes on failure.
I32 FPQ16MulChecked(I64 a_q16,
                    I64 b_q16,
                    I64 *out_q16)
{
    U64 abs_a;
    U64 abs_b;
    U64 round_bias;
    U64 abs_prod;
    U64 rounded_mag;
    U64 limit;
    I64 is_negative;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (!a_q16 || !b_q16)
    {
        *out_q16 = 0;
        return FP_Q16_OK;
    }

    abs_a = FPAbsToU64(a_q16);
    abs_b = FPAbsToU64(b_q16);
    is_negative = ((a_q16 < 0) ^ (b_q16 < 0));

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    // Detect overflow before signed multiplication.
    if (abs_a > (U64_MAX_VALUE / abs_b))
        return FP_Q16_ERR_OVERFLOW;

    abs_prod = abs_a * abs_b;
    round_bias = (U64)1 << (FP_Q16_SHIFT - 1);

    // Rounded magnitude in Q16.
    if (abs_prod > U64_MAX_VALUE - round_bias)
        rounded_mag = (U64_MAX_VALUE >> FP_Q16_SHIFT);
    else
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT;

    if (rounded_mag > limit)
        return FP_Q16_ERR_OVERFLOW;

    return FPTryApplySignFromU64Checked(rounded_mag,
                                        is_negative,
                                        out_q16);
}

// Checked signed multiply with power-of-two divisor and deterministic
// half-away-from-zero rounding:
//   out_value = round_half_away_from_zero((lhs * rhs) / 2^shift)
//
// Contract:
// - out_value must be non-null.
// - shift must be in [0, 62].
// - Uses unsigned-magnitude arithmetic to avoid signed UB.
// - Returns OVERFLOW if multiply magnitude overflows or rounded magnitude
//   cannot be represented in signed I64 with final sign.
I32 FPQ16MulShiftRoundChecked(I64 lhs,
                              I64 rhs,
                              I64 shift,
                              I64 *out_value)
{
    U64 abs_lhs;
    U64 abs_rhs;
    U64 signed_limit;
    U64 product_mag;
    U64 quotient;
    U64 remainder;
    U64 remainder_mask;
    U64 half;
    I64 is_negative;

    if (!out_value)
        return FP_Q16_ERR_NULL_PTR;
    if (shift < 0 || shift > 62)
        return FP_Q16_ERR_BAD_PARAM;

    if (!lhs || !rhs)
    {
        *out_value = 0;
        return FP_Q16_OK;
    }

    abs_lhs = FPAbsToU64(lhs);
    abs_rhs = FPAbsToU64(rhs);
    is_negative = ((lhs < 0) ^ (rhs < 0));
    signed_limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        signed_limit = 0x8000000000000000;

    if (abs_lhs > (U64_MAX_VALUE / abs_rhs))
        return FP_Q16_ERR_OVERFLOW;

    product_mag = abs_lhs * abs_rhs;

    if (!shift)
        return FPTryApplySignFromU64Checked(product_mag,
                                            is_negative,
                                            out_value);

    quotient = product_mag >> shift;
    remainder_mask = (((U64)1) << shift) - 1;
    remainder = product_mag & remainder_mask;
    half = ((U64)1) << (shift - 1);

    // Half-away-from-zero: ties increment magnitude before sign restore.
    if (remainder >= half)
    {
        if (quotient >= signed_limit)
            return FP_Q16_ERR_OVERFLOW;
        quotient++;
    }

    return FPTryApplySignFromU64Checked(quotient,
                                        is_negative,
                                        out_value);
}

// Checked + saturating Q16 multiply:
//   out_q16 = sat_i64(round_half_up((a_q16 * b_q16) / 2^16))
//
// Contract:
// - On NULL output pointer: return NULL_PTR and perform no write.
// - On in-range result: write output and return OK.
// - On saturation: write saturated bound and return OVERFLOW.
I32 FPQ16MulSatChecked(I64 a_q16,
                       I64 b_q16,
                       I64 *out_q16)
{
    U64 abs_a;
    U64 abs_b;
    U64 abs_prod;
    U64 rounded_mag;
    U64 frac_mask;
    U64 frac_part;
    U64 limit;
    I64 is_negative;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (!a_q16 || !b_q16)
    {
        *out_q16 = 0;
        return FP_Q16_OK;
    }

    abs_a = FPAbsToU64(a_q16);
    abs_b = FPAbsToU64(b_q16);
    is_negative = ((a_q16 < 0) ^ (b_q16 < 0));

    if (abs_a > (U64_MAX_VALUE / abs_b))
    {
        if (is_negative)
            *out_q16 = I64_MIN_VALUE;
        else
            *out_q16 = I64_MAX_VALUE;
        return FP_Q16_ERR_OVERFLOW;
    }

    abs_prod = abs_a * abs_b;
    frac_mask = (((U64)1) << FP_Q16_SHIFT) - 1;
    frac_part = abs_prod & frac_mask;

    rounded_mag = abs_prod >> FP_Q16_SHIFT;
    if (frac_part >= (((U64)1) << (FP_Q16_SHIFT - 1)))
        rounded_mag++;

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    if (rounded_mag > limit)
    {
        if (is_negative)
            *out_q16 = I64_MIN_VALUE;
        else
            *out_q16 = I64_MAX_VALUE;
        return FP_Q16_ERR_OVERFLOW;
    }

    *out_q16 = FPSignApplyFromU64(rounded_mag, is_negative);
    return FP_Q16_OK;
}

// Checked + saturating Q16 multiply-accumulate:
//   out_q16 = sat_i64(acc_q16 + FPQ16MulSatChecked(a_q16, b_q16))
//
// Contract:
// - Uses saturating multiply above.
// - Detects signed addition overflow explicitly.
// - Returns OVERFLOW when multiply or add saturated.
// - Writes exactly once on all non-NULL exits.
I32 FPQ16MacSatChecked(I64 acc_q16,
                       I64 a_q16,
                       I64 b_q16,
                       I64 *out_q16)
{
    I64 mul_q16;
    I64 sum_q16;
    I32 mul_status;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    mul_status = FPQ16MulSatChecked(a_q16, b_q16, &mul_q16);

    if (mul_q16 > 0 && acc_q16 > I64_MAX_VALUE - mul_q16)
    {
        *out_q16 = I64_MAX_VALUE;
        return FP_Q16_ERR_OVERFLOW;
    }
    if (mul_q16 < 0 && acc_q16 < I64_MIN_VALUE - mul_q16)
    {
        *out_q16 = I64_MIN_VALUE;
        return FP_Q16_ERR_OVERFLOW;
    }

    sum_q16 = acc_q16 + mul_q16;
    *out_q16 = sum_q16;

    if (mul_status != FP_Q16_OK)
        return FP_Q16_ERR_OVERFLOW;

    return FP_Q16_OK;
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

// Checked Q16 divide:
//   out_q16 = round_half_up((num_q16 / den_q16) * 2^16)
// Returns DOMAIN on divide-by-zero and OVERFLOW on range violations.
// On any error this performs no output write.
I32 FPQ16DivChecked(I64 num_q16,
                    I64 den_q16,
                    I64 *out_q16)
{
    U64 abs_num;
    U64 abs_den;
    U64 limit;
    U64 int_part;
    U64 rem;
    U64 result_mag;
    I64 is_negative;
    I64 bit;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (!den_q16)
        return FP_Q16_ERR_DOMAIN;

    abs_num = FPAbsToU64(num_q16);
    abs_den = FPAbsToU64(den_q16);
    is_negative = ((num_q16 < 0) ^ (den_q16 < 0));

    limit = (U64)I64_MAX_VALUE;
    if (is_negative)
        limit = 0x8000000000000000;

    int_part = abs_num / abs_den;
    if (int_part > (limit >> FP_Q16_SHIFT))
        return FP_Q16_ERR_OVERFLOW;

    result_mag = int_part << FP_Q16_SHIFT;
    rem = abs_num % abs_den;

    // Long division emits exactly 16 fractional bits.
    for (bit = FP_Q16_SHIFT - 1; bit >= 0; bit--)
    {
        rem <<= 1;
        if (rem >= abs_den)
        {
            rem -= abs_den;
            if (result_mag > (limit - (((U64)1) << bit)))
                return FP_Q16_ERR_OVERFLOW;
            result_mag |= ((U64)1) << bit;
        }
    }

    // Half-up rounding from the remaining fraction.
    if (rem >= ((abs_den + 1) >> 1))
    {
        if (result_mag == limit)
            return FP_Q16_ERR_OVERFLOW;
        result_mag++;
    }

    return FPTryApplySignFromU64Checked(result_mag,
                                        is_negative,
                                        out_q16);
}

// Commit-only checked Q16 divide wrapper.
//
// This stages FPQ16DivChecked into a local scalar first, then performs
// immutable snapshot parity checks over {numerator_q16, denominator_q16}
// before a single atomic publish into *out_q16.
//
// Contract:
// - out_q16 must be non-null.
// - Returns exactly the same status domain as FPQ16DivChecked.
// - On any failure this performs no out_q16 write.
I32 FPQ16DivCheckedNoPartialCommitOnly(I64 numerator_q16,
                                       I64 denominator_q16,
                                       I64 *out_q16)
{
    I64 snapshot_numerator_q16;
    I64 snapshot_denominator_q16;
    I64 staged_out_q16;
    I32 status;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_numerator_q16 = numerator_q16;
    snapshot_denominator_q16 = denominator_q16;

    status = FPQ16DivChecked(numerator_q16,
                             denominator_q16,
                             &staged_out_q16);
    if (status)
        return status;

    if (snapshot_numerator_q16 != numerator_q16 ||
        snapshot_denominator_q16 != denominator_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_q16 = staged_out_q16;
    return FP_Q16_OK;
}

// Diagnostics-only preflight companion for FPQ16DivCheckedNoPartialCommitOnly.
//
// This runs the canonical commit-only divider into a local staging lane to
// preserve DOMAIN/OVERFLOW semantics, then publishes only the canonical scalar
// requirement metadata `{required_out_cells=1}` when immutable input snapshots
// still match.
//
// Contract:
// - out_required_out_cells must be non-null.
// - Returns exactly the same status domain as FPQ16DivCheckedNoPartialCommitOnly.
// - Performs no publish on failure.
I32 FPQ16DivCheckedNoPartialCommitOnlyPreflightOnly(I64 numerator_q16,
                                                    I64 denominator_q16,
                                                    I64 *out_required_out_cells)
{
    I64 snapshot_numerator_q16;
    I64 snapshot_denominator_q16;
    I64 required_out_cells;
    I64 staged_out_q16;
    I32 status;

    if (!out_required_out_cells)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_numerator_q16 = numerator_q16;
    snapshot_denominator_q16 = denominator_q16;
    required_out_cells = 1;

    status = FPQ16DivCheckedNoPartialCommitOnly(numerator_q16,
                                                 denominator_q16,
                                                 &staged_out_q16);
    if (status)
        return status;

    if (snapshot_numerator_q16 != numerator_q16 ||
        snapshot_denominator_q16 != denominator_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_out_cells = required_out_cells;
    return FP_Q16_OK;
}

// Strict diagnostics parity gate over
// FPQ16DivCheckedNoPartialCommitOnlyPreflightOnly and
// FPQ16DivCheckedNoPartialCommitOnly.
//
// This preserves divide domain/overflow semantics by staging both delegated
// helpers, revalidates immutable input snapshots, and publishes exactly one
// diagnostics tuple value `{required_out_cells=1}` only on strict parity.
//
// Contract:
// - out_required_out_cells must be non-null.
// - Returns same status domain as delegated divide helpers.
// - Performs no diagnostics publish on failure.
I32 FPQ16DivCheckedNoPartialCommitOnlyPreflightOnlyParity(I64 numerator_q16,
                                                          I64 denominator_q16,
                                                          I64 *out_required_out_cells)
{
    I64 snapshot_numerator_q16;
    I64 snapshot_denominator_q16;
    I64 staged_preflight_required_out_cells;
    I64 staged_commit_out_q16;
    I64 recomputed_required_out_cells;
    I32 status;

    if (!out_required_out_cells)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_numerator_q16 = numerator_q16;
    snapshot_denominator_q16 = denominator_q16;

    status = FPQ16DivCheckedNoPartialCommitOnlyPreflightOnly(
        numerator_q16,
        denominator_q16,
        &staged_preflight_required_out_cells);
    if (status)
        return status;

    status = FPQ16DivCheckedNoPartialCommitOnly(numerator_q16,
                                                denominator_q16,
                                                &staged_commit_out_q16);
    if (status)
        return status;

    if (snapshot_numerator_q16 != numerator_q16 ||
        snapshot_denominator_q16 != denominator_q16)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_out_cells = 1;

    if (staged_preflight_required_out_cells != recomputed_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (recomputed_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_out_cells = staged_preflight_required_out_cells;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16DivCheckedNoPartialCommitOnlyPreflightOnlyParity.
//
// This stages delegated diagnostics into a local scalar, revalidates immutable
// `{numerator_q16,denominator_q16}` snapshots, and atomically publishes only
// canonical `{required_out_cells=1}` on strict parity success.
//
// Contract:
// - out_required_out_cells must be non-null.
// - Returns same status domain as delegated diagnostics divide helpers.
// - Performs no diagnostics publish on failure.
I32 FPQ16DivCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(I64 numerator_q16,
                                                                    I64 denominator_q16,
                                                                    I64 *out_required_out_cells)
{
    I64 snapshot_numerator_q16;
    I64 snapshot_denominator_q16;
    I64 staged_required_out_cells;
    I64 recomputed_required_out_cells;
    I32 status;

    if (!out_required_out_cells)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_numerator_q16 = numerator_q16;
    snapshot_denominator_q16 = denominator_q16;

    status = FPQ16DivCheckedNoPartialCommitOnlyPreflightOnlyParity(
        numerator_q16,
        denominator_q16,
        &staged_required_out_cells);
    if (status)
        return status;

    if (snapshot_numerator_q16 != numerator_q16 ||
        snapshot_denominator_q16 != denominator_q16)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_out_cells = 1;

    if (staged_required_out_cells != recomputed_required_out_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (recomputed_required_out_cells != 1)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_out_cells = staged_required_out_cells;
    return FP_Q16_OK;
}

// Checked elementwise Q16 multiply helper for inference vector paths.
// Contracts:
//   - count >= 0
//   - lhs_q16/rhs_q16/out_q16 are valid arrays for count lanes
//   - out_q16 is unchanged if any lane fails
I32 FPQ16MulArrayChecked(I64 *lhs_q16,
                         I64 *rhs_q16,
                         I64 *out_q16,
                         I64 count)
{
    I64 i;
    I64 lane_value;
    I32 err;

    if (!lhs_q16 || !rhs_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (lhs_q16 == rhs_q16 || lhs_q16 == out_q16 || rhs_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight all lanes first so failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulChecked(lhs_q16[i],
                              rhs_q16[i],
                              &lane_value);
        if (err)
            return err;
    }

    for (i = 0; i < count; i++)
    {
        err = FPQ16MulChecked(lhs_q16[i],
                              rhs_q16[i],
                              &out_q16[i]);
        if (err)
            return err;
    }

    return FP_Q16_OK;
}

// Checked elementwise Q16 multiply helper with strict no-partial semantics:
//   out_q16[i] = round_half_up((lhs_q16[i] * rhs_q16[i]) / 2^16)
//
// Contract:
// - Null pointers return NULL_PTR and perform no writes.
// - count < 0 returns BAD_PARAM and performs no writes.
// - Any overlap among {lhs_q16,rhs_q16,out_q16} is rejected.
// - The full lane set is preflighted before committing output writes.
// - On first lane overflow/domain failure, returns that status with no writes.
I32 FPQ16MulCheckedNoPartialArray(I64 *lhs_q16,
                                  I64 *rhs_q16,
                                  I64 *out_q16,
                                  I64 count)
{
    I64 i;
    I64 last_index;
    I64 lane_value;
    U64 span_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    I32 err;

    if (!lhs_q16 || !rhs_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Reject exact alias pairs up-front so callers get deterministic BAD_PARAM
    // before pointer-span math.
    if (lhs_q16 == rhs_q16 || lhs_q16 == out_q16 || rhs_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span math before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    span_bytes = ((U64)count) << 3;
    lhs_base = (U64)lhs_q16;
    rhs_base = (U64)rhs_q16;
    out_base = (U64)out_q16;

    if (lhs_base > (U64_MAX_VALUE - span_bytes) ||
        rhs_base > (U64_MAX_VALUE - span_bytes) ||
        out_base > (U64_MAX_VALUE - span_bytes))
        return FP_Q16_ERR_OVERFLOW;

    lhs_end = lhs_base + span_bytes;
    rhs_end = rhs_base + span_bytes;
    out_end = out_base + span_bytes;

    // Reject any span overlap to preserve strict no-partial semantics.
    if ((lhs_end > rhs_base && rhs_end > lhs_base) ||
        (lhs_end > out_base && out_end > lhs_base) ||
        (rhs_end > out_base && out_end > rhs_base))
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight all lanes first so hard failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulChecked(lhs_q16[i],
                              rhs_q16[i],
                              &lane_value);
        if (err)
            return err;
    }

    for (i = 0; i < count; i++)
    {
        err = FPQ16MulChecked(lhs_q16[i],
                              rhs_q16[i],
                              &out_q16[i]);
        if (err)
            return err;
    }

    return FP_Q16_OK;
}

// Checked elementwise integer multiply/shift rounding helper with strict
// no-partial semantics:
//   out_values[i] = round_half_away_from_zero((lhs[i] * rhs[i]) / 2^shift)
//
// Contract:
// - Null pointers return NULL_PTR and perform no writes.
// - count/capacities < 0 return BAD_PARAM and perform no writes.
// - Enforces count <= {lhs_capacity,rhs_capacity,out_capacity}.
// - Rejects any overlap among {lhs_values,rhs_values,out_values} lane spans.
// - Uses immutable snapshot parity over {count,capacities,shift} before commit.
// - Preflights all lanes first, then commits all lanes.
I32 FPQ16MulShiftRoundCheckedNoPartialArray(I64 *lhs_values,
                                            I64 lhs_capacity,
                                            I64 *rhs_values,
                                            I64 rhs_capacity,
                                            I64 *out_values,
                                            I64 out_capacity,
                                            I64 count,
                                            I64 shift)
{
    I64 snapshot_count;
    I64 snapshot_lhs_capacity;
    I64 snapshot_rhs_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_shift;
    I64 required_lhs;
    I64 required_rhs;
    I64 required_out;
    I64 i;
    I64 lane_value;
    U64 span_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    I32 status;

    if (!lhs_values || !rhs_values || !out_values)
        return FP_Q16_ERR_NULL_PTR;
    if (lhs_capacity < 0 || rhs_capacity < 0 || out_capacity < 0 || count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (lhs_values == rhs_values || lhs_values == out_values || rhs_values == out_values)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_count = count;
    snapshot_lhs_capacity = lhs_capacity;
    snapshot_rhs_capacity = rhs_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_shift = shift;

    required_lhs = snapshot_count;
    required_rhs = snapshot_count;
    required_out = snapshot_count;

    if (required_lhs > snapshot_lhs_capacity ||
        required_rhs > snapshot_rhs_capacity ||
        required_out > snapshot_out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (!snapshot_count)
        return FP_Q16_OK;

    if (snapshot_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    span_bytes = ((U64)snapshot_count) << 3;
    lhs_base = (U64)lhs_values;
    rhs_base = (U64)rhs_values;
    out_base = (U64)out_values;

    if (lhs_base > (U64_MAX_VALUE - span_bytes) ||
        rhs_base > (U64_MAX_VALUE - span_bytes) ||
        out_base > (U64_MAX_VALUE - span_bytes))
        return FP_Q16_ERR_OVERFLOW;

    lhs_end = lhs_base + span_bytes;
    rhs_end = rhs_base + span_bytes;
    out_end = out_base + span_bytes;

    if ((lhs_end > rhs_base && rhs_end > lhs_base) ||
        (lhs_end > out_base && out_end > lhs_base) ||
        (rhs_end > out_base && out_end > rhs_base))
        return FP_Q16_ERR_BAD_PARAM;

    // Phase 1: preflight all lanes; never publish out_values on failure.
    for (i = 0; i < snapshot_count; i++)
    {
        status = FPQ16MulShiftRoundChecked(lhs_values[i],
                                           rhs_values[i],
                                           snapshot_shift,
                                           &lane_value);
        if (status)
            return status;
    }

    // Immutable tuple revalidation before commit publish.
    if (snapshot_count != count ||
        snapshot_lhs_capacity != lhs_capacity ||
        snapshot_rhs_capacity != rhs_capacity ||
        snapshot_out_capacity != out_capacity ||
        snapshot_shift != shift)
        return FP_Q16_ERR_BAD_PARAM;

    // Phase 2: commit every lane.
    for (i = 0; i < snapshot_count; i++)
    {
        status = FPQ16MulShiftRoundChecked(lhs_values[i],
                                           rhs_values[i],
                                           snapshot_shift,
                                           &out_values[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over FPQ16MulShiftRoundCheckedNoPartialArray.
//
// This composes the checked no-partial array multiply+shift+round first, then
// stages canonical requirements tuple:
//   required_lhs = count
//   required_rhs = count
//   required_out = count
// and only publishes after immutable snapshot revalidation and strict parity.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - Negative capacities/count return BAD_PARAM and perform no diagnostics writes.
// - Enforces count <= {lhs_capacity,rhs_capacity,out_capacity}.
// - Rejects diagnostics output alias/overlap against active lane spans.
// - Atomic diagnostics publish of {required_lhs,required_rhs,required_out}.
I32 FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnly(I64 *lhs_values,
                                                      I64 lhs_capacity,
                                                      I64 *rhs_values,
                                                      I64 rhs_capacity,
                                                      I64 *out_values,
                                                      I64 out_capacity,
                                                      I64 count,
                                                      I64 shift,
                                                      I64 *out_required_lhs,
                                                      I64 *out_required_rhs,
                                                      I64 *out_required_out)
{
    I64 *snapshot_lhs_values;
    I64 *snapshot_rhs_values;
    I64 *snapshot_out_values;
    I64 snapshot_lhs_capacity;
    I64 snapshot_rhs_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_count;
    I64 snapshot_shift;
    I64 staged_required_lhs;
    I64 staged_required_rhs;
    I64 staged_required_out;
    I64 recomputed_required_lhs;
    I64 recomputed_required_rhs;
    I64 recomputed_required_out;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_lhs_base;
    U64 req_lhs_end;
    U64 req_rhs_base;
    U64 req_rhs_end;
    U64 req_out_base;
    U64 req_out_end;
    I32 status;

    if (!lhs_values || !rhs_values || !out_values ||
        !out_required_lhs || !out_required_rhs || !out_required_out)
        return FP_Q16_ERR_NULL_PTR;

    if (lhs_capacity < 0 || rhs_capacity < 0 || out_capacity < 0 || count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_lhs == out_required_rhs ||
        out_required_lhs == out_required_out ||
        out_required_rhs == out_required_out)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_lhs == lhs_values || out_required_lhs == rhs_values ||
        out_required_lhs == out_values || out_required_rhs == lhs_values ||
        out_required_rhs == rhs_values || out_required_rhs == out_values ||
        out_required_out == lhs_values || out_required_out == rhs_values ||
        out_required_out == out_values)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > lhs_capacity || count > rhs_capacity || count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_values, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_values, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_values, count, &out_base, &out_end);
        if (status)
            return status;

        req_lhs_base = (U64)out_required_lhs;
        req_rhs_base = (U64)out_required_rhs;
        req_out_base = (U64)out_required_out;

        if (req_lhs_base > (U64_MAX_VALUE - 7) ||
            req_rhs_base > (U64_MAX_VALUE - 7) ||
            req_out_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_lhs_end = req_lhs_base + 8;
        req_rhs_end = req_rhs_base + 8;
        req_out_end = req_out_base + 8;

        if (FPAddressRangesOverlap(req_lhs_base, req_lhs_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_lhs_base, req_lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_lhs_base, req_lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_rhs_base, req_rhs_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_rhs_base, req_rhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_rhs_base, req_rhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_out_base, req_out_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_out_base, req_out_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_out_base, req_out_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_lhs_base, req_lhs_end, req_rhs_base, req_rhs_end) ||
            FPAddressRangesOverlap(req_lhs_base, req_lhs_end, req_out_base, req_out_end) ||
            FPAddressRangesOverlap(req_rhs_base, req_rhs_end, req_out_base, req_out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_values = lhs_values;
    snapshot_rhs_values = rhs_values;
    snapshot_out_values = out_values;
    snapshot_lhs_capacity = lhs_capacity;
    snapshot_rhs_capacity = rhs_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_count = count;
    snapshot_shift = shift;

    status = FPQ16MulShiftRoundCheckedNoPartialArray(lhs_values,
                                                     lhs_capacity,
                                                     rhs_values,
                                                     rhs_capacity,
                                                     out_values,
                                                     out_capacity,
                                                     count,
                                                     shift);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_values != lhs_values ||
        snapshot_rhs_values != rhs_values ||
        snapshot_out_values != out_values ||
        snapshot_lhs_capacity != lhs_capacity ||
        snapshot_rhs_capacity != rhs_capacity ||
        snapshot_out_capacity != out_capacity ||
        snapshot_count != count ||
        snapshot_shift != shift)
        return FP_Q16_ERR_BAD_PARAM;

    staged_required_lhs = count;
    staged_required_rhs = count;
    staged_required_out = count;

    recomputed_required_lhs = snapshot_count;
    recomputed_required_rhs = snapshot_count;
    recomputed_required_out = snapshot_count;

    if (staged_required_lhs != recomputed_required_lhs ||
        staged_required_rhs != recomputed_required_rhs ||
        staged_required_out != recomputed_required_out)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_lhs = staged_required_lhs;
    *out_required_rhs = staged_required_rhs;
    *out_required_out = staged_required_out;
    return FP_Q16_OK;
}

// Geometry diagnostics helper for FPQ16MulCheckedNoPartialArray.
//
// Returns canonical lane requirements without touching out_q16:
//   required_cells = count
//   required_bytes = count * sizeof(I64)
//
// Contract:
// - Null pointers return NULL_PTR and perform no writes.
// - count < 0 returns BAD_PARAM and perform no writes.
// - Rejects any overlap among {lhs_q16,rhs_q16,out_q16} lane spans.
// - Rejects overlap/alias between diagnostics outputs and any lane span.
// - Uses checked integer math for byte geometry.
// - Publishes diagnostics scalars exactly once on success.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytes(I64 *lhs_q16,
                                               I64 *rhs_q16,
                                               I64 *out_q16,
                                               I64 count,
                                               I64 *out_required_cells,
                                               I64 *out_required_bytes)
{
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I64 recomputed_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (lhs_q16 == rhs_q16 || lhs_q16 == out_q16 || rhs_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(lhs_base, lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(rhs_base, rhs_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    staged_required_cells = count;
    staged_required_bytes = count << 3;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over FPQ16MulCheckedNoPartialArray.
//
// This composes the checked no-partial multiply first, then stages canonical
// diagnostics tuple `{required_cells, required_bytes}` and only publishes after
// immutable snapshot revalidation:
//   required_cells = count
//   required_bytes = count * sizeof(I64)
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count/out_capacity < 0 return BAD_PARAM and perform no diagnostics writes.
// - Rejects out_required alias/overlap with lane-array spans.
// - Enforces count <= out_capacity before delegated compute.
// - Atomic diagnostics publish only after delegated compute + snapshot parity.
I32 FPQ16MulCheckedNoPartialArrayCommitOnly(I64 *lhs_q16,
                                            I64 *rhs_q16,
                                            I64 *out_q16,
                                            I64 count,
                                            I64 out_capacity,
                                            I64 *out_required_cells,
                                            I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 snapshot_out_capacity;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0 || out_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;
    snapshot_out_capacity = out_capacity;

    status = FPQ16MulCheckedNoPartialArray(lhs_q16,
                                           rhs_q16,
                                           out_q16,
                                           count);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_cells = snapshot_count;
    recomputed_required_bytes = snapshot_count << 3;

    staged_required_cells = count;
    staged_required_bytes = count << 3;

    if (staged_required_cells != recomputed_required_cells ||
        staged_required_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion for the commit-only multiply contract.
//
// This helper preserves the same geometry/output contract as
// FPQ16MulCheckedNoPartialArrayCommitOnly without touching out_q16 data lanes.
// It validates full lane geometry and lane multiply preconditions via
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly, then recomputes the
// canonical tuple from immutable snapshots before one-shot diagnostics publish.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count/out_capacity < 0 return BAD_PARAM and perform no writes.
// - count > out_capacity returns BAD_PARAM and perform no writes.
// - Enforces alias/overlap safety for diagnostics outputs.
// - Publishes {required_cells,required_bytes} exactly once on success.
I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly(I64 *lhs_q16,
                                                         I64 *rhs_q16,
                                                         I64 *out_q16,
                                                         I64 count,
                                                         I64 out_capacity,
                                                         I64 *out_required_cells,
                                                         I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 snapshot_out_capacity;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I64 recomputed_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0 || out_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;
    snapshot_out_capacity = out_capacity;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_rhs_q16 != rhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_count > snapshot_out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_cells = snapshot_count;

    status = FPTryMulI64Checked(snapshot_count,
                                8,
                                &recomputed_required_bytes_checked);
    if (status)
        return status;

    if (snapshot_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    recomputed_required_bytes = snapshot_count << 3;

    if (recomputed_required_bytes_checked != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells != recomputed_required_cells ||
        staged_required_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16MulCheckedNoPartialArrayRequiredBytes.
//
// Stages canonical geometry tuple locally, revalidates immutable caller
// snapshots, and only then atomically publishes diagnostics scalars.
//
// Contract:
// - Composes FPQ16MulCheckedNoPartialArrayRequiredBytes first.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Enforces diagnostics output alias-safety before final publish.
// - Commits {required_cells,required_bytes} exactly once on success.
// - Preserves strict no-partial diagnostics writes on delegated failures.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly(I64 *lhs_q16,
                                                         I64 *rhs_q16,
                                                         I64 *out_q16,
                                                         I64 count,
                                                         I64 *out_required_cells,
                                                         I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I64 recomputed_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;

    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;
        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytes(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_rhs_q16 != rhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (snapshot_count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_cells = snapshot_count;

    status = FPTryMulI64Checked(snapshot_count,
                                8,
                                &recomputed_required_bytes_checked);
    if (status)
        return status;

    if (snapshot_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    recomputed_required_bytes = snapshot_count << 3;

    if (recomputed_required_bytes_checked != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells != recomputed_required_cells ||
        staged_required_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;
        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only parity gate over
// FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly and
// FPQ16MulCheckedNoPartialArrayCommitOnly.
//
// Stages diagnostics from both tuple producers, revalidates immutable input
// snapshots, enforces exact tuple parity, and atomically publishes outputs.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count/out_capacity < 0 or count > out_capacity return BAD_PARAM.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count,out_capacity} snapshots.
// - Requires exact parity on {required_cells,required_bytes} tuples.
// - Performs exactly one diagnostics publish on success.
I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(I64 *lhs_q16,
                                                               I64 *rhs_q16,
                                                               I64 *out_q16,
                                                               I64 count,
                                                               I64 out_capacity,
                                                               I64 *out_required_cells,
                                                               I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 snapshot_out_capacity;
    I64 staged_from_preflight_cells;
    I64 staged_from_preflight_bytes;
    I64 staged_from_commit_cells;
    I64 staged_from_commit_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0 || out_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;
    snapshot_out_capacity = out_capacity;

    status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        &staged_from_preflight_cells,
        &staged_from_preflight_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16MulCheckedNoPartialArrayCommitOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        &staged_from_commit_cells,
        &staged_from_commit_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_from_preflight_cells != staged_from_commit_cells ||
        staged_from_preflight_bytes != staged_from_commit_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_from_preflight_cells;
    *out_required_bytes = staged_from_preflight_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly.
//
// Stages canonical diagnostics tuple, revalidates immutable input snapshots,
// enforces diagnostics-output alias safety, and atomically publishes exactly
// once on success.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count/out_capacity < 0 or count > out_capacity return BAD_PARAM.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count,out_capacity} snapshots.
// - Enforces diagnostics-output alias/overlap safety before final publish.
// - Preserves strict no-partial diagnostics writes on delegated failures.
I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(I64 *lhs_q16,
                                                                          I64 *rhs_q16,
                                                                          I64 *out_q16,
                                                                          I64 count,
                                                                          I64 out_capacity,
                                                                          I64 *out_required_cells,
                                                                          I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 snapshot_out_capacity;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0 || out_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;
    snapshot_out_capacity = out_capacity;

    status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only parity gate over
// FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly
// and FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly.
//
// Stages tuple outputs from both delegated helpers, revalidates immutable input
// snapshots, enforces exact tuple parity, and atomically publishes diagnostics.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count/out_capacity < 0 or count > out_capacity return BAD_PARAM.
// - count > (I64_MAX_VALUE >> 3) returns OVERFLOW.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count,out_capacity} snapshots.
// - Requires tuple parity on {required_cells,required_bytes} before publish.
// - Preserves strict no-partial diagnostics writes on delegated failures.
I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(I64 *lhs_q16,
                                                                                              I64 *rhs_q16,
                                                                                              I64 *out_q16,
                                                                                              I64 count,
                                                                                              I64 out_capacity,
                                                                                              I64 *out_required_cells,
                                                                                              I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 snapshot_out_capacity;
    I64 staged_from_preflight_cells;
    I64 staged_from_preflight_bytes;
    I64 staged_from_commit_only_cells;
    I64 staged_from_commit_only_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I64 recomputed_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0 || out_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;
    snapshot_out_capacity = out_capacity;

    status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        &staged_from_preflight_cells,
        &staged_from_preflight_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        &staged_from_commit_only_cells,
        &staged_from_commit_only_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_from_preflight_cells != staged_from_commit_only_cells ||
        staged_from_preflight_bytes != staged_from_commit_only_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_cells = snapshot_count;

    status = FPMulI64Checked(snapshot_count,
                             8,
                             &recomputed_required_bytes_checked);
    if (status != FP_Q16_OK)
        return status;
    recomputed_required_bytes = recomputed_required_bytes_checked;

    if (staged_from_preflight_cells != recomputed_required_cells ||
        staged_from_preflight_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_from_preflight_cells;
    *out_required_bytes = staged_from_preflight_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity.
//
// Stages parity-verified diagnostics tuple locally, revalidates immutable input
// snapshots, recomputes canonical required tuple math from count, and atomically
// publishes only when all parity/canonical checks hold.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count/out_capacity < 0 or count > out_capacity return BAD_PARAM.
// - count > (I64_MAX_VALUE >> 3) returns OVERFLOW and performs no diagnostics writes.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count,out_capacity} snapshots.
// - Enforces diagnostics-output alias safety before final publish.
// - Commits {required_cells,required_bytes} exactly once on success.
I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(I64 *lhs_q16,
                                                                                                        I64 *rhs_q16,
                                                                                                        I64 *out_q16,
                                                                                                        I64 count,
                                                                                                        I64 out_capacity,
                                                                                                        I64 *out_required_cells,
                                                                                                        I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 snapshot_out_capacity;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 canonical_required_cells;
    I64 canonical_required_bytes;
    I64 canonical_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0 || out_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > out_capacity)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    req_cells_base = (U64)out_required_cells;
    req_bytes_base = (U64)out_required_bytes;

    if (req_cells_base > (U64_MAX_VALUE - 7) ||
        req_bytes_base > (U64_MAX_VALUE - 7))
        return FP_Q16_ERR_OVERFLOW;

    req_cells_end = req_cells_base + 8;
    req_bytes_end = req_bytes_base + 8;

    if (FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(lhs_base, lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(rhs_base, rhs_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;
    snapshot_out_capacity = out_capacity;

    status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count ||
        snapshot_out_capacity != out_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    canonical_required_cells = snapshot_count;

    status = FPTryMulI64Checked(snapshot_count,
                                8,
                                &canonical_required_bytes_checked);
    if (status)
        return status;

    if (snapshot_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    canonical_required_bytes = snapshot_count << 3;

    if (canonical_required_bytes != canonical_required_bytes_checked)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells != canonical_required_cells ||
        staged_required_bytes != canonical_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly.
//
// Stages canonical diagnostics through commit-only helper, revalidates
// immutable caller snapshots, recomputes canonical geometry from count,
// and atomically publishes tuple outputs only on exact parity.
//
// Contract:
// - Never writes out_q16 (strict preflight/no-partial semantics).
// - Composes FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly first.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Recomputes required tuple with checked integer math.
// - Performs exactly one diagnostics publish on success.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly(I64 *lhs_q16,
                                                                      I64 *rhs_q16,
                                                                      I64 *out_q16,
                                                                      I64 count,
                                                                      I64 *out_required_cells,
                                                                      I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I64 recomputed_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (lhs_q16 == rhs_q16 || lhs_q16 == out_q16 || rhs_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(lhs_base, lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(rhs_base, rhs_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_rhs_q16 != rhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_cells = snapshot_count;

    status = FPTryMulI64Checked(snapshot_count,
                                8,
                                &recomputed_required_bytes_checked);
    if (status)
        return status;

    if (snapshot_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    recomputed_required_bytes = snapshot_count << 3;

    if (recomputed_required_bytes_checked != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells != recomputed_required_cells ||
        staged_required_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only parity gate over
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly and
// canonical required-bytes tuple recomputation.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count < 0 returns BAD_PARAM and performs no diagnostics writes.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Requires exact tuple parity on {required_cells,required_bytes}.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParity(I64 *lhs_q16,
                                                                             I64 *rhs_q16,
                                                                             I64 *out_q16,
                                                                             I64 count,
                                                                             I64 *out_required_cells,
                                                                             I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_from_preflight_cells;
    I64 staged_from_preflight_bytes;
    I64 staged_from_required_cells;
    I64 staged_from_required_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;

    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_from_preflight_cells,
        &staged_from_preflight_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytes(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_from_required_cells,
        &staged_from_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_from_preflight_cells != staged_from_required_cells ||
        staged_from_preflight_bytes != staged_from_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    *out_required_cells = staged_from_preflight_cells;
    *out_required_bytes = staged_from_preflight_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParity.
//
// Stages canonical parity tuple, revalidates immutable input snapshots, and
// atomically publishes diagnostics outputs exactly once on success.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count < 0 returns BAD_PARAM and performs no diagnostics writes.
// - count > (I64_MAX_VALUE >> 3) returns OVERFLOW and performs no diagnostics writes.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Enforces diagnostics-output alias safety before final publish.
// - Preserves strict no-partial diagnostics writes on delegated failures.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly(I64 *lhs_q16,
                                                                                       I64 *rhs_q16,
                                                                                       I64 *out_q16,
                                                                                       I64 count,
                                                                                       I64 *out_required_cells,
                                                                                       I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParity(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_rhs_q16 != rhs_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly.
//
// Stages commit-only parity diagnostics, revalidates immutable caller snapshots,
// recomputes canonical required tuple from count, and atomically publishes only
// when staged diagnostics exactly match canonical tuple math.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count < 0 returns BAD_PARAM and performs no diagnostics writes.
// - count > (I64_MAX_VALUE >> 3) returns OVERFLOW and performs no diagnostics writes.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Enforces diagnostics-output alias safety before final publish.
// - Preserves strict no-partial diagnostics writes on delegated failures.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(I64 *lhs_q16,
                                                                                                     I64 *rhs_q16,
                                                                                                     I64 *out_q16,
                                                                                                     I64 count,
                                                                                                     I64 *out_required_cells,
                                                                                                     I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I64 recomputed_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(lhs_base, lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(rhs_base, rhs_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    recomputed_required_cells = snapshot_count;

    status = FPTryMulI64Checked(snapshot_count,
                                8,
                                &recomputed_required_bytes_checked);
    if (status)
        return status;
    recomputed_required_bytes = snapshot_count << 3;

    if (recomputed_required_bytes_checked != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells != recomputed_required_cells ||
        staged_required_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}


// Diagnostics-only parity gate over
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly
// and FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly.
//
// Stages both tuples from immutable snapshots and publishes only if both
// implementations agree exactly on {required_cells,required_bytes}.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count < 0 returns BAD_PARAM and performs no diagnostics writes.
// - count > (I64_MAX_VALUE >> 3) returns OVERFLOW and performs no diagnostics writes.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Enforces diagnostics-output alias safety before final publish.
// - Preserves strict no-partial diagnostics writes on delegated failures.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(I64 *lhs_q16,
                                                                                                           I64 *rhs_q16,
                                                                                                           I64 *out_q16,
                                                                                                           I64 count,
                                                                                                           I64 *out_required_cells,
                                                                                                           I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_from_preflight_cells;
    I64 staged_from_preflight_bytes;
    I64 staged_from_commit_only_cells;
    I64 staged_from_commit_only_bytes;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;

    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(lhs_base, lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(rhs_base, rhs_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_from_preflight_cells,
        &staged_from_preflight_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_from_commit_only_cells,
        &staged_from_commit_only_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_from_preflight_cells != staged_from_commit_only_cells ||
        staged_from_preflight_bytes != staged_from_commit_only_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    *out_required_cells = staged_from_preflight_cells;
    *out_required_bytes = staged_from_preflight_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity.
//
// Stages parity-verified diagnostics tuple locally, revalidates immutable input
// snapshots, recomputes canonical required tuple math from count, and atomically
// publishes only when all parity/canonical checks hold.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count < 0 returns BAD_PARAM and performs no diagnostics writes.
// - count > (I64_MAX_VALUE >> 3) returns OVERFLOW and performs no diagnostics writes.
// - Revalidates immutable {lhs_q16,rhs_q16,out_q16,count} snapshots.
// - Enforces diagnostics-output alias safety before final publish.
// - Commits {required_cells,required_bytes} exactly once on success.
I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(I64 *lhs_q16,
                                                                                                                     I64 *rhs_q16,
                                                                                                                     I64 *out_q16,
                                                                                                                     I64 count,
                                                                                                                     I64 *out_required_cells,
                                                                                                                     I64 *out_required_bytes)
{
    I64 *snapshot_lhs_q16;
    I64 *snapshot_rhs_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 canonical_required_cells;
    I64 canonical_required_bytes;
    I64 canonical_required_bytes_checked;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    U64 req_cells_base;
    U64 req_cells_end;
    U64 req_bytes_base;
    U64 req_bytes_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 ||
        !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(lhs_base, lhs_end, out_base, out_end) ||
            FPAddressRangesOverlap(rhs_base, rhs_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_lhs_q16 = lhs_q16;
    snapshot_rhs_q16 = rhs_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_lhs_q16 != lhs_q16 ||
        snapshot_rhs_q16 != rhs_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    canonical_required_cells = snapshot_count;

    status = FPTryMulI64Checked(snapshot_count,
                                8,
                                &canonical_required_bytes_checked);
    if (status)
        return status;

    if (snapshot_count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    canonical_required_bytes = snapshot_count << 3;

    if (canonical_required_bytes != canonical_required_bytes_checked)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells != canonical_required_cells ||
        staged_required_bytes != canonical_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(rhs_q16, count, &rhs_base, &rhs_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        req_cells_base = (U64)out_required_cells;
        req_bytes_base = (U64)out_required_bytes;

        if (req_cells_base > (U64_MAX_VALUE - 7) ||
            req_bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        req_cells_end = req_cells_base + 8;
        req_bytes_end = req_bytes_base + 8;

        if (FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, lhs_base, lhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, rhs_base, rhs_end) ||
            FPAddressRangesOverlap(req_bytes_base, req_bytes_end, out_base, out_end) ||
            FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    if (out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == lhs_q16 || out_required_cells == rhs_q16 ||
        out_required_cells == out_q16 || out_required_bytes == lhs_q16 ||
        out_required_bytes == rhs_q16 || out_required_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Checked + saturating elementwise Q16 multiply helper with strict
// stage-then-commit semantics:
//   out_q16[i] = sat_i64(round_half_up((lhs_q16[i] * rhs_q16[i]) / 2^16))
//
// Contract:
// - Null pointers return NULL_PTR and perform no writes.
// - count < 0 returns BAD_PARAM and performs no writes.
// - The full lane set is preflighted before committing any output writes.
// - If one or more lanes saturate, output is still fully committed and
//   OVERFLOW is returned after publish.
I32 FPQ16MulSatCheckedNoPartialArray(I64 *lhs_q16,
                                     I64 *rhs_q16,
                                     I64 *out_q16,
                                     I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_value;
    I32 err;
    I32 overflow_seen;

    if (!lhs_q16 || !rhs_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span math before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)lhs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)rhs_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight all lanes first so hard failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulSatChecked(lhs_q16[i],
                                 rhs_q16[i],
                                 &lane_value);
        if (err != FP_Q16_OK && err != FP_Q16_ERR_OVERFLOW)
            return err;
    }

    overflow_seen = 0;
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulSatChecked(lhs_q16[i],
                                 rhs_q16[i],
                                 &out_q16[i]);
        if (err == FP_Q16_ERR_OVERFLOW)
            overflow_seen = 1;
        else if (err != FP_Q16_OK)
            return err;
    }

    if (overflow_seen)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Diagnostics helper over FPQ16MulSatCheckedNoPartialArray that computes the
// caller-visible output byte requirement:
//   required_output_bytes = count * sizeof(I64)
//
// Contract:
// - Composes FPQ16MulSatCheckedNoPartialArray first.
// - Derives required_output_bytes with checked integer math.
// - Publishes required_output_bytes exactly once on success/overflow.
// - Performs no required-bytes write on hard failures.
I32 FPQ16MulSatCheckedNoPartialArrayRequiredBytes(I64 *lhs_q16,
                                                  I64 *rhs_q16,
                                                  I64 *out_q16,
                                                  I64 count,
                                                  I64 *out_required_output_bytes)
{
    I64 staged_required_output_bytes;
    U64 span_bytes;
    U64 required_base;
    U64 required_end;
    U64 lhs_base;
    U64 lhs_end;
    U64 rhs_base;
    U64 rhs_end;
    U64 out_base;
    U64 out_end;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == lhs_q16 ||
        out_required_output_bytes == rhs_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve no-partial semantics by rejecting any overlap between the
    // diagnostics scalar and lane-array spans before delegated compute.
    if (count)
    {
        if (count > (I64_MAX_VALUE >> 3))
            return FP_Q16_ERR_OVERFLOW;

        span_bytes = ((U64)count) << 3;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        lhs_base = (U64)lhs_q16;
        rhs_base = (U64)rhs_q16;
        out_base = (U64)out_q16;

        if (lhs_base > (U64_MAX_VALUE - span_bytes) ||
            rhs_base > (U64_MAX_VALUE - span_bytes) ||
            out_base > (U64_MAX_VALUE - span_bytes))
            return FP_Q16_ERR_OVERFLOW;

        lhs_end = lhs_base + span_bytes;
        rhs_end = rhs_base + span_bytes;
        out_end = out_base + span_bytes;

        if ((required_end > lhs_base && lhs_end > required_base) ||
            (required_end > rhs_base && rhs_end > required_base) ||
            (required_end > out_base && out_end > required_base))
            return FP_Q16_ERR_BAD_PARAM;
    }

    status = FPQ16MulSatCheckedNoPartialArray(lhs_q16,
                                              rhs_q16,
                                              out_q16,
                                              count);
    if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)
        return status;

    staged_required_output_bytes = count << 3;
    *out_required_output_bytes = staged_required_output_bytes;

    if (status == FP_Q16_ERR_OVERFLOW)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Compute byte span [base_addr, end_addr_exclusive) for an I64 lane array.
// The caller guarantees:
// - base_ptr is non-null
// - count > 0
I32 FPArrayI64SpanChecked(I64 *base_ptr,
                          I64 count,
                          U64 *out_base_addr,
                          U64 *out_end_addr_exclusive)
{
    U64 base_addr;
    U64 byte_count;

    if (!base_ptr || !out_base_addr || !out_end_addr_exclusive)
        return FP_Q16_ERR_NULL_PTR;
    if (count <= 0)
        return FP_Q16_ERR_BAD_PARAM;

    if ((U64)count > (U64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    base_addr = (U64)base_ptr;
    byte_count = ((U64)count) << 3;

    if (base_addr > (U64_MAX_VALUE - byte_count))
        return FP_Q16_ERR_OVERFLOW;

    *out_base_addr = base_addr;
    *out_end_addr_exclusive = base_addr + byte_count;
    return FP_Q16_OK;
}

I32 FPAddressRangesOverlap(U64 a_base,
                           U64 a_end_exclusive,
                           U64 b_base,
                           U64 b_end_exclusive)
{
    if (a_end_exclusive <= a_base)
        return 0;
    if (b_end_exclusive <= b_base)
        return 0;

    if (a_end_exclusive <= b_base)
        return 0;
    if (b_end_exclusive <= a_base)
        return 0;
    return 1;
}

// Checked + saturating Q16 MAC over arrays with strict stage-then-commit
// semantics and explicit alias/bounds validation:
//   out_q16[i] = sat_i64(acc_q16[i] + mul_q16(a_q16[i], b_q16[i]))
//
// Contract:
// - Null pointers return NULL_PTR and perform no writes.
// - count < 0 returns BAD_PARAM and performs no writes.
// - Any pointer alias among {acc_q16,a_q16,b_q16,out_q16} is rejected.
// - The full lane set is preflighted before committing output writes.
// - If one or more lanes saturate, output is fully committed and OVERFLOW
//   is returned after publish.
I32 FPQ16MacSatCheckedNoPartialArray(I64 *acc_q16,
                                     I64 *a_q16,
                                     I64 *b_q16,
                                     I64 *out_q16,
                                     I64 count)
{
    I64 i;
    I64 lane_value;
    U64 acc_base;
    U64 acc_end;
    U64 a_base;
    U64 a_end;
    U64 b_base;
    U64 b_end;
    U64 out_base;
    U64 out_end;
    I32 err;
    I32 overflow_seen;

    if (!acc_q16 || !a_q16 || !b_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Fast-path exact-alias rejection before span math. This keeps
    // deterministic BAD_PARAM behavior for identical pointers and
    // documents the no-alias contract at call-site granularity.
    if (acc_q16 == a_q16 || acc_q16 == b_q16 || acc_q16 == out_q16 ||
        a_q16 == b_q16 || a_q16 == out_q16 || b_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (!count)
        return FP_Q16_OK;
    // Guard pointer-span math before indexed lane access and reject
    // any overlap between input/output ranges.
    err = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);
    if (err)
        return err;
    err = FPArrayI64SpanChecked(a_q16, count, &a_base, &a_end);
    if (err)
        return err;
    err = FPArrayI64SpanChecked(b_q16, count, &b_base, &b_end);
    if (err)
        return err;
    err = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
    if (err)
        return err;

    if (FPAddressRangesOverlap(acc_base, acc_end, a_base, a_end) ||
        FPAddressRangesOverlap(acc_base, acc_end, b_base, b_end) ||
        FPAddressRangesOverlap(acc_base, acc_end, out_base, out_end) ||
        FPAddressRangesOverlap(a_base, a_end, b_base, b_end) ||
        FPAddressRangesOverlap(a_base, a_end, out_base, out_end) ||
        FPAddressRangesOverlap(b_base, b_end, out_base, out_end))
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight all lanes first so hard failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MacSatChecked(acc_q16[i],
                                 a_q16[i],
                                 b_q16[i],
                                 &lane_value);
        if (err != FP_Q16_OK && err != FP_Q16_ERR_OVERFLOW)
            return err;
    }

    overflow_seen = 0;
    for (i = 0; i < count; i++)
    {
        err = FPQ16MacSatChecked(acc_q16[i],
                                 a_q16[i],
                                 b_q16[i],
                                 &out_q16[i]);
        if (err == FP_Q16_ERR_OVERFLOW)
            overflow_seen = 1;
        else if (err != FP_Q16_OK)
            return err;
    }

    if (overflow_seen)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Diagnostics helper over FPQ16MacSatCheckedNoPartialArray that computes the
// caller-visible output byte requirement:
//   required_output_bytes = count * sizeof(I64)
//
// Contract:
// - Composes FPQ16MacSatCheckedNoPartialArray first.
// - Derives required_output_bytes with checked integer math.
// - Publishes required_output_bytes exactly once on success/overflow.
// - Performs no required-bytes write on hard failures.
I32 FPQ16MacSatCheckedNoPartialArrayRequiredBytes(I64 *acc_q16,
                                                  I64 *a_q16,
                                                  I64 *b_q16,
                                                  I64 *out_q16,
                                                  I64 count,
                                                  I64 *out_required_output_bytes)
{
    I64 staged_required_output_bytes;
    U64 acc_base;
    U64 acc_end;
    U64 a_base;
    U64 a_end;
    U64 b_base;
    U64 b_end;
    U64 out_base;
    U64 out_end;
    U64 required_base;
    U64 required_end;
    I32 status;

    if (!acc_q16 || !a_q16 || !b_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    if (out_required_output_bytes == acc_q16 ||
        out_required_output_bytes == a_q16 ||
        out_required_output_bytes == b_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve no-partial semantics by rejecting any overlap between the
    // diagnostics scalar and lane-array spans before delegated compute.
    if (count)
    {
        status = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(a_q16, count, &a_base, &a_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(b_q16, count, &b_base, &b_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, acc_base, acc_end) ||
            FPAddressRangesOverlap(required_base, required_end, a_base, a_end) ||
            FPAddressRangesOverlap(required_base, required_end, b_base, b_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    status = FPQ16MacSatCheckedNoPartialArray(acc_q16,
                                              a_q16,
                                              b_q16,
                                              out_q16,
                                              count);
    if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)
        return status;

    staged_required_output_bytes = count << 3;
    *out_required_output_bytes = staged_required_output_bytes;

    if (status == FP_Q16_ERR_OVERFLOW)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16MacSatCheckedNoPartialArrayRequiredBytes.
//
// This stages required_output_bytes locally, revalidates immutable call-site
// snapshots, and only then publishes the diagnostics scalar to the caller.
//
// Contract:
// - Composes FPQ16MacSatCheckedNoPartialArrayRequiredBytes first.
// - Treats {OK, OVERFLOW} as publishable statuses.
// - Preserves no-partial diagnostics writes on hard failures.
I32 FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnly(I64 *acc_q16,
                                                            I64 *a_q16,
                                                            I64 *b_q16,
                                                            I64 *out_q16,
                                                            I64 count,
                                                            I64 *out_required_output_bytes)
{
    I64 snapshot_count;
    I64 *snapshot_acc_q16;
    I64 *snapshot_a_q16;
    I64 *snapshot_b_q16;
    I64 *snapshot_out_q16;
    I64 staged_required_output_bytes;
    U64 acc_base;
    U64 acc_end;
    U64 a_base;
    U64 a_end;
    U64 b_base;
    U64 b_end;
    U64 out_base;
    U64 out_end;
    U64 required_base;
    U64 required_end;
    I32 status;

    if (!acc_q16 || !a_q16 || !b_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == acc_q16 ||
        out_required_output_bytes == a_q16 ||
        out_required_output_bytes == b_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(a_q16, count, &a_base, &a_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(b_q16, count, &b_base, &b_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, acc_base, acc_end) ||
            FPAddressRangesOverlap(required_base, required_end, a_base, a_end) ||
            FPAddressRangesOverlap(required_base, required_end, b_base, b_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_count = count;
    snapshot_acc_q16 = acc_q16;
    snapshot_a_q16 = a_q16;
    snapshot_b_q16 = b_q16;
    snapshot_out_q16 = out_q16;

    status = FPQ16MacSatCheckedNoPartialArrayRequiredBytes(acc_q16,
                                                           a_q16,
                                                           b_q16,
                                                           out_q16,
                                                           count,
                                                           &staged_required_output_bytes);
    if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)
        return status;

    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_acc_q16 != acc_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_a_q16 != a_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_b_q16 != b_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == acc_q16 ||
        out_required_output_bytes == a_q16 ||
        out_required_output_bytes == b_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(a_q16, count, &a_base, &a_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(b_q16, count, &b_base, &b_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, acc_base, acc_end) ||
            FPAddressRangesOverlap(required_base, required_end, a_base, a_end) ||
            FPAddressRangesOverlap(required_base, required_end, b_base, b_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    *out_required_output_bytes = staged_required_output_bytes;

    if (status == FP_Q16_ERR_OVERFLOW)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnly.
//
// Stages diagnostics via commit-only helper, revalidates immutable call-site
// snapshots, recomputes canonical geometry from count, and atomically
// publishes required_output_bytes only on strict parity success.
//
// Contract:
// - Null pointers return NULL_PTR and perform no diagnostics writes.
// - count < 0 returns BAD_PARAM.
// - Revalidates immutable {acc_q16,a_q16,b_q16,out_q16,count} snapshots.
// - Recomputes required_output_bytes with checked integer math.
// - Performs exactly one diagnostics publish on success/overflow.
I32 FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly(I64 *acc_q16,
                                                                          I64 *a_q16,
                                                                          I64 *b_q16,
                                                                          I64 *out_q16,
                                                                          I64 count,
                                                                          I64 *out_required_output_bytes)
{
    I64 *snapshot_acc_q16;
    I64 *snapshot_a_q16;
    I64 *snapshot_b_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_output_bytes;
    I64 staged_required_cells;
    I64 recomputed_required_output_bytes;
    I64 recomputed_required_output_bytes_checked;
    I64 scratch_cells;
    I64 scratch_bytes;
    I64 *scratch_out_q16;
    U64 acc_base;
    U64 acc_end;
    U64 a_base;
    U64 a_end;
    U64 b_base;
    U64 b_end;
    U64 out_base;
    U64 out_end;
    U64 required_base;
    U64 required_end;
    I32 status;

    if (!acc_q16 || !a_q16 || !b_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == acc_q16 ||
        out_required_output_bytes == a_q16 ||
        out_required_output_bytes == b_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(a_q16, count, &a_base, &a_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(b_q16, count, &b_base, &b_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, acc_base, acc_end) ||
            FPAddressRangesOverlap(required_base, required_end, a_base, a_end) ||
            FPAddressRangesOverlap(required_base, required_end, b_base, b_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_acc_q16 = acc_q16;
    snapshot_a_q16 = a_q16;
    snapshot_b_q16 = b_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    // Preflight-only contract: never write caller out_q16. Compose the
    // commit-only helper with a private scratch output span and discard it.
    scratch_cells = count;
    if (scratch_cells < 1)
        scratch_cells = 1;

    status = FPTryMulI64Checked(scratch_cells,
                                8,
                                &scratch_bytes);
    if (status)
        return status;

    scratch_out_q16 = MAlloc(scratch_bytes);
    if (!scratch_out_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnly(acc_q16,
                                                                      a_q16,
                                                                      b_q16,
                                                                      scratch_out_q16,
                                                                      count,
                                                                      &staged_required_output_bytes);
    if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)
    {
        Free(scratch_out_q16);
        return status;
    }

    if (snapshot_acc_q16 != acc_q16 ||
        snapshot_a_q16 != a_q16 ||
        snapshot_b_q16 != b_q16 ||
        snapshot_out_q16 != out_q16 ||
        snapshot_count != count)
    {
        Free(scratch_out_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    staged_required_cells = snapshot_count;
    if (staged_required_cells < 0)
    {
        Free(scratch_out_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    status = FPTryMulI64Checked(staged_required_cells,
                                8,
                                &recomputed_required_output_bytes_checked);
    if (status)
    {
        Free(scratch_out_q16);
        return status;
    }

    if (staged_required_cells > (I64_MAX_VALUE >> 3))
    {
        Free(scratch_out_q16);
        return FP_Q16_ERR_OVERFLOW;
    }
    recomputed_required_output_bytes = staged_required_cells << 3;

    if (recomputed_required_output_bytes != recomputed_required_output_bytes_checked)
    {
        Free(scratch_out_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    if (staged_required_output_bytes != recomputed_required_output_bytes)
    {
        Free(scratch_out_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    if (out_required_output_bytes == acc_q16 ||
        out_required_output_bytes == a_q16 ||
        out_required_output_bytes == b_q16 ||
        out_required_output_bytes == out_q16)
    {
        Free(scratch_out_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    if (count)
    {
        status = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);
        if (status)
        {
            Free(scratch_out_q16);
            return status;
        }
        status = FPArrayI64SpanChecked(a_q16, count, &a_base, &a_end);
        if (status)
        {
            Free(scratch_out_q16);
            return status;
        }
        status = FPArrayI64SpanChecked(b_q16, count, &b_base, &b_end);
        if (status)
        {
            Free(scratch_out_q16);
            return status;
        }
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
        {
            Free(scratch_out_q16);
            return status;
        }

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
        {
            Free(scratch_out_q16);
            return FP_Q16_ERR_OVERFLOW;
        }
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, acc_base, acc_end) ||
            FPAddressRangesOverlap(required_base, required_end, a_base, a_end) ||
            FPAddressRangesOverlap(required_base, required_end, b_base, b_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
        {
            Free(scratch_out_q16);
            return FP_Q16_ERR_BAD_PARAM;
        }
    }

    *out_required_output_bytes = staged_required_output_bytes;
    Free(scratch_out_q16);

    if (status == FP_Q16_ERR_OVERFLOW)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Checked elementwise Q16 divide helper with no-partial-write contract.
I32 FPQ16DivArrayChecked(I64 *num_q16,
                         I64 *den_q16,
                         I64 *out_q16,
                         I64 count)
{
    I64 i;
    I64 lane_value;
    I32 err;

    if (!num_q16 || !den_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    for (i = 0; i < count; i++)
    {
        err = FPQ16DivChecked(num_q16[i],
                              den_q16[i],
                              &lane_value);
        if (err)
            return err;
    }

    for (i = 0; i < count; i++)
    {
        err = FPQ16DivChecked(num_q16[i],
                              den_q16[i],
                              &out_q16[i]);
        if (err)
            return err;
    }

    return FP_Q16_OK;
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

// Checked elementwise one-rounding mul/div helper:
//   out_q16[i] = round_half_up((a_q16[i] * b_q16[i]) / d_q16[i])
//
// Contracts:
// - All pointers are non-null and each points to at least count I64 lanes.
// - count >= 0.
// - Per-lane math reuses FPQ16MulDivRoundedChecked (same overflow/domain rules).
// - No-partial-write: a full preflight pass runs before any out writes.
I32 FPQ16MulDivArrayRoundedChecked(I64 *a_q16,
                                   I64 *b_q16,
                                   I64 *d_q16,
                                   I64 *out_q16,
                                   I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_value;
    I32 err;

    if (!a_q16 || !b_q16 || !d_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span math before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)a_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)b_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)d_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight all lanes first so failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulDivRoundedChecked(a_q16[i],
                                        b_q16[i],
                                        d_q16[i],
                                        &lane_value);
        if (err)
            return err;
    }

    for (i = 0; i < count; i++)
    {
        err = FPQ16MulDivRoundedChecked(a_q16[i],
                                        b_q16[i],
                                        d_q16[i],
                                        &out_q16[i]);
        if (err)
            return err;
    }

    return FP_Q16_OK;
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

// Checked elementwise one-rounding mul/div helper with positive integer
// denominators:
//   out_q16[i] = round_half_up((a_q16[i] * b_q16[i]) / d_int[i])
// where every d_int[i] is required to be > 0.
//
// Contracts:
// - All pointers are non-null and each points to at least count I64 lanes.
// - count >= 0.
// - Per-lane math reuses FPQ16MulDivRoundedByPositiveIntChecked.
// - No-partial-write: a full preflight pass runs before any out writes.
I32 FPQ16MulDivArrayRoundedByPositiveIntChecked(I64 *a_q16,
                                                I64 *b_q16,
                                                I64 *d_int,
                                                I64 *out_q16,
                                                I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_value;
    I32 err;

    if (!a_q16 || !b_q16 || !d_int || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span math before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)a_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)b_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)d_int > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight all lanes first so failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulDivRoundedByPositiveIntChecked(a_q16[i],
                                                     b_q16[i],
                                                     d_int[i],
                                                     &lane_value);
        if (err)
            return err;
    }

    for (i = 0; i < count; i++)
    {
        err = FPQ16MulDivRoundedByPositiveIntChecked(a_q16[i],
                                                     b_q16[i],
                                                     d_int[i],
                                                     &out_q16[i]);
        if (err)
            return err;
    }

    return FP_Q16_OK;
}

// Checked elementwise one-rounding mul/div helper with positive integer
// denominators encoded as exact Q16 whole numbers:
//   d_q16[i] = d_int[i] << 16, with d_int[i] > 0
//   out_q16[i] = round_half_up((a_q16[i] * b_q16[i]) / (d_int[i] * 2^16))
//
// Contracts:
// - All pointers are non-null and each points to at least count I64 lanes.
// - count >= 0.
// - Every denominator lane must be strictly positive and have zero fractional
//   bits in Q16 representation.
// - No-partial-write: a full preflight pass runs before any out writes.
I32 FPQ16MulDivArrayRoundedByPositiveIntFromQ16DenChecked(I64 *a_q16,
                                                          I64 *b_q16,
                                                          I64 *d_q16,
                                                          I64 *out_q16,
                                                          I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_value;
    I32 err;

    if (!a_q16 || !b_q16 || !d_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span math before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)a_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)b_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)d_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Preflight all lanes first so failures never partially write output.
    for (i = 0; i < count; i++)
    {
        err = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(a_q16[i],
                                                                b_q16[i],
                                                                d_q16[i],
                                                                &lane_value);
        if (err)
            return err;
    }

    for (i = 0; i < count; i++)
    {
        err = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(a_q16[i],
                                                                b_q16[i],
                                                                d_q16[i],
                                                                &out_q16[i]);
        if (err)
            return err;
    }

    return FP_Q16_OK;
}

// Checked no-alias wrapper for Q16-denominator positive-int array mul/div.
// This hardens immutable-source call sites by requiring output storage to be
// distinct from every input lane buffer before delegating into the checked
// core helper that performs full preflight and lane math.
I32 FPQ16MulDivArrayRoundedByPositiveIntFromQ16DenCheckedNoAlias(I64 *a_q16,
                                                                 I64 *b_q16,
                                                                 I64 *d_q16,
                                                                 I64 *out_q16,
                                                                 I64 count)
{
    if (!a_q16 || !b_q16 || !d_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_q16 == a_q16 || out_q16 == b_q16 || out_q16 == d_q16)
        return FP_Q16_ERR_BAD_PARAM;

    return FPQ16MulDivArrayRoundedByPositiveIntFromQ16DenChecked(a_q16,
                                                                  b_q16,
                                                                  d_q16,
                                                                  out_q16,
                                                                  count);
}

// Checked integer-only Q16 exponential approximation.
//
// Range reduction helper used by exp approximation core.
//
// Splits x_q16 into:
//   x_q16 = k*ln(2) + r_q16
// where k is nearest integer to x/ln(2) and |r_q16| <= ln(2)/2.
//
// Contracts:
// - All output pointers are required.
// - Uses checked integer math for k*ln(2) reconstruction and subtraction.
// - Commits outputs exactly once after all invariants pass.
I32 FPQ16ExpApproxRangeReduceChecked(I64 x_q16,
                                     I64 *out_k,
                                     I64 *out_r_q16)
{
    I64 ln2_q16;
    I64 half_ln2_q16;
    I64 k;
    I64 k_ln2_q16;
    I64 residual_q16;
    U64 abs_k_u64;

    if (!out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (out_k == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    ln2_q16 = 45426;       // round(ln(2) * 65536)
    half_ln2_q16 = 22713;  // round(ln(2)/2 * 65536)

    // Round to nearest integer k = round(x / ln(2)) using integer-only math.
    if (x_q16 >= 0)
        k = (x_q16 + half_ln2_q16) / ln2_q16;
    else
        k = -(((-x_q16) + half_ln2_q16) / ln2_q16);

    abs_k_u64 = FPAbsToU64(k);
    if (abs_k_u64 > (U64_MAX_VALUE / (U64)ln2_q16))
        return FP_Q16_ERR_OVERFLOW;

    if (k >= 0)
        k_ln2_q16 = (I64)(abs_k_u64 * (U64)ln2_q16);
    else
        k_ln2_q16 = -(I64)(abs_k_u64 * (U64)ln2_q16);

    if ((k_ln2_q16 > 0 && x_q16 < (I64_MIN_VALUE + k_ln2_q16)) ||
        (k_ln2_q16 < 0 && x_q16 > (I64_MAX_VALUE + k_ln2_q16)))
        return FP_Q16_ERR_OVERFLOW;

    residual_q16 = x_q16 - k_ln2_q16;

    // Because k is nearest-integer rounded, residual must stay within
    // +/- ln(2)/2 and is safe for polynomial core evaluation.
    if (residual_q16 < -half_ln2_q16 || residual_q16 > half_ln2_q16)
        return FP_Q16_ERR_BAD_PARAM;

    *out_k = k;
    *out_r_q16 = residual_q16;
    return FP_Q16_OK;
}

// Checked lane-wise exp range-reduction helper with strict no-partial-write
// semantics over parallel {x_q16,k,r_q16} arrays.
//
// Contracts:
// - Null pointers return NULL_PTR with no writes.
// - count < 0 returns BAD_PARAM with no writes.
// - Any alias/overlap among input/output ranges is rejected.
// - All lanes are preflighted before any output lane is committed.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArray(I64 *x_q16,
                                                   I64 *out_k,
                                                   I64 *out_r_q16,
                                                   I64 count)
{
    I64 i;
    I64 staged_k;
    I64 staged_r_q16;
    U64 x_base;
    U64 x_end;
    U64 k_base;
    U64 k_end;
    U64 r_base;
    U64 r_end;
    I32 status;

    if (!x_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve deterministic BAD_PARAM classification for exact-alias tuples
    // before span arithmetic.
    if (x_q16 == out_k || x_q16 == out_r_q16 || out_k == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (!count)
        return FP_Q16_OK;

    status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
    if (status)
        return status;
    status = FPArrayI64SpanChecked(out_k, count, &k_base, &k_end);
    if (status)
        return status;
    status = FPArrayI64SpanChecked(out_r_q16, count, &r_base, &r_end);
    if (status)
        return status;

    if (FPAddressRangesOverlap(x_base, x_end, k_base, k_end) ||
        FPAddressRangesOverlap(x_base, x_end, r_base, r_end) ||
        FPAddressRangesOverlap(k_base, k_end, r_base, r_end))
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight every lane first so hard failures never partially publish.
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpApproxRangeReduceChecked(x_q16[i],
                                                  &staged_k,
                                                  &staged_r_q16);
        if (status)
            return status;
    }

    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpApproxRangeReduceChecked(x_q16[i],
                                                  &out_k[i],
                                                  &out_r_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16ExpApproxRangeReduceCheckedNoPartialArray.
//
// Staged totals model:
//   staged_count          = count
//   staged_required_cells = 2 * count      (k + r lanes)
//   staged_required_bytes = staged_required_cells * sizeof(I64)
//
// Contract:
// - Composes FPQ16ExpApproxRangeReduceCheckedNoPartialArray first.
// - Revalidates immutable snapshots for {x_q16,out_k,out_r_q16,count}.
// - Atomically publishes {count,required_cells,required_bytes} exactly once
//   on success.
// - Preserves strict no-partial semantics on hard failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnly(I64 *x_q16,
                                                             I64 *out_k,
                                                             I64 *out_r_q16,
                                                             I64 count,
                                                             I64 *out_count,
                                                             I64 *out_required_cells,
                                                             I64 *out_required_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_k;
    I64 *snapshot_out_r_q16;
    I64 snapshot_count;
    I64 staged_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    U64 x_base;
    U64 x_end;
    U64 k_base;
    U64 k_end;
    U64 r_base;
    U64 r_end;
    U64 count_base;
    U64 count_end;
    U64 cells_base;
    U64 cells_end;
    U64 bytes_base;
    U64 bytes_end;
    I32 status;

    if (!out_count || !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == out_required_cells ||
        out_count == out_required_bytes ||
        out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == x_q16 || out_count == out_k || out_count == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == x_q16 || out_required_cells == out_k || out_required_cells == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_bytes == x_q16 || out_required_bytes == out_k || out_required_bytes == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_k, count, &k_base, &k_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_r_q16, count, &r_base, &r_end);
        if (status)
            return status;

        count_base = (U64)out_count;
        cells_base = (U64)out_required_cells;
        bytes_base = (U64)out_required_bytes;

        if (count_base > (U64_MAX_VALUE - 7) ||
            cells_base > (U64_MAX_VALUE - 7) ||
            bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        count_end = count_base + 8;
        cells_end = cells_base + 8;
        bytes_end = bytes_base + 8;

        if (FPAddressRangesOverlap(count_base, count_end, x_base, x_end) ||
            FPAddressRangesOverlap(count_base, count_end, k_base, k_end) ||
            FPAddressRangesOverlap(count_base, count_end, r_base, r_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, x_base, x_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, k_base, k_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, r_base, r_end) ||
            FPAddressRangesOverlap(bytes_base, bytes_end, x_base, x_end) ||
            FPAddressRangesOverlap(bytes_base, bytes_end, k_base, k_end) ||
            FPAddressRangesOverlap(bytes_base, bytes_end, r_base, r_end) ||
            FPAddressRangesOverlap(count_base, count_end, cells_base, cells_end) ||
            FPAddressRangesOverlap(count_base, count_end, bytes_base, bytes_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, bytes_base, bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_x_q16 = x_q16;
    snapshot_out_k = out_k;
    snapshot_out_r_q16 = out_r_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArray(x_q16,
                                                            out_k,
                                                            out_r_q16,
                                                            count);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_k != out_k)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_r_q16 != out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > (I64_MAX_VALUE >> 1))
        return FP_Q16_ERR_OVERFLOW;

    staged_count = count;
    staged_required_cells = count << 1;

    if (staged_required_cells > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_required_bytes = staged_required_cells << 3;

    *out_count = staged_count;
    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnly.
//
// This preflight helper validates that each input lane can pass checked
// range reduction, computes canonical diagnostics
//   {count, required_cells = 2*count, required_bytes = required_cells*8}
// and atomically publishes diagnostics only after all checks pass.
//
// Contract:
// - Never writes {out_k,out_r_q16}; these buffers are validated but treated as
//   immutable snapshots for no-write preflight parity.
// - Enforces non-alias for x/k/r arrays, diagnostics scalar aliases, and
//   diagnostics-scalar-vs-array overlap for count>0.
// - Preserves strict no-partial diagnostics writes on all failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnly(I64 *x_q16,
                                                                           I64 *out_k,
                                                                           I64 *out_r_q16,
                                                                           I64 count,
                                                                           I64 *out_count,
                                                                           I64 *out_required_cells,
                                                                           I64 *out_required_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_k;
    I64 *snapshot_out_r_q16;
    I64 snapshot_count;
    I64 staged_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 staged_required_cells_recomputed;
    I64 staged_required_bytes_recomputed;
    I64 lane_k;
    I64 lane_r;
    U64 x_base;
    U64 x_end;
    U64 k_base;
    U64 k_end;
    U64 r_base;
    U64 r_end;
    U64 count_base;
    U64 count_end;
    U64 cells_base;
    U64 cells_end;
    U64 bytes_base;
    U64 bytes_end;
    I64 i;
    I32 status;

    if (!x_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (!out_count || !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_k || x_q16 == out_r_q16 || out_k == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == out_required_cells ||
        out_count == out_required_bytes ||
        out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == x_q16 || out_count == out_k || out_count == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_cells == x_q16 || out_required_cells == out_k || out_required_cells == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_bytes == x_q16 || out_required_bytes == out_k || out_required_bytes == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_k, count, &k_base, &k_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_r_q16, count, &r_base, &r_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(x_base, x_end, k_base, k_end) ||
            FPAddressRangesOverlap(x_base, x_end, r_base, r_end) ||
            FPAddressRangesOverlap(k_base, k_end, r_base, r_end))
            return FP_Q16_ERR_BAD_PARAM;

        count_base = (U64)out_count;
        cells_base = (U64)out_required_cells;
        bytes_base = (U64)out_required_bytes;

        if (count_base > (U64_MAX_VALUE - 7) ||
            cells_base > (U64_MAX_VALUE - 7) ||
            bytes_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;

        count_end = count_base + 8;
        cells_end = cells_base + 8;
        bytes_end = bytes_base + 8;

        if (FPAddressRangesOverlap(count_base, count_end, x_base, x_end) ||
            FPAddressRangesOverlap(count_base, count_end, k_base, k_end) ||
            FPAddressRangesOverlap(count_base, count_end, r_base, r_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, x_base, x_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, k_base, k_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, r_base, r_end) ||
            FPAddressRangesOverlap(bytes_base, bytes_end, x_base, x_end) ||
            FPAddressRangesOverlap(bytes_base, bytes_end, k_base, k_end) ||
            FPAddressRangesOverlap(bytes_base, bytes_end, r_base, r_end) ||
            FPAddressRangesOverlap(count_base, count_end, cells_base, cells_end) ||
            FPAddressRangesOverlap(count_base, count_end, bytes_base, bytes_end) ||
            FPAddressRangesOverlap(cells_base, cells_end, bytes_base, bytes_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_x_q16 = x_q16;
    snapshot_out_k = out_k;
    snapshot_out_r_q16 = out_r_q16;
    snapshot_count = count;

    // Validate each lane with checked range reduction but do not commit any
    // lane output buffers in this preflight companion.
    for (i = 0; i < count; ++i)
    {
        status = FPQ16ExpApproxRangeReduceChecked(x_q16[i],
                                                  &lane_k,
                                                  &lane_r);
        if (status)
            return status;
    }

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_k != out_k)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_r_q16 != out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > (I64_MAX_VALUE >> 1))
        return FP_Q16_ERR_OVERFLOW;

    staged_count = count;
    staged_required_cells = count << 1;

    status = FPTryMulI64Checked(staged_count, 2, &staged_required_cells_recomputed);
    if (status)
        return status;
    if (staged_required_cells != staged_required_cells_recomputed)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_required_cells > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_required_bytes = staged_required_cells << 3;

    status = FPTryMulI64Checked(staged_required_cells, 8, &staged_required_bytes_recomputed);
    if (status)
        return status;
    if (staged_required_bytes != staged_required_bytes_recomputed)
        return FP_Q16_ERR_BAD_PARAM;

    *out_count = staged_count;
    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only parity gate over
// FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnly and
// FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnly.
//
// Contract:
// - Stages {required_cells,required_bytes} via no-write preflight helper.
// - Revalidates immutable snapshots {count,x_q16,out_k,out_r_q16}.
// - Stages a second diagnostics tuple through commit-only composition.
// - Requires exact tuple parity on {count,required_cells,required_bytes}
//   before atomic diagnostics publish.
// - Preserves strict no-partial diagnostics writes on all failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParity(I64 *x_q16,
                                                                                 I64 *out_k,
                                                                                 I64 *out_r_q16,
                                                                                 I64 count,
                                                                                 I64 *out_count,
                                                                                 I64 *out_required_cells,
                                                                                 I64 *out_required_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_k;
    I64 *snapshot_out_r_q16;
    I64 snapshot_count;
    I64 preflight_count;
    I64 preflight_required_cells;
    I64 preflight_required_bytes;
    I64 commit_count;
    I64 commit_required_cells;
    I64 commit_required_bytes;
    I32 status;

    if (!x_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (!out_count || !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_k || x_q16 == out_r_q16 || out_k == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == out_required_cells ||
        out_count == out_required_bytes ||
        out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == x_q16 ||
        out_count == out_k ||
        out_count == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_cells == x_q16 ||
        out_required_cells == out_k ||
        out_required_cells == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (out_required_bytes == x_q16 ||
        out_required_bytes == out_k ||
        out_required_bytes == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_x_q16 = x_q16;
    snapshot_out_k = out_k;
    snapshot_out_r_q16 = out_r_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnly(
        x_q16,
        out_k,
        out_r_q16,
        count,
        &preflight_count,
        &preflight_required_cells,
        &preflight_required_bytes);
    if (status)
        return status;

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnly(
        x_q16,
        out_k,
        out_r_q16,
        count,
        &commit_count,
        &commit_required_cells,
        &commit_required_bytes);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_k != out_k)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_r_q16 != out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (preflight_count != commit_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (preflight_required_cells != commit_required_cells)
        return FP_Q16_ERR_BAD_PARAM;
    if (preflight_required_bytes != commit_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_count = preflight_count;
    *out_required_cells = preflight_required_cells;
    *out_required_bytes = preflight_required_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParity.
//
// Contract:
// - Stages canonical {count,required_cells,required_bytes} through parity gate.
// - Revalidates immutable snapshots {count,x_q16,out_k,out_r_q16}.
// - Publishes staged diagnostics atomically on success.
// - Preserves strict no-partial diagnostics writes on all failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(I64 *x_q16,
                                                                                           I64 *out_k,
                                                                                           I64 *out_r_q16,
                                                                                           I64 count,
                                                                                           I64 *out_count,
                                                                                           I64 *out_required_cells,
                                                                                           I64 *out_required_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_k;
    I64 *snapshot_out_r_q16;
    I64 snapshot_count;
    I64 staged_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I32 status;

    if (!x_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (!out_count || !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_k || x_q16 == out_r_q16 || out_k == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == out_required_cells ||
        out_count == out_required_bytes ||
        out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == x_q16 ||
        out_count == out_k ||
        out_count == out_r_q16 ||
        out_required_cells == x_q16 ||
        out_required_cells == out_k ||
        out_required_cells == out_r_q16 ||
        out_required_bytes == x_q16 ||
        out_required_bytes == out_k ||
        out_required_bytes == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_x_q16 = x_q16;
    snapshot_out_k = out_k;
    snapshot_out_r_q16 = out_r_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParity(
        x_q16,
        out_k,
        out_r_q16,
        count,
        &staged_count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_k != out_k)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_r_q16 != out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    *out_count = staged_count;
    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly.
//
// Contract:
// - Never writes {out_k,out_r_q16}; they are treated as immutable snapshots.
// - Stages canonical diagnostics tuple through no-write parity helper.
// - Recomputes canonical tuple from immutable {count} with checked math.
// - Atomically publishes diagnostics once after parity + snapshot validation.
// - Preserves strict no-partial diagnostics writes on all failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(I64 *x_q16,
                                                                                                        I64 *out_k,
                                                                                                        I64 *out_r_q16,
                                                                                                        I64 count,
                                                                                                        I64 *out_count,
                                                                                                        I64 *out_required_cells,
                                                                                                        I64 *out_required_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_k;
    I64 *snapshot_out_r_q16;
    I64 snapshot_count;
    I64 staged_count;
    I64 staged_required_cells;
    I64 staged_required_bytes;
    I64 canonical_count;
    I64 canonical_required_cells;
    I64 canonical_required_bytes;
    I64 recomputed_required_cells;
    I64 recomputed_required_bytes;
    I32 status;

    if (!x_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (!out_count || !out_required_cells || !out_required_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_k || x_q16 == out_r_q16 || out_k == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == out_required_cells ||
        out_count == out_required_bytes ||
        out_required_cells == out_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_count == x_q16 ||
        out_count == out_k ||
        out_count == out_r_q16 ||
        out_required_cells == x_q16 ||
        out_required_cells == out_k ||
        out_required_cells == out_r_q16 ||
        out_required_bytes == x_q16 ||
        out_required_bytes == out_k ||
        out_required_bytes == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_x_q16 = x_q16;
    snapshot_out_k = out_k;
    snapshot_out_r_q16 = out_r_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(
        x_q16,
        out_k,
        out_r_q16,
        count,
        &staged_count,
        &staged_required_cells,
        &staged_required_bytes);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_k != out_k)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_r_q16 != out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    canonical_count = count;
    if (canonical_count > (I64_MAX_VALUE >> 1))
        return FP_Q16_ERR_OVERFLOW;

    canonical_required_cells = canonical_count << 1;
    status = FPTryMulI64Checked(canonical_count, 2, &recomputed_required_cells);
    if (status)
        return status;
    if (canonical_required_cells != recomputed_required_cells)
        return FP_Q16_ERR_BAD_PARAM;

    if (canonical_required_cells > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    canonical_required_bytes = canonical_required_cells << 3;
    status = FPTryMulI64Checked(canonical_required_cells, 8, &recomputed_required_bytes);
    if (status)
        return status;
    if (canonical_required_bytes != recomputed_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_count != canonical_count)
        return FP_Q16_ERR_BAD_PARAM;
    if (staged_required_cells != canonical_required_cells)
        return FP_Q16_ERR_BAD_PARAM;
    if (staged_required_bytes != canonical_required_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_count = staged_count;
    *out_required_cells = staged_required_cells;
    *out_required_bytes = staged_required_bytes;
    return FP_Q16_OK;
}

// Diagnostics helper over FPQ16ExpApproxRangeReduceCheckedNoPartialArray that
// computes byte requirement for one output lane array in the paired
// {out_k,out_r_q16} outputs:
//   required_output_bytes = count * sizeof(I64)
//
// Contract:
// - Composes FPQ16ExpApproxRangeReduceCheckedNoPartialArray first.
// - Derives required_output_bytes with checked integer math.
// - Publishes required_output_bytes exactly once on success/overflow.
// - Performs no required-bytes write on hard failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes(I64 *x_q16,
                                                                I64 *out_k,
                                                                I64 *out_r_q16,
                                                                I64 count,
                                                                I64 *out_required_output_bytes)
{
    I64 staged_required_output_bytes;
    U64 required_base;
    U64 required_end;
    U64 x_base;
    U64 x_end;
    U64 k_base;
    U64 k_end;
    U64 r_base;
    U64 r_end;
    I32 status;

    if (!x_q16 || !out_k || !out_r_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == x_q16 ||
        out_required_output_bytes == out_k ||
        out_required_output_bytes == out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve no-partial semantics by rejecting any overlap between the
    // diagnostics scalar and lane-array spans before delegated compute.
    if (count)
    {
        status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_k, count, &k_base, &k_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_r_q16, count, &r_base, &r_end);
        if (status)
            return status;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, x_base, x_end) ||
            FPAddressRangesOverlap(required_base, required_end, k_base, k_end) ||
            FPAddressRangesOverlap(required_base, required_end, r_base, r_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArray(x_q16,
                                                            out_k,
                                                            out_r_q16,
                                                            count);
    if (status)
        return status;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    staged_required_output_bytes = count << 3;
    *out_required_output_bytes = staged_required_output_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes.
//
// This stages required_output_bytes locally, revalidates immutable call-site
// snapshots, and only then publishes the diagnostics scalar to the caller.
//
// Contract:
// - Composes FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes first.
// - Revalidates immutable {x_q16,out_k,out_r_q16,count} snapshots.
// - Preserves no-partial diagnostics writes on hard failures.
I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytesCommitOnly(I64 *x_q16,
                                                                          I64 *out_k,
                                                                          I64 *out_r_q16,
                                                                          I64 count,
                                                                          I64 *out_required_output_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_k;
    I64 *snapshot_out_r_q16;
    I64 snapshot_count;
    I64 staged_required_output_bytes;
    I32 status;

    if (!out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_x_q16 = x_q16;
    snapshot_out_k = out_k;
    snapshot_out_r_q16 = out_r_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes(
        x_q16,
        out_k,
        out_r_q16,
        count,
        &staged_required_output_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_k != out_k)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_r_q16 != out_r_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_required_output_bytes;
    return FP_Q16_OK;
}

// Checked integer-only Q16 exponential approximation.
//
// Math model:
//   x = k*ln(2) + r,  where k is nearest integer and |r| <= ln(2)/2
//   exp(x) = 2^k * exp(r)
//   exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24   (all terms in Q16)
//
// Contracts:
// - Null output pointer returns NULL_PTR with no write.
// - Uses checked Q16 multiply helpers for polynomial terms.
// - Saturates to I64_MAX_VALUE and returns OVERFLOW on upper overflow.
// - Underflow to sub-Q16 magnitudes returns 0.
// - Commits caller output exactly once.
I32 FPQ16ExpApproxChecked(I64 x_q16,
                          I64 *out_q16)
{
    I32 status;
    I64 k;
    I64 shift_mag;
    I64 residual_q16;
    I64 residual2_q16;
    I64 residual3_q16;
    I64 residual4_q16;
    I64 poly_q16;
    I64 staged_out_q16;

    if (!out_q16)
        return FP_Q16_ERR_NULL_PTR;

    status = FPQ16ExpApproxRangeReduceChecked(x_q16,
                                              &k,
                                              &residual_q16);
    if (status != FP_Q16_OK)
        return status;

    if (FPQ16MulChecked(residual_q16,
                        residual_q16,
                        &residual2_q16))
        return FP_Q16_ERR_OVERFLOW;

    if (FPQ16MulChecked(residual2_q16,
                        residual_q16,
                        &residual3_q16))
        return FP_Q16_ERR_OVERFLOW;

    if (FPQ16MulChecked(residual3_q16,
                        residual_q16,
                        &residual4_q16))
        return FP_Q16_ERR_OVERFLOW;

    // Direct polynomial accumulation keeps every approximation term visible.
    poly_q16 = FP_Q16_ONE;
    if (residual_q16 > 0 && poly_q16 > (I64_MAX_VALUE - residual_q16))
        return FP_Q16_ERR_OVERFLOW;
    if (residual_q16 < 0 && poly_q16 < (I64_MIN_VALUE - residual_q16))
        return FP_Q16_ERR_OVERFLOW;
    poly_q16 += residual_q16;

    residual2_q16 = residual2_q16 / 2;
    residual3_q16 = residual3_q16 / 6;
    residual4_q16 = residual4_q16 / 24;

    if (residual2_q16 > 0 && poly_q16 > (I64_MAX_VALUE - residual2_q16))
        return FP_Q16_ERR_OVERFLOW;
    if (residual2_q16 < 0 && poly_q16 < (I64_MIN_VALUE - residual2_q16))
        return FP_Q16_ERR_OVERFLOW;
    poly_q16 += residual2_q16;

    if (residual3_q16 > 0 && poly_q16 > (I64_MAX_VALUE - residual3_q16))
        return FP_Q16_ERR_OVERFLOW;
    if (residual3_q16 < 0 && poly_q16 < (I64_MIN_VALUE - residual3_q16))
        return FP_Q16_ERR_OVERFLOW;
    poly_q16 += residual3_q16;

    if (residual4_q16 > 0 && poly_q16 > (I64_MAX_VALUE - residual4_q16))
        return FP_Q16_ERR_OVERFLOW;
    if (residual4_q16 < 0 && poly_q16 < (I64_MIN_VALUE - residual4_q16))
        return FP_Q16_ERR_OVERFLOW;
    poly_q16 += residual4_q16;

    if (poly_q16 <= 0)
    {
        *out_q16 = 0;
        return FP_Q16_OK;
    }

    if (k >= 0)
    {
        if (k >= 63)
        {
            *out_q16 = I64_MAX_VALUE;
            return FP_Q16_ERR_OVERFLOW;
        }

        if (poly_q16 > (I64_MAX_VALUE >> k))
        {
            *out_q16 = I64_MAX_VALUE;
            return FP_Q16_ERR_OVERFLOW;
        }

        staged_out_q16 = poly_q16 << k;
    }
    else
    {
        shift_mag = -k;
        if (shift_mag >= 63)
            staged_out_q16 = 0;
        else
            staged_out_q16 = FPQ16RoundShiftSigned(poly_q16, shift_mag);
    }

    if (staged_out_q16 < 0)
    {
        *out_q16 = 0;
        return FP_Q16_OK;
    }

    *out_q16 = staged_out_q16;
    return FP_Q16_OK;
}

// Checked lane-wise exp approximation helper with strict no-partial-write
// semantics over parallel {x_q16,out_q16} arrays.
//
// Contract:
// - Null pointers return NULL_PTR with no writes.
// - count < 0 returns BAD_PARAM with no writes.
// - Any alias/overlap between input/output ranges is rejected.
// - All lanes are preflighted before any output lane is committed.
// - OVERFLOW from one or more lanes is reported after full publish.
I32 FPQ16ExpApproxCheckedNoPartialArray(I64 *x_q16,
                                        I64 *out_q16,
                                        I64 count)
{
    I64 i;
    I64 staged_lane_q16;
    U64 x_base;
    U64 x_end;
    U64 out_base;
    U64 out_end;
    I32 status;
    I32 overflow_seen;

    if (!x_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (!count)
        return FP_Q16_OK;

    status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
    if (status)
        return status;
    status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
    if (status)
        return status;

    if (FPAddressRangesOverlap(x_base, x_end, out_base, out_end))
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight every lane first so hard failures never partially publish.
    overflow_seen = 0;
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpApproxChecked(x_q16[i],
                                       &staged_lane_q16);
        if (status == FP_Q16_ERR_OVERFLOW)
            overflow_seen = 1;
        else if (status != FP_Q16_OK)
            return status;
    }

    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpApproxChecked(x_q16[i],
                                       &out_q16[i]);
        if (status == FP_Q16_ERR_OVERFLOW)
            overflow_seen = 1;
        else if (status != FP_Q16_OK)
            return status;
    }

    if (overflow_seen)
        return FP_Q16_ERR_OVERFLOW;
    return FP_Q16_OK;
}

// Checked lane-wise exp approximation with strict no-partial-write semantics.
//
// Math model per lane:
//   out_q16[i] = FPQ16ExpApproxChecked(x_q16[i])
//
// Contracts:
// - Null pointers return NULL_PTR with no writes.
// - count < 0 returns BAD_PARAM with no writes.
// - Any alias/overlap between input/output spans is rejected.
// - Two-pass execution: preflight every lane first, then commit all lanes.
I32 FPQ16ExpApproxCheckedNoPartialArray(I64 *x_q16,
                                        I64 *out_q16,
                                        I64 count)
{
    I64 i;
    I64 staged_lane_q16;
    U64 x_base;
    U64 x_end;
    U64 out_base;
    U64 out_end;
    I32 status;

    if (!x_q16 || !out_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (x_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (!count)
        return FP_Q16_OK;

    status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
    if (status)
        return status;
    status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
    if (status)
        return status;

    if (FPAddressRangesOverlap(x_base, x_end, out_base, out_end))
        return FP_Q16_ERR_BAD_PARAM;

    // Preflight all lanes first so hard failures never partially publish.
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpApproxChecked(x_q16[i], &staged_lane_q16);
        if (status)
            return status;
    }

    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpApproxChecked(x_q16[i], &out_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Diagnostics helper over FPQ16ExpApproxCheckedNoPartialArray that computes
// byte requirement for the output array:
//   required_output_bytes = count * sizeof(I64)
//
// Contract:
// - Composes FPQ16ExpApproxCheckedNoPartialArray first.
// - Derives required_output_bytes with checked integer math.
// - Publishes required_output_bytes exactly once on success.
// - Performs no required-bytes write on hard failures and overflow paths.
I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes(I64 *x_q16,
                                                     I64 *out_q16,
                                                     I64 count,
                                                     I64 *out_required_output_bytes)
{
    I64 staged_required_output_bytes;
    U64 required_base;
    U64 required_end;
    U64 x_base;
    U64 x_end;
    U64 out_base;
    U64 out_end;
    I32 status;

    if (!x_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == x_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve no-partial semantics by rejecting overlap between diagnostics
    // scalar output and lane-array spans before delegated compute.
    if (count)
    {
        status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
        if (status)
            return status;
        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, x_base, x_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    status = FPQ16ExpApproxCheckedNoPartialArray(x_q16,
                                                 out_q16,
                                                 count);
    if (status)
        return status;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    staged_required_output_bytes = count << 3;
    *out_required_output_bytes = staged_required_output_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes.
//
// This stages required_output_bytes locally, revalidates immutable call-site
// snapshots, and only then publishes the diagnostics scalar to the caller.
//
// Contract:
// - Composes FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes first.
// - Revalidates immutable {x_q16,out_q16,count} snapshots.
// - Commits required_output_bytes exactly once on success.
// - Preserves no-partial diagnostics writes on delegated hard failures.
I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly(I64 *x_q16,
                                                               I64 *out_q16,
                                                               I64 count,
                                                               I64 *out_required_output_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_output_bytes;
    I32 status;

    if (!out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_x_q16 = x_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes(
        x_q16,
        out_q16,
        count,
        &staged_required_output_bytes);
    if (status != FP_Q16_OK)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_required_output_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only parity gate over
// FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly and
// FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes.
//
// Contract:
// - Stages required_output_bytes through commit-only wrapper.
// - Revalidates immutable {x_q16,out_q16,count} snapshots.
// - Recomputes canonical required_output_bytes through base helper.
// - Enforces exact tuple parity before atomic diagnostics publish.
// - Preserves strict no-partial diagnostics writes on all failures.
I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParity(I64 *x_q16,
                                                                     I64 *out_q16,
                                                                     I64 count,
                                                                     I64 *out_required_output_bytes)
{
    U64 required_base;
    U64 required_end;
    U64 x_base;
    U64 x_end;
    U64 out_base;
    U64 out_end;
    I64 *snapshot_x_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_commit_required_output_bytes;
    I64 staged_canonical_required_output_bytes;
    I32 status;

    if (!x_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == x_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    // Keep strict no-partial semantics for final diagnostics publish by
    // rejecting scalar-output overlap against caller array spans.
    if (count)
    {
        status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(x_base, x_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, x_base, x_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_x_q16 = x_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly(
        x_q16,
        out_q16,
        count,
        &staged_commit_required_output_bytes);
    if (status)
        return status;

    status = FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes(
        x_q16,
        out_q16,
        count,
        &staged_canonical_required_output_bytes);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (staged_commit_required_output_bytes != staged_canonical_required_output_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_commit_required_output_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParity.
//
// This helper preserves strict preflight semantics by delegating to the
// existing commit-only parity wrapper, recomputing canonical
// required_output_bytes with checked math, and enforcing parity before a
// single diagnostics publish.
//
// Contract:
// - Never writes out_q16 (strict preflight/no-partial semantics).
// - Revalidates immutable {x_q16,out_q16,count} snapshots.
// - Enforces exact parity on {required_output_bytes} before publish.
// - Performs exactly one diagnostics commit on success.
I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParityPreflightOnly(I64 *x_q16,
                                                                                   I64 *out_q16,
                                                                                   I64 count,
                                                                                   I64 *out_required_output_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_parity_required_output_bytes;
    I64 staged_canonical_required_output_bytes;
    I32 status;

    if (!x_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == x_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_x_q16 = x_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParity(
        x_q16,
        out_q16,
        count,
        &staged_parity_required_output_bytes);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPTryMulI64Checked(count,
                                8,
                                &staged_canonical_required_output_bytes);
    if (status)
        return status;

    if (staged_parity_required_output_bytes != staged_canonical_required_output_bytes)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_parity_required_output_bytes;
    return FP_Q16_OK;
}

// Commit-only diagnostics wrapper over
// FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParityPreflightOnly.
//
// Contract:
// - Stages required_output_bytes through the parity preflight helper.
// - Revalidates immutable {x_q16,out_q16,count} snapshots.
// - Performs exactly one diagnostics publish on success.
// - Preserves strict no-partial semantics on all failures.
I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParityPreflightOnlyCommitOnly(I64 *x_q16,
                                                                                             I64 *out_q16,
                                                                                             I64 count,
                                                                                             I64 *out_required_output_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_output_bytes;
    I32 status;

    if (!out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_x_q16 = x_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    status = FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParityPreflightOnly(
        x_q16,
        out_q16,
        count,
        &staged_required_output_bytes);
    if (status)
        return status;

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_required_output_bytes;
    return FP_Q16_OK;
}

// Diagnostics-only no-write companion over
// FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly.
//
// This preflight helper validates each source lane with scalar checked exp,
// keeps out_q16 immutable, computes canonical required_output_bytes with
// checked integer math, and atomically publishes diagnostics on success.
//
// Contract:
// - Never writes out_q16 (strict preflight/no-partial semantics).
// - Revalidates immutable {x_q16,out_q16,count} snapshots.
// - Computes required_output_bytes = count * sizeof(I64) with checked math.
// - Performs exactly one diagnostics commit on success.
I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly(I64 *x_q16,
                                                                             I64 *out_q16,
                                                                             I64 count,
                                                                             I64 *out_required_output_bytes)
{
    I64 *snapshot_x_q16;
    I64 *snapshot_out_q16;
    I64 snapshot_count;
    I64 staged_required_output_bytes;
    I64 staged_required_output_bytes_recomputed;
    I64 staged_lane_q16;
    U64 required_base;
    U64 required_end;
    U64 x_base;
    U64 x_end;
    U64 out_base;
    U64 out_end;
    I64 i;
    I32 status;

    if (!x_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (x_q16 == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (out_required_output_bytes == x_q16 ||
        out_required_output_bytes == out_q16)
        return FP_Q16_ERR_BAD_PARAM;

    if (count)
    {
        status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);
        if (status)
            return status;

        status = FPArrayI64SpanChecked(out_q16, count, &out_base, &out_end);
        if (status)
            return status;

        if (FPAddressRangesOverlap(x_base, x_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;

        required_base = (U64)out_required_output_bytes;
        if (required_base > (U64_MAX_VALUE - 7))
            return FP_Q16_ERR_OVERFLOW;
        required_end = required_base + 8;

        if (FPAddressRangesOverlap(required_base, required_end, x_base, x_end) ||
            FPAddressRangesOverlap(required_base, required_end, out_base, out_end))
            return FP_Q16_ERR_BAD_PARAM;
    }

    snapshot_x_q16 = x_q16;
    snapshot_out_q16 = out_q16;
    snapshot_count = count;

    // Preflight validation: ensure every lane is representable by scalar exp,
    // but do not commit any lane outputs into out_q16.
    for (i = 0; i < count; ++i)
    {
        status = FPQ16ExpApproxChecked(x_q16[i],
                                       &staged_lane_q16);
        if (status)
            return status;
    }

    if (snapshot_x_q16 != x_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_out_q16 != out_q16)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_required_output_bytes = count << 3;

    status = FPTryMulI64Checked(count,
                                8,
                                &staged_required_output_bytes_recomputed);
    if (status)
        return status;
    if (staged_required_output_bytes != staged_required_output_bytes_recomputed)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_required_output_bytes;
    return FP_Q16_OK;
}
