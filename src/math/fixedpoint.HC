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
                                                  U64 *out_required_output_bytes)
{
    I64 snapshot_count;
    U64 staged_required_output_bytes;
    I32 status;

    if (!lhs_q16 || !rhs_q16 || !out_q16 || !out_required_output_bytes)
        return FP_Q16_ERR_NULL_PTR;

    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;

    snapshot_count = count;

    if ((U64)snapshot_count > (U64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_required_output_bytes = ((U64)snapshot_count) << 3;

    status = FPQ16MulSatCheckedNoPartialArray(lhs_q16,
                                              rhs_q16,
                                              out_q16,
                                              snapshot_count);
    if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)
        return status;

    // Immutable diagnostics contract: caller-shared scalar inputs must stay
    // stable across checked composition before publishing required bytes.
    if (snapshot_count != count)
        return FP_Q16_ERR_BAD_PARAM;

    *out_required_output_bytes = staged_required_output_bytes;
    return status;
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
