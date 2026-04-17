// intexp.HC
// Integer-only Q16.16 exponential approximation for TempleOS HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)

#define FP_Q16_OK            0
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_OVERFLOW  4

#define EXP_Q16_LN2       45426   // round(ln(2) * 65536)
#define EXP_Q16_MAX_INPUT 655360  // +10.0 in Q16
#define EXP_Q16_MIN_INPUT -655360 // -10.0 in Q16

#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF
#define U64_MAX_VALUE 0xFFFFFFFFFFFFFFFF

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

// Vector checked clamp helper for Q16 exp input-domain saturation.
// Applies the same scalar clamp contract lane-by-lane:
//   out[i] = clamp(in[i], EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT)
// This keeps scalar and vector behavior identical by delegating each lane to
// FPQ16ExpClampToInputDomainChecked instead of duplicating clamp logic.
I32 FPQ16ExpClampArrayToInputDomainChecked(I64 *input_q16,
                                           I64 *output_q16,
                                           I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic used by indexed array access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpClampToInputDomainChecked(input_q16[i],
                                                   &output_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Checked Q16 exp evaluation for callers that already performed input-domain
// clamping. This helper refuses out-of-domain inputs instead of clamping:
//   requires EXP_Q16_MIN_INPUT <= clamped_input_q16 <= EXP_Q16_MAX_INPUT
//   returns BAD_PARAM on domain violations
// This keeps split-phase clamp+eval vector paths explicit and auditable.
I32 FPQ16ExpFromClampedInputChecked(I64 clamped_input_q16,
                                    I64 *out_exp_q16)
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

    if (!out_exp_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (clamped_input_q16 < EXP_Q16_MIN_INPUT ||
        clamped_input_q16 > EXP_Q16_MAX_INPUT)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve scalar saturation contract exactly on domain boundaries.
    if (clamped_input_q16 >= EXP_Q16_MAX_INPUT)
    {
        *out_exp_q16 = I64_MAX_VALUE;
        return FP_Q16_OK;
    }
    if (clamped_input_q16 <= EXP_Q16_MIN_INPUT)
    {
        *out_exp_q16 = 0;
        return FP_Q16_OK;
    }

    // Range reduction: x = k*ln(2) + r, where r in [0, ln(2)).
    k = clamped_input_q16 / EXP_Q16_LN2;
    if (clamped_input_q16 < 0 && (clamped_input_q16 % EXP_Q16_LN2))
        k--;

    r = clamped_input_q16 - (k * EXP_Q16_LN2);

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
        {
            *out_exp_q16 = I64_MAX_VALUE;
            return FP_Q16_OK;
        }

        if (poly > (I64_MAX_VALUE >> k))
        {
            *out_exp_q16 = I64_MAX_VALUE;
            return FP_Q16_OK;
        }

        *out_exp_q16 = poly << k;
        return FP_Q16_OK;
    }

    if (k <= -63)
    {
        *out_exp_q16 = 0;
        return FP_Q16_OK;
    }

    *out_exp_q16 = poly >> (-k);
    return FP_Q16_OK;
}

// Checked vector exp helper for softmax precompute lanes.
// For each lane i, this composes the scalar checked path directly:
//   clamped = clamp(input_q16[i], EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT)
//   output_q16[i] = exp(clamped)
// This preserves scalar saturation/domain semantics exactly while keeping all
// pointer/count arithmetic guarded for deterministic TempleOS-safe behavior.
I32 FPQ16ExpArrayChecked(I64 *input_q16,
                         I64 *output_q16,
                         I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 clamped_lane;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic used by indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpClampToInputDomainChecked(input_q16[i],
                                                   &clamped_lane);
        if (status)
            return status;

        status = FPQ16ExpFromClampedInputChecked(clamped_lane,
                                                 &output_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Checked vector exp helper for callers that already pre-clamped lanes into
// the scalar-domain contract:
//   EXP_Q16_MIN_INPUT <= input_q16[i] <= EXP_Q16_MAX_INPUT
// On any out-of-domain lane this returns BAD_PARAM and performs zero writes,
// so split-phase clamp/eval callers never observe partially-updated output.
I32 FPQ16ExpArrayFromClampedInputChecked(I64 *input_q16,
                                         I64 *output_q16,
                                         I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic used by indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Fail fast before any writes so output remains untouched on BAD_PARAM.
    for (i = 0; i < count; i++)
    {
        if (input_q16[i] < EXP_Q16_MIN_INPUT ||
            input_q16[i] > EXP_Q16_MAX_INPUT)
            return FP_Q16_ERR_BAD_PARAM;
    }

    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpFromClampedInputChecked(input_q16[i],
                                                 &output_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Checked no-alias vector exp helper for pre-clamped lanes.
// Contract is explicit for immutable-source call sites:
//   - input_q16 and output_q16 must both be non-null
//   - input_q16 and output_q16 must be different buffers
//   - all lanes must already satisfy EXP_Q16_MIN_INPUT..EXP_Q16_MAX_INPUT
// This composes the existing pre-clamped helper so pointer-span overflow
// checks and lane math stay centralized in one auditable implementation.
I32 FPQ16ExpArrayCheckedNoAliasChecked(I64 *input_q16,
                                       I64 *output_q16,
                                       I64 count)
{
    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (input_q16 == output_q16)
        return FP_Q16_ERR_BAD_PARAM;

    return FPQ16ExpArrayFromClampedInputChecked(input_q16,
                                                output_q16,
                                                count);
}

I64 FPQ16Exp(I64 x)
{
    I32 clamp_status;
    I64 clamped_x;
    I64 out_exp_q16;

    clamp_status = FPQ16ExpClampToInputDomainChecked(x,
                                                     &clamped_x);
    if (clamp_status)
        return 0;

    if (FPQ16ExpFromClampedInputChecked(clamped_x, &out_exp_q16))
        return 0;

    return out_exp_q16;
}
