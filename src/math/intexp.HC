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
#define EXP_Q16_POLY_C0   FP_Q16_ONE
#define EXP_Q16_POLY_C2_DIV 2
#define EXP_Q16_POLY_C3_DIV 6
#define EXP_Q16_POLY_C4_DIV 24
#define EXP_Q16_LEFT_SHIFT_SAT_K   30
#define EXP_Q16_RIGHT_SHIFT_ZERO_K -63

#define EXP_Q16_SAT_MIN_OUTPUT 0
#define EXP_Q16_SAT_MAX_OUTPUT I64_MAX_VALUE

#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF
#define U64_MAX_VALUE 0xFFFFFFFFFFFFFFFF

I64 FPQ16Mul(I64 a, I64 b);
I32 FPQ16MulChecked(I64 a_q16, I64 b_q16, I64 *out_q16);
I32 FPQ16ExpArrayCheckedNoPartialPreflight(I64 *input_q16,
                                           I64 *output_q16,
                                           I64 count,
                                           I64 *out_staged_bytes);

// Checked polynomial core for exp(r) on the range-reduced residual domain:
//   0 <= r < ln(2)
// Uses monotonic positive-term integer coefficients in Q16:
//   exp(r) ~= 1 + r + r^2/2 + r^3/6 + r^4/24
// All products are checked (no silent saturation) so every failure surface is
// explicit and auditable.
I32 FPQ16ExpPolyCoreLn2ResidualQ16Checked(I64 residual_q16,
                                          I64 *out_poly_q16)
{
    I64 r2;
    I64 r3;
    I64 r4;
    I64 t1;
    I64 t2;
    I64 t3;
    I64 t4;
    I64 poly;
    I32 status;

    if (!out_poly_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (residual_q16 < 0 || residual_q16 >= EXP_Q16_LN2)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16MulChecked(residual_q16,
                             residual_q16,
                             &r2);
    if (status)
        return status;

    status = FPQ16MulChecked(r2,
                             residual_q16,
                             &r3);
    if (status)
        return status;

    status = FPQ16MulChecked(r3,
                             residual_q16,
                             &r4);
    if (status)
        return status;

    t1 = residual_q16;
    t2 = r2 / EXP_Q16_POLY_C2_DIV;
    t3 = r3 / EXP_Q16_POLY_C3_DIV;
    t4 = r4 / EXP_Q16_POLY_C4_DIV;

    // Terms are non-negative on residual domain; checked adds keep overflow
    // behavior explicit before pow2 scaling.
    poly = EXP_Q16_POLY_C0;
    if (poly > (I64_MAX_VALUE - t1))
        return FP_Q16_ERR_OVERFLOW;
    poly += t1;

    if (poly > (I64_MAX_VALUE - t2))
        return FP_Q16_ERR_OVERFLOW;
    poly += t2;

    if (poly > (I64_MAX_VALUE - t3))
        return FP_Q16_ERR_OVERFLOW;
    poly += t3;

    if (poly > (I64_MAX_VALUE - t4))
        return FP_Q16_ERR_OVERFLOW;
    poly += t4;

    *out_poly_q16 = poly;
    return FP_Q16_OK;
}

// Checked all-or-nothing wrapper for the ln2-residual polynomial core.
// This preserves exact status surfaces from the checked core while ensuring
// caller-visible output is only committed on success.
I32 FPQ16ExpPolyCoreLn2ResidualQ16CheckedNoPartial(I64 residual_q16,
                                                   I64 *out_poly_q16)
{
    I64 staged_poly_q16;
    I32 status;

    if (!out_poly_q16)
        return FP_Q16_ERR_NULL_PTR;

    staged_poly_q16 = 0;
    status = FPQ16ExpPolyCoreLn2ResidualQ16Checked(residual_q16,
                                                   &staged_poly_q16);
    if (status)
        return status;

    *out_poly_q16 = staged_poly_q16;
    return FP_Q16_OK;
}

// Checked vector ln(2)-residual polynomial core evaluator.
// For each lane i:
//   residual_q16[i] -> out_poly_q16[i]
// where scalar-domain constraints are preserved exactly:
//   0 <= residual_q16[i] < EXP_Q16_LN2
//
// No-partial-write semantics:
//   - Pass 1 validates every lane through the scalar checked core.
//   - Pass 2 commits outputs only after full preflight success.
I32 FPQ16ExpPolyCoreLn2ResidualQ16ArrayChecked(I64 *residual_q16,
                                               I64 *out_poly_q16,
                                               I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staged_poly_q16;
    I32 status;

    if (!residual_q16 || !out_poly_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)residual_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_poly_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Pass 1: validate every lane without caller-buffer writes.
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpPolyCoreLn2ResidualQ16Checked(residual_q16[i],
                                                       &staged_poly_q16);
        if (status)
            return status;
    }

    // Pass 2: deterministic commit after successful preflight.
    // Use scalar no-partial staging per lane so caller output write happens
    // exactly once per lane (after successful scalar evaluation).
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpPolyCoreLn2ResidualQ16CheckedNoPartial(residual_q16[i],
                                                                &staged_poly_q16);
        if (status)
            return status;

        out_poly_q16[i] = staged_poly_q16;
    }

    return FP_Q16_OK;
}

// Checked no-partial wrapper for ln(2)-residual vector polynomial core.
// Delegates scalar-domain math to the canonical checked array evaluator,
// but stages all output lanes so caller-visible out buffer is untouched on
// any failure.
I32 FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartial(I64 *residual_q16,
                                                        I64 *out_poly_q16,
                                                        I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staging_bytes;
    I64 *staging_out_poly_q16;
    I32 status;

    if (!residual_q16 || !out_poly_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)residual_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_poly_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staging_bytes = count << 3;
    if (staging_bytes <= 0)
        return FP_Q16_ERR_OVERFLOW;

    staging_out_poly_q16 = MAlloc(staging_bytes);
    if (!staging_out_poly_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpPolyCoreLn2ResidualQ16ArrayChecked(residual_q16,
                                                        staging_out_poly_q16,
                                                        count);
    if (status) {
        Free(staging_out_poly_q16);
        return status;
    }

    for (i = 0; i < count; i++)
        out_poly_q16[i] = staging_out_poly_q16[i];

    Free(staging_out_poly_q16);
    return FP_Q16_OK;
}

// Compatibility alias retained for existing call sites/tests.
I32 FPQ16ExpPolyCoreResidualLn2Q16Checked(I64 residual_q16,
                                          I64 *out_poly_q16)
{
    return FPQ16ExpPolyCoreLn2ResidualQ16Checked(residual_q16,
                                                 out_poly_q16);
}

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

// Checked base-2 range split for clamped Q16 exp inputs.
// Decomposes x into:
//   x = k*ln(2) + r
// where k is floor(x/ln2) and remainder r satisfies 0 <= r < ln(2).
//
// This keeps range-reduction math centralized and auditable for every exp
// caller while validating the ln(2) base constant at runtime.
I32 FPQ16ExpSplitBase2FromClampedInputChecked(I64 clamped_input_q16,
                                              I64 *out_k,
                                              I64 *out_r_q16)
{
    I64 k;
    I64 r;

    if (!out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (clamped_input_q16 < EXP_Q16_MIN_INPUT ||
        clamped_input_q16 > EXP_Q16_MAX_INPUT)
        return FP_Q16_ERR_BAD_PARAM;

    if (EXP_Q16_LN2 <= 0)
        return FP_Q16_ERR_BAD_PARAM;

    // Floor division so negative inputs preserve [0, ln(2)) remainder.
    k = clamped_input_q16 / EXP_Q16_LN2;
    if (clamped_input_q16 < 0 && (clamped_input_q16 % EXP_Q16_LN2))
        k--;

    r = clamped_input_q16 - (k * EXP_Q16_LN2);
    if (r < 0 || r >= EXP_Q16_LN2)
        return FP_Q16_ERR_BAD_PARAM;

    *out_k = k;
    *out_r_q16 = r;
    return FP_Q16_OK;
}

// Checked ln(2)-based range reducer for exp callers that start from an
// unclamped Q16 input. Performs canonical scalar-domain clamp first, then
// decomposes the clamped lane into:
//   clamped_x = k*ln(2) + r
//   0 <= r < ln(2)
// This keeps all FPQ16Exp range-reduction call sites on one auditable path.
I32 FPQ16ExpRangeReduceLn2Checked(I64 input_q16,
                                  I64 *out_k,
                                  I64 *out_r_q16)
{
    I64 clamped_q16;
    I32 status;

    if (!out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;

    status = FPQ16ExpClampToInputDomainChecked(input_q16,
                                               &clamped_q16);
    if (status)
        return status;

    return FPQ16ExpSplitBase2FromClampedInputChecked(clamped_q16,
                                                     out_k,
                                                     out_r_q16);
}

// Checked all-or-nothing wrapper for ln(2) range reduction.
// Stages {k,r} so callers can rely on strict no-partial-write behavior:
//   - On any error, out_k/out_r_q16 remain unchanged.
//   - On success, staged values are committed atomically.
// Success/error surfaces intentionally match FPQ16ExpRangeReduceLn2Checked.
I32 FPQ16ExpRangeReduceLn2CheckedNoPartial(I64 input_q16,
                                           I64 *out_k,
                                           I64 *out_r_q16)
{
    I64 staged_k;
    I64 staged_r_q16;
    I32 status;

    if (!out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;

    // Stage into local scratch only. This wrapper must not depend on prior
    // caller contents of {out_k,out_r_q16}; it only commits on success.
    staged_k = 0;
    staged_r_q16 = 0;

    status = FPQ16ExpRangeReduceLn2Checked(input_q16,
                                           &staged_k,
                                           &staged_r_q16);
    if (status)
        return status;

    *out_k = staged_k;
    *out_r_q16 = staged_r_q16;
    return FP_Q16_OK;
}

// Checked vector ln(2) range reducer for Q16 exp paths.
// For each lane i:
//   input_q16[i] -> {out_k[i], out_r_q16[i]}
// where the scalar contract is preserved exactly:
//   clamped_x = clamp(input_q16[i], EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT)
//   clamped_x = out_k[i] * ln(2) + out_r_q16[i]
//   0 <= out_r_q16[i] < ln(2)
//
// No-partial-write semantics:
//   - First pass validates every lane through scalar checked reduction.
//   - Second pass commits outputs only after global preflight succeeds.
I32 FPQ16ExpRangeReduceLn2ArrayChecked(I64 *input_q16,
                                       I64 *out_k,
                                       I64 *out_r_q16,
                                       I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staged_k;
    I64 staged_r_q16;
    I32 status;

    if (!input_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_k > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_r_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    // Pass 1: full-lane validation without caller-buffer writes.
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpRangeReduceLn2Checked(input_q16[i],
                                               &staged_k,
                                               &staged_r_q16);
        if (status)
            return status;
    }

    // Pass 2: deterministic commit after successful preflight.
    for (i = 0; i < count; i++)
    {
        status = FPQ16ExpRangeReduceLn2Checked(input_q16[i],
                                               &out_k[i],
                                               &out_r_q16[i]);
        if (status)
            return status;
    }

    return FP_Q16_OK;
}

// Checked all-or-nothing vector wrapper for ln(2) range reduction.
// This wrapper preserves exact lane math/error surfaces from
// FPQ16ExpRangeReduceLn2ArrayChecked while strengthening write semantics:
//   - No caller writes to out_k/out_r_q16 on any failure.
//   - Full-lane staged results are committed only after checked core success.
// Pointer-span overflow guards match the checked core before any indexed reads
// or writes, keeping array contracts auditable and deterministic.
I32 FPQ16ExpRangeReduceLn2ArrayCheckedNoPartial(I64 *input_q16,
                                                I64 *out_k,
                                                I64 *out_r_q16,
                                                I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staged_bytes;
    I64 *staged_k;
    I64 *staged_r_q16;
    I32 status;

    if (!input_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before any indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_k > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_r_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_bytes = count << 3;
    if (!staged_bytes)
        return FP_Q16_ERR_OVERFLOW;

    staged_k = MAlloc(staged_bytes);
    if (!staged_k)
        return FP_Q16_ERR_OVERFLOW;

    staged_r_q16 = MAlloc(staged_bytes);
    if (!staged_r_q16)
    {
        Free(staged_k);
        return FP_Q16_ERR_OVERFLOW;
    }

    status = FPQ16ExpRangeReduceLn2ArrayChecked(input_q16,
                                                staged_k,
                                                staged_r_q16,
                                                count);
    if (status)
    {
        Free(staged_r_q16);
        Free(staged_k);
        return status;
    }

    for (i = 0; i < count; i++)
    {
        out_k[i] = staged_k[i];
        out_r_q16[i] = staged_r_q16[i];
    }

    Free(staged_r_q16);
    Free(staged_k);
    return FP_Q16_OK;
}

// Canonical default-capacity wrapper for vector ln(2) range reduction.
// Default contract:
//   - `length` is the active lane span (no extra capacity parameter).
//   - Pointer-span overflow guards are preflighted here so default callers
//     get deterministic guard behavior before checked-core dispatch.
//   - Success/error semantics remain exact pass-through to
//     FPQ16ExpRangeReduceLn2ArrayChecked.
I32 FPQ16ExpRangeReduceLn2ArrayCheckedDefault(I64 *input_q16,
                                              I64 *out_k,
                                              I64 *out_r_q16,
                                              I64 length)
{
    I64 last_index;
    I64 last_byte_offset;

    if (!input_q16 || !out_k || !out_r_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (length < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!length)
        return FP_Q16_OK;

    // Guard indexed span before checked-core dispatch.
    last_index = length - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_k > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)out_r_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    return FPQ16ExpRangeReduceLn2ArrayChecked(input_q16,
                                              out_k,
                                              out_r_q16,
                                              length);
}

// Checked power-of-two scaler for Q16 exponent assembly paths.
// Computes:
//   out_scaled_q16 = base_q16 * (2^k)
// with signed-k support:
//   k >= 0  -> checked left shift (overflow returns OVERFLOW)
//   k < 0   -> arithmetic right shift (very negative shifts clamp to 0)
//
// Contracts:
//   - base_q16 must be non-negative for exp assembly domains.
//   - base_q16 == 0 always returns 0 (zero-preserving semantics).
I32 FPQ16ExpMulPow2Checked(I64 base_q16,
                           I64 k,
                           I64 *out_scaled_q16)
{
    I64 right_shift;

    if (!out_scaled_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (base_q16 < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!base_q16)
    {
        *out_scaled_q16 = 0;
        return FP_Q16_OK;
    }

    if (k >= 0)
    {
        if (k >= 63)
            return FP_Q16_ERR_OVERFLOW;

        if (base_q16 > (I64_MAX_VALUE >> k))
            return FP_Q16_ERR_OVERFLOW;

        *out_scaled_q16 = base_q16 << k;
        return FP_Q16_OK;
    }

    // Very large negative shifts deterministically underflow to zero.
    if (k <= EXP_Q16_RIGHT_SHIFT_ZERO_K)
    {
        *out_scaled_q16 = 0;
        return FP_Q16_OK;
    }

    right_shift = -k;
    if (right_shift >= 63)
    {
        *out_scaled_q16 = 0;
        return FP_Q16_OK;
    }

    *out_scaled_q16 = base_q16 >> right_shift;
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
    I64 poly;
    I64 scaled;
    I32 status;

    if (!out_exp_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (clamped_input_q16 < EXP_Q16_MIN_INPUT ||
        clamped_input_q16 > EXP_Q16_MAX_INPUT)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve scalar saturation contract exactly on domain boundaries.
    if (clamped_input_q16 >= EXP_Q16_MAX_INPUT)
    {
        *out_exp_q16 = EXP_Q16_SAT_MAX_OUTPUT;
        return FP_Q16_OK;
    }
    if (clamped_input_q16 <= EXP_Q16_MIN_INPUT)
    {
        *out_exp_q16 = EXP_Q16_SAT_MIN_OUTPUT;
        return FP_Q16_OK;
    }

    status = FPQ16ExpRangeReduceLn2Checked(clamped_input_q16,
                                           &k,
                                           &r);
    if (status)
        return status;

    status = FPQ16ExpPolyCoreLn2ResidualQ16Checked(r,
                                                   &poly);
    if (status)
        return status;

    if (k >= EXP_Q16_LEFT_SHIFT_SAT_K)
    {
        *out_exp_q16 = EXP_Q16_SAT_MAX_OUTPUT;
        return FP_Q16_OK;
    }

    status = FPQ16ExpMulPow2Checked(poly,
                                    k,
                                    &scaled);
    if (status == FP_Q16_ERR_OVERFLOW)
    {
        // Keep scalar exp saturation behavior stable for overflowed scaling.
        if (k >= 0)
            *out_exp_q16 = EXP_Q16_SAT_MAX_OUTPUT;
        else
            *out_exp_q16 = EXP_Q16_SAT_MIN_OUTPUT;
        return FP_Q16_OK;
    }
    if (status)
        return status;

    *out_exp_q16 = scaled;
    return FP_Q16_OK;
}

// Checked all-or-nothing wrapper for clamped-input scalar exp.
// Stages the output so callers get strict no-partial-write behavior:
//   - on error, *out_exp_q16 is unchanged
//   - on success, staged exp output commits atomically
// Status/error surfaces are intentionally identical to
// FPQ16ExpFromClampedInputChecked.
I32 FPQ16ExpFromClampedInputCheckedNoPartial(I64 clamped_input_q16,
                                             I64 *out_exp_q16)
{
    I64 staged_exp_q16;
    I32 status;

    if (!out_exp_q16)
        return FP_Q16_ERR_NULL_PTR;

    staged_exp_q16 = 0;
    status = FPQ16ExpFromClampedInputChecked(clamped_input_q16,
                                             &staged_exp_q16);
    if (status)
        return status;

    *out_exp_q16 = staged_exp_q16;
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

// Canonical default-capacity wrapper for unclamped vector exp.
// This wrapper intentionally keeps semantics pass-through to the checked core:
//   - count defines the active lane span
//   - success/error contracts are identical
//   - clamp+eval lane math remains centralized in one auditable core helper
I32 FPQ16ExpArrayCheckedDefault(I64 *input_q16,
                                I64 *output_q16,
                                I64 count)
{
    I32 status;

    // Keep wrapper behavior intentionally transparent and pass-through:
    // checked core owns all pointer/count guards and lane math, while this
    // default-capacity entrypoint provides one stable WS1 call surface.
    status = FPQ16ExpArrayChecked(input_q16,
                                  output_q16,
                                  count);
    return status;
}

// Shared checked preflight for default-capacity no-partial vector exp wrappers.
// Computes staging allocation bytes for successful non-zero spans while keeping
// pointer/count overflow guards centralized and auditable.
I32 FPQ16ExpArrayCheckedDefaultNoPartialPreflight(I64 *input_q16,
                                                  I64 *output_q16,
                                                  I64 count,
                                                  I64 *out_staging_bytes)
{
    I64 last_index;
    I64 last_byte_offset;

    if (!out_staging_bytes)
        return FP_Q16_ERR_NULL_PTR;

    *out_staging_bytes = 0;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    *out_staging_bytes = count << 3;
    if (*out_staging_bytes <= 0)
    {
        *out_staging_bytes = 0;
        return FP_Q16_ERR_OVERFLOW;
    }

    return FP_Q16_OK;
}

// Checked all-or-nothing default-capacity wrapper for unclamped vector exp.
// Stages full-lane output from FPQ16ExpArrayCheckedDefault and commits to
// caller output only after complete success.
//
// Contract:
//   - preserves exact success/error surfaces from FPQ16ExpArrayCheckedDefault
//   - no writes to output_q16[] on any error
//   - no extra capacity parameter; count is active lane span
I32 FPQ16ExpArrayCheckedDefaultNoPartial(I64 *input_q16,
                                         I64 *output_q16,
                                         I64 count)
{
    I64 i;
    I64 staging_bytes;
    I64 *staged_output_q16;
    I32 status;

    status = FPQ16ExpArrayCheckedDefaultNoPartialPreflight(input_q16,
                                                           output_q16,
                                                           count,
                                                           &staging_bytes);
    if (status)
        return status;
    if (!count)
        return FP_Q16_OK;

    staged_output_q16 = MAlloc(staging_bytes);
    if (!staged_output_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpArrayCheckedDefault(input_q16,
                                         staged_output_q16,
                                         count);
    if (status)
    {
        Free(staged_output_q16);
        return status;
    }

    for (i = 0; i < count; i++)
        output_q16[i] = staged_output_q16[i];

    Free(staged_output_q16);
    return FP_Q16_OK;
}

// Checked all-or-nothing wrapper for unclamped vector exp.
// Stages full lane output from FPQ16ExpArrayChecked before commit so callers
// never observe partial writes on any error path.
// Semantics are intentionally pass-through to FPQ16ExpArrayChecked:
//   - exact clamp + scalar eval behavior per lane
//   - exact status/error surfaces
//   - output commit only after full checked-core success
I32 FPQ16ExpArrayCheckedNoPartial(I64 *input_q16,
                                  I64 *output_q16,
                                  I64 count)
{
    I64 i;
    I64 staged_bytes;
    I64 *staged_output_q16;
    I32 status;

    status = FPQ16ExpArrayCheckedNoPartialPreflight(input_q16,
                                                    output_q16,
                                                    count,
                                                    &staged_bytes);
    if (status)
        return status;
    if (!count)
        return FP_Q16_OK;

    staged_output_q16 = MAlloc(staged_bytes);
    if (!staged_output_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpArrayChecked(input_q16,
                                  staged_output_q16,
                                  count);
    if (status)
    {
        Free(staged_output_q16);
        return status;
    }

    for (i = 0; i < count; i++)
        output_q16[i] = staged_output_q16[i];

    Free(staged_output_q16);
    return FP_Q16_OK;
}

// Shared checked preflight for staged no-partial vector exp wrappers.
// Validates pointer/count/overflow contracts and returns required staging
// bytes for non-zero lane spans.
I32 FPQ16ExpArrayCheckedNoPartialPreflight(I64 *input_q16,
                                           I64 *output_q16,
                                           I64 count,
                                           I64 *out_staged_bytes)
{
    I64 last_index;
    I64 last_byte_offset;

    if (!out_staged_bytes)
        return FP_Q16_ERR_NULL_PTR;

    *out_staged_bytes = 0;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    *out_staged_bytes = count << 3;
    if (*out_staged_bytes <= 0)
    {
        *out_staged_bytes = 0;
        return FP_Q16_ERR_OVERFLOW;
    }

    return FP_Q16_OK;
}

// Canonical default-capacity all-or-nothing wrapper for unclamped vector exp.
// Default contract:
//   - `count` is the active lane span (no extra capacity parameter).
//   - Success/error behavior is exact pass-through to
//     FPQ16ExpArrayCheckedNoPartial.
//   - No-partial-write semantics stay centralized in one checked helper.
I32 FPQ16ExpArrayCheckedNoPartialDefault(I64 *input_q16,
                                         I64 *output_q16,
                                         I64 count)
{
    I32 status;

    status = FPQ16ExpArrayCheckedNoPartial(input_q16,
                                           output_q16,
                                           count);
    return status;
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

// Checked all-or-nothing wrapper for pre-clamped vector exp.
// Keeps scalar/domain semantics from FPQ16ExpArrayFromClampedInputChecked,
// but strengthens write behavior:
//   - no writes to caller output on any error path
//   - staged lane results commit only after full checked-core success
I32 FPQ16ExpArrayFromClampedInputCheckedNoPartial(I64 *input_q16,
                                                  I64 *output_q16,
                                                  I64 count)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staged_bytes;
    I64 *staged_output_q16;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!count)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    if (count > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_bytes = count << 3;
    if (!staged_bytes)
        return FP_Q16_ERR_OVERFLOW;

    staged_output_q16 = MAlloc(staged_bytes);
    if (!staged_output_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpArrayFromClampedInputChecked(input_q16,
                                                  staged_output_q16,
                                                  count);
    if (status)
    {
        Free(staged_output_q16);
        return status;
    }

    for (i = 0; i < count; i++)
        output_q16[i] = staged_output_q16[i];

    Free(staged_output_q16);
    return FP_Q16_OK;
}

// Canonical default-capacity wrapper for pre-clamped vector exp.
// This wrapper intentionally keeps semantics pass-through to the checked core:
//   - length defines the active lane span
//   - success/error contracts are identical
//   - no-partial write guarantees remain those of the checked core helper
I32 FPQ16ExpArrayFromClampedInputCheckedDefault(I64 *input_q16,
                                                I64 *output_q16,
                                                I64 length)
{
    return FPQ16ExpArrayFromClampedInputChecked(input_q16,
                                                output_q16,
                                                length);
}

// Checked all-or-nothing default-capacity wrapper for pre-clamped vector exp.
// Stages the default-capacity checked-core outputs and commits to caller buffer
// only after full success, so any lane/domain/overflow failure leaves
// output_q16[] untouched.
I32 FPQ16ExpArrayFromClampedInputCheckedDefaultNoPartial(I64 *input_q16,
                                                         I64 *output_q16,
                                                         I64 length)
{
    I64 i;
    I64 last_index;
    I64 last_byte_offset;
    I64 staging_bytes;
    I64 *staging_output_q16;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (length < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (!length)
        return FP_Q16_OK;

    // Guard pointer-span arithmetic before indexed lane access.
    last_index = length - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    if (length > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staging_bytes = length << 3;
    if (staging_bytes <= 0)
        return FP_Q16_ERR_OVERFLOW;

    staging_output_q16 = MAlloc(staging_bytes);
    if (!staging_output_q16)
        return FP_Q16_ERR_OVERFLOW;

    status = FPQ16ExpArrayFromClampedInputCheckedDefault(input_q16,
                                                         staging_output_q16,
                                                         length);
    if (status)
    {
        Free(staging_output_q16);
        return status;
    }

    for (i = 0; i < length; i++)
        output_q16[i] = staging_output_q16[i];

    Free(staging_output_q16);
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
