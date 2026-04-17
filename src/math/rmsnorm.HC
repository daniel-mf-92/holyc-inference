// rmsnorm.HC
// Integer-only Q16.16 RMSNorm for HolyC inference.

#define FP_Q16_SHIFT 16
#define FP_Q16_ONE   (1 << FP_Q16_SHIFT)
#define I64_MAX_VALUE 0x7FFFFFFFFFFFFFFF
#define FP_Q16_ERR_NULL_PTR  1
#define FP_Q16_ERR_BAD_PARAM 2
#define FP_Q16_ERR_DOMAIN    3
#define FP_Q16_ERR_OVERFLOW  4
#define U64_MAX_VALUE 0xFFFFFFFFFFFFFFFF
#define FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE 0
#define FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE     1

I64 FPQ16Mul(I64 a, I64 b);
I64 FPQ16Div(I64 num, I64 den);
I64 FPQ16FromInt(I64 x);
I64 FPQ16Sqrt(I64 x);
I32 FPQ16MulChecked(I64 a_q16,
                    I64 b_q16,
                    I64 *out_q16);
I32 FPQ16MulDivRoundedChecked(I64 a_q16,
                              I64 b_q16,
                              I64 d_q16,
                              I64 *out_q16);
I32 FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(I64 a_q16,
                                                     I64 b_q16,
                                                     I64 d_q16,
                                                     I64 *out_q16);

// WS1-04 RMSNorm assumptions (Q16 tensors):
// - `count` is the feature width for one normalized row/token/head slice.
// - `input_q16`, `gamma_q16`, `output_q16` each contain at least `count`
//   contiguous I64 Q16 lanes.
// - mean-square and denominator are scalar per-slice values.
// TODO(WS4/WS5): Add tiled/strided variants for batched matmul/attention paths.
// TODO(WS4/WS5): Add AVX2 lane kernels once scalar checked semantics are frozen.

// Compute checked RMSNorm inverse denominator from one Q16 slice:
//   mean_sq_q16 = (sum_i input[i]^2) / count
//   denom_q16   = sqrt(mean_sq_q16 + eps_q16)
//   inv_q16     = 1.0_q16 / denom_q16
// All intermediate math is integer-only with explicit overflow/domain checks.
I32 FPQ16RMSNormComputeInvDenomChecked(I64 *input_q16,
                                       I64 count,
                                       I64 eps_q16,
                                       I64 *inv_denom_q16_out,
                                       I64 *denom_q16_out)
{
    I64 lane_index;
    I64 lane_sq_q16;
    I64 sum_sq_q16;
    I64 count_q16;
    I64 mean_sq_q16;
    I64 denom_arg_q16;
    I64 denom_q16;
    I64 inv_denom_q16;
    I32 status;

    if (!input_q16 || !inv_denom_q16_out || !denom_q16_out)
        return FP_Q16_ERR_NULL_PTR;
    if (count <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (eps_q16 < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (count > (I64_MAX_VALUE >> FP_Q16_SHIFT))
        return FP_Q16_ERR_OVERFLOW;

    sum_sq_q16 = 0;
    for (lane_index = 0; lane_index < count; lane_index++)
    {
        status = FPQ16MulChecked(input_q16[lane_index],
                                 input_q16[lane_index],
                                 &lane_sq_q16);
        if (status)
            return status;

        if (sum_sq_q16 > I64_MAX_VALUE - lane_sq_q16)
            return FP_Q16_ERR_OVERFLOW;

        sum_sq_q16 += lane_sq_q16;
    }

    count_q16 = FPQ16FromInt(count);
    if (count_q16 <= 0)
        return FP_Q16_ERR_OVERFLOW;

    mean_sq_q16 = FPQ16Div(sum_sq_q16, count_q16);
    if (mean_sq_q16 < 0)
        return FP_Q16_ERR_DOMAIN;

    if (mean_sq_q16 > I64_MAX_VALUE - eps_q16)
        return FP_Q16_ERR_OVERFLOW;

    denom_arg_q16 = mean_sq_q16 + eps_q16;
    denom_q16 = FPQ16Sqrt(denom_arg_q16);
    if (denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    inv_denom_q16 = FPQ16Div(FP_Q16_ONE, denom_q16);
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    *inv_denom_q16_out = inv_denom_q16;
    *denom_q16_out = denom_q16;

    return 0;
}

// Checked full RMSNorm (no-alias output contract):
//   out[i] = ((input[i] / sqrt(mean(input^2)+eps)) * gamma[i])
// The function performs a no-write validation pass first, then commits writes,
// so any error returns with `output_q16` unchanged.
I32 FPQ16RMSNormChecked(I64 *input_q16,
                        I64 *gamma_q16,
                        I64 *output_q16,
                        I64 count,
                        I64 eps_q16)
{
    I64 lane_index;
    I64 inv_denom_q16;
    I64 denom_q16;
    I64 norm_lane_q16;
    I64 weighted_lane_q16;
    I32 status;

    if (!input_q16 || !gamma_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (eps_q16 < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (input_q16 == output_q16)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16RMSNormComputeInvDenomChecked(input_q16,
                                                count,
                                                eps_q16,
                                                &inv_denom_q16,
                                                &denom_q16);
    if (status)
        return status;

    // Validation pass: prove every lane arithmetic operation succeeds before
    // writing output, preserving no-partial-write behavior.
    for (lane_index = 0; lane_index < count; lane_index++)
    {
        status = FPQ16MulChecked(input_q16[lane_index],
                                 inv_denom_q16,
                                 &norm_lane_q16);
        if (status)
            return status;

        status = FPQ16MulChecked(norm_lane_q16,
                                 gamma_q16[lane_index],
                                 &weighted_lane_q16);
        if (status)
            return status;
    }

    for (lane_index = 0; lane_index < count; lane_index++)
    {
        status = FPQ16MulChecked(input_q16[lane_index],
                                 inv_denom_q16,
                                 &norm_lane_q16);
        if (status)
            return status;

        status = FPQ16MulChecked(norm_lane_q16,
                                 gamma_q16[lane_index],
                                 &weighted_lane_q16);
        if (status)
            return status;

        output_q16[lane_index] = weighted_lane_q16;
    }

    return 0;
}

// Checked full RMSNorm in-place variant:
//   input[i] = ((input[i] / sqrt(mean(input^2)+eps)) * gamma[i])
// This mirrors FPQ16RMSNormChecked arithmetic and error contracts while
// allowing alias-safe writes to the input buffer.
// Behavior:
//   - Computes inverse denominator from the original input slice.
//   - Runs a no-write validation pass across all lanes.
//   - Commits in-place writes only if every lane pre-validates.
I32 FPQ16RMSNormCheckedInPlace(I64 *input_q16,
                               I64 *gamma_q16,
                               I64 count,
                               I64 eps_q16)
{
    I64 lane_index;
    I64 inv_denom_q16;
    I64 denom_q16;
    I64 norm_lane_q16;
    I64 weighted_lane_q16;
    I32 status;

    if (!input_q16 || !gamma_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count <= 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (eps_q16 < 0)
        return FP_Q16_ERR_BAD_PARAM;

    status = FPQ16RMSNormComputeInvDenomChecked(input_q16,
                                                count,
                                                eps_q16,
                                                &inv_denom_q16,
                                                &denom_q16);
    if (status)
        return status;

    // Validation pass over original input values: fail fast on overflow/domain
    // before mutating any lane in-place.
    for (lane_index = 0; lane_index < count; lane_index++)
    {
        status = FPQ16MulChecked(input_q16[lane_index],
                                 inv_denom_q16,
                                 &norm_lane_q16);
        if (status)
            return status;

        status = FPQ16MulChecked(norm_lane_q16,
                                 gamma_q16[lane_index],
                                 &weighted_lane_q16);
        if (status)
            return status;
    }

    // Commit phase: same checked math, now writing back lane-by-lane.
    for (lane_index = 0; lane_index < count; lane_index++)
    {
        status = FPQ16MulChecked(input_q16[lane_index],
                                 inv_denom_q16,
                                 &norm_lane_q16);
        if (status)
            return status;

        status = FPQ16MulChecked(norm_lane_q16,
                                 gamma_q16[lane_index],
                                 &weighted_lane_q16);
        if (status)
            return status;

        input_q16[lane_index] = weighted_lane_q16;
    }

    return 0;
}

// Apply precomputed RMSNorm inverse denominator to a Q16 vector with explicit
// checked error surfaces. This uses one-rounding math:
//   out[i] = round_half_up((input[i] * inv_denom_q16) / den_q16)
// where den_q16 is a positive Q16 whole integer (d_int << 16).
I32 FPQ16RMSNormApplyInvDenomChecked(I64 *input_q16,
                                     I64 *output_q16,
                                     I64 count,
                                     I64 inv_denom_q16,
                                     I64 den_q16)
{
    I64 i;
    I64 scaled_q16;
    I32 status;

    if (!input_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    // Delegate denominator contract + checked one-rounding to fixedpoint core.
    // That keeps RMSNorm scaling error behavior identical to shared math helpers.
    for (i = 0; i < count; i++)
    {
        status = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(input_q16[i],
                                                                   inv_denom_q16,
                                                                   den_q16,
                                                                   &scaled_q16);
        if (status)
            return status;

        output_q16[i] = scaled_q16;
    }

    return 0;
}

// Apply precomputed RMSNorm inverse denominator and per-channel gamma to a
// Q16 vector with explicit checked error surfaces:
//   out[i] = round_half_up(round_half_up((input[i] * inv_denom_q16) / den_q16)
//                          * gamma[i] / FP_Q16_ONE)
// The first stage enforces the positive whole-Q16 denominator contract used by
// WS5 norm call chains. The second stage is checked Q16*Q16 scaling.
I32 FPQ16RMSNormApplyInvDenomWeightedChecked(I64 *input_q16,
                                             I64 *gamma_q16,
                                             I64 *output_q16,
                                             I64 count,
                                             I64 inv_denom_q16,
                                             I64 den_q16)
{
    I64 i;
    I64 scaled_q16;
    I64 weighted_q16;
    I32 status;

    if (!input_q16 || !gamma_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    for (i = 0; i < count; i++)
    {
        status = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(input_q16[i],
                                                                   inv_denom_q16,
                                                                   den_q16,
                                                                   &scaled_q16);
        if (status)
            return status;

        status = FPQ16MulDivRoundedChecked(scaled_q16,
                                           gamma_q16[i],
                                           FP_Q16_ONE,
                                           &weighted_q16);
        if (status)
            return status;

        output_q16[i] = weighted_q16;
    }

    return 0;
}

// In-place variant of weighted inverse-denominator RMSNorm scaling:
//   input[i] = round_half_up(round_half_up((input[i] * inv_denom_q16) / den_q16)
//                            * gamma[i] / FP_Q16_ONE)
// This preserves the same checked contracts as the out-of-place helper while
// explicitly supporting alias-safe writes when caller uses one buffer.
I32 FPQ16RMSNormApplyInvDenomWeightedInPlaceChecked(I64 *input_q16,
                                                    I64 *gamma_q16,
                                                    I64 count,
                                                    I64 inv_denom_q16,
                                                    I64 den_q16)
{
    I64 i;
    I64 scaled_q16;
    I64 weighted_q16;
    I32 status;

    if (!input_q16 || !gamma_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    for (i = 0; i < count; i++)
    {
        // Read one element, compute checked scaling, then write back to the
        // same slot. No later iteration depends on previous slot values.
        status = FPQ16MulDivRoundedByPositiveIntFromQ16DenChecked(input_q16[i],
                                                                   inv_denom_q16,
                                                                   den_q16,
                                                                   &scaled_q16);
        if (status)
            return status;

        status = FPQ16MulDivRoundedChecked(scaled_q16,
                                           gamma_q16[i],
                                           FP_Q16_ONE,
                                           &weighted_q16);
        if (status)
            return status;

        input_q16[i] = weighted_q16;
    }

    return 0;
}

// Shared preflight for auto-dispatch weighted inverse-denominator RMSNorm.
// This centralizes parameter/domain checks so alias and non-alias paths
// surface identical BAD_PARAM/DOMAIN/OVERFLOW behavior before any writes.
I32 FPQ16RMSNormApplyInvDenomWeightedAutoValidateChecked(I64 *input_q16,
                                                         I64 *gamma_q16,
                                                         I64 *output_q16,
                                                         I64 count,
                                                         I64 inv_denom_q16,
                                                         I64 den_q16)
{
    I64 last_index;
    I64 last_byte_offset;

    if (!input_q16 || !gamma_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;
    if (count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (inv_denom_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;

    // Keep denominator contract explicit at auto-dispatch boundary:
    // - positive
    // - exact Q16 whole integer (d_int << 16)
    // - d_int strictly positive after decode
    if (den_q16 <= 0)
        return FP_Q16_ERR_DOMAIN;
    if (den_q16 & (FP_Q16_ONE - 1))
        return FP_Q16_ERR_BAD_PARAM;
    if ((den_q16 >> FP_Q16_SHIFT) <= 0)
        return FP_Q16_ERR_DOMAIN;

    if (!count)
        return 0;

    // Guard pointer-span math used by indexed access in both dispatch paths.
    // count-1 element index and byte offsets must stay in signed range.
    last_index = count - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)gamma_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    return 0;
}

// Select auto-dispatch execution path deterministically from pointer aliasing.
// The contract is explicit and checked:
// - in-place path only when input and output pointers are identical
// - out-of-place path for all other pointer pairs
// This helper isolates path selection from math kernels so dispatch behavior
// stays stable as validation and execution logic evolve independently.
I32 FPQ16RMSNormApplyInvDenomWeightedAutoSelectPathChecked(I64 *input_q16,
                                                           I64 *output_q16,
                                                           I32 *path_out)
{
    if (!input_q16 || !output_q16 || !path_out)
        return FP_Q16_ERR_NULL_PTR;

    if (input_q16 == output_q16)
    {
        *path_out = FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE;
        return 0;
    }

    *path_out = FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE;
    return 0;
}

// Auto-dispatch weighted inverse-denominator RMSNorm scaling:
// - If input/output alias, use the in-place checked kernel.
// - If they do not alias, use the out-of-place checked kernel.
// This keeps the math path fully transparent while preserving the exact
// checked error surfaces of the selected canonical helper.
I32 FPQ16RMSNormApplyInvDenomWeightedAutoChecked(I64 *input_q16,
                                                 I64 *gamma_q16,
                                                 I64 *output_q16,
                                                 I64 count,
                                                 I64 inv_denom_q16,
                                                 I64 den_q16)
{
    I32 status;
    I32 selected_path;

    status = FPQ16RMSNormApplyInvDenomWeightedAutoValidateChecked(input_q16,
                                                                   gamma_q16,
                                                                   output_q16,
                                                                   count,
                                                                   inv_denom_q16,
                                                                   den_q16);
    if (status)
        return status;

    status = FPQ16RMSNormApplyInvDenomWeightedAutoSelectPathChecked(input_q16,
                                                                     output_q16,
                                                                     &selected_path);
    if (status)
        return status;

    if (selected_path == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE)
    {
        return FPQ16RMSNormApplyInvDenomWeightedInPlaceChecked(input_q16,
                                                                gamma_q16,
                                                                count,
                                                                inv_denom_q16,
                                                                den_q16);
    }

    if (selected_path != FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE)
        return FP_Q16_ERR_BAD_PARAM;

    return FPQ16RMSNormApplyInvDenomWeightedChecked(input_q16,
                                                    gamma_q16,
                                                    output_q16,
                                                    count,
                                                    inv_denom_q16,
                                                    den_q16);
}

// Auto-dispatch weighted inverse-denominator RMSNorm scaling with explicit
// selected-path output for parity harnesses and callers that need to assert
// alias-driven execution mode.
// - On success: writes path id to `path_out` and executes selected kernel.
// - On failure: preserves canonical checked error surfaces from shared
//   validation/selection/math helpers.
I32 FPQ16RMSNormApplyInvDenomWeightedAutoCheckedWithPathOut(I64 *input_q16,
                                                            I64 *gamma_q16,
                                                            I64 *output_q16,
                                                            I64 count,
                                                            I64 inv_denom_q16,
                                                            I64 den_q16,
                                                            I32 *path_out)
{
    I32 status;
    I32 selected_path;

    if (!path_out)
        return FP_Q16_ERR_NULL_PTR;

    status = FPQ16RMSNormApplyInvDenomWeightedAutoValidateChecked(input_q16,
                                                                   gamma_q16,
                                                                   output_q16,
                                                                   count,
                                                                   inv_denom_q16,
                                                                   den_q16);
    if (status)
        return status;

    status = FPQ16RMSNormApplyInvDenomWeightedAutoSelectPathChecked(input_q16,
                                                                     output_q16,
                                                                     &selected_path);
    if (status)
        return status;

    *path_out = selected_path;

    if (selected_path == FP_Q16_RMSNORM_AUTO_PATH_IN_PLACE)
    {
        return FPQ16RMSNormApplyInvDenomWeightedInPlaceChecked(input_q16,
                                                                gamma_q16,
                                                                count,
                                                                inv_denom_q16,
                                                                den_q16);
    }

    if (selected_path != FP_Q16_RMSNORM_AUTO_PATH_OUT_OF_PLACE)
        return FP_Q16_ERR_BAD_PARAM;

    return FPQ16RMSNormApplyInvDenomWeightedChecked(input_q16,
                                                    gamma_q16,
                                                    output_q16,
                                                    count,
                                                    inv_denom_q16,
                                                    den_q16);
}

U0 FPQ16RMSNorm(I64 *input, I64 *weight, I64 *output, I64 count, I64 eps_q16)
{
    I64 i;
    I64 sq;
    I64 sum_sq;
    I64 mean_sq;
    I64 denom;
    I64 inv_denom;
    I64 norm;

    if (!input || !weight || !output || count <= 0)
        return;

    if (eps_q16 < 0)
        eps_q16 = 0;

    sum_sq = 0;
    for (i = 0; i < count; i++)
    {
        sq = FPQ16Mul(input[i], input[i]);
        if (sum_sq > I64_MAX_VALUE - sq)
            sum_sq = I64_MAX_VALUE;
        else
            sum_sq += sq;
    }

    mean_sq = FPQ16Div(sum_sq, FPQ16FromInt(count));

    if (mean_sq > I64_MAX_VALUE - eps_q16)
        mean_sq = I64_MAX_VALUE;
    else
        mean_sq += eps_q16;

    denom = FPQ16Sqrt(mean_sq);
    if (denom <= 0)
    {
        for (i = 0; i < count; i++)
            output[i] = 0;
        return;
    }

    inv_denom = FPQ16Div(FP_Q16_ONE, denom);

    for (i = 0; i < count; i++)
    {
        norm = FPQ16Mul(input[i], inv_denom);
        output[i] = FPQ16Mul(norm, weight[i]);
    }
}
