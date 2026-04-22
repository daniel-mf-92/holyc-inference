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

I32 FPQ16RMSNormTryAddI64Checked(I64 lhs,
                                 I64 rhs,
                                 I64 *out_sum)
{
    if (!out_sum)
        return FP_Q16_ERR_NULL_PTR;

    if (rhs > 0)
    {
        if (lhs > I64_MAX_VALUE - rhs)
            return FP_Q16_ERR_OVERFLOW;
    }
    else if (rhs < 0)
    {
        if (lhs < (-I64_MAX_VALUE - 1) - rhs)
            return FP_Q16_ERR_OVERFLOW;
    }

    *out_sum = lhs + rhs;
    return 0;
}

I32 FPQ16RMSNormTryMulI64Checked(I64 lhs,
                                 I64 rhs,
                                 I64 *out_prod)
{
    I64 product;

    if (!out_prod)
        return FP_Q16_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return 0;
    }

    if (lhs == (-I64_MAX_VALUE - 1))
    {
        if (rhs != 1)
        {
            if (rhs != -1)
                return FP_Q16_ERR_OVERFLOW;
            return FP_Q16_ERR_OVERFLOW;
        }
    }
    if (rhs == (-I64_MAX_VALUE - 1))
    {
        if (lhs != 1)
        {
            if (lhs != -1)
                return FP_Q16_ERR_OVERFLOW;
            return FP_Q16_ERR_OVERFLOW;
        }
    }

    product = lhs * rhs;
    if (product / rhs != lhs)
        return FP_Q16_ERR_OVERFLOW;

    *out_prod = product;
    return 0;
}

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
    U64 input_span_start;
    U64 input_span_end;
    U64 gamma_span_start;
    U64 gamma_span_end;
    U64 output_span_start;
    U64 output_span_end;
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

// Checked batched row-wise RMSNorm kernel for Q16 buffers.
//
// Contract:
// - each row contains `lane_count` contiguous Q16 lanes
// - row bases advance by explicit row strides (in I64 lanes)
// - capacity checks are strict: computed required cells must fit every tensor
//
// Math per row:
//   inv = 1 / sqrt(mean(input_row^2) + eps)
//   out_lane = round_q16(round_q16(input_lane * inv) * gamma_lane)
//
// Execution model is two-phase:
// 1) preflight all rows/lane arithmetic with no writes
// 2) commit all row outputs only after global preflight succeeds
I32 FPQ16RMSNormRowsChecked(I64 *input_q16,
                            I64 input_capacity,
                            I64 input_row_stride,
                            I64 *gamma_q16,
                            I64 gamma_capacity,
                            I64 gamma_row_stride,
                            I64 *output_q16,
                            I64 output_capacity,
                            I64 output_row_stride,
                            I64 row_count,
                            I64 lane_count,
                            I64 eps_q16)
{
    I64 last_row;
    I64 last_index;
    I64 last_byte_offset;
    I64 required_input_cells;
    I64 required_gamma_cells;
    I64 required_output_cells;
    I64 row_index;
    I64 lane_index;
    I64 input_row_base;
    I64 gamma_row_base;
    I64 output_row_base;
    I64 next_input_row_base;
    I64 next_gamma_row_base;
    I64 next_output_row_base;
    I64 inv_denom_q16;
    I64 denom_q16;
    I64 norm_lane_q16;
    I64 weighted_lane_q16;
    I32 status;

    if (!input_q16 || !gamma_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;

    if (input_capacity < 0 || gamma_capacity < 0 || output_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (input_row_stride < 0 || gamma_row_stride < 0 || output_row_stride < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (eps_q16 < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return 0;

    if (input_row_stride < lane_count || gamma_row_stride < lane_count ||
        output_row_stride < lane_count)
        return FP_Q16_ERR_BAD_PARAM;

    // Alias safety: in-place rows are allowed only when row bases are identical.
    if (input_q16 == output_q16 && input_row_stride != output_row_stride)
        return FP_Q16_ERR_BAD_PARAM;
    if (gamma_q16 == output_q16)
        return FP_Q16_ERR_BAD_PARAM;

    last_row = row_count - 1;

    status = FPQ16RMSNormTryMulI64Checked(last_row,
                                          input_row_stride,
                                          &required_input_cells);
    if (status)
        return status;
    status = FPQ16RMSNormTryAddI64Checked(required_input_cells,
                                          lane_count,
                                          &required_input_cells);
    if (status)
        return status;

    status = FPQ16RMSNormTryMulI64Checked(last_row,
                                          gamma_row_stride,
                                          &required_gamma_cells);
    if (status)
        return status;
    status = FPQ16RMSNormTryAddI64Checked(required_gamma_cells,
                                          lane_count,
                                          &required_gamma_cells);
    if (status)
        return status;

    status = FPQ16RMSNormTryMulI64Checked(last_row,
                                          output_row_stride,
                                          &required_output_cells);
    if (status)
        return status;
    status = FPQ16RMSNormTryAddI64Checked(required_output_cells,
                                          lane_count,
                                          &required_output_cells);
    if (status)
        return status;

    if (required_input_cells > input_capacity)
        return FP_Q16_ERR_BAD_PARAM;
    if (required_gamma_cells > gamma_capacity)
        return FP_Q16_ERR_BAD_PARAM;
    if (required_output_cells > output_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    // Guard pointer-span arithmetic for full required windows.
    last_index = required_input_cells - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)input_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    last_index = required_gamma_cells - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)gamma_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    last_index = required_output_cells - 1;
    if (last_index > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))
        return FP_Q16_ERR_OVERFLOW;

    input_span_start = (U64)input_q16;
    gamma_span_start = (U64)gamma_q16;
    output_span_start = (U64)output_q16;

    input_span_end = input_span_start + ((U64)required_input_cells << 3);
    gamma_span_end = gamma_span_start + ((U64)required_gamma_cells << 3);
    output_span_end = output_span_start + ((U64)required_output_cells << 3);

    // Reject partial overlap between read and write windows. Exact alias is
    // still handled by the explicit in-place contract above.
    if (input_q16 != output_q16)
    {
        if (!(input_span_end <= output_span_start ||
              output_span_end <= input_span_start))
            return FP_Q16_ERR_BAD_PARAM;
    }

    // `gamma` is read-only; any overlap with output would violate deterministic
    // gamma sampling during commit, so keep this strictly disjoint.
    if (!(gamma_span_end <= output_span_start ||
          output_span_end <= gamma_span_start))
        return FP_Q16_ERR_BAD_PARAM;

    input_row_base = 0;
    gamma_row_base = 0;
    output_row_base = 0;

    // Global preflight: prove all rows can normalize without overflow/domain
    // failures before any output write occurs.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = FPQ16RMSNormComputeInvDenomChecked(input_q16 + input_row_base,
                                                    lane_count,
                                                    eps_q16,
                                                    &inv_denom_q16,
                                                    &denom_q16);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FPQ16MulChecked(input_q16[input_row_base + lane_index],
                                     inv_denom_q16,
                                     &norm_lane_q16);
            if (status)
                return status;

            status = FPQ16MulChecked(norm_lane_q16,
                                     gamma_q16[gamma_row_base + lane_index],
                                     &weighted_lane_q16);
            if (status)
                return status;
        }

        status = FPQ16RMSNormTryAddI64Checked(input_row_base,
                                              input_row_stride,
                                              &next_input_row_base);
        if (status)
            return status;
        status = FPQ16RMSNormTryAddI64Checked(gamma_row_base,
                                              gamma_row_stride,
                                              &next_gamma_row_base);
        if (status)
            return status;
        status = FPQ16RMSNormTryAddI64Checked(output_row_base,
                                              output_row_stride,
                                              &next_output_row_base);
        if (status)
            return status;

        input_row_base = next_input_row_base;
        gamma_row_base = next_gamma_row_base;
        output_row_base = next_output_row_base;
    }

    input_row_base = 0;
    gamma_row_base = 0;
    output_row_base = 0;

    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = FPQ16RMSNormComputeInvDenomChecked(input_q16 + input_row_base,
                                                    lane_count,
                                                    eps_q16,
                                                    &inv_denom_q16,
                                                    &denom_q16);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = FPQ16MulChecked(input_q16[input_row_base + lane_index],
                                     inv_denom_q16,
                                     &norm_lane_q16);
            if (status)
                return status;

            status = FPQ16MulChecked(norm_lane_q16,
                                     gamma_q16[gamma_row_base + lane_index],
                                     &weighted_lane_q16);
            if (status)
                return status;

            output_q16[output_row_base + lane_index] = weighted_lane_q16;
        }

        status = FPQ16RMSNormTryAddI64Checked(input_row_base,
                                              input_row_stride,
                                              &next_input_row_base);
        if (status)
            return status;
        status = FPQ16RMSNormTryAddI64Checked(gamma_row_base,
                                              gamma_row_stride,
                                              &next_gamma_row_base);
        if (status)
            return status;
        status = FPQ16RMSNormTryAddI64Checked(output_row_base,
                                              output_row_stride,
                                              &next_output_row_base);
        if (status)
            return status;

        input_row_base = next_input_row_base;
        gamma_row_base = next_gamma_row_base;
        output_row_base = next_output_row_base;
    }

    return 0;
}

// Checked batched row-wise RMSNorm with strict no-partial publish.
//
// Difference from FPQ16RMSNormRowsChecked:
// - caches one inverse denominator per row during preflight
// - stages every output lane into scratch
// - performs one final publish pass into caller output only after all rows
//   fully validate and stage
//
// Math per row:
//   inv[row] = 1 / sqrt(mean(input_row^2) + eps)
//   staged[out_row + lane] = round_q16(input_lane * inv[row] * gamma_lane)
I32 FPQ16RMSNormRowsCheckedNoPartial(I64 *input_q16,
                                     I64 input_capacity,
                                     I64 input_row_stride,
                                     I64 *gamma_q16,
                                     I64 gamma_capacity,
                                     I64 gamma_row_stride,
                                     I64 *output_q16,
                                     I64 output_capacity,
                                     I64 output_row_stride,
                                     I64 row_count,
                                     I64 lane_count,
                                     I64 eps_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I64 snapshot_input_capacity;
    I64 snapshot_gamma_capacity;
    I64 snapshot_output_capacity;
    I64 snapshot_input_row_stride;
    I64 snapshot_gamma_row_stride;
    I64 snapshot_output_row_stride;
    I64 *snapshot_input_ptr;
    I64 *snapshot_gamma_ptr;
    I64 *snapshot_output_ptr;
    I64 last_row;
    I64 required_output_cells;
    I64 row_index;
    I64 lane_index;
    I64 output_row_base;
    I64 next_output_row_base;
    I64 staged_output_bytes;
    I64 *staged_output_q16;
    I32 status;

    if (!input_q16 || !gamma_q16 || !output_q16)
        return FP_Q16_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_input_capacity = input_capacity;
    snapshot_gamma_capacity = gamma_capacity;
    snapshot_output_capacity = output_capacity;
    snapshot_input_row_stride = input_row_stride;
    snapshot_gamma_row_stride = gamma_row_stride;
    snapshot_output_row_stride = output_row_stride;
    snapshot_input_ptr = input_q16;
    snapshot_gamma_ptr = gamma_q16;
    snapshot_output_ptr = output_q16;

    if (snapshot_input_capacity < 0 ||
        snapshot_gamma_capacity < 0 ||
        snapshot_output_capacity < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_input_row_stride < 0 ||
        snapshot_gamma_row_stride < 0 ||
        snapshot_output_row_stride < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_row_count < 0 || snapshot_lane_count < 0)
        return FP_Q16_ERR_BAD_PARAM;
    if (eps_q16 < 0)
        return FP_Q16_ERR_BAD_PARAM;

    if (!snapshot_row_count || !snapshot_lane_count)
        return 0;

    if (snapshot_input_row_stride < snapshot_lane_count ||
        snapshot_gamma_row_stride < snapshot_lane_count ||
        snapshot_output_row_stride < snapshot_lane_count)
        return FP_Q16_ERR_BAD_PARAM;

    // Preserve alias contract from FPQ16RMSNormRowsChecked.
    if (snapshot_input_ptr == snapshot_output_ptr &&
        snapshot_input_row_stride != snapshot_output_row_stride)
        return FP_Q16_ERR_BAD_PARAM;
    if (snapshot_gamma_ptr == snapshot_output_ptr)
        return FP_Q16_ERR_BAD_PARAM;

    last_row = snapshot_row_count - 1;

    status = FPQ16RMSNormTryMulI64Checked(last_row,
                                          snapshot_output_row_stride,
                                          &required_output_cells);
    if (status)
        return status;
    status = FPQ16RMSNormTryAddI64Checked(required_output_cells,
                                          snapshot_lane_count,
                                          &required_output_cells);
    if (status)
        return status;

    if (required_output_cells > snapshot_output_capacity)
        return FP_Q16_ERR_BAD_PARAM;

    if (required_output_cells > (I64_MAX_VALUE >> 3))
        return FP_Q16_ERR_OVERFLOW;
    staged_output_bytes = required_output_cells << 3;

    staged_output_q16 = MAlloc(staged_output_bytes);
    if (!staged_output_q16)
        return FP_Q16_ERR_NULL_PTR;

    // Phase 1 (preflight): run full checked kernel into private staging so
    // caller output is untouched on every error path.
    status = FPQ16RMSNormRowsChecked(input_q16,
                                     input_capacity,
                                     input_row_stride,
                                     gamma_q16,
                                     gamma_capacity,
                                     gamma_row_stride,
                                     staged_output_q16,
                                     required_output_cells,
                                     output_row_stride,
                                     row_count,
                                     lane_count,
                                     eps_q16);
    if (status)
    {
        Free(staged_output_q16);
        return status;
    }

    // Immutable snapshot guard before publish.
    if (snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count ||
        snapshot_input_capacity != input_capacity ||
        snapshot_gamma_capacity != gamma_capacity ||
        snapshot_output_capacity != output_capacity ||
        snapshot_input_row_stride != input_row_stride ||
        snapshot_gamma_row_stride != gamma_row_stride ||
        snapshot_output_row_stride != output_row_stride ||
        snapshot_input_ptr != input_q16 ||
        snapshot_gamma_ptr != gamma_q16 ||
        snapshot_output_ptr != output_q16)
    {
        Free(staged_output_q16);
        return FP_Q16_ERR_BAD_PARAM;
    }

    // Publish preflight: prove row-base progression cannot overflow so commit
    // has no failing branches.
    output_row_base = 0;
    for (row_index = 0; row_index < snapshot_row_count; row_index++)
    {
        status = FPQ16RMSNormTryAddI64Checked(output_row_base,
                                              snapshot_output_row_stride,
                                              &next_output_row_base);
        if (status)
        {
            Free(staged_output_q16);
            return status;
        }

        output_row_base = next_output_row_base;
    }

    // Phase 2 (commit): one publish pass into caller output.
    output_row_base = 0;
    for (row_index = 0; row_index < snapshot_row_count; row_index++)
    {
        for (lane_index = 0; lane_index < snapshot_lane_count; lane_index++)
            output_q16[output_row_base + lane_index] = staged_output_q16[output_row_base + lane_index];

        output_row_base += snapshot_output_row_stride;
    }

    Free(staged_output_q16);
    return 0;
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
