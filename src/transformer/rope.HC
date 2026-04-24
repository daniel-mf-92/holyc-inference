// rope.HC
// Transformer RoPE helpers (integer-only HolyC).
//
// Pairwise apply kernel:
//   x' = x*cos(theta) - y*sin(theta)
//   y' = x*sin(theta) + y*cos(theta)
//
// Contracts:
// - Q16 lanes only (I64 integer domain, no floating point).
// - deterministic lane-order semantics: lane_x_index must be < lane_y_index.
// - all multiplies/adds/subtracts are checked through shared helpers.

#define ROPE_Q16_OK            0
#define ROPE_Q16_ERR_NULL_PTR  1
#define ROPE_Q16_ERR_BAD_PARAM 2

I32 RoPEQ16LookupSinCosChecked(I64 angle_q16,
                               I64 *out_sin_q16,
                               I64 *out_cos_q16);
I32 RoPEQ16MulChecked(I64 lhs_q16,
                      I64 rhs_q16,
                      I64 *out_q16);
I32 RoPETryAddI64Checked(I64 lhs,
                         I64 rhs,
                         I64 *out_sum);
I32 RoPETrySubI64Checked(I64 lhs,
                         I64 rhs,
                         I64 *out_diff);

I32 RoPEQ16ApplyPairwiseChecked(I64 *head_cells_q16,
                                I64 head_cell_capacity,
                                I64 lane_x_index,
                                I64 lane_y_index,
                                I64 angle_q16)
{
    I64 x_q16;
    I64 y_q16;
    I64 sin_q16;
    I64 cos_q16;
    I64 x_cos_q16;
    I64 y_sin_q16;
    I64 x_sin_q16;
    I64 y_cos_q16;
    I64 x_rot_q16;
    I64 y_rot_q16;
    I32 status;

    if (!head_cells_q16)
        return ROPE_Q16_ERR_NULL_PTR;

    if (head_cell_capacity < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    if (lane_x_index < 0 || lane_y_index < 0)
        return ROPE_Q16_ERR_BAD_PARAM;

    if (lane_x_index >= head_cell_capacity ||
        lane_y_index >= head_cell_capacity)
        return ROPE_Q16_ERR_BAD_PARAM;

    // Deterministic lane-order contract keeps call sites auditable and
    // avoids orientation ambiguity for pairwise rotary updates.
    if (lane_x_index >= lane_y_index)
        return ROPE_Q16_ERR_BAD_PARAM;

    x_q16 = head_cells_q16[lane_x_index];
    y_q16 = head_cells_q16[lane_y_index];

    // Shared lookup helper performs checked LUT indexing from angle_q16.
    status = RoPEQ16LookupSinCosChecked(angle_q16,
                                        &sin_q16,
                                        &cos_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x_q16,
                               cos_q16,
                               &x_cos_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(y_q16,
                               sin_q16,
                               &y_sin_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(x_q16,
                               sin_q16,
                               &x_sin_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPEQ16MulChecked(y_q16,
                               cos_q16,
                               &y_cos_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETrySubI64Checked(x_cos_q16,
                                  y_sin_q16,
                                  &x_rot_q16);
    if (status != ROPE_Q16_OK)
        return status;

    status = RoPETryAddI64Checked(x_sin_q16,
                                  y_cos_q16,
                                  &y_rot_q16);
    if (status != ROPE_Q16_OK)
        return status;

    head_cells_q16[lane_x_index] = x_rot_q16;
    head_cells_q16[lane_y_index] = y_rot_q16;
    return ROPE_Q16_OK;
}
