// attention.HC
// Integer-only Q16 attention score helpers for HolyC inference.
//
// This file computes per-token Q·K dot rows for one query head row:
//   score_q32[t] = Sum_i ( q_row_q16[i] * k_rows_q16[t*stride + i] )
//
// Inputs are Q16.16 fixed-point values in I64 lanes.
// The product of two Q16 lanes is Q32.32 (stored in I64 integer domain).
// We keep the full Q32 accumulator (no right shift) so softmax/logit stages
// can decide where and how to rescale.

#define ATTN_Q16_OK            0
#define ATTN_Q16_ERR_NULL_PTR  1
#define ATTN_Q16_ERR_BAD_PARAM 2
#define ATTN_Q16_ERR_OVERFLOW  3

#define ATTN_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define ATTN_I64_MIN_VALUE -0x8000000000000000
#define ATTN_U64_MAX_VALUE  0xFFFFFFFFFFFFFFFF
#define ATTN_Q16_SHIFT       16

I32 AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 in_score_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocDefaultCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityDefault(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityDefault(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-636 contract:
    // derive default commit-stage cell capacity from live packed stage slab
    // (`query_row_count * token_count`) and route diagnostics through the
    // canonical explicit-stride required-bytes/commit-capacity helper.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafePreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);


I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytes(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityDefault(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_row_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells)
{
    I64 in_row_cells;
    I64 out_row_cells;
    I64 last_row_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_last_row_base_index ||
        !out_required_in_cells || !out_required_out_cells ||
        !out_required_stage_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 0 || out_score_stride < 0 || row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
    {
        *out_last_row_base_index = 0;
        *out_required_in_cells = 0;
        *out_required_out_cells = 0;
        *out_required_stage_cells = 0;
        return ATTN_Q16_OK;
    }

    if (in_score_stride < 1 || out_score_stride < 1 || row_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(out_row_cells,
                                       1,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (row_stride < in_row_cells || row_stride < out_row_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       row_stride,
                                       &last_row_base_index);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(last_row_base_index,
                                       in_row_cells,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryAddI64Checked(last_row_base_index,
                                       out_row_cells,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    *out_last_row_base_index = last_row_base_index;
    *out_required_in_cells = required_in_cells;
    *out_required_out_cells = required_out_cells;
    *out_required_stage_cells = required_stage_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsChecked(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartial(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_row_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
    I64 row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly(
    I64 row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnlyPreflightOnly(
    I64 row_count,
    I64 token_count,
    I64 staged_scores_capacity,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 commit_stage_cell_capacity;
    I64 commit_stage_byte_capacity;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-634 default-capacity strided commit contract:
    // derive commit scratch capacities from live packed stage slab
    // (`query_row_count * token_count`) and caller stage storage bytes
    // (`staged_scores_capacity * sizeof(I64)`), then route through the
    // canonical strided no-alloc commit-capacity wrapper.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &commit_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacityPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 staged_scores_capacity,
    I64 *out_commit_stage_cell_capacity,
    I64 *out_commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I64 commit_stage_byte_capacity;
    I32 status;

    if (!out_commit_stage_cell_capacity || !out_commit_stage_byte_capacity ||
        !out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-643 preflight-only default-capacity diagnostics contract:
    // derive default commit scratch capacities from the live packed stage slab
    // (`query_row_count * token_count`) and caller staging capacity bytes
    // (`staged_scores_capacity * sizeof(I64)`), route required-bytes diagnostics
    // through the shared explicit-stride commit-capacity helper, and expose both
    // derived capacities + required stage/out diagnostics with no compute/commit.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &commit_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    *out_commit_stage_cell_capacity = commit_stage_cell_capacity;
    *out_commit_stage_byte_capacity = commit_stage_byte_capacity;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideCommitOnly(
    I64 query_row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStridePreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStridePreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_required_q_cells || !out_required_k_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-575 default-stride batched diagnostics contract:
    // - query rows are tightly packed by `head_dim` lanes;
    // - key rows are tightly packed by `head_dim` lanes;
    // - output rows are token-major with stride `token_count`.
    // Route to the shared checked preflight helper so required-cell
    // diagnostics stay composition-identical with explicit stride expansion.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    return AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAlloc(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAlloc(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells)
{
    // IQ-577 wrapper: keep the legacy queue/API name while routing all
    // diagnostics through the canonical default-stride no-alloc preflight
    // implementation.
    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocRequiredBytes(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity);

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
    I64 query_row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity);

U64 AttentionAbsToU64(I64 x)
{
    if (x >= 0)
        return (U64)x;

    // Keep I64_MIN representable using two's-complement identity.
    return ((U64)(-(x + 1))) + 1;
}

I32 AttentionApplySignFromMagnitudeChecked(U64 mag, Bool is_negative, I64 *out_value)
{
    if (!out_value)
        return ATTN_Q16_ERR_NULL_PTR;

    if (is_negative)
    {
        if (mag > 0x8000000000000000)
            return ATTN_Q16_ERR_OVERFLOW;

        if (mag == 0x8000000000000000)
        {
            *out_value = ATTN_I64_MIN_VALUE;
            return ATTN_Q16_OK;
        }

        *out_value = -(I64)mag;
        return ATTN_Q16_OK;
    }

    if (mag > (U64)ATTN_I64_MAX_VALUE)
        return ATTN_Q16_ERR_OVERFLOW;

    *out_value = (I64)mag;
    return ATTN_Q16_OK;
}

I32 AttentionTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return ATTN_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > ATTN_I64_MAX_VALUE - rhs)
        return ATTN_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < ATTN_I64_MIN_VALUE - rhs)
        return ATTN_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return ATTN_Q16_OK;
}

I32 AttentionTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    U64 abs_lhs;
    U64 abs_rhs;
    U64 abs_prod;
    Bool is_negative;

    if (!out_prod)
        return ATTN_Q16_ERR_NULL_PTR;

    abs_lhs = AttentionAbsToU64(lhs);
    abs_rhs = AttentionAbsToU64(rhs);
    is_negative = ((lhs < 0) ^ (rhs < 0));

    if (abs_lhs && abs_rhs && abs_lhs > (ATTN_U64_MAX_VALUE / abs_rhs))
        return ATTN_Q16_ERR_OVERFLOW;

    abs_prod = abs_lhs * abs_rhs;
    return AttentionApplySignFromMagnitudeChecked(abs_prod,
                                                  is_negative,
                                                  out_prod);
}

I32 AttentionTryAddU64Checked(U64 lhs, U64 rhs, U64 *out_sum)
{
    if (!out_sum)
        return ATTN_Q16_ERR_NULL_PTR;

    if (lhs > ATTN_U64_MAX_VALUE - rhs)
        return ATTN_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return ATTN_Q16_OK;
}

I32 AttentionByteRangeEndChecked(I64 base_addr,
                                 I64 span_bytes,
                                 I64 *out_end_addr)
{
    if (!out_end_addr)
        return ATTN_Q16_ERR_NULL_PTR;

    if (span_bytes < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    return AttentionTryAddI64Checked(base_addr,
                                     span_bytes,
                                     out_end_addr);
}

I32 AttentionByteRangesOverlap(I64 a_base,
                               I64 a_end,
                               I64 b_base,
                               I64 b_end)
{
    if (a_base >= a_end || b_base >= b_end)
        return 0;

    if (a_base < b_end && b_base < a_end)
        return 1;

    return 0;
}

I32 AttentionQ32ScaleByQ16Checked(I64 value_q32,
                                  I64 scale_q16,
                                  I64 *out_value_q32)
{
    U64 abs_value;
    U64 abs_scale;
    U64 abs_prod;
    U64 round_bias;
    U64 rounded;
    Bool is_negative;

    if (!out_value_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    abs_value = AttentionAbsToU64(value_q32);
    abs_scale = AttentionAbsToU64(scale_q16);
    is_negative = ((value_q32 < 0) ^ (scale_q16 < 0));

    if (abs_value && abs_scale && abs_value > (ATTN_U64_MAX_VALUE / abs_scale))
        return ATTN_Q16_ERR_OVERFLOW;

    abs_prod = abs_value * abs_scale;
    round_bias = ((U64)1) << (ATTN_Q16_SHIFT - 1);

    if (abs_prod > ATTN_U64_MAX_VALUE - round_bias)
        rounded = ATTN_U64_MAX_VALUE >> ATTN_Q16_SHIFT;
    else
        rounded = (abs_prod + round_bias) >> ATTN_Q16_SHIFT;

    return AttentionApplySignFromMagnitudeChecked(rounded,
                                                  is_negative,
                                                  out_value_q32);
}

I32 AttentionQ16ComputeQKDotRowChecked(I64 *q_row_q16,
                                       I64 q_row_capacity,
                                       I64 *k_rows_q16,
                                       I64 k_rows_capacity,
                                       I64 token_count,
                                       I64 k_row_stride_q16,
                                       I64 head_dim,
                                       I64 *out_scores_q32,
                                       I64 out_scores_capacity,
                                       I64 out_score_stride)
{
    I64 required_k_cells;
    I64 required_out_cells;
    I64 token_index;
    I64 token_base;
    I64 out_base;
    I64 lane_index;
    I64 product_q32;
    I64 score_q32;
    I32 status;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || k_row_stride_q16 < 0 || head_dim < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Query row must expose the active head dimension.
    if (head_dim > q_row_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // If we have tokens, each K row stride must cover the active head lanes,
    // and output stride must reserve at least one cell per token score.
    if (token_count > 0 && k_row_stride_q16 < head_dim)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count > 0 && out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Full K-matrix span check: token_count rows spaced by stride.
    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_k_cells > k_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Output span check for deterministic row-major score layout:
    // score[t] stored at out_scores_q32[t * out_score_stride].
    if (token_count == 0)
        return ATTN_Q16_OK;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(required_out_cells,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           k_row_stride_q16,
                                           &token_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
            return status;

        score_q32 = 0;
        for (lane_index = 0; lane_index < head_dim; lane_index++)
        {
            status = AttentionTryMulI64Checked(q_row_q16[lane_index],
                                               k_rows_q16[token_base + lane_index],
                                               &product_q32);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(score_q32,
                                               product_q32,
                                               &score_q32);
            if (status != ATTN_Q16_OK)
                return status;
        }

        out_scores_q32[out_base] = score_q32;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleChecked(I64 *in_scores_q32,
                                       I64 in_scores_capacity,
                                       I64 token_count,
                                       I64 in_score_stride,
                                       I64 score_scale_q16,
                                       I64 *out_scores_q32,
                                       I64 out_scores_capacity,
                                       I64 out_score_stride)
{
    I64 required_in_cells;
    I64 required_out_cells;
    I64 token_index;
    I64 in_base;
    I64 out_base;
    I64 scaled_score_q32;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || in_score_stride < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
        return ATTN_Q16_OK;

    // Strides must expose one scalar score per token.
    if (in_score_stride < 1 || out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Validate full source span: token_count rows spaced by in_score_stride.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(required_in_cells,
                                       1,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Validate full destination span: token_count rows spaced by out_score_stride.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryAddI64Checked(required_out_cells,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           in_score_stride,
                                           &in_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionQ32ScaleByQ16Checked(in_scores_q32[in_base],
                                               score_scale_q16,
                                               &scaled_score_q32);
        if (status != ATTN_Q16_OK)
            return status;

        out_scores_q32[out_base] = scaled_score_q32;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleCheckedNoPartial(I64 *in_scores_q32,
                                                I64 in_scores_capacity,
                                                I64 token_count,
                                                I64 in_score_stride,
                                                I64 score_scale_q16,
                                                I64 *out_scores_q32,
                                                I64 out_scores_capacity,
                                                I64 out_score_stride)
{
    I64 last_in_base_index;
    I64 last_out_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 token_index;
    I64 out_base;
    I64 *staged_scores_q32;
    I32 status;

    status = AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        in_score_stride,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        &last_in_base_index,
        &last_out_base_index,
        &required_in_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Preflight guarantees span fit, but keep explicit contract checks here
    // so staged wrapper behavior stays self-contained and auditable.
    if (required_in_cells > in_scores_capacity ||
        required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
        return ATTN_Q16_OK;

    // Alias-safe no-partial staging: compute all scaled scores in an isolated
    // dense buffer first, then commit to caller output only on full success.
    stage_cells = token_count;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ApplyScoreScaleChecked(in_scores_q32,
                                                in_scores_capacity,
                                                token_count,
                                                in_score_stride,
                                                score_scale_q16,
                                                staged_scores_q32,
                                                stage_cells,
                                                1);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        out_scores_q32[out_base] = staged_scores_q32[token_index];
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 in_score_stride,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells)
{
    I64 total_in_cells;
    I64 total_out_cells;
    I64 last_in_base_index;
    I64 last_out_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_last_in_base_index ||
        !out_last_out_base_index || !out_required_in_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || in_score_stride < 0 || out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
    {
        *out_last_in_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_in_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    // One score lane per token for both source and destination paths.
    if (in_score_stride < 1 || out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Explicit full-span multiply gates keep stride math auditable before
    // calling the checked scaler core.
    status = AttentionTryMulI64Checked(token_count,
                                       in_score_stride,
                                       &total_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count,
                                       out_score_stride,
                                       &total_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &last_in_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_in_base_index,
                                       1,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &last_out_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_out_base_index,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_last_in_base_index = last_in_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_in_cells = required_in_cells;
    *out_required_out_cells = required_out_cells;

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleCheckedDefaultStride(I64 *in_scores_q32,
                                                    I64 in_scores_capacity,
                                                    I64 token_count,
                                                    I64 score_scale_q16,
                                                    I64 *out_scores_q32,
                                                    I64 out_scores_capacity)
{
    I64 default_score_stride;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-548 default-stride contract for row-major attention score buffers:
    // score[token_index] is addressed at base + token_index * token_count.
    // The checked core owns all overflow and span validation for this stride.
    default_score_stride = token_count;

    return AttentionQ16ApplyScoreScaleChecked(in_scores_q32,
                                              in_scores_capacity,
                                              token_count,
                                              default_score_stride,
                                              score_scale_q16,
                                              out_scores_q32,
                                              out_scores_capacity,
                                              default_score_stride);
}

I32 AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_in_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells)
{
    I64 default_score_stride;

    if (!in_scores_q32 || !out_scores_q32 || !out_last_in_base_index ||
        !out_last_out_base_index || !out_required_in_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-557 default-stride no-partial preflight helper:
    // score lanes are token-major at stride = token_count on both source and
    // destination; delegate to canonical no-partial preflight for all checked
    // overflow/span diagnostics.
    default_score_stride = token_count;

    return AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        default_score_stride,
        out_scores_q32,
        out_scores_capacity,
        default_score_stride,
        out_last_in_base_index,
        out_last_out_base_index,
        out_required_in_cells,
        out_required_out_cells);
}

I32 AttentionQ16ApplyScoreScaleCheckedNoPartialDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_score_stride;
    I64 last_in_base_index;
    I64 last_out_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 token_index;
    I64 out_base;
    I64 *staged_scores_q32;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-565 no-partial default-stride contract:
    // - source and destination strides are token_count (token-major lanes)
    // - run default-stride no-partial preflight first for canonical checked
    //   diagnostics reused by staged explicit compositions.
    // - compute scaled scores into dense staging first, then commit to
    //   token-major destination so failures never partially mutate output.
    default_score_stride = token_count;

    status = AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        &last_in_base_index,
        &last_out_base_index,
        &required_in_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!token_count)
        return ATTN_Q16_OK;

    stage_cells = token_count;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ApplyScoreScaleChecked(in_scores_q32,
                                                in_scores_capacity,
                                                token_count,
                                                default_score_stride,
                                                score_scale_q16,
                                                staged_scores_q32,
                                                stage_cells,
                                                1);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           default_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        out_scores_q32[out_base] = staged_scores_q32[token_index];
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsChecked(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 in_row_cells;
    I64 out_row_cells;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 row_base;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 0 || out_score_stride < 0 || row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    if (row_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 1 || out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Compute per-row active span for source and destination score layouts.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(out_row_cells,
                                       1,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Rows are laid out at fixed row_stride; each row must fit its active
    // token span so batched execution remains deterministic and auditable.
    if (row_stride < in_row_cells || row_stride < out_row_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       row_stride,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_in_cells,
                                       in_row_cells,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       out_row_cells,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-571: deterministic row-major dispatch with checked per-row score
    // scaling and explicit span guards before first caller-visible write.
    row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = AttentionQ16ApplyScoreScaleChecked(in_scores_q32 + row_base,
                                                    in_row_cells,
                                                    token_count,
                                                    in_score_stride,
                                                    score_scale_q16,
                                                    out_scores_q32 + row_base,
                                                    out_row_cells,
                                                    out_score_stride);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryAddI64Checked(row_base,
                                           row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
            return status;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_score_stride;
    I64 default_row_stride;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-588 default-stride contract for row-batched score scaling:
    // score lanes and row slabs both advance at token_count cells.
    default_row_stride = token_count;
    default_score_stride = token_count;

    return AttentionQ16ApplyScoreScaleRowsChecked(in_scores_q32,
                                                  in_scores_capacity,
                                                  row_count,
                                                  token_count,
                                                  default_score_stride,
                                                  default_score_stride,
                                                  default_row_stride,
                                                  score_scale_q16,
                                                  out_scores_q32,
                                                  out_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStride(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_score_stride;
    I64 default_row_stride;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-593 staged no-partial default-stride rows contract:
    // score lanes and row slabs are both token-major (stride=token_count).
    // Delegate to the strict no-partial rows core so staged all-or-nothing
    // write behavior is identical to explicit default-stride composition.
    default_score_stride = token_count;
    default_row_stride = token_count;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartial(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_row_stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartial(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 in_score_stride,
    I64 out_score_stride,
    I64 row_stride,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 last_row_base_index;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 in_row_cells;
    I64 stage_bytes;
    I64 row_index;
    I64 row_base;
    I64 next_row_base;
    I64 stage_row_base;
    I64 token_index;
    I64 out_index;
    I64 stage_index;
    I64 *staged_scores_q32;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (in_score_stride < 0 || out_score_stride < 0 || row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        out_scores_q32,
        out_scores_capacity,
        &last_row_base_index,
        &required_in_cells,
        &required_out_cells,
        &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    // Row-local source span used by per-row no-partial scaling calls.
    status = AttentionTryMulI64Checked(token_count - 1,
                                       in_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(required_stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    // Stage each row into a tight row-major slab first. Any failure returns
    // without touching caller output, preserving strict no-partial semantics.
    row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        status = AttentionQ16ApplyScoreScaleCheckedNoPartial(
            in_scores_q32 + row_base,
            in_row_cells,
            token_count,
            in_score_stride,
            score_scale_q16,
            staged_scores_q32 + stage_row_base,
            token_count,
            1);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        status = AttentionTryAddI64Checked(row_base,
                                           row_stride,
                                           &next_row_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        row_base = next_row_base;
    }

    // Commit only active token cells; caller row padding remains unchanged.
    row_base = 0;
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
            {
                Free(staged_scores_q32);
                return status;
            }

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
            {
                Free(staged_scores_q32);
                return status;
            }

            out_scores_q32[out_index] = staged_scores_q32[stage_index];
        }

        status = AttentionTryAddI64Checked(row_base,
                                           row_stride,
                                           &next_row_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        row_base = next_row_base;
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells)
{
    I64 default_score_stride;
    I64 default_row_stride;
    I64 in_row_cells;
    I64 out_row_cells;
    I64 required_in_cells;
    I64 required_out_cells;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_required_in_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
    {
        *out_required_in_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    // IQ-572 default-stride no-partial diagnostics contract:
    // score rows are token-major, so score stride and row stride are both
    // token_count. This helper computes checked aggregate source/destination
    // spans for all rows before any staged no-partial score writes.
    default_score_stride = token_count;
    default_row_stride = token_count;

    if (default_score_stride < 1 || default_row_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       default_score_stride,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(in_row_cells,
                                       1,
                                       &in_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       default_score_stride,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(out_row_cells,
                                       1,
                                       &out_row_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (default_row_stride < in_row_cells || default_row_stride < out_row_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       default_row_stride,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_in_cells,
                                       in_row_cells,
                                       &required_in_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       default_row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       out_row_cells,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_in_cells > in_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_required_in_cells = required_in_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 default_score_stride;
    I64 default_row_stride;
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_in_bytes;
    I64 required_out_bytes;
    U64 staged_begin;
    U64 staged_end;
    U64 in_begin;
    U64 in_end;
    U64 out_begin;
    U64 out_end;
    I32 status;

    // IQ-589 no-alloc no-partial default-stride contract:
    // all score and row strides are derived from token_count so callers can
    // stage deterministic row-major slabs without hidden allocations.
    default_score_stride = token_count;
    default_row_stride = token_count;

    status =
        AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        staged_scores_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells,
        );
    if (status != ATTN_Q16_OK)
        return status;

    // For default-stride score scaling the source and destination row spans
    // are both token_count-based, so required source cells match required
    // destination cells. Keep this explicit for later alias-window checks.
    required_in_cells = required_out_cells;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    status = AttentionTryMulI64Checked(required_in_cells,
                                       sizeof(I64),
                                       &required_in_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryMulI64Checked(required_out_cells,
                                       sizeof(I64),
                                       &required_out_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_bytes <= 0 || required_in_bytes <= 0 ||
        required_out_bytes <= 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // No-partial contract: staging must be disjoint from live sources/sinks.
    staged_begin = (U64)staged_scores_q32;
    in_begin = (U64)in_scores_q32;
    out_begin = (U64)out_scores_q32;

    status = AttentionTryAddU64Checked(staged_begin,
                                       (U64)required_stage_bytes,
                                       &staged_end);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddU64Checked(in_begin,
                                       (U64)required_in_bytes,
                                       &in_end);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddU64Checked(out_begin,
                                       (U64)required_out_bytes,
                                       &out_end);
    if (status != ATTN_Q16_OK)
        return status;

    if (staged_begin < in_end && in_begin < staged_end)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_begin < out_end && out_begin < staged_end)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Stage first; commit only if every row scales successfully.
    status = AttentionQ16ApplyScoreScaleRowsChecked(in_scores_q32,
                                                    in_scores_capacity,
                                                    row_count,
                                                    token_count,
                                                    default_score_stride,
                                                    default_score_stride,
                                                    default_row_stride,
                                                    score_scale_q16,
                                                    staged_scores_q32,
                                                    staged_scores_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells)
{
    I64 default_score_stride;
    I64 default_row_stride;
    I64 last_row_base_index;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_required_in_cells ||
        !out_required_out_cells || !out_required_stage_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0 ||
        stage_cell_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-594 no-alloc default-stride preflight contract:
    // derive score/row strides from token_count, then reuse the strict
    // no-partial row preflight helper so all span/overflow checks stay
    // centralized before staged compute or commit paths run.
    default_score_stride = token_count;
    default_row_stride = token_count;

    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        default_score_stride,
        default_score_stride,
        default_row_stride,
        out_scores_q32,
        out_scores_capacity,
        &last_row_base_index,
        out_required_in_cells,
        out_required_out_cells,
        out_required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (*out_required_stage_cells > stage_cell_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocDefaultCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_in_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells)
{
    I64 default_stage_cell_capacity;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_required_in_cells ||
        !out_required_out_cells || !out_required_stage_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-628 default-capacity preflight-only contract:
    // derive staging capacity from the live row-major score slab
    // (`row_count * token_count`) with checked overflow guards, then route all
    // diagnostics through the canonical no-alloc default-stride preflight
    // helper without compute/commit side effects.
    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &default_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        out_required_in_cells,
        out_required_out_cells,
        out_required_stage_cells);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 required_in_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32 || !out_required_stage_cells ||
        !out_required_stage_bytes || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0 ||
        stage_cell_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-621 required-bytes helper contract for default-stride no-alloc rows:
    // 1) reuse the canonical no-alloc preflight helper for checked in/out/stage
    //    span validation;
    // 2) convert required stage cells to bytes with checked I64 multiply;
    // 3) publish only the diagnostics needed by no-alloc staging callers.
    status =
        AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
            in_scores_q32,
            in_scores_capacity,
            row_count,
            token_count,
            out_scores_q32,
            out_scores_capacity,
            stage_cell_capacity,
            &required_in_cells,
            &required_out_cells,
            &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(required_stage_cells,
                                       sizeof(I64),
                                       &required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

// Default-capacity wrapper for no-alloc required-bytes diagnostics.
//
// IQ-622 contract:
// - derive `stage_cell_capacity = row_count * token_count` using checked
//   overflow-safe integer math;
// - route all diagnostics/null/shape contracts through the canonical
//   ...NoAllocRequiredBytes helper so behavior stays composition-identical.
I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 default_stage_cell_capacity;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &default_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells);
}

// Default commit-cell-capacity wrapper for no-alloc required-bytes commit
// diagnostics.
I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityDefault(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_required_stage_cells;
    I64 commit_required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-623 commit-capacity diagnostics contract:
    // derive strict required stage cells/bytes through the canonical no-alloc
    // required-bytes helper, then validate those diagnostics against caller
    // commit scratch limits before commit-only dispatch.
    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        &commit_required_stage_cells,
        &commit_required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (commit_required_stage_bytes > commit_stage_byte_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_commit_required_stage_cells = commit_required_stage_cells;
    *out_commit_required_stage_bytes = commit_required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I64 required_q_bytes;
    I64 required_k_bytes;
    I64 required_out_bytes;
    I64 staging_capacity_bytes;
    U64 staged_begin;
    U64 staged_end;
    U64 q_begin;
    U64 q_end;
    U64 k_begin;
    U64 k_end;
    U64 out_begin;
    U64 out_end;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &staging_capacity_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    // IQ-624 no-alloc commit-capacity execution contract:
    // 1) enforce caller commit scratch limits through the shared
    //    ...NoAllocRequiredBytesCommitCapacity diagnostics path;
    // 2) compute staged default-stride scaled rows into caller scratch;
    // 3) commit staged live cells atomically through no-alloc commit-only.
    status = AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;
    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Preserve strict no-partial semantics: staging scratch must not alias
    // source or destination buffers when compute+commit execute in stages.
    if (staged_scores_q32 == in_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == out_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity(
    I64 *in_scores_q32,
    I64 in_scores_capacity,
    I64 row_count,
    I64 token_count,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 default_commit_stage_cell_capacity;
    I64 default_commit_stage_byte_capacity;
    I32 status;

    if (!in_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (in_scores_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-625 default-capacity commit contract:
    // derive commit scratch limits from live row/token geometry
    // (`row_count * token_count`) and caller staging bytes
    // (`staged_scores_capacity * sizeof(I64)`), then route all
    // no-partial semantics through the canonical commit-capacity wrapper.
    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &default_commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &default_commit_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
    I64 row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 required_stage_cells;
    I64 required_out_cells;
    I32 status;

    if (!staged_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (staged_scores_capacity < 0 || out_scores_capacity < 0 ||
        stage_cell_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status =
        AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnlyPreflightOnly(
            row_count,
            token_count,
            staged_scores_capacity,
            out_scores_capacity,
            &required_stage_cells,
            &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // IQ-595 no-alloc commit-only contract:
    // caller announces scratch/stage budget in cells, and commit path must
    // reject writes if required staged slab exceeds that budget.
    if (required_stage_cells > stage_cell_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnly(
    I64 row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_row_stride;
    I64 required_stage_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 token_index;
    I64 row_base;
    I64 stage_row_base;
    I64 out_index;
    I64 stage_index;
    I32 status;

    if (!staged_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    status =
        AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnlyPreflightOnly(
            row_count,
            token_count,
            staged_scores_capacity,
            out_scores_capacity,
            &required_stage_cells,
            &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!row_count || !token_count)
        return ATTN_Q16_OK;

    // IQ-592 commit-only contract for default-stride row-major slabs:
    // both staged and destination rows are tightly packed by token_count.
    // Validate all checked index math before first caller-visible write.
    default_row_stride = token_count;

    // Pass 1: checked index validation only (no writes).
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           default_row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;
        }
    }

    // Pass 2: commit live token lanes only; no row padding writes.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           default_row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;

            out_scores_q32[out_index] = staged_scores_q32[stage_index];
        }
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideCommitOnlyPreflightOnly(
    I64 row_count,
    I64 token_count,
    I64 staged_scores_capacity,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells)
{
    I64 default_row_stride;
    I64 required_stage_cells;
    I64 required_out_cells;
    I32 status;

    if (!out_required_stage_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!row_count || !token_count)
    {
        *out_required_stage_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    default_row_stride = token_count;

    status = AttentionTryMulI64Checked(row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(row_count - 1,
                                       default_row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       token_count,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_required_stage_cells = required_stage_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowChecked(I64 *q_row_q16,
                                          I64 q_row_capacity,
                                          I64 *k_rows_q16,
                                          I64 k_rows_capacity,
                                          I64 token_count,
                                          I64 k_row_stride_q16,
                                          I64 head_dim,
                                          I64 score_scale_q16,
                                          I64 *out_scores_q32,
                                          I64 out_scores_capacity,
                                          I64 out_score_stride)
{
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_k_cells;
    I64 required_out_cells;
    I32 status;

    status = AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        &last_k_base_index,
        &last_out_base_index,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Stage 1: compute unscaled Q·K row in Q32 lanes.
    status = AttentionQ16ComputeQKDotRowChecked(q_row_q16,
                                                q_row_capacity,
                                                k_rows_q16,
                                                k_rows_capacity,
                                                token_count,
                                                k_row_stride_q16,
                                                head_dim,
                                                out_scores_q32,
                                                out_scores_capacity,
                                                out_score_stride);
    if (status != ATTN_Q16_OK)
        return status;

    // Stage 2: apply checked Q16 scale in-place over the same row layout so
    // callers get deterministic pre-softmax score buffers.
    return AttentionQ16ApplyScoreScaleChecked(out_scores_q32,
                                              out_scores_capacity,
                                              token_count,
                                              out_score_stride,
                                              score_scale_q16,
                                              out_scores_q32,
                                              out_scores_capacity,
                                              out_score_stride);
}

I32 AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 required_k_cells;
    I64 required_out_cells;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I32 status;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_last_k_base_index || !out_last_out_base_index ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || k_row_stride_q16 < 0 || head_dim < 0 ||
        out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (head_dim > q_row_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
    {
        *out_last_k_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_k_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    if (k_row_stride_q16 < head_dim)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (out_score_stride < 1)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Keep K/output span checks identical to QK-row checked core so wrappers
    // can preflight with exact success/error diagnostics before score writes.
    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_k_cells > k_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       k_row_stride_q16,
                                       &last_k_base_index);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(token_count - 1,
                                       out_score_stride,
                                       &last_out_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_out_base_index,
                                       1,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_last_k_base_index = last_k_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 default_k_row_stride_q16;
    I64 default_out_score_stride;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_last_k_base_index || !out_last_out_base_index ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-559 default-stride no-partial diagnostics:
    // derive the same row-major strides used by staged no-partial wrappers
    // so callers can validate full K/output spans before score writes.
    default_k_row_stride_q16 = head_dim;
    default_out_score_stride = token_count;

    return AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_k_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_score_stride,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    // IQ-563 no-partial row diagnostics are intentionally identical to the
    // checked row preflight contract. Staged wrappers use this helper before
    // any score writes so all required spans and base indices are validated
    // up-front with exact overflow/error behavior.
    return AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_k_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStride(I64 *q_row_q16,
                                                       I64 q_row_capacity,
                                                       I64 *k_rows_q16,
                                                       I64 k_rows_capacity,
                                                       I64 token_count,
                                                       I64 head_dim,
                                                       I64 score_scale_q16,
                                                       I64 *out_scores_q32,
                                                       I64 out_scores_capacity)
{
    I64 default_k_row_stride_q16;
    I64 default_out_score_stride;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-564 default row-major no-partial bridge contracts:
    // - K rows are tightly packed by head_dim lanes.
    // - Output score row is spaced by token_count lanes.
    // Delegate through the no-partial row helper so this default wrapper
    // preserves staged all-or-nothing semantics while keeping the same
    // stride derivation and checked overflow/span validation.
    default_k_row_stride_q16 = head_dim;
    default_out_score_stride = token_count;

    return AttentionQ16ComputeScaledQKRowCheckedNoPartial(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride);
}

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartial(I64 *q_row_q16,
                                                   I64 q_row_capacity,
                                                   I64 *k_rows_q16,
                                                   I64 k_rows_capacity,
                                                   I64 token_count,
                                                   I64 k_row_stride_q16,
                                                   I64 head_dim,
                                                   I64 score_scale_q16,
                                                   I64 *out_scores_q32,
                                                   I64 out_scores_capacity,
                                                   I64 out_score_stride)
{
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_k_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 required_out_cells;
    I64 token_index;
    I64 out_base;
    I64 *staged_scores_q32;
    I32 status;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || k_row_stride_q16 < 0 || head_dim < 0 ||
        out_score_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowCheckedNoPartialPreflightOnly(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
        &last_k_base_index,
        &last_out_base_index,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!token_count)
        return ATTN_Q16_OK;

    // Stage into a dense contiguous token row first so failures never mutate
    // caller output. Commit to strided output only after full success.
    stage_cells = token_count;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ComputeScaledQKRowChecked(q_row_q16,
                                                   q_row_capacity,
                                                   k_rows_q16,
                                                   k_rows_capacity,
                                                   token_count,
                                                   k_row_stride_q16,
                                                   head_dim,
                                                   score_scale_q16,
                                                   staged_scores_q32,
                                                   stage_cells,
                                                   1);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    for (token_index = 0; token_index < token_count; token_index++)
    {
        status = AttentionTryMulI64Checked(token_index,
                                           out_score_stride,
                                           &out_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        out_scores_q32[out_base] = staged_scores_q32[token_index];
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialDefaultStride(
    I64 *q_row_q16,
    I64 q_row_capacity,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_k_row_stride_q16;
    I64 default_out_score_stride;

    if (!q_row_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_row_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-554 default-stride no-partial contract:
    // - key rows are tightly packed with head_dim cells per token.
    // - output scores are token-major with token_count stride.
    // Delegate to the checked no-partial core so staged all-or-nothing
    // semantics and overflow validation stay identical to explicit composition.
    default_k_row_stride_q16 = head_dim;
    default_out_score_stride = token_count;

    return AttentionQ16ComputeScaledQKRowCheckedNoPartial(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_last_q_base_index || !out_last_k_base_index ||
        !out_last_out_base_index || !out_required_q_cells ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (query_row_count > 0 && query_row_stride_q16 < head_dim)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count > 0 && token_count > 0 && out_row_stride < token_count)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!query_row_count)
    {
        *out_last_q_base_index = 0;
        *out_last_k_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_q_cells = 0;
        *out_required_k_cells = 0;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       query_row_stride_q16,
                                       &last_q_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_q_base_index,
                                       head_dim,
                                       &required_q_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_q_cells > q_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_k_cells > k_rows_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!token_count)
    {
        *out_last_q_base_index = last_q_base_index;
        *out_last_k_base_index = 0;
        *out_last_out_base_index = 0;
        *out_required_q_cells = required_q_cells;
        *out_required_k_cells = required_k_cells;
        *out_required_out_cells = 0;
        return ATTN_Q16_OK;
    }

    status = AttentionTryMulI64Checked(token_count - 1,
                                       k_row_stride_q16,
                                       &last_k_base_index);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       out_row_stride,
                                       &last_out_base_index);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(last_out_base_index,
                                       token_count,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_last_q_base_index = last_q_base_index;
    *out_last_k_base_index = last_k_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_q_cells = required_q_cells;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_last_q_base_index,
    I64 *out_last_k_base_index,
    I64 *out_last_out_base_index,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I32 status;

    if (!out_last_q_base_index || !out_last_k_base_index ||
        !out_last_out_base_index || !out_required_q_cells ||
        !out_required_k_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    // IQ-587: preserve strict no-partial diagnostics semantics by staging
    // all preflight outputs locally and committing to caller pointers only on
    // full success.
    status = AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    *out_last_q_base_index = last_q_base_index;
    *out_last_k_base_index = last_k_base_index;
    *out_last_out_base_index = last_out_base_index;
    *out_required_q_cells = required_q_cells;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsChecked(I64 *q_rows_q16,
                                           I64 q_rows_capacity,
                                           I64 query_row_count,
                                           I64 query_row_stride_q16,
                                           I64 *k_rows_q16,
                                           I64 k_rows_capacity,
                                           I64 token_count,
                                           I64 k_row_stride_q16,
                                           I64 head_dim,
                                           I64 score_scale_q16,
                                           I64 *out_scores_q32,
                                           I64 out_scores_capacity,
                                           I64 out_row_stride)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 q_row_base;
    I64 out_row_base;
    I32 status;

    status = AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count)
        return ATTN_Q16_OK;

    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           query_row_stride_q16,
                                           &q_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           out_row_stride,
                                           &out_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionQ16ComputeScaledQKRowChecked(q_rows_q16 + q_row_base,
                                                       head_dim,
                                                       k_rows_q16,
                                                       k_rows_capacity,
                                                       token_count,
                                                       k_row_stride_q16,
                                                       head_dim,
                                                       score_scale_q16,
                                                       out_scores_q32 + out_row_base,
                                                       token_count,
                                                       1);
        if (status != ATTN_Q16_OK)
            return status;
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartial(I64 *q_rows_q16,
                                                    I64 q_rows_capacity,
                                                    I64 query_row_count,
                                                    I64 query_row_stride_q16,
                                                    I64 *k_rows_q16,
                                                    I64 k_rows_capacity,
                                                    I64 token_count,
                                                    I64 k_row_stride_q16,
                                                    I64 head_dim,
                                                    I64 score_scale_q16,
                                                    I64 *out_scores_q32,
                                                    I64 out_scores_capacity,
                                                    I64 out_row_stride)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 stage_cells;
    I64 stage_bytes;
    I64 row_index;
    I64 token_index;
    I64 row_base;
    I64 out_index;
    I64 *staged_scores_q32;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Empty query batches never touch output.
    if (!query_row_count)
        return ATTN_Q16_OK;

    // If there are no tokens, row cores do zero work and perform no writes.
    if (!token_count)
    {
        return AttentionQ16ComputeScaledQKRowsChecked(
            q_rows_q16,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_scores_q32,
            out_scores_capacity,
            out_row_stride);
    }

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    // Stage complete row-major score slab so any failure leaves caller output
    // untouched. RowsChecked owns all math/overflow checks.
    stage_cells = required_out_cells;
    status = AttentionTryMulI64Checked(stage_cells,
                                       sizeof(I64),
                                       &stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (stage_bytes <= 0)
        return ATTN_Q16_ERR_OVERFLOW;

    staged_scores_q32 = MAlloc(stage_bytes);
    if (!staged_scores_q32)
        return ATTN_Q16_ERR_OVERFLOW;

    status = AttentionQ16ComputeScaledQKRowsChecked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        stage_cells,
        out_row_stride);
    if (status != ATTN_Q16_OK)
    {
        Free(staged_scores_q32);
        return status;
    }

    // Commit only live token cells (preserve caller padding between rows).
    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           out_row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
        {
            Free(staged_scores_q32);
            return status;
        }

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
            {
                Free(staged_scores_q32);
                return status;
            }

            out_scores_q32[out_index] = staged_scores_q32[out_index];
        }
    }

    Free(staged_scores_q32);
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_required_q_cells || !out_required_k_cells ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-562 default-stride no-partial diagnostics contract:
    // - query rows are tightly packed by `head_dim` lanes.
    // - key rows are tightly packed by `head_dim` lanes.
    // - output rows are token-major with `token_count` stride.
    // This function computes all span diagnostics up-front so staged wrappers
    // can reject before any score writes to caller output buffers.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells);
    return status;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells)
{
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32 ||
        !out_required_stage_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (stage_cell_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-596 default-stride no-alloc preflight contract:
    // use the checked no-partial default-stride diagnostics helper so
    // q/k/out span guards are centralized before no-alloc staged commits.
    status = AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_cells > stage_cell_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_required_stage_cells = required_stage_cells;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_out_cells)
{
    I64 default_stage_cell_capacity;
    I32 status;

    if (!out_required_stage_cells || !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-627 default-capacity no-alloc preflight contract:
    // derive caller stage-cell capacity from exact live default-stride score
    // slab (`query_row_count * token_count`) with checked overflow math, then
    // delegate diagnostics and span validation to the canonical no-alloc
    // preflight helper so error/no-partial semantics stay identical.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &default_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        out_required_stage_cells,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocRequiredBytes(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        stage_cell_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-644 preflight-only required-bytes contract:
    // extend default-stride no-alloc diagnostics with checked stage-byte
    // output while preserving shape/span and error semantics by delegating to
    // the canonical ...NoAllocRequiredBytes helper.
    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    // IQ-601 no-alloc required-bytes contract:
    // reuse the strict no-alloc preflight for span/shape validation, then
    // convert checked stage-cell diagnostics into byte diagnostics for caller
    // managed staging buffers used by commit-only pipelines.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        &required_stage_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(required_stage_cells,
                                       sizeof(I64),
                                       &required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 default_stage_cell_capacity;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-605 default-capacity contract:
    // derive caller stage capacity from exact live score slab footprint
    // (`query_row_count * token_count`) with checked overflow arithmetic,
    // then delegate all no-alloc required-bytes diagnostics to the shared
    // helper so null/shape/span semantics stay identical.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &default_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells);
}

// Default commit-cell-capacity wrapper for no-alloc required-bytes commit
// diagnostics.
//
// IQ-615 contract:
// - derive commit_stage_cell_capacity = query_row_count * token_count using
//   checked overflow arithmetic;
// - preserve no-allocation semantics by routing all diagnostics through
//   ...NoAllocRequiredBytesCommitCapacity;
// - keep NULL/shape/capacity diagnostics identical to explicit composition.
I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityDefault(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacityPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 staged_scores_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_stage_cell_capacity;
    I64 commit_stage_byte_capacity;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-629 preflight-only default-capacity contract:
    // derive default commit scratch capacities from the live query/token slab
    // and caller staging capacity, then route diagnostics through the shared
    // ...NoAllocRequiredBytesCommitCapacity helper with no compute/commit.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &commit_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 commit_required_stage_cells;
    I64 commit_required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-606 commit-capacity diagnostics contract:
    // derive strict required stage cells/bytes through the canonical no-alloc
    // required-bytes helper, then validate those diagnostics against caller
    // commit scratch limits before routing to commit-only paths.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        &commit_required_stage_cells,
        &commit_required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (commit_required_stage_bytes > commit_stage_byte_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_commit_required_stage_cells = commit_required_stage_cells;
    *out_commit_required_stage_bytes = commit_required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAlloc(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 stage_cell_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 required_stage_cells;
    I64 required_out_cells;
    I64 required_stage_bytes;
    I64 staging_capacity_bytes;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-604 no-alloc no-partial default-stride contract:
    // 1) preflight all q/k/out/staging spans with checked math;
    // 2) compute full batch into caller-owned staging slab;
    // 3) commit through no-alloc commit-only helper for all-or-nothing writes.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        &required_stage_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (query_row_count)
    {
        status = AttentionTryMulI64Checked(query_row_count - 1,
                                           query_row_stride_q16,
                                           &required_q_cells);
        if (status != ATTN_Q16_OK)
            return status;
        status = AttentionTryAddI64Checked(required_q_cells,
                                           head_dim,
                                           &required_q_cells);
        if (status != ATTN_Q16_OK)
            return status;
    }
    else
        required_q_cells = 0;

    status = AttentionTryMulI64Checked(token_count,
                                       k_row_stride_q16,
                                       &required_k_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;
    if (staged_scores_capacity < required_stage_cells)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(required_stage_cells,
                                       sizeof(I64),
                                       &required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &staging_capacity_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_stage_bytes > staging_capacity_bytes)
        return ATTN_Q16_ERR_BAD_PARAM;

    // Staging must not alias live tensors or destination output.
    if (staged_scores_q32 == q_rows_q16)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == k_rows_q16)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == out_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 default_stage_cell_capacity;
    I64 staging_capacity_bytes;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &default_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &staging_capacity_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    // IQ-607 default-capacity no-alloc contract:
    // derive stage capacity from the exact live score slab
    // (`query_row_count * token_count`) and route staging diagnostics
    // through commit-capacity required-bytes validation before compute.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_stage_cell_capacity,
        staging_capacity_bytes,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return ATTN_Q16_ERR_BAD_PARAM;

    // No-partial guarantee for destination output requires caller staging to
    // be separate from source/destination tensor buffers.
    if (staged_scores_q32 == q_rows_q16)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == k_rows_q16)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == out_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I64 staging_capacity_bytes;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &staging_capacity_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    // IQ-616 commit-capacity no-alloc contract:
    // 1) enforce caller commit scratch contracts through the shared
    //    required-bytes helper (`commit_stage_cell_capacity`,
    //    `commit_stage_byte_capacity`);
    // 2) compute into caller-provided staging buffer with default strides;
    // 3) commit staged live score slab atomically via no-alloc commit-only.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;
    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(required_q_cells,
                                       sizeof(I64),
                                       &required_q_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryMulI64Checked(required_k_cells,
                                       sizeof(I64),
                                       &required_k_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryMulI64Checked(required_out_cells,
                                       sizeof(I64),
                                       &required_out_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_bytes <= 0 || required_q_bytes <= 0 ||
        required_k_bytes <= 0 || required_out_bytes <= 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // No-partial contract: staging must not overlap live Q/K/output spans.
    // This guards interior-pointer aliasing, not just exact pointer equality.
    staged_begin = (U64)staged_scores_q32;
    q_begin = (U64)q_rows_q16;
    k_begin = (U64)k_rows_q16;
    out_begin = (U64)out_scores_q32;

    status = AttentionTryAddU64Checked(staged_begin,
                                       (U64)required_stage_bytes,
                                       &staged_end);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddU64Checked(q_begin,
                                       (U64)required_q_bytes,
                                       &q_end);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddU64Checked(k_begin,
                                       (U64)required_k_bytes,
                                       &k_end);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddU64Checked(out_begin,
                                       (U64)required_out_bytes,
                                       &out_end);
    if (status != ATTN_Q16_OK)
        return status;

    if (staged_begin < q_end && q_begin < staged_end)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_begin < k_end && k_begin < staged_end)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_begin < out_end && out_begin < staged_end)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 default_commit_stage_cell_capacity;
    I64 default_commit_stage_byte_capacity;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-617 default-capacity commit contract:
    // derive commit scratch limits from the live default-stride batch shape
    // (`query_row_count * token_count`) and caller stage buffer cells
    // (`staged_scores_capacity * sizeof(I64)`), then route all no-partial
    // compute/commit semantics through the canonical commit-capacity wrapper.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &default_commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &default_commit_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes)
{
    I64 last_q_base_index;
    I64 last_k_base_index;
    I64 last_out_base_index;
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I32 status;

    if (!out_required_q_cells || !out_required_k_cells || !out_required_out_cells ||
        !out_required_stage_cells || !out_required_stage_bytes)
        return ATTN_Q16_ERR_NULL_PTR;

    // IQ-637 explicit-stride no-alloc preflight contract:
    // 1) reuse the canonical no-partial span checker for Q/K/out windows;
    // 2) derive packed staging requirements (`query_row_count * token_count`);
    // 3) stage every diagnostic locally and only commit on full success.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &last_q_base_index,
        &last_k_base_index,
        &last_out_base_index,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(required_stage_cells,
                                       sizeof(I64),
                                       &required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    *out_required_q_cells = required_q_cells;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;
    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 commit_required_stage_cells;
    I64 commit_required_stage_bytes;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells,
        &commit_required_stage_cells,
        &commit_required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    if (commit_required_stage_cells > commit_stage_cell_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_required_stage_bytes > commit_stage_byte_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_commit_required_stage_cells = commit_required_stage_cells;
    *out_commit_required_stage_bytes = commit_required_stage_bytes;
    *out_required_out_cells = required_out_cells;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 default_stage_cell_capacity;
    I64 default_stage_byte_capacity;
    I32 status;

    if (!out_required_stage_cells || !out_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-635 default-capacity diagnostics contract:
    // derive default stage capacities directly from live packed stage shape,
    // then route all diagnostics through the canonical explicit commit-capacity
    // required-bytes helper so guard/overflow semantics remain identical.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &default_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(default_stage_cell_capacity,
                                       sizeof(I64),
                                       &default_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        default_stage_cell_capacity,
        default_stage_byte_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytes(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *out_required_q_cells,
    I64 *out_required_k_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_cells,
    I64 *out_required_stage_bytes)
{
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I32 status;

    if (!out_required_q_cells || !out_required_k_cells || !out_required_out_cells ||
        !out_required_stage_cells || !out_required_stage_bytes)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-638 explicit-stride required-bytes diagnostics contract:
    // 1) preflight q/k/out span + stage byte requirements through the
    //    canonical no-alloc preflight helper exactly once;
    // 2) stage all diagnostics locally and commit output pointers only
    //    on full success to preserve strict no-partial semantics.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells,
        &required_stage_cells,
        &required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    *out_required_q_cells = required_q_cells;
    *out_required_k_cells = required_k_cells;
    *out_required_out_cells = required_out_cells;
    *out_required_stage_cells = required_stage_cells;
    *out_required_stage_bytes = required_stage_bytes;
    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I64 staging_capacity_bytes;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &staging_capacity_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    // IQ-630/IQ-632 strided no-alloc commit-capacity contract:
    // 1) derive strict required stage/out diagnostics via shared required-bytes
    //    commit-capacity helper;
    // 2) compute into packed staging rows (`out_row_stride = token_count`);
    // 3) commit packed staging rows into caller strided output atomically.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;
    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > staging_capacity_bytes)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (staged_scores_q32 == q_rows_q16)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == k_rows_q16)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_q32 == out_scores_q32)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionQ16ComputeScaledQKRowsChecked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
        token_count);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitOnly(
        query_row_count,
        token_count,
        out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 required_out_cells;
    I32 status;

    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafePreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        &required_stage_cells,
        &required_stage_bytes,
        &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafePreflightOnly(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 commit_stage_cell_capacity,
    I64 commit_stage_byte_capacity,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_commit_required_stage_cells,
    I64 *out_commit_required_stage_bytes,
    I64 *out_required_out_cells)
{
    I64 required_q_cells;
    I64 required_k_cells;
    I64 required_out_cells;
    I64 required_stage_cells;
    I64 required_stage_bytes;
    I64 q_span_bytes;
    I64 k_span_bytes;
    I64 out_span_bytes;
    I64 stage_capacity_bytes;
    I64 q_base_addr;
    I64 q_end_addr;
    I64 k_base_addr;
    I64 k_end_addr;
    I64 out_base_addr;
    I64 out_end_addr;
    I64 stage_base_addr;
    I64 stage_end_addr;
    I32 status;

    if (!out_commit_required_stage_cells || !out_commit_required_stage_bytes ||
        !out_required_out_cells)
        return ATTN_Q16_ERR_NULL_PTR;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (commit_stage_cell_capacity < 0 || commit_stage_byte_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-645 alias-safe contract:
    // 1) run canonical explicit-stride no-alloc preflight diagnostics;
    // 2) reject any live-span overlap between staged buffer window and
    //    q/k/out windows (interior-pointer alias safe);
    // 3) delegate compute+commit to canonical commit-capacity helper.
    status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        &required_q_cells,
        &required_k_cells,
        &required_out_cells,
        &required_stage_cells,
        &required_stage_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_cells > commit_stage_cell_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_stage_bytes > commit_stage_byte_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    *out_commit_required_stage_cells = required_stage_cells;
    *out_commit_required_stage_bytes = required_stage_bytes;
    *out_required_out_cells = required_out_cells;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    if (!staged_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;
    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &stage_capacity_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    if (required_stage_bytes > stage_capacity_bytes)
        return ATTN_Q16_ERR_BAD_PARAM;

    status = AttentionTryMulI64Checked(required_q_cells,
                                       sizeof(I64),
                                       &q_span_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryMulI64Checked(required_k_cells,
                                       sizeof(I64),
                                       &k_span_bytes);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryMulI64Checked(required_out_cells,
                                       sizeof(I64),
                                       &out_span_bytes);
    if (status != ATTN_Q16_OK)
        return status;

    q_base_addr = (I64)q_rows_q16;
    k_base_addr = (I64)k_rows_q16;
    out_base_addr = (I64)out_scores_q32;
    stage_base_addr = (I64)staged_scores_q32;

    status = AttentionByteRangeEndChecked(q_base_addr,
                                          q_span_bytes,
                                          &q_end_addr);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionByteRangeEndChecked(k_base_addr,
                                          k_span_bytes,
                                          &k_end_addr);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionByteRangeEndChecked(out_base_addr,
                                          out_span_bytes,
                                          &out_end_addr);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionByteRangeEndChecked(stage_base_addr,
                                          stage_capacity_bytes,
                                          &stage_end_addr);
    if (status != ATTN_Q16_OK)
        return status;

    if (AttentionByteRangesOverlap(stage_base_addr,
                                   stage_end_addr,
                                   q_base_addr,
                                   q_end_addr))
        return ATTN_Q16_ERR_BAD_PARAM;
    if (AttentionByteRangesOverlap(stage_base_addr,
                                   stage_end_addr,
                                   k_base_addr,
                                   k_end_addr))
        return ATTN_Q16_ERR_BAD_PARAM;
    if (AttentionByteRangesOverlap(stage_base_addr,
                                   stage_end_addr,
                                   out_base_addr,
                                   out_end_addr))
        return ATTN_Q16_ERR_BAD_PARAM;

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacity(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 query_row_stride_q16,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 k_row_stride_q16,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity,
    I64 out_row_stride,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity)
{
    I64 commit_stage_cell_capacity;
    I64 commit_stage_byte_capacity;
    I32 status;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0 ||
        staged_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || query_row_stride_q16 < 0 || token_count < 0 ||
        k_row_stride_q16 < 0 || head_dim < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-651 default-capacity alias-safe wrapper contract:
    // derive commit capacities from packed stage shape and caller staging
    // bytes, then dispatch through the canonical alias-safe commit helper.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &commit_stage_cell_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(staged_scores_capacity,
                                       sizeof(I64),
                                       &commit_stage_byte_capacity);
    if (status != ATTN_Q16_OK)
        return status;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitOnly(
    I64 query_row_count,
    I64 token_count,
    I64 out_row_stride,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 required_stage_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 token_index;
    I64 out_row_base;
    I64 stage_row_base;
    I64 out_index;
    I64 stage_index;
    I32 status;

    if (!staged_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (query_row_count < 0 || token_count < 0 || out_row_stride < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       out_row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       token_count,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // First pass validates all packed-stage and strided-output indices.
    // Any overflow aborts before caller output is modified.
    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           out_row_stride,
                                           &out_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(out_row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;
        }
    }

    // Commit pass copies live token lanes only, preserving row padding in
    // caller destination buffers for strided layouts.
    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           out_row_stride,
                                           &out_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(out_row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;

            out_scores_q32[out_index] = staged_scores_q32[stage_index];
        }
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideCommitOnly(
    I64 query_row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_out_row_stride;
    I64 required_stage_cells;
    I64 required_out_cells;
    I64 row_index;
    I64 token_index;
    I64 row_base;
    I64 stage_row_base;
    I64 out_index;
    I64 stage_index;
    I32 status;

    if (!staged_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (query_row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    // IQ-591 commit-only contract:
    // staged rows are tightly packed by `token_count`, and destination rows
    // use the same default stride. Commit only live token lanes with checked
    // `{row_base + token_index}` math so no-partial callers keep deterministic
    // all-or-nothing semantics.
    default_out_row_stride = token_count;

    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       default_out_row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       token_count,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    // First pass validates every source/destination index. If any checked
    // index operation fails, return without touching caller output.
    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           default_out_row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;
        }
    }

    // Commit pass copies only live token lanes; destination padding remains
    // untouched by construction because writes are limited to `token_count`.
    for (row_index = 0; row_index < query_row_count; row_index++)
    {
        status = AttentionTryMulI64Checked(row_index,
                                           default_out_row_stride,
                                           &row_base);
        if (status != ATTN_Q16_OK)
            return status;

        status = AttentionTryMulI64Checked(row_index,
                                           token_count,
                                           &stage_row_base);
        if (status != ATTN_Q16_OK)
            return status;

        for (token_index = 0; token_index < token_count; token_index++)
        {
            status = AttentionTryAddI64Checked(row_base,
                                               token_index,
                                               &out_index);
            if (status != ATTN_Q16_OK)
                return status;

            status = AttentionTryAddI64Checked(stage_row_base,
                                               token_index,
                                               &stage_index);
            if (status != ATTN_Q16_OK)
                return status;

            out_scores_q32[out_index] = staged_scores_q32[stage_index];
        }
    }

    return ATTN_Q16_OK;
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(
    I64 query_row_count,
    I64 token_count,
    I64 *staged_scores_q32,
    I64 staged_scores_capacity,
    I64 stage_cell_capacity,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_out_row_stride;
    I64 required_stage_cells;
    I64 required_out_cells;
    I32 status;

    if (!staged_scores_q32 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (query_row_count < 0 || token_count < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (staged_scores_capacity < 0 || stage_cell_capacity < 0 ||
        out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    if (!query_row_count || !token_count)
        return ATTN_Q16_OK;

    // IQ-599 no-alloc commit-only contract:
    // - staged rows are tightly packed by `token_count`;
    // - destination rows use default token-major stride;
    // - caller-provided stage capacity is validated before commit.
    status = AttentionTryMulI64Checked(query_row_count,
                                       token_count,
                                       &required_stage_cells);
    if (status != ATTN_Q16_OK)
        return status;

    default_out_row_stride = token_count;
    status = AttentionTryMulI64Checked(query_row_count - 1,
                                       default_out_row_stride,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;
    status = AttentionTryAddI64Checked(required_out_cells,
                                       token_count,
                                       &required_out_cells);
    if (status != ATTN_Q16_OK)
        return status;

    if (required_stage_cells > staged_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_stage_cells > stage_cell_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (required_out_cells > out_scores_capacity)
        return ATTN_Q16_ERR_BAD_PARAM;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideCommitOnly(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-560 default-stride batch contract:
    // - query rows are tightly packed head vectors.
    // - key rows are tightly packed head vectors.
    // - output rows are token-major with token_count row stride.
    // Delegate to the checked rows core so overflow and row-major
    // layout behavior are identical to explicit composition.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    return AttentionQ16ComputeScaledQKRowsChecked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride);
}

I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStride(
    I64 *q_rows_q16,
    I64 q_rows_capacity,
    I64 query_row_count,
    I64 *k_rows_q16,
    I64 k_rows_capacity,
    I64 token_count,
    I64 head_dim,
    I64 score_scale_q16,
    I64 *out_scores_q32,
    I64 out_scores_capacity)
{
    I64 default_query_row_stride_q16;
    I64 default_k_row_stride_q16;
    I64 default_out_row_stride;

    if (!q_rows_q16 || !k_rows_q16 || !out_scores_q32)
        return ATTN_Q16_ERR_NULL_PTR;

    if (q_rows_capacity < 0 || k_rows_capacity < 0 || out_scores_capacity < 0)
        return ATTN_Q16_ERR_BAD_PARAM;
    if (query_row_count < 0 || token_count < 0 || head_dim < 0)
        return ATTN_Q16_ERR_BAD_PARAM;

    // IQ-561 default-stride no-partial batch contract:
    // - query rows are tightly packed head vectors.
    // - key rows are tightly packed head vectors.
    // - output rows are token-major with token_count row stride.
    // Delegate to the strict no-partial rows core so staged all-or-nothing
    // write semantics and overflow checks exactly match explicit composition.
    default_query_row_stride_q16 = head_dim;
    default_k_row_stride_q16 = head_dim;
    default_out_row_stride = token_count;

    return AttentionQ16ComputeScaledQKRowsCheckedNoPartial(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride);
}
