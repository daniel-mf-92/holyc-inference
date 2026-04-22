// model.HC
// Integer-only forward inference pipeline for TempleOS HolyC runtime.
//
// IQ-1045: embed -> N transformer blocks -> final RMSNorm -> logits
// with strict checked no-partial publication semantics.

#define MODEL_Q16_OK            0
#define MODEL_Q16_ERR_NULL_PTR  1
#define MODEL_Q16_ERR_BAD_PARAM 2
#define MODEL_Q16_ERR_OVERFLOW  4

#define MODEL_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define MODEL_I64_MIN_VALUE -0x8000000000000000

I32 FPQ16RMSNormChecked(I64 *input_q16,
                        I64 *gamma_q16,
                        I64 *output_q16,
                        I64 count,
                        I64 eps_q16);

I32 TransformerBlockQ16ApplyCheckedNoPartialCommitOnly(
    I64 *input_q16,
    I64 input_capacity,
    I64 *attn_out_q16,
    I64 attn_out_capacity,
    I64 *ffn_out_q16,
    I64 ffn_out_capacity,
    I64 *gamma_attn_q16,
    I64 gamma_attn_capacity,
    I64 *gamma_ffn_q16,
    I64 gamma_ffn_capacity,
    I64 eps_q16,
    I64 *workspace_q16,
    I64 workspace_capacity,
    I64 *out_q16,
    I64 out_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_required_qkv_cells,
    I64 *out_required_attn_scores,
    I64 *out_required_ffn_cells,
    I64 *out_required_out_cells,
    I64 *out_required_stage_bytes);

Bool GGUFTensorByteSpansOverlap(U8 *a_base,
                                U64 a_size,
                                U8 *b_base,
                                U64 b_size);

I32 ModelTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return MODEL_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > MODEL_I64_MAX_VALUE - rhs)
        return MODEL_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < MODEL_I64_MIN_VALUE - rhs)
        return MODEL_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return MODEL_Q16_OK;
}

I32 ModelTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return MODEL_Q16_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return MODEL_Q16_OK;
    }

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > (MODEL_I64_MAX_VALUE / rhs))
                return MODEL_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (rhs < (MODEL_I64_MIN_VALUE / lhs))
                return MODEL_Q16_ERR_OVERFLOW;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < (MODEL_I64_MIN_VALUE / rhs))
                return MODEL_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (lhs != 0 && rhs < (MODEL_I64_MAX_VALUE / lhs))
                return MODEL_Q16_ERR_OVERFLOW;
        }
    }

    *out_prod = lhs * rhs;
    return MODEL_Q16_OK;
}

// IQ-1045 commit-only forward pass.
//
// Pipeline:
// 1) Copy embedding slab into private ping buffer.
// 2) For each layer: checked transformer block compose into pong buffer.
// 3) Final RMSNorm row-by-row into private pong buffer.
// 4) Commit logits only after every checked stage succeeds.
//
// All capacities are measured in I64 cells. Dense slab size is:
// dense_cells = row_count * lane_count.
I32 InferenceForwardQ16CheckedNoPartial(
    I64 *embed_q16,
    I64 embed_capacity,
    I64 *block_attn_q16,
    I64 block_attn_capacity,
    I64 *block_ffn_q16,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 *gamma_attn_q16,
    I64 gamma_attn_capacity,
    I64 *gamma_ffn_q16,
    I64 gamma_ffn_capacity,
    I64 block_eps_q16,
    I64 *final_norm_gamma_q16,
    I64 final_norm_gamma_capacity,
    I64 final_norm_eps_q16,
    I64 *workspace_q16,
    I64 workspace_capacity,
    I64 *block_workspace_q16,
    I64 block_workspace_capacity,
    I64 *logits_q16,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count)
{
    I64 dense_cells;
    I64 doubled_dense_cells;
    I64 required_block_tensor_cells;
    I64 dense_bytes;
    I64 row_index;
    I64 lane_index;
    I64 row_base;
    I64 dense_index;
    I64 layer_index;
    I64 layer_base;

    I64 required_qkv_cells;
    I64 required_attn_scores;
    I64 required_ffn_cells;
    I64 required_out_cells;
    I64 required_stage_bytes;

    I64 *current_q16;
    I64 *next_q16;
    I64 *tmp_q16;
    I32 status;

    if (!embed_q16 || !block_attn_q16 || !block_ffn_q16 ||
        !gamma_attn_q16 || !gamma_ffn_q16 ||
        !final_norm_gamma_q16 || !workspace_q16 ||
        !block_workspace_q16 || !logits_q16)
        return MODEL_Q16_ERR_NULL_PTR;

    if (embed_capacity < 0 || block_attn_capacity < 0 || block_ffn_capacity < 0 ||
        gamma_attn_capacity < 0 || gamma_ffn_capacity < 0 ||
        final_norm_gamma_capacity < 0 || workspace_capacity < 0 ||
        block_workspace_capacity < 0 || logits_capacity < 0)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || lane_count < 0 || block_count < 0)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (block_eps_q16 < 0 || final_norm_eps_q16 < 0)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (!row_count || !lane_count)
        return MODEL_Q16_OK;

    if (gamma_attn_capacity < lane_count ||
        gamma_ffn_capacity < lane_count ||
        final_norm_gamma_capacity < lane_count)
        return MODEL_Q16_ERR_BAD_PARAM;

    status = ModelTryMulI64Checked(row_count,
                                   lane_count,
                                   &dense_cells);
    if (status)
        return status;

    status = ModelTryMulI64Checked(dense_cells,
                                   2,
                                   &doubled_dense_cells);
    if (status)
        return status;

    status = ModelTryMulI64Checked(dense_cells,
                                   sizeof(I64),
                                   &dense_bytes);
    if (status)
        return status;

    status = ModelTryMulI64Checked(block_count,
                                   dense_cells,
                                   &required_block_tensor_cells);
    if (status)
        return status;

    if (embed_capacity < dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (block_attn_capacity < required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (block_ffn_capacity < required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (workspace_capacity < doubled_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (logits_capacity < dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (dense_bytes <= 0)
        return MODEL_Q16_ERR_OVERFLOW;

    if (workspace_q16 == logits_q16)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (GGUFTensorByteSpansOverlap((U8 *)workspace_q16,
                                   (U64)dense_bytes,
                                   (U8 *)logits_q16,
                                   (U64)dense_bytes))
        return MODEL_Q16_ERR_BAD_PARAM;
    if (GGUFTensorByteSpansOverlap((U8 *)(workspace_q16 + dense_cells),
                                   (U64)dense_bytes,
                                   (U8 *)logits_q16,
                                   (U64)dense_bytes))
        return MODEL_Q16_ERR_BAD_PARAM;

    current_q16 = workspace_q16;
    next_q16 = workspace_q16 + dense_cells;

    // Stage embedding in private workspace first.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = ModelTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = ModelTryAddI64Checked(row_base,
                                           lane_index,
                                           &dense_index);
            if (status)
                return status;
            current_q16[dense_index] = embed_q16[dense_index];
        }
    }

    // Apply each transformer block in sequence using ping-pong buffers.
    for (layer_index = 0; layer_index < block_count; layer_index++)
    {
        status = ModelTryMulI64Checked(layer_index,
                                       dense_cells,
                                       &layer_base);
        if (status)
            return status;

        status = TransformerBlockQ16ApplyCheckedNoPartialCommitOnly(
            current_q16,
            dense_cells,
            block_attn_q16 + layer_base,
            dense_cells,
            block_ffn_q16 + layer_base,
            dense_cells,
            gamma_attn_q16,
            gamma_attn_capacity,
            gamma_ffn_q16,
            gamma_ffn_capacity,
            block_eps_q16,
            block_workspace_q16,
            block_workspace_capacity,
            next_q16,
            dense_cells,
            row_count,
            lane_count,
            &required_qkv_cells,
            &required_attn_scores,
            &required_ffn_cells,
            &required_out_cells,
            &required_stage_bytes);
        if (status)
            return status;

        if (required_out_cells != dense_cells)
            return MODEL_Q16_ERR_BAD_PARAM;

        tmp_q16 = current_q16;
        current_q16 = next_q16;
        next_q16 = tmp_q16;
    }

    // Final RMSNorm rows into private next_q16 buffer.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = ModelTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        status = FPQ16RMSNormChecked(current_q16 + row_base,
                                     final_norm_gamma_q16,
                                     next_q16 + row_base,
                                     lane_count,
                                     final_norm_eps_q16);
        if (status)
            return status;
    }

    // Commit logits only after full forward pipeline succeeds.
    for (row_index = 0; row_index < row_count; row_index++)
    {
        status = ModelTryMulI64Checked(row_index,
                                       lane_count,
                                       &row_base);
        if (status)
            return status;

        for (lane_index = 0; lane_index < lane_count; lane_index++)
        {
            status = ModelTryAddI64Checked(row_base,
                                           lane_index,
                                           &dense_index);
            if (status)
                return status;
            logits_q16[dense_index] = next_q16[dense_index];
        }
    }

    return MODEL_Q16_OK;
}

// IQ-1053 diagnostics-only preflight helper for forward-pass contracts.
//
// Computes canonical dense/workspace/block/logits cell requirements for
// `InferenceForwardQ16CheckedNoPartial` without touching model tensors.
// Output publication is strict no-partial: every output remains unchanged on
// failure, and staged diagnostics are committed atomically after immutable
// snapshot revalidation.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 staged_dense_cells;
    I64 staged_required_workspace_cells;
    I64 staged_required_block_tensor_cells;
    I64 staged_required_logits_cells;

    I64 snapshot_embed_capacity;
    I64 snapshot_block_attn_capacity;
    I64 snapshot_block_ffn_capacity;
    I64 snapshot_block_count;
    I64 snapshot_gamma_attn_capacity;
    I64 snapshot_gamma_ffn_capacity;
    I64 snapshot_final_norm_gamma_capacity;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (embed_capacity < 0 || block_attn_capacity < 0 || block_ffn_capacity < 0 ||
        block_count < 0 || gamma_attn_capacity < 0 || gamma_ffn_capacity < 0 ||
        final_norm_gamma_capacity < 0 || workspace_capacity < 0 ||
        logits_capacity < 0 || row_count < 0 || lane_count < 0)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_embed_capacity = embed_capacity;
    snapshot_block_attn_capacity = block_attn_capacity;
    snapshot_block_ffn_capacity = block_ffn_capacity;
    snapshot_block_count = block_count;
    snapshot_gamma_attn_capacity = gamma_attn_capacity;
    snapshot_gamma_ffn_capacity = gamma_ffn_capacity;
    snapshot_final_norm_gamma_capacity = final_norm_gamma_capacity;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;

    if (!row_count || !lane_count)
    {
        staged_dense_cells = 0;
        staged_required_workspace_cells = 0;
        staged_required_block_tensor_cells = 0;
        staged_required_logits_cells = 0;
    }
    else
    {
        status = ModelTryMulI64Checked(row_count,
                                       lane_count,
                                       &staged_dense_cells);
        if (status)
            return status;

        status = ModelTryMulI64Checked(staged_dense_cells,
                                       2,
                                       &staged_required_workspace_cells);
        if (status)
            return status;

        status = ModelTryMulI64Checked(block_count,
                                       staged_dense_cells,
                                       &staged_required_block_tensor_cells);
        if (status)
            return status;

        staged_required_logits_cells = staged_dense_cells;

        if (gamma_attn_capacity < lane_count ||
            gamma_ffn_capacity < lane_count ||
            final_norm_gamma_capacity < lane_count)
            return MODEL_Q16_ERR_BAD_PARAM;

        if (embed_capacity < staged_dense_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (block_attn_capacity < staged_required_block_tensor_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (block_ffn_capacity < staged_required_block_tensor_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (workspace_capacity < staged_required_workspace_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (logits_capacity < staged_required_logits_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
    }

    if (snapshot_embed_capacity != embed_capacity ||
        snapshot_block_attn_capacity != block_attn_capacity ||
        snapshot_block_ffn_capacity != block_ffn_capacity ||
        snapshot_block_count != block_count ||
        snapshot_gamma_attn_capacity != gamma_attn_capacity ||
        snapshot_gamma_ffn_capacity != gamma_ffn_capacity ||
        snapshot_final_norm_gamma_capacity != final_norm_gamma_capacity ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = staged_dense_cells;
    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_required_block_tensor_cells = staged_required_block_tensor_cells;
    *out_required_logits_cells = staged_required_logits_cells;
    return MODEL_Q16_OK;
}

// IQ-1054 diagnostics parity gate for forward preflight requirements.
//
// Contract:
// - compose staged requirements via
//   `InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly`;
// - recompute canonical requirements directly from immutable geometry snapshot;
// - publish `{dense,workspace,block_tensor,logits}` only on strict tuple parity.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 commit_dense_cells;
    I64 commit_required_workspace_cells;
    I64 commit_required_block_tensor_cells;
    I64 commit_required_logits_cells;

    I64 canonical_dense_cells;
    I64 canonical_required_workspace_cells;
    I64 canonical_required_block_tensor_cells;
    I64 canonical_required_logits_cells;

    I64 snapshot_embed_capacity;
    I64 snapshot_block_attn_capacity;
    I64 snapshot_block_ffn_capacity;
    I64 snapshot_block_count;
    I64 snapshot_gamma_attn_capacity;
    I64 snapshot_gamma_ffn_capacity;
    I64 snapshot_final_norm_gamma_capacity;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;

    I64 *snapshot_out_dense_cells;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_required_block_tensor_cells;
    I64 *snapshot_out_required_logits_cells;

    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_embed_capacity = embed_capacity;
    snapshot_block_attn_capacity = block_attn_capacity;
    snapshot_block_ffn_capacity = block_ffn_capacity;
    snapshot_block_count = block_count;
    snapshot_gamma_attn_capacity = gamma_attn_capacity;
    snapshot_gamma_ffn_capacity = gamma_ffn_capacity;
    snapshot_final_norm_gamma_capacity = final_norm_gamma_capacity;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;

    snapshot_out_dense_cells = out_dense_cells;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells;
    snapshot_out_required_logits_cells = out_required_logits_cells;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &commit_dense_cells,
        &commit_required_workspace_cells,
        &commit_required_block_tensor_cells,
        &commit_required_logits_cells);
    if (status)
        return status;

    if (!row_count || !lane_count)
    {
        canonical_dense_cells = 0;
        canonical_required_workspace_cells = 0;
        canonical_required_block_tensor_cells = 0;
        canonical_required_logits_cells = 0;
    }
    else
    {
        status = ModelTryMulI64Checked(row_count,
                                       lane_count,
                                       &canonical_dense_cells);
        if (status)
            return status;

        status = ModelTryMulI64Checked(canonical_dense_cells,
                                       2,
                                       &canonical_required_workspace_cells);
        if (status)
            return status;

        status = ModelTryMulI64Checked(block_count,
                                       canonical_dense_cells,
                                       &canonical_required_block_tensor_cells);
        if (status)
            return status;

        canonical_required_logits_cells = canonical_dense_cells;

        if (gamma_attn_capacity < lane_count ||
            gamma_ffn_capacity < lane_count ||
            final_norm_gamma_capacity < lane_count)
            return MODEL_Q16_ERR_BAD_PARAM;

        if (embed_capacity < canonical_dense_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (block_attn_capacity < canonical_required_block_tensor_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (block_ffn_capacity < canonical_required_block_tensor_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (workspace_capacity < canonical_required_workspace_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
        if (logits_capacity < canonical_required_logits_cells)
            return MODEL_Q16_ERR_BAD_PARAM;
    }

    if (snapshot_embed_capacity != embed_capacity ||
        snapshot_block_attn_capacity != block_attn_capacity ||
        snapshot_block_ffn_capacity != block_ffn_capacity ||
        snapshot_block_count != block_count ||
        snapshot_gamma_attn_capacity != gamma_attn_capacity ||
        snapshot_gamma_ffn_capacity != gamma_ffn_capacity ||
        snapshot_final_norm_gamma_capacity != final_norm_gamma_capacity ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (snapshot_out_dense_cells != out_dense_cells ||
        snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_required_block_tensor_cells != out_required_block_tensor_cells ||
        snapshot_out_required_logits_cells != out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (commit_dense_cells < 0 ||
        commit_required_workspace_cells < 0 ||
        commit_required_block_tensor_cells < 0 ||
        commit_required_logits_cells < 0 ||
        canonical_dense_cells < 0 ||
        canonical_required_workspace_cells < 0 ||
        canonical_required_block_tensor_cells < 0 ||
        canonical_required_logits_cells < 0)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (commit_dense_cells != canonical_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (commit_required_workspace_cells != canonical_required_workspace_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (commit_required_block_tensor_cells != canonical_required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (commit_required_logits_cells != canonical_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = commit_dense_cells;
    *out_required_workspace_cells = commit_required_workspace_cells;
    *out_required_block_tensor_cells = commit_required_block_tensor_cells;
    *out_required_logits_cells = commit_required_logits_cells;
    return MODEL_Q16_OK;
}

// IQ-1055 commit-only diagnostics wrapper over IQ-1054 parity gate.
//
// Stages parity-verified diagnostics tuple locally, revalidates immutable
// geometry/capacity snapshots, and atomically publishes outputs only once on
// success. Any failure preserves strict no-partial diagnostics writes.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 commit_dense_cells;
    I64 commit_required_workspace_cells;
    I64 commit_required_block_tensor_cells;
    I64 commit_required_logits_cells;

    I64 canonical_dense_cells;
    I64 canonical_required_workspace_cells;
    I64 canonical_required_block_tensor_cells;
    I64 canonical_required_logits_cells;

    I64 snapshot_embed_capacity;
    I64 snapshot_block_attn_capacity;
    I64 snapshot_block_ffn_capacity;
    I64 snapshot_block_count;
    I64 snapshot_gamma_attn_capacity;
    I64 snapshot_gamma_ffn_capacity;
    I64 snapshot_final_norm_gamma_capacity;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;

    I64 *snapshot_out_dense_cells;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_required_block_tensor_cells;
    I64 *snapshot_out_required_logits_cells;

    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_embed_capacity = embed_capacity;
    snapshot_block_attn_capacity = block_attn_capacity;
    snapshot_block_ffn_capacity = block_ffn_capacity;
    snapshot_block_count = block_count;
    snapshot_gamma_attn_capacity = gamma_attn_capacity;
    snapshot_gamma_ffn_capacity = gamma_ffn_capacity;
    snapshot_final_norm_gamma_capacity = final_norm_gamma_capacity;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;

    snapshot_out_dense_cells = out_dense_cells;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells;
    snapshot_out_required_logits_cells = out_required_logits_cells;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &commit_dense_cells,
        &commit_required_workspace_cells,
        &commit_required_block_tensor_cells,
        &commit_required_logits_cells);
    if (status)
        return status;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &canonical_dense_cells,
        &canonical_required_workspace_cells,
        &canonical_required_block_tensor_cells,
        &canonical_required_logits_cells);
    if (status)
        return status;

    if (snapshot_embed_capacity != embed_capacity ||
        snapshot_block_attn_capacity != block_attn_capacity ||
        snapshot_block_ffn_capacity != block_ffn_capacity ||
        snapshot_block_count != block_count ||
        snapshot_gamma_attn_capacity != gamma_attn_capacity ||
        snapshot_gamma_ffn_capacity != gamma_ffn_capacity ||
        snapshot_final_norm_gamma_capacity != final_norm_gamma_capacity ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (snapshot_out_dense_cells != out_dense_cells ||
        snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_required_block_tensor_cells != out_required_block_tensor_cells ||
        snapshot_out_required_logits_cells != out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (commit_dense_cells != canonical_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (commit_required_workspace_cells != canonical_required_workspace_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (commit_required_block_tensor_cells != canonical_required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (commit_required_logits_cells != canonical_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = commit_dense_cells;
    *out_required_workspace_cells = commit_required_workspace_cells;
    *out_required_block_tensor_cells = commit_required_block_tensor_cells;
    *out_required_logits_cells = commit_required_logits_cells;
    return MODEL_Q16_OK;
}


// IQ-1058 diagnostics-only no-write companion over IQ-1055 commit-only parity wrapper.
//
// Contract:
// - call `...ParityCommitOnly` into private staging tuple only;
// - recompute canonical tuple through `...PreflightOnlyParity` with immutable snapshots;
// - publish outputs atomically only when staged and canonical tuples match exactly.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 staged_dense_cells;
    I64 staged_required_workspace_cells;
    I64 staged_required_block_tensor_cells;
    I64 staged_required_logits_cells;

    I64 canonical_dense_cells;
    I64 canonical_required_workspace_cells;
    I64 canonical_required_block_tensor_cells;
    I64 canonical_required_logits_cells;

    I64 snapshot_embed_capacity;
    I64 snapshot_block_attn_capacity;
    I64 snapshot_block_ffn_capacity;
    I64 snapshot_block_count;
    I64 snapshot_gamma_attn_capacity;
    I64 snapshot_gamma_ffn_capacity;
    I64 snapshot_final_norm_gamma_capacity;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;

    I64 *snapshot_out_dense_cells;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_required_block_tensor_cells;
    I64 *snapshot_out_required_logits_cells;

    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_embed_capacity = embed_capacity;
    snapshot_block_attn_capacity = block_attn_capacity;
    snapshot_block_ffn_capacity = block_ffn_capacity;
    snapshot_block_count = block_count;
    snapshot_gamma_attn_capacity = gamma_attn_capacity;
    snapshot_gamma_ffn_capacity = gamma_ffn_capacity;
    snapshot_final_norm_gamma_capacity = final_norm_gamma_capacity;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;

    snapshot_out_dense_cells = out_dense_cells;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells;
    snapshot_out_required_logits_cells = out_required_logits_cells;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &staged_dense_cells,
        &staged_required_workspace_cells,
        &staged_required_block_tensor_cells,
        &staged_required_logits_cells);
    if (status)
        return status;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &canonical_dense_cells,
        &canonical_required_workspace_cells,
        &canonical_required_block_tensor_cells,
        &canonical_required_logits_cells);
    if (status)
        return status;

    if (snapshot_embed_capacity != embed_capacity ||
        snapshot_block_attn_capacity != block_attn_capacity ||
        snapshot_block_ffn_capacity != block_ffn_capacity ||
        snapshot_block_count != block_count ||
        snapshot_gamma_attn_capacity != gamma_attn_capacity ||
        snapshot_gamma_ffn_capacity != gamma_ffn_capacity ||
        snapshot_final_norm_gamma_capacity != final_norm_gamma_capacity ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (snapshot_out_dense_cells != out_dense_cells ||
        snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_required_block_tensor_cells != out_required_block_tensor_cells ||
        snapshot_out_required_logits_cells != out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (staged_dense_cells != canonical_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_workspace_cells != canonical_required_workspace_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_block_tensor_cells != canonical_required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_logits_cells != canonical_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = staged_dense_cells;
    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_required_block_tensor_cells = staged_required_block_tensor_cells;
    *out_required_logits_cells = staged_required_logits_cells;
    return MODEL_Q16_OK;
}


// IQ-1059 diagnostics-only parity gate over commit-only preflight wrappers.
//
// Compose:
// - `...ParityCommitOnlyPreflightOnly` (diagnostics-only no-write companion)
// - canonical `...ParityCommitOnly` (commit-only parity wrapper)
//
// Publish output tuple only when both staged tuples match exactly and immutable
// `{row_count,lane_count,block_count,workspace_capacity,logits_capacity}`
// snapshots remain unchanged.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 staged_preflight_dense_cells;
    I64 staged_preflight_required_workspace_cells;
    I64 staged_preflight_required_block_tensor_cells;
    I64 staged_preflight_required_logits_cells;

    I64 staged_commit_dense_cells;
    I64 staged_commit_required_workspace_cells;
    I64 staged_commit_required_block_tensor_cells;
    I64 staged_commit_required_logits_cells;

    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I64 snapshot_block_count;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;

    I64 *snapshot_out_dense_cells;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_required_block_tensor_cells;
    I64 *snapshot_out_required_logits_cells;

    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_block_count = block_count;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;

    snapshot_out_dense_cells = out_dense_cells;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells;
    snapshot_out_required_logits_cells = out_required_logits_cells;

    status =
        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
            embed_capacity,
            block_attn_capacity,
            block_ffn_capacity,
            block_count,
            gamma_attn_capacity,
            gamma_ffn_capacity,
            final_norm_gamma_capacity,
            workspace_capacity,
            logits_capacity,
            row_count,
            lane_count,
            &staged_preflight_dense_cells,
            &staged_preflight_required_workspace_cells,
            &staged_preflight_required_block_tensor_cells,
            &staged_preflight_required_logits_cells);
    if (status)
        return status;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &staged_commit_dense_cells,
        &staged_commit_required_workspace_cells,
        &staged_commit_required_block_tensor_cells,
        &staged_commit_required_logits_cells);
    if (status)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count ||
        snapshot_block_count != block_count ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (snapshot_out_dense_cells != out_dense_cells ||
        snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_required_block_tensor_cells != out_required_block_tensor_cells ||
        snapshot_out_required_logits_cells != out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (staged_preflight_dense_cells != staged_commit_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_preflight_required_workspace_cells != staged_commit_required_workspace_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_preflight_required_block_tensor_cells != staged_commit_required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_preflight_required_logits_cells != staged_commit_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = staged_preflight_dense_cells;
    *out_required_workspace_cells = staged_preflight_required_workspace_cells;
    *out_required_block_tensor_cells = staged_preflight_required_block_tensor_cells;
    *out_required_logits_cells = staged_preflight_required_logits_cells;
    return MODEL_Q16_OK;
}


// IQ-1060 commit-only diagnostics wrapper over IQ-1059 parity gate.
//
// Stages tuple output from `...ParityCommitOnlyPreflightOnlyParity` privately,
// recomputes canonical tuple via `...CommitOnlyPreflightOnly`, and atomically
// publishes only when immutable shape snapshots and tuple parity both hold.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 staged_dense_cells;
    I64 staged_required_workspace_cells;
    I64 staged_required_block_tensor_cells;
    I64 staged_required_logits_cells;

    I64 canonical_dense_cells;
    I64 canonical_required_workspace_cells;
    I64 canonical_required_block_tensor_cells;
    I64 canonical_required_logits_cells;

    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I64 snapshot_block_count;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;

    I64 *snapshot_out_dense_cells;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_required_block_tensor_cells;
    I64 *snapshot_out_required_logits_cells;

    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_block_count = block_count;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;

    snapshot_out_dense_cells = out_dense_cells;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells;
    snapshot_out_required_logits_cells = out_required_logits_cells;

    status =
        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
            embed_capacity,
            block_attn_capacity,
            block_ffn_capacity,
            block_count,
            gamma_attn_capacity,
            gamma_ffn_capacity,
            final_norm_gamma_capacity,
            workspace_capacity,
            logits_capacity,
            row_count,
            lane_count,
            &staged_dense_cells,
            &staged_required_workspace_cells,
            &staged_required_block_tensor_cells,
            &staged_required_logits_cells);
    if (status)
        return status;

    status = InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnly(
        embed_capacity,
        block_attn_capacity,
        block_ffn_capacity,
        block_count,
        gamma_attn_capacity,
        gamma_ffn_capacity,
        final_norm_gamma_capacity,
        workspace_capacity,
        logits_capacity,
        row_count,
        lane_count,
        &canonical_dense_cells,
        &canonical_required_workspace_cells,
        &canonical_required_block_tensor_cells,
        &canonical_required_logits_cells);
    if (status)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count ||
        snapshot_block_count != block_count ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (snapshot_out_dense_cells != out_dense_cells ||
        snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_required_block_tensor_cells != out_required_block_tensor_cells ||
        snapshot_out_required_logits_cells != out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (staged_dense_cells != canonical_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_workspace_cells != canonical_required_workspace_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_block_tensor_cells != canonical_required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_logits_cells != canonical_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = staged_dense_cells;
    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_required_block_tensor_cells = staged_required_block_tensor_cells;
    *out_required_logits_cells = staged_required_logits_cells;
    return MODEL_Q16_OK;
}


// IQ-1062 diagnostics-only no-write companion over IQ-1060 commit-only wrapper.
//
// Compose:
// - staged tuple via `...ParityCommitOnlyPreflightOnlyParityCommitOnly`;
// - canonical tuple via `...ParityCommitOnlyPreflightOnlyParity`.
//
// Publish output only when immutable shape snapshots and strict tuple parity
// both hold. On any mismatch, return error with no partial writes.
I32 InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 embed_capacity,
    I64 block_attn_capacity,
    I64 block_ffn_capacity,
    I64 block_count,
    I64 gamma_attn_capacity,
    I64 gamma_ffn_capacity,
    I64 final_norm_gamma_capacity,
    I64 workspace_capacity,
    I64 logits_capacity,
    I64 row_count,
    I64 lane_count,
    I64 *out_dense_cells,
    I64 *out_required_workspace_cells,
    I64 *out_required_block_tensor_cells,
    I64 *out_required_logits_cells)
{
    I64 staged_dense_cells;
    I64 staged_required_workspace_cells;
    I64 staged_required_block_tensor_cells;
    I64 staged_required_logits_cells;

    I64 canonical_dense_cells;
    I64 canonical_required_workspace_cells;
    I64 canonical_required_block_tensor_cells;
    I64 canonical_required_logits_cells;

    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    I64 snapshot_block_count;
    I64 snapshot_workspace_capacity;
    I64 snapshot_logits_capacity;

    I64 *snapshot_out_dense_cells;
    I64 *snapshot_out_required_workspace_cells;
    I64 *snapshot_out_required_block_tensor_cells;
    I64 *snapshot_out_required_logits_cells;

    I32 status;

    if (!out_dense_cells || !out_required_workspace_cells ||
        !out_required_block_tensor_cells || !out_required_logits_cells)
        return MODEL_Q16_ERR_NULL_PTR;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_block_count = block_count;
    snapshot_workspace_capacity = workspace_capacity;
    snapshot_logits_capacity = logits_capacity;

    snapshot_out_dense_cells = out_dense_cells;
    snapshot_out_required_workspace_cells = out_required_workspace_cells;
    snapshot_out_required_block_tensor_cells = out_required_block_tensor_cells;
    snapshot_out_required_logits_cells = out_required_logits_cells;

    status =
        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
            embed_capacity,
            block_attn_capacity,
            block_ffn_capacity,
            block_count,
            gamma_attn_capacity,
            gamma_ffn_capacity,
            final_norm_gamma_capacity,
            workspace_capacity,
            logits_capacity,
            row_count,
            lane_count,
            &staged_dense_cells,
            &staged_required_workspace_cells,
            &staged_required_block_tensor_cells,
            &staged_required_logits_cells);
    if (status)
        return status;

    status =
        InferenceForwardQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
            embed_capacity,
            block_attn_capacity,
            block_ffn_capacity,
            block_count,
            gamma_attn_capacity,
            gamma_ffn_capacity,
            final_norm_gamma_capacity,
            workspace_capacity,
            logits_capacity,
            row_count,
            lane_count,
            &canonical_dense_cells,
            &canonical_required_workspace_cells,
            &canonical_required_block_tensor_cells,
            &canonical_required_logits_cells);
    if (status)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count ||
        snapshot_block_count != block_count ||
        snapshot_workspace_capacity != workspace_capacity ||
        snapshot_logits_capacity != logits_capacity)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (snapshot_out_dense_cells != out_dense_cells ||
        snapshot_out_required_workspace_cells != out_required_workspace_cells ||
        snapshot_out_required_block_tensor_cells != out_required_block_tensor_cells ||
        snapshot_out_required_logits_cells != out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (out_dense_cells == out_required_workspace_cells ||
        out_dense_cells == out_required_block_tensor_cells ||
        out_dense_cells == out_required_logits_cells ||
        out_required_workspace_cells == out_required_block_tensor_cells ||
        out_required_workspace_cells == out_required_logits_cells ||
        out_required_block_tensor_cells == out_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    if (staged_dense_cells != canonical_dense_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_workspace_cells != canonical_required_workspace_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_block_tensor_cells != canonical_required_block_tensor_cells)
        return MODEL_Q16_ERR_BAD_PARAM;
    if (staged_required_logits_cells != canonical_required_logits_cells)
        return MODEL_Q16_ERR_BAD_PARAM;

    *out_dense_cells = staged_dense_cells;
    *out_required_workspace_cells = staged_required_workspace_cells;
    *out_required_block_tensor_cells = staged_required_block_tensor_cells;
    *out_required_logits_cells = staged_required_logits_cells;
    return MODEL_Q16_OK;
}
