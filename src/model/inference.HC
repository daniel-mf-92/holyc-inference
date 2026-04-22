// inference.HC
// Integer-only inference generation diagnostics wrappers for TempleOS runtime.

#define SAMPLING_Q16_OK            0
#define SAMPLING_Q16_ERR_NULL_PTR  1
#define SAMPLING_Q16_ERR_BAD_PARAM 2
#define SAMPLING_Q16_ERR_OVERFLOW  4

#define SAMPLING_Q16_SHIFT 16
#define SAMPLING_Q16_ONE   (1 << SAMPLING_Q16_SHIFT)

I32 InferenceGenerateTokensPreflightChecked(
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 workspace_stage_logits_capacity,
    I64 workspace_topk_logits_capacity,
    I64 workspace_topk_index_capacity,
    I64 random_q16_capacity,
    I64 generated_capacity,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity);

// IQ-1049 diagnostics parity gate for general top-k/top-p generation contracts.
//
// Contract:
// - no-partial diagnostics publish: outputs remain untouched on failure;
// - compose canonical preflight requirements and recompute commit-path tuple;
// - enforce immutable runtime snapshot parity before atomic publish.
I32 InferenceGenerateTokensCheckedTopKTopPCommitOnlyPreflightOnlyParity(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 temperature_q16,
    I64 top_k,
    I64 top_p_q16,
    I64 repetition_penalty_q16,
    I64 *random_q16_values,
    I64 random_q16_capacity,
    I64 *workspace_stage_logits_q16,
    I64 workspace_stage_logits_capacity,
    I64 *workspace_topk_logits_q16,
    I64 workspace_topk_logits_capacity,
    I64 *workspace_topk_indices,
    I64 workspace_topk_index_capacity,
    I64 *out_generated_tokens,
    I64 generated_capacity,
    I64 *out_required_logits,
    I64 *out_required_tokens,
    I64 *out_final_token_count)
{
    I64 required_step_logits_cells;
    I64 required_history_capacity;
    I64 required_stage_logits_capacity;
    I64 required_topk_capacity;
    I64 required_random_capacity;
    I64 required_generated_capacity;
    I64 commit_required_logits;
    I64 commit_required_tokens;
    I64 commit_final_token_count;

    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_temperature_q16;
    I64 snapshot_top_k;
    I64 snapshot_top_p_q16;
    I64 snapshot_repetition_penalty_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;

    I64 *snapshot_step_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_random_q16_values;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_out_generated_tokens;
    I64 *snapshot_out_required_logits;
    I64 *snapshot_out_required_tokens;
    I64 *snapshot_out_final_token_count;
    I32 status;

    if (!out_required_logits || !out_required_tokens || !out_final_token_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_logits == out_required_tokens ||
        out_required_logits == out_final_token_count ||
        out_required_tokens == out_final_token_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_temperature_q16 = temperature_q16;
    snapshot_top_k = top_k;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_repetition_penalty_q16 = repetition_penalty_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    snapshot_step_logits_q16 = step_logits_q16;
    snapshot_token_history = token_history;
    snapshot_random_q16_values = random_q16_values;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_out_generated_tokens = out_generated_tokens;
    snapshot_out_required_logits = out_required_logits;
    snapshot_out_required_tokens = out_required_tokens;
    snapshot_out_final_token_count = out_final_token_count;

    status = InferenceGenerateTokensPreflightChecked(
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history_capacity,
        token_history_count,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        random_q16_capacity,
        generated_capacity,
        &required_step_logits_cells,
        &required_history_capacity,
        &required_stage_logits_capacity,
        &required_topk_capacity,
        &required_random_capacity,
        &required_generated_capacity);
    if (status)
        return status;

    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        commit_required_logits = vocab_size * max_new_tokens;
    }
    else
    {
        commit_required_logits = 0;
    }

    commit_required_tokens = max_new_tokens;
    commit_final_token_count = token_history_count + max_new_tokens;
    if (commit_final_token_count < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_temperature_q16 != temperature_q16 ||
        snapshot_top_k != top_k ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_repetition_penalty_q16 != repetition_penalty_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_step_logits_q16 != step_logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_random_q16_values != random_q16_values ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_out_generated_tokens != out_generated_tokens ||
        snapshot_out_required_logits != out_required_logits ||
        snapshot_out_required_tokens != out_required_tokens ||
        snapshot_out_final_token_count != out_final_token_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (required_step_logits_cells != commit_required_logits)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_generated_capacity != commit_required_tokens)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_history_capacity != commit_final_token_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_random_capacity != commit_required_tokens)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_logits = commit_required_logits;
    *out_required_tokens = commit_required_tokens;
    *out_final_token_count = commit_final_token_count;
    return SAMPLING_Q16_OK;
}
