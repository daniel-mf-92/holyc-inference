// sampling.HC
// Integer-only sampling helpers for HolyC inference.

#define SAMPLING_Q16_OK            0
#define SAMPLING_Q16_ERR_NULL_PTR  1
#define SAMPLING_Q16_ERR_BAD_PARAM 2
#define SAMPLING_Q16_ERR_OVERFLOW  4
#define SAMPLING_Q16_ERR_DOMAIN    5

#define SAMPLING_Q16_SHIFT 16
#define SAMPLING_Q16_ONE   (1 << SAMPLING_Q16_SHIFT)

// IQ-750 explicit stage-coded generation-step errors.
// Returned code = STAGE_BASE | underlying checked status.
// Example: repetition stage bad-param => 0x0102.
#define GENERATION_STEP_STAGE_REPETITION_BASE 0x0100
#define GENERATION_STEP_STAGE_TEMPERATURE_BASE 0x0200
#define GENERATION_STEP_STAGE_TOPK_BASE       0x0300
#define GENERATION_STEP_STAGE_SOFTMAX_BASE    0x0400
#define GENERATION_STEP_STAGE_TOPP_CUTOFF_BASE 0x0500
#define GENERATION_STEP_STAGE_TOPP_SAMPLE_BASE 0x0600

I32 FPQ16MulDivRoundedChecked(I64 a_q16,
                              I64 b_q16,
                              I64 d_q16,
                              I64 *out_q16);
I32 FPQ16TopKSelectIndicesChecked(I64 *logits_q16,
                                  I64 lane_count,
                                  I64 k,
                                  I64 *out_indices,
                                  I64 out_index_capacity);
I32 FPQ16SoftmaxFromPreclampedChecked(I64 *preclamped_logits_q16,
                                      I64 *exp_lanes_q16,
                                      I64 lane_count,
                                      I64 *probs_q16);
I32 FPQ16TopPSampleIndexChecked(I64 *probs_q16,
                                I64 lane_count,
                                I64 prefix_len,
                                I64 random_q16,
                                I64 *out_index);
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
I32 InferenceGenerateTokensPreflightCheckedDefaultTopK(
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
I32 InferenceGenerateTokensPreflightCheckedDefaultTopP(
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
I32 InferenceGenerateTokensPreflightCheckedTopKDefault(
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
I32 InferenceGenerateTokensPreflightCheckedGreedyDefaultTopK(
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
I32 InferenceGenerateTokensPreflightCheckedGreedyDefaultTopKDefaultCapacity(
    I64 vocab_size,
    I64 max_new_tokens,
    I64 token_history_count,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity);
I32 InferenceGenerateTokensPreflightCheckedGreedyDefault(
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
I32 InferenceGenerateTokensCheckedGreedyDefaultTopKDefaultCapacity(
    I64 *step_logits_q16,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_count,
    I64 repetition_penalty_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 *workspace_topk_logits_q16,
    I64 *workspace_topk_indices,
    I64 *out_generated_tokens,
    I64 *out_generated_count);
I32 InferenceGenerateTokensCheckedTopKDefault(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 workspace_stage_logits_capacity,
    I64 *workspace_topk_logits_q16,
    I64 workspace_topk_logits_capacity,
    I64 *workspace_topk_indices,
    I64 workspace_topk_index_capacity,
    I64 *out_generated_tokens,
    I64 *out_generated_count);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyDefaultCapacity(
    I64 *step_logits_q16,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_count,
    I64 top_p_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 *workspace_topk_logits_q16,
    I64 *workspace_topk_indices,
    I64 *out_generated_tokens,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);
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
    I64 *out_final_token_count);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityPreflightOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnlyParity(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity);

I32 GenerationStepComposeStageError(I32 stage_base, I32 stage_status)
{
    if (!stage_status)
        return SAMPLING_Q16_OK;

    if (stage_status < 0 || stage_status > 0xFF)
        return stage_base | SAMPLING_Q16_ERR_BAD_PARAM;

    return stage_base | stage_status;
}

// Temperature scaling for sampling logits (Q16 lanes):
//   out_logits_q16[i] = round_half_up(logits_q16[i] / temperature_q16)
//                      = round_half_up((logits_q16[i] * 1.0_q16) / temperature_q16)
//
// Contracts:
// - temperature_q16 must be strictly positive.
// - capacities/counts are in I64 lanes and must be non-negative.
// - lane_count lanes are read from logits and written to out.
// - no-partial-write: all lanes are preflighted before commit writes begin.
I32 SamplingQ16ApplyTemperatureChecked(I64 *logits_q16,
                                       I64 logits_capacity,
                                       I64 lane_count,
                                       I64 temperature_q16,
                                       I64 *out_logits_q16,
                                       I64 out_capacity)
{
    I64 lane_index;
    I64 last_index;
    I64 last_byte_offset;
    I64 scaled_lane_q16;
    I32 status;

    if (!logits_q16 || !out_logits_q16)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || out_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (lane_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!lane_count)
        return SAMPLING_Q16_OK;

    if (lane_count > logits_capacity || lane_count > out_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Guard span/index math before per-lane pointer indexing.
    last_index = lane_count - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;
    if ((U64)out_logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Preflight every lane so overflow/domain failures never partially update
    // caller-visible output buffers.
    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = FPQ16MulDivRoundedChecked(logits_q16[lane_index],
                                           SAMPLING_Q16_ONE,
                                           temperature_q16,
                                           &scaled_lane_q16);
        if (status)
            return status;
    }

    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        status = FPQ16MulDivRoundedChecked(logits_q16[lane_index],
                                           SAMPLING_Q16_ONE,
                                           temperature_q16,
                                           &out_logits_q16[lane_index]);
        if (status)
            return status;
    }

    return SAMPLING_Q16_OK;
}

// Checked deterministic top-k index selection for sampling.
//
// Purpose:
//   Select the top-k token indices from a vocab-sized logit vector using
//   integer-only ordering and deterministic stable tie-break semantics.
//
// Ordering contract:
//   - Primary key: larger logit wins.
//   - Tie-break: lower token id wins.
//
// Checked contracts:
//   - logits_q16 and out_indices are non-null.
//   - logits_capacity, vocab_size, k, out_index_capacity are non-negative.
//   - vocab_size <= logits_capacity (only caller-owned initialized lanes read).
//   - k <= vocab_size and k <= out_index_capacity.
//   - No partial writes on all validation failures.
//
// Notes:
//   - The wrapper enforces sampling-specific vocabulary bounds, then delegates
//     ordering and stable tie-break selection to the canonical softmax helper
//     so top-k math exists in one audited place.
I32 SamplingTopKSelectIndicesChecked(I64 *logits_q16,
                                     I64 logits_capacity,
                                     I64 vocab_size,
                                     I64 k,
                                     I64 *out_indices,
                                     I64 out_index_capacity)
{
    if (!logits_q16 || !out_indices)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k < 0 || out_index_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k > out_index_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Canonical stable top-k implementation (descending logit, lower-id tie
    // break) is centralized in softmax.HC for single-source determinism.
    return FPQ16TopKSelectIndicesChecked(logits_q16,
                                         vocab_size,
                                         k,
                                         out_indices,
                                         out_index_capacity);
}

// Diagnostics-only preflight for caller-staged top-k commit pipelines.
//
// Contract:
// - validates immutable input geometry for top-k index+score commit paths;
// - publishes required tuple `{required_indices,required_scores,selected_count}`
//   only on full success;
// - zero-write on all failures.
I32 SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnly(
    I64 *logits_q16,
    I64 logits_capacity,
    I64 vocab_size,
    I64 k,
    I64 out_index_capacity,
    I64 out_score_capacity,
    I64 *out_required_indices,
    I64 *out_required_scores,
    I64 *out_selected_count)
{
    I64 last_index;
    I64 last_byte_offset;
    I64 required_indices;
    I64 required_scores;
    I64 selected_count;

    if (!out_required_indices || !out_required_scores || !out_selected_count)
        return SAMPLING_Q16_ERR_NULL_PTR;
    if (!logits_q16)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k < 0 || out_index_capacity < 0 || out_score_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    required_indices = k;
    required_scores = k;
    selected_count = k;

    if (required_indices > out_index_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_scores > out_score_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (vocab_size)
    {
        // Guard pointer-span arithmetic for indexed logit reads used by
        // commit-time score gather.
        last_index = vocab_size - 1;
        if (last_index > 0x0FFFFFFFFFFFFFFF)
            return SAMPLING_Q16_ERR_OVERFLOW;

        last_byte_offset = last_index << 3;
        if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
            return SAMPLING_Q16_ERR_OVERFLOW;
    }

    *out_required_indices = required_indices;
    *out_required_scores = required_scores;
    *out_selected_count = selected_count;
    return SAMPLING_Q16_OK;
}

// Commit-only top-k selection with staged no-partial writes.
//
// The helper composes checked top-k index selection plus score gather into
// scratch buffers, then commits `{indices,scores,selected_count}` only after
// all checks succeed.
I32 SamplingTopKQ16CheckedNoPartialCommitOnly(
    I64 *logits_q16,
    I64 logits_capacity,
    I64 vocab_size,
    I64 k,
    I64 *out_topk_indices,
    I64 out_index_capacity,
    I64 *out_topk_scores_q16,
    I64 out_score_capacity,
    I64 *out_selected_count)
{
    I64 required_indices;
    I64 required_scores;
    I64 selected_count;
    I64 staged_index_bytes;
    I64 staged_score_bytes;
    I64 i;
    I64 index_i;
    I64 *staged_indices;
    I64 *staged_scores;
    I32 status;

    if (!out_topk_indices || !out_topk_scores_q16 || !out_selected_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    status = SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_capacity,
        vocab_size,
        k,
        out_index_capacity,
        out_score_capacity,
        &required_indices,
        &required_scores,
        &selected_count);
    if (status)
        return status;

    if (!selected_count)
    {
        *out_selected_count = 0;
        return SAMPLING_Q16_OK;
    }

    if (required_indices > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    if (required_scores > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    staged_index_bytes = required_indices << 3;
    staged_score_bytes = required_scores << 3;

    staged_indices = MAlloc(staged_index_bytes);
    if (!staged_indices)
        return SAMPLING_Q16_ERR_NULL_PTR;

    staged_scores = MAlloc(staged_score_bytes);
    if (!staged_scores)
    {
        Free(staged_indices);
        return SAMPLING_Q16_ERR_NULL_PTR;
    }

    status = SamplingTopKSelectIndicesChecked(logits_q16,
                                              logits_capacity,
                                              vocab_size,
                                              k,
                                              staged_indices,
                                              required_indices);
    if (status)
    {
        Free(staged_scores);
        Free(staged_indices);
        return status;
    }

    for (i = 0; i < selected_count; i++)
    {
        index_i = staged_indices[i];
        if (index_i < 0 || index_i >= vocab_size)
        {
            Free(staged_scores);
            Free(staged_indices);
            return SAMPLING_Q16_ERR_BAD_PARAM;
        }

        staged_scores[i] = logits_q16[index_i];
    }

    for (i = 0; i < selected_count; i++)
    {
        out_topk_indices[i] = staged_indices[i];
        out_topk_scores_q16[i] = staged_scores[i];
    }
    *out_selected_count = selected_count;

    Free(staged_scores);
    Free(staged_indices);
    return SAMPLING_Q16_OK;
}

// Diagnostics-only parity gate over top-k preflight and commit paths.
//
// Contracts:
// - enforces immutable input snapshot parity across both subpaths;
// - requires exact tuple parity for
//   `{required_indices,required_scores,selected_count}`;
// - atomically publishes tuple outputs only on strict parity success.
I32 SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
    I64 *logits_q16,
    I64 logits_capacity,
    I64 vocab_size,
    I64 k,
    I64 out_index_capacity,
    I64 out_score_capacity,
    I64 *out_required_indices,
    I64 *out_required_scores,
    I64 *out_selected_count)
{
    I64 snapshot_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_k;
    I64 snapshot_out_index_capacity;
    I64 snapshot_out_score_capacity;
    I64 *snapshot_logits_q16;

    I64 pre_required_indices;
    I64 pre_required_scores;
    I64 pre_selected_count;

    I64 commit_selected_count;
    I64 commit_required_indices;
    I64 commit_required_scores;
    I64 *staged_indices;
    I64 *staged_scores;
    I64 staged_indices_bytes;
    I64 staged_scores_bytes;
    I64 dummy_index;
    I64 dummy_score;
    I32 status;

    if (!out_required_indices || !out_required_scores || !out_selected_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    snapshot_logits_q16 = logits_q16;
    snapshot_logits_capacity = logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_k = k;
    snapshot_out_index_capacity = out_index_capacity;
    snapshot_out_score_capacity = out_score_capacity;

    status = SamplingTopKQ16CheckedNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_capacity,
        vocab_size,
        k,
        out_index_capacity,
        out_score_capacity,
        &pre_required_indices,
        &pre_required_scores,
        &pre_selected_count);
    if (status)
        return status;

    if (snapshot_logits_q16 != logits_q16)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_logits_capacity != logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_vocab_size != vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_k != k)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_out_index_capacity != out_index_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_out_score_capacity != out_score_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    dummy_index = 0;
    dummy_score = 0;
    staged_indices = &dummy_index;
    staged_scores = &dummy_score;
    staged_indices_bytes = 0;
    staged_scores_bytes = 0;

    if (pre_selected_count)
    {
        if (pre_selected_count > 0x0FFFFFFFFFFFFFFF)
            return SAMPLING_Q16_ERR_OVERFLOW;

        staged_indices_bytes = pre_selected_count << 3;
        staged_scores_bytes = pre_selected_count << 3;

        staged_indices = MAlloc(staged_indices_bytes);
        if (!staged_indices)
            return SAMPLING_Q16_ERR_NULL_PTR;

        staged_scores = MAlloc(staged_scores_bytes);
        if (!staged_scores)
        {
            Free(staged_indices);
            return SAMPLING_Q16_ERR_NULL_PTR;
        }
    }

    status = SamplingTopKQ16CheckedNoPartialCommitOnly(logits_q16,
                                                       logits_capacity,
                                                       vocab_size,
                                                       k,
                                                       staged_indices,
                                                       out_index_capacity,
                                                       staged_scores,
                                                       out_score_capacity,
                                                       &commit_selected_count);
    if (status)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return status;
    }

    if (snapshot_logits_q16 != logits_q16)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }
    if (snapshot_logits_capacity != logits_capacity)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }
    if (snapshot_vocab_size != vocab_size)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }
    if (snapshot_k != k)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    commit_required_indices = commit_selected_count;
    commit_required_scores = commit_selected_count;

    if (commit_required_indices != pre_required_indices)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }
    if (commit_required_scores != pre_required_scores)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }
    if (commit_selected_count != pre_selected_count)
    {
        if (pre_selected_count)
        {
            Free(staged_scores);
            Free(staged_indices);
        }
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    *out_required_indices = pre_required_indices;
    *out_required_scores = pre_required_scores;
    *out_selected_count = pre_selected_count;

    if (pre_selected_count)
    {
        Free(staged_scores);
        Free(staged_indices);
    }
    return SAMPLING_Q16_OK;
}

// Checked deterministic top-p cutoff selector over descending probability
// lanes.
//
// Output contract:
//   - Writes exactly one cutoff index to *out_cutoff_index on success.
//   - Returned index is the first crossing i where
//       sum(probs_q16[0..i]) >= top_p_q16.
//   - If cumulative mass never crosses due finite-lane residue, result clamps
//     deterministically to lane_count - 1.
//   - No partial writes on all failure paths.
//
// Checked guards:
//   - probs_q16 and out_cutoff_index are non-null.
//   - probs_capacity/lane_count are non-negative.
//   - lane_count > 0 and lane_count <= probs_capacity.
//   - top_p_q16 in (0, 1.0_q16].
//   - Pointer-span arithmetic for probs_q16[lane_count] cannot overflow U64.
//   - Each probability lane is in [0, 1.0_q16].
//   - Probabilities are monotone non-increasing.
//   - Cumulative mass is monotone, cannot overflow I64, never exceeds 1.0_q16,
//     and total mass must equal 1.0_q16.
I32 SamplingTopPSelectCutoffChecked(I64 *probs_q16,
                                    I64 probs_capacity,
                                    I64 lane_count,
                                    I64 top_p_q16,
                                    I64 *out_cutoff_index)
{
    I64 lane_index;
    I64 last_index;
    I64 last_byte_offset;
    I64 lane_prob_q16;
    I64 prev_prob_q16;
    I64 cumulative_q16;
    I64 prev_cumulative_q16;
    I64 cutoff_index;

    if (!probs_q16 || !out_cutoff_index)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (probs_capacity < 0 || lane_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (!lane_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (lane_count > probs_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Guard span/index arithmetic before lane loads.
    last_index = lane_count - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)probs_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    cumulative_q16 = 0;
    prev_cumulative_q16 = 0;
    prev_prob_q16 = SAMPLING_Q16_ONE;
    cutoff_index = -1;

    for (lane_index = 0; lane_index < lane_count; lane_index++)
    {
        lane_prob_q16 = probs_q16[lane_index];
        if (lane_prob_q16 < 0 || lane_prob_q16 > SAMPLING_Q16_ONE)
            return SAMPLING_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > prev_prob_q16)
            return SAMPLING_Q16_ERR_BAD_PARAM;
        if (lane_prob_q16 > (0x7FFFFFFFFFFFFFFF - cumulative_q16))
            return SAMPLING_Q16_ERR_OVERFLOW;

        cumulative_q16 += lane_prob_q16;
        if (cumulative_q16 < prev_cumulative_q16)
            return SAMPLING_Q16_ERR_OVERFLOW;
        if (cumulative_q16 > SAMPLING_Q16_ONE)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (cutoff_index < 0 && cumulative_q16 >= top_p_q16)
            cutoff_index = lane_index;

        prev_prob_q16 = lane_prob_q16;
        prev_cumulative_q16 = cumulative_q16;
    }

    if (cumulative_q16 != SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (cutoff_index < 0)
        cutoff_index = lane_count - 1;

    *out_cutoff_index = cutoff_index;
    return SAMPLING_Q16_OK;
}

// Checked in-place repetition-penalty pass over logits using token history.
//
// Integer penalty rule (Q16):
//   - For positive logits: divide by penalty (reduces favored repeats).
//   - For negative logits: multiply by penalty (pushes further negative).
//
// Determinism / no-partial semantics:
//   - Each token id present in history is penalized at most once.
//   - Duplicate history entries are skipped after first occurrence.
//   - Full preflight pass validates pointers, bounds, history token ids, and
//     per-token fixed-point arithmetic before any in-place commit writes.
I32 SamplingApplyRepetitionPenaltyChecked(I64 *logits_q16,
                                          I64 logits_capacity,
                                          I64 vocab_size,
                                          I64 *token_history,
                                          I64 token_history_capacity,
                                          I64 token_history_count,
                                          I64 penalty_q16)
{
    I64 history_index;
    I64 prior_index;
    I64 token_id;
    I64 last_index;
    I64 last_byte_offset;
    I64 adjusted_logit_q16;
    I64 base_logit_q16;
    I32 status;
    Bool already_seen;

    if (!logits_q16)
        return SAMPLING_Q16_ERR_NULL_PTR;
    if (token_history_count > 0 && !token_history)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_capacity < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_count > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    // Repetition penalty is a divisor/multiplier ratio and must be >= 1.0_q16.
    // Values below one would reward repeats and violate intended semantics.
    if (penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!vocab_size || !token_history_count)
        return SAMPLING_Q16_OK;

    // Guard logits pointer span for vocab-sized indexed loads/stores.
    last_index = vocab_size - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Guard token-history pointer span for indexed reads.
    last_index = token_history_count - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_byte_offset = last_index << 3;
    if ((U64)token_history > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Preflight: validate token ids, dedup policy, and arithmetic viability.
    for (history_index = 0; history_index < token_history_count; history_index++)
    {
        token_id = token_history[history_index];
        if (token_id < 0 || token_id >= vocab_size)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        already_seen = FALSE;
        for (prior_index = 0; prior_index < history_index; prior_index++)
        {
            if (token_history[prior_index] == token_id)
            {
                already_seen = TRUE;
                break;
            }
        }

        if (already_seen)
            continue;

        base_logit_q16 = logits_q16[token_id];
        if (base_logit_q16 < 0)
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               penalty_q16,
                                               SAMPLING_Q16_ONE,
                                               &adjusted_logit_q16);
        }
        else
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               SAMPLING_Q16_ONE,
                                               penalty_q16,
                                               &adjusted_logit_q16);
        }

        if (status)
            return status;
    }

    // Commit: same dedup/arith path, now writing in place.
    for (history_index = 0; history_index < token_history_count; history_index++)
    {
        token_id = token_history[history_index];

        already_seen = FALSE;
        for (prior_index = 0; prior_index < history_index; prior_index++)
        {
            if (token_history[prior_index] == token_id)
            {
                already_seen = TRUE;
                break;
            }
        }

        if (already_seen)
            continue;

        base_logit_q16 = logits_q16[token_id];
        if (base_logit_q16 < 0)
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               penalty_q16,
                                               SAMPLING_Q16_ONE,
                                               &adjusted_logit_q16);
        }
        else
        {
            status = FPQ16MulDivRoundedChecked(base_logit_q16,
                                               SAMPLING_Q16_ONE,
                                               penalty_q16,
                                               &adjusted_logit_q16);
        }

        if (status)
            return status;
        logits_q16[token_id] = adjusted_logit_q16;
    }

    return SAMPLING_Q16_OK;
}

// Checked composed next-token selector from integer logits.
//
// Pipeline (all integer-only, no allocation):
//   1) stage_logits <- logits (immutable caller logits preserved)
//   2) optional repetition penalty in-place on stage_logits
//   3) temperature scaling in-place on stage_logits
//   4) deterministic stable top-k token index selection
//   5) gather top-k logits -> topk_probs workspace
//   6) softmax over top-k logits (Q16 probabilities)
//   7) top-p cutoff over descending top-k probabilities
//   8) deterministic sample inside top-p prefix using random_q16
//
// Determinism / safety:
//   - Tie-break policy is inherited from SamplingTopKSelectIndicesChecked:
//     larger logit first, lower token id on ties.
//   - *out_token_id is written exactly once on success.
//   - Failures return checked error code and do not partially publish output.
I32 SamplingSelectNextTokenChecked(I64 *logits_q16,
                                   I64 logits_capacity,
                                   I64 vocab_size,
                                   I64 *token_history,
                                   I64 token_history_capacity,
                                   I64 token_history_count,
                                   I64 temperature_q16,
                                   I64 top_k,
                                   I64 top_p_q16,
                                   I64 repetition_penalty_q16,
                                   I64 random_q16,
                                   I64 *stage_logits_q16,
                                   I64 stage_logits_capacity,
                                   I64 *topk_probs_q16,
                                   I64 topk_probs_capacity,
                                   I64 *topk_indices,
                                   I64 topk_index_capacity,
                                   I64 *out_token_id)
{
    I64 lane_index;
    I64 last_index;
    I64 last_byte_offset;
    I64 top_p_cutoff_index;
    I64 nucleus_prefix_len;
    I64 sampled_rank;
    I64 sampled_token_id;
    I32 status;

    if (!logits_q16 || !stage_logits_q16 || !topk_probs_q16 || !topk_indices ||
        !out_token_id)
        return SAMPLING_Q16_ERR_NULL_PTR;
    if (token_history_count > 0 && !token_history)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_capacity < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (stage_logits_capacity < 0 || topk_probs_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (topk_index_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (vocab_size > stage_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (token_history_count > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k > topk_index_capacity || top_k > topk_probs_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (random_q16 < 0 || random_q16 >= SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Guard pointer-span math before copy/gather loops.
    last_index = vocab_size - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;
    if ((U64)stage_logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_index = top_k - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)topk_probs_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;
    if ((U64)topk_indices > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Stage immutable caller logits into scratch before in-place transforms.
    for (lane_index = 0; lane_index < vocab_size; lane_index++)
        stage_logits_q16[lane_index] = logits_q16[lane_index];

    // Repetition penalty before temperature mirrors common LLM decoding order:
    // punish seen tokens first, then smooth/sharpen with temperature.
    status = SamplingApplyRepetitionPenaltyChecked(stage_logits_q16,
                                                   stage_logits_capacity,
                                                   vocab_size,
                                                   token_history,
                                                   token_history_capacity,
                                                   token_history_count,
                                                   repetition_penalty_q16);
    if (status)
        return status;

    status = SamplingQ16ApplyTemperatureChecked(stage_logits_q16,
                                                stage_logits_capacity,
                                                vocab_size,
                                                temperature_q16,
                                                stage_logits_q16,
                                                stage_logits_capacity);
    if (status)
        return status;

    status = SamplingTopKSelectIndicesChecked(stage_logits_q16,
                                              stage_logits_capacity,
                                              vocab_size,
                                              top_k,
                                              topk_indices,
                                              topk_index_capacity);
    if (status)
        return status;

    // Gather top-k logits in stable rank order so probabilities stay descending.
    for (lane_index = 0; lane_index < top_k; lane_index++)
    {
        sampled_token_id = topk_indices[lane_index];
        if (sampled_token_id < 0 || sampled_token_id >= vocab_size)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        topk_probs_q16[lane_index] = stage_logits_q16[sampled_token_id];
    }

    // Alias-permitted call: same topk buffer is reused for preclamped logits,
    // exp staging, and final probabilities to keep the runtime allocation-free.
    status = FPQ16SoftmaxFromPreclampedChecked(topk_probs_q16,
                                               topk_probs_q16,
                                               top_k,
                                               topk_probs_q16);
    if (status)
        return status;

    status = SamplingTopPSelectCutoffChecked(topk_probs_q16,
                                             topk_probs_capacity,
                                             top_k,
                                             top_p_q16,
                                             &top_p_cutoff_index);
    if (status)
        return status;

    nucleus_prefix_len = top_p_cutoff_index + 1;
    status = FPQ16TopPSampleIndexChecked(topk_probs_q16,
                                         top_k,
                                         nucleus_prefix_len,
                                         random_q16,
                                         &sampled_rank);
    if (status)
        return status;

    if (sampled_rank < 0 || sampled_rank >= top_k)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    sampled_token_id = topk_indices[sampled_rank];
    if (sampled_token_id < 0 || sampled_token_id >= vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_token_id = sampled_token_id;
    return SAMPLING_Q16_OK;
}

// Execute exactly one generation step from logits to one sampled token id.
//
// This is the canonical WS7 step-level entrypoint with explicit stage-coded
// failures. It shares integer-only transforms with SamplingSelectNextTokenChecked
// but exposes which transform stage rejected the input.
//
// Stage-coded failures (base | status):
// - repetition penalty: 0x0100
// - temperature scale:  0x0200
// - top-k selection:    0x0300
// - softmax:            0x0400
// - top-p cutoff scan:  0x0500
// - top-p sample:       0x0600
I32 GenerationStepChecked(I64 *logits_q16,
                          I64 logits_capacity,
                          I64 vocab_size,
                          I64 *token_history,
                          I64 token_history_capacity,
                          I64 token_history_count,
                          I64 temperature_q16,
                          I64 top_k,
                          I64 top_p_q16,
                          I64 repetition_penalty_q16,
                          I64 random_q16,
                          I64 *workspace_stage_logits_q16,
                          I64 workspace_stage_logits_capacity,
                          I64 *workspace_topk_logits_q16,
                          I64 workspace_topk_logits_capacity,
                          I64 *workspace_topk_indices,
                          I64 workspace_topk_index_capacity,
                          I64 *out_token_id)
{
    I64 lane_index;
    I64 last_index;
    I64 last_byte_offset;
    I64 top_p_cutoff_index;
    I64 nucleus_prefix_len;
    I64 sampled_rank;
    I64 sampled_token_id;
    I32 status;

    if (!logits_q16 || !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_token_id)
        return SAMPLING_Q16_ERR_NULL_PTR;
    if (token_history_count > 0 && !token_history)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (logits_capacity < 0 || vocab_size < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_capacity < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (workspace_stage_logits_capacity < 0 || workspace_topk_logits_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (workspace_topk_index_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (vocab_size > logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (vocab_size > workspace_stage_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (token_history_count > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k > workspace_topk_index_capacity ||
        top_k > workspace_topk_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (random_q16 < 0 || random_q16 >= SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Checked pointer-span guards before any copy/gather writes.
    last_index = vocab_size - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;
    if ((U64)workspace_stage_logits_q16 >
        (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    last_index = top_k - 1;
    if (last_index > 0x0FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    last_byte_offset = last_index << 3;
    if ((U64)workspace_topk_logits_q16 >
        (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;
    if ((U64)workspace_topk_indices >
        (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
        return SAMPLING_Q16_ERR_OVERFLOW;

    // Stage immutable caller logits into mutable workspace.
    for (lane_index = 0; lane_index < vocab_size; lane_index++)
        workspace_stage_logits_q16[lane_index] = logits_q16[lane_index];

    status = SamplingApplyRepetitionPenaltyChecked(workspace_stage_logits_q16,
                                                   workspace_stage_logits_capacity,
                                                   vocab_size,
                                                   token_history,
                                                   token_history_capacity,
                                                   token_history_count,
                                                   repetition_penalty_q16);
    if (status)
        return GenerationStepComposeStageError(
            GENERATION_STEP_STAGE_REPETITION_BASE,
            status);

    status = SamplingQ16ApplyTemperatureChecked(workspace_stage_logits_q16,
                                                workspace_stage_logits_capacity,
                                                vocab_size,
                                                temperature_q16,
                                                workspace_stage_logits_q16,
                                                workspace_stage_logits_capacity);
    if (status)
        return GenerationStepComposeStageError(
            GENERATION_STEP_STAGE_TEMPERATURE_BASE,
            status);

    status = SamplingTopKSelectIndicesChecked(workspace_stage_logits_q16,
                                              workspace_stage_logits_capacity,
                                              vocab_size,
                                              top_k,
                                              workspace_topk_indices,
                                              workspace_topk_index_capacity);
    if (status)
        return GenerationStepComposeStageError(
            GENERATION_STEP_STAGE_TOPK_BASE,
            status);

    for (lane_index = 0; lane_index < top_k; lane_index++)
    {
        sampled_token_id = workspace_topk_indices[lane_index];
        if (sampled_token_id < 0 || sampled_token_id >= vocab_size)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        workspace_topk_logits_q16[lane_index] =
            workspace_stage_logits_q16[sampled_token_id];
    }

    status = FPQ16SoftmaxFromPreclampedChecked(workspace_topk_logits_q16,
                                               workspace_topk_logits_q16,
                                               top_k,
                                               workspace_topk_logits_q16);
    if (status)
        return GenerationStepComposeStageError(
            GENERATION_STEP_STAGE_SOFTMAX_BASE,
            status);

    status = SamplingTopPSelectCutoffChecked(workspace_topk_logits_q16,
                                             workspace_topk_logits_capacity,
                                             top_k,
                                             top_p_q16,
                                             &top_p_cutoff_index);
    if (status)
        return GenerationStepComposeStageError(
            GENERATION_STEP_STAGE_TOPP_CUTOFF_BASE,
            status);

    nucleus_prefix_len = top_p_cutoff_index + 1;
    status = FPQ16TopPSampleIndexChecked(workspace_topk_logits_q16,
                                         top_k,
                                         nucleus_prefix_len,
                                         random_q16,
                                         &sampled_rank);
    if (status)
        return GenerationStepComposeStageError(
            GENERATION_STEP_STAGE_TOPP_SAMPLE_BASE,
            status);

    if (sampled_rank < 0 || sampled_rank >= top_k)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    sampled_token_id = workspace_topk_indices[sampled_rank];
    if (sampled_token_id < 0 || sampled_token_id >= vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_token_id = sampled_token_id;
    return SAMPLING_Q16_OK;
}

// Run an autoregressive generation loop for a fixed number of steps.
//
// Inputs are precomputed per-step logits laid out row-major:
//   step_logits_q16[step * vocab_size + token_id]
//
// Checked / no-partial contract:
// - Stage generated token ids in token_history tail for step-to-step context.
// - Do not write out_generated_tokens/out_generated_count until every
//   GenerationStepChecked call succeeds.
// - On any failure, return immediately with caller-visible generated outputs
//   unchanged.
I32 GenerationRunChecked(I64 *step_logits_q16,
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
                         I64 random_values_capacity,
                         I64 *workspace_stage_logits_q16,
                         I64 workspace_stage_logits_capacity,
                         I64 *workspace_topk_logits_q16,
                         I64 workspace_topk_logits_capacity,
                         I64 *workspace_topk_indices,
                         I64 workspace_topk_index_capacity,
                         I64 *out_generated_tokens,
                         I64 out_generated_tokens_capacity,
                         I64 *out_generated_count)
{
    I64 step_index;
    I64 required_logits_cells;
    I64 required_history_cells;
    I64 required_generated_cells;
    I64 last_index;
    I64 last_byte_offset;
    I64 sampled_token_id;
    I32 status;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens || !out_generated_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (step_logits_capacity < 0 || vocab_size < 0 || max_new_tokens < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_capacity < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (random_values_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (workspace_stage_logits_capacity < 0 || workspace_topk_logits_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (workspace_topk_index_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (out_generated_tokens_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (vocab_size <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_count > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (max_new_tokens > out_generated_tokens_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (max_new_tokens > random_values_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        required_logits_cells = vocab_size * max_new_tokens;
    }
    else
    {
        required_logits_cells = 0;
    }

    required_history_cells = token_history_count + max_new_tokens;
    if (required_history_cells < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;
    if (required_history_cells > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    required_generated_cells = max_new_tokens;
    if (required_logits_cells > step_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Guard indexed spans before stepping.
    if (required_logits_cells)
    {
        last_index = required_logits_cells - 1;
        if (last_index > 0x0FFFFFFFFFFFFFFF)
            return SAMPLING_Q16_ERR_OVERFLOW;
        last_byte_offset = last_index << 3;
        if ((U64)step_logits_q16 > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
            return SAMPLING_Q16_ERR_OVERFLOW;
    }

    if (required_history_cells)
    {
        last_index = required_history_cells - 1;
        if (last_index > 0x0FFFFFFFFFFFFFFF)
            return SAMPLING_Q16_ERR_OVERFLOW;
        last_byte_offset = last_index << 3;
        if ((U64)token_history > (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
            return SAMPLING_Q16_ERR_OVERFLOW;
    }

    if (required_generated_cells)
    {
        last_index = required_generated_cells - 1;
        if (last_index > 0x0FFFFFFFFFFFFFFF)
            return SAMPLING_Q16_ERR_OVERFLOW;
        last_byte_offset = last_index << 3;
        if ((U64)out_generated_tokens >
            (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
            return SAMPLING_Q16_ERR_OVERFLOW;
        if ((U64)random_q16_values >
            (0xFFFFFFFFFFFFFFFF - (U64)last_byte_offset))
            return SAMPLING_Q16_ERR_OVERFLOW;
    }

    // Stage into token_history tail so repetition penalty sees generated tokens
    // without touching caller-visible generated outputs until final commit.
    for (step_index = 0; step_index < max_new_tokens; step_index++)
    {
        status = GenerationStepChecked(step_logits_q16 + (step_index * vocab_size),
                                       vocab_size,
                                       vocab_size,
                                       token_history,
                                       token_history_capacity,
                                       token_history_count + step_index,
                                       temperature_q16,
                                       top_k,
                                       top_p_q16,
                                       repetition_penalty_q16,
                                       random_q16_values[step_index],
                                       workspace_stage_logits_q16,
                                       workspace_stage_logits_capacity,
                                       workspace_topk_logits_q16,
                                       workspace_topk_logits_capacity,
                                       workspace_topk_indices,
                                       workspace_topk_index_capacity,
                                       &sampled_token_id);
        if (status)
            return status;

        token_history[token_history_count + step_index] = sampled_token_id;
    }

    // Atomic publish: generated outputs only become visible after full success.
    for (step_index = 0; step_index < max_new_tokens; step_index++)
        out_generated_tokens[step_index] = token_history[token_history_count + step_index];

    *out_generated_count = max_new_tokens;
    return SAMPLING_Q16_OK;
}

// WS7 public entrypoint wrapper for token generation.
//
// Default-capacity contracts:
// - live history capacity = token_history_count + max_new_tokens
// - stage logits capacity = vocab_size
// - top-k logits/index capacity = vocab_size
// - random/output generated token capacity = max_new_tokens
//
// The wrapper keeps failure diagnostics deterministic for API callers by
// explicitly classifying additive/multiplicative size math as OVERFLOW and
// capacity/shape contract violations as BAD_PARAM before dispatch.
I32 InferenceGenerateTokensChecked(I64 *step_logits_q16,
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
                                   I64 *workspace_stage_logits_q16,
                                   I64 workspace_stage_logits_capacity,
                                   I64 *workspace_topk_logits_q16,
                                   I64 workspace_topk_logits_capacity,
                                   I64 *workspace_topk_indices,
                                   I64 workspace_topk_index_capacity,
                                   I64 *out_generated_tokens,
                                   I64 *out_generated_count)
{
    I64 required_step_logits_cells;
    I64 required_history_capacity;
    I64 required_stage_logits_capacity;
    I64 required_topk_capacity;
    I64 required_random_capacity;
    I64 required_generated_capacity;
    I32 status;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens || !out_generated_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    status = InferenceGenerateTokensPreflightChecked(
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history_capacity,
        token_history_count,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        max_new_tokens,
        max_new_tokens,
        &required_step_logits_cells,
        &required_history_capacity,
        &required_stage_logits_capacity,
        &required_topk_capacity,
        &required_random_capacity,
        &required_generated_capacity);
    if (status)
        return status;

    return GenerationRunChecked(step_logits_q16,
                                step_logits_capacity,
                                vocab_size,
                                max_new_tokens,
                                token_history,
                                required_history_capacity,
                                token_history_count,
                                temperature_q16,
                                top_k,
                                top_p_q16,
                                repetition_penalty_q16,
                                random_q16_values,
                                required_random_capacity,
                                workspace_stage_logits_q16,
                                required_stage_logits_capacity,
                                workspace_topk_logits_q16,
                                required_topk_capacity,
                                workspace_topk_indices,
                                required_topk_capacity,
                                out_generated_tokens,
                                required_generated_capacity,
                                out_generated_count);
}

// WS7 API sugar wrapper: force top-k to full vocab.
//
// This exposes a minimal public API where caller only provides top-p and
// temperature controls. `top_k=vocab_size` keeps full-vocabulary eligibility
// while preserving the exact checked/no-partial semantics of
// `InferenceGenerateTokensChecked`.
I32 InferenceGenerateTokensCheckedDefaultTopK(I64 *step_logits_q16,
                                              I64 step_logits_capacity,
                                              I64 vocab_size,
                                              I64 max_new_tokens,
                                              I64 *token_history,
                                              I64 token_history_capacity,
                                              I64 token_history_count,
                                              I64 temperature_q16,
                                              I64 top_p_q16,
                                              I64 repetition_penalty_q16,
                                              I64 *random_q16_values,
                                              I64 *workspace_stage_logits_q16,
                                              I64 workspace_stage_logits_capacity,
                                              I64 *workspace_topk_logits_q16,
                                              I64 workspace_topk_logits_capacity,
                                              I64 *workspace_topk_indices,
                                              I64 workspace_topk_index_capacity,
                                              I64 *out_generated_tokens,
                                              I64 *out_generated_count)
{
    return InferenceGenerateTokensChecked(step_logits_q16,
                                          step_logits_capacity,
                                          vocab_size,
                                          max_new_tokens,
                                          token_history,
                                          token_history_capacity,
                                          token_history_count,
                                          temperature_q16,
                                          vocab_size,
                                          top_p_q16,
                                          repetition_penalty_q16,
                                          random_q16_values,
                                          workspace_stage_logits_q16,
                                          workspace_stage_logits_capacity,
                                          workspace_topk_logits_q16,
                                          workspace_topk_logits_capacity,
                                          workspace_topk_indices,
                                          workspace_topk_index_capacity,
                                          out_generated_tokens,
                                          out_generated_count);
}

// WS7 API sugar wrapper: force top-p to full probability mass.
//
// Caller still controls `top_k` and temperature; setting `top_p_q16=1.0`
// disables nucleus truncation while preserving the exact checked and
// no-partial semantics of `InferenceGenerateTokensChecked`.
I32 InferenceGenerateTokensCheckedDefaultTopP(I64 *step_logits_q16,
                                              I64 step_logits_capacity,
                                              I64 vocab_size,
                                              I64 max_new_tokens,
                                              I64 *token_history,
                                              I64 token_history_capacity,
                                              I64 token_history_count,
                                              I64 temperature_q16,
                                              I64 top_k,
                                              I64 repetition_penalty_q16,
                                              I64 *random_q16_values,
                                              I64 *workspace_stage_logits_q16,
                                              I64 workspace_stage_logits_capacity,
                                              I64 *workspace_topk_logits_q16,
                                              I64 workspace_topk_logits_capacity,
                                              I64 *workspace_topk_indices,
                                              I64 workspace_topk_index_capacity,
                                              I64 *out_generated_tokens,
                                              I64 *out_generated_count)
{
    return InferenceGenerateTokensChecked(step_logits_q16,
                                          step_logits_capacity,
                                          vocab_size,
                                          max_new_tokens,
                                          token_history,
                                          token_history_capacity,
                                          token_history_count,
                                          temperature_q16,
                                          top_k,
                                          SAMPLING_Q16_ONE,
                                          repetition_penalty_q16,
                                          random_q16_values,
                                          workspace_stage_logits_q16,
                                          workspace_stage_logits_capacity,
                                          workspace_topk_logits_q16,
                                          workspace_topk_logits_capacity,
                                          workspace_topk_indices,
                                          workspace_topk_index_capacity,
                                          out_generated_tokens,
                                          out_generated_count);
}

// WS7 API sugar wrapper: deterministic top-k-default generation.
//
// Fixed sampling controls for this entry point:
// - temperature_q16 = SAMPLING_Q16_ONE
// - top_k = vocab_size (full-vocabulary eligibility)
// - repetition_penalty_q16 = SAMPLING_Q16_ONE
//
// Caller still provides `top_p_q16` and all workspace/history capacities.
// The wrapper preserves exact checked/no-partial semantics by delegating to
// canonical `InferenceGenerateTokensChecked` with only sampling constants
// substituted.
I32 InferenceGenerateTokensCheckedTopKDefault(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 workspace_stage_logits_capacity,
    I64 *workspace_topk_logits_q16,
    I64 workspace_topk_logits_capacity,
    I64 *workspace_topk_indices,
    I64 workspace_topk_index_capacity,
    I64 *out_generated_tokens,
    I64 *out_generated_count)
{
    return InferenceGenerateTokensChecked(step_logits_q16,
                                          step_logits_capacity,
                                          vocab_size,
                                          max_new_tokens,
                                          token_history,
                                          token_history_capacity,
                                          token_history_count,
                                          SAMPLING_Q16_ONE,
                                          vocab_size,
                                          top_p_q16,
                                          SAMPLING_Q16_ONE,
                                          random_q16_values,
                                          workspace_stage_logits_q16,
                                          workspace_stage_logits_capacity,
                                          workspace_topk_logits_q16,
                                          workspace_topk_logits_capacity,
                                          workspace_topk_indices,
                                          workspace_topk_index_capacity,
                                          out_generated_tokens,
                                          out_generated_count);
}

// WS7 diagnostics-only companion for top-k-default checked generation.
//
// Preflight invariants:
// - pointer contract mirrors checked runtime wrapper inputs/workspaces;
// - top-p domain must remain (0, 1.0_q16];
// - capacity diagnostics are computed through canonical top-k-default
//   preflight and staged locally before publish.
//
// No-partial diagnostics contract:
// - required-capacity outputs are not modified unless all checks pass;
// - generation/token-history buffers are never written.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity)
{
    I64 required_step_logits_cells;
    I64 required_history_capacity;
    I64 required_stage_logits_capacity;
    I64 required_topk_capacity;
    I64 required_random_capacity;
    I64 required_generated_capacity;
    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_top_k;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;
    I64 snapshot_top_p_q16;
    I32 status;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity ||
        !out_required_random_capacity || !out_required_generated_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_top_k = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;
    snapshot_top_p_q16 = top_p_q16;

    status = InferenceGenerateTokensPreflightCheckedTopKDefault(
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

    // Immutable snapshot revalidation before diagnostics publish.
    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_top_k != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity ||
        snapshot_top_p_q16 != top_p_q16)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = required_step_logits_cells;
    *out_required_history_capacity = required_history_capacity;
    *out_required_stage_logits_capacity = required_stage_logits_capacity;
    *out_required_topk_capacity = required_topk_capacity;
    *out_required_random_capacity = required_random_capacity;
    *out_required_generated_capacity = required_generated_capacity;
    return SAMPLING_Q16_OK;
}

// WS7 diagnostics-only default-capacity wrapper for top-k-default preflight.
//
// Derived capacities (integer-only, overflow-checked):
// - step_logits_capacity = vocab_size * max_new_tokens
// - token_history_capacity = token_history_count + max_new_tokens
// - workspace stage/top-k capacities = vocab_size
// - random/generated capacities = max_new_tokens
//
// No-partial diagnostics contract: delegate publishes required-capacity tuple
// only on full success; no generation/history buffers are modified.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyDefaultCapacity(
    I64 *step_logits_q16,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_count,
    I64 top_p_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 *workspace_topk_logits_q16,
    I64 *workspace_topk_indices,
    I64 *out_generated_tokens,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity)
{
    I64 step_logits_capacity;
    I64 token_history_capacity;

    if (vocab_size < 0 || max_new_tokens < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    token_history_capacity = token_history_count + max_new_tokens;
    if (token_history_capacity < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;

    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        step_logits_capacity = vocab_size * max_new_tokens;
    }
    else
    {
        step_logits_capacity = 0;
    }

    return InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        max_new_tokens,
        workspace_stage_logits_q16,
        vocab_size,
        workspace_topk_logits_q16,
        vocab_size,
        workspace_topk_indices,
        vocab_size,
        out_generated_tokens,
        max_new_tokens,
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 diagnostics commit-only helper for top-k-default preflight requirements.
//
// Contract:
// - compose canonical `...PreflightOnly` and stage full diagnostics tuple;
// - publish only selected fields `{required_step_logits_cells,required_generated_capacity,required_random_capacity}`;
// - preserve no-partial semantics by writing caller outputs only after full validation.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 staged_required_step_logits_cells;
    I64 staged_required_history_capacity;
    I64 staged_required_stage_logits_capacity;
    I64 staged_required_topk_capacity;
    I64 staged_required_random_capacity;
    I64 staged_required_generated_capacity;

    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &staged_required_step_logits_cells,
        &staged_required_history_capacity,
        &staged_required_stage_logits_capacity,
        &staged_required_topk_capacity,
        &staged_required_random_capacity,
        &staged_required_generated_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_required_step_logits_cells;
    *out_required_generated_capacity = staged_required_generated_capacity;
    *out_required_random_capacity = staged_required_random_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-843 parity gate for top-k-default diagnostics wrappers.
//
// Contract:
// 1) Stage the selected diagnostics tuple from commit-only wrapper.
// 2) Stage canonical diagnostics tuple from preflight-only wrapper.
// 3) Revalidate immutable generation/capacity snapshots.
// 4) Require exact tuple parity before atomically publishing outputs.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 commit_required_step_logits_cells;
    I64 commit_required_generated_capacity;
    I64 commit_required_random_capacity;

    I64 canonical_required_step_logits_cells;
    I64 canonical_required_history_capacity;
    I64 canonical_required_stage_logits_capacity;
    I64 canonical_required_topk_capacity;
    I64 canonical_required_random_capacity;
    I64 canonical_required_generated_capacity;

    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &commit_required_step_logits_cells,
        &commit_required_generated_capacity,
        &commit_required_random_capacity);
    if (status)
        return status;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &canonical_required_step_logits_cells,
        &canonical_required_history_capacity,
        &canonical_required_stage_logits_capacity,
        &canonical_required_topk_capacity,
        &canonical_required_random_capacity,
        &canonical_required_generated_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (commit_required_step_logits_cells != canonical_required_step_logits_cells)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (commit_required_generated_capacity != canonical_required_generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (commit_required_random_capacity != canonical_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = commit_required_step_logits_cells;
    *out_required_generated_capacity = commit_required_generated_capacity;
    *out_required_random_capacity = commit_required_random_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-850 strict commit-only diagnostics wrapper for top-k-default parity helper.
//
// Contract:
// 1) Stage required-capacity tuple via parity helper into local outputs.
// 2) Revalidate immutable generation/capacity snapshot arguments.
// 3) Atomically publish staged diagnostics only on full success.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;

    I64 staged_required_step_logits_cells;
    I64 staged_required_generated_capacity;
    I64 staged_required_random_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    staged_required_step_logits_cells = *out_required_step_logits_cells;
    staged_required_generated_capacity = *out_required_generated_capacity;
    staged_required_random_capacity = *out_required_random_capacity;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &staged_required_step_logits_cells,
        &staged_required_generated_capacity,
        &staged_required_random_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_required_step_logits_cells;
    *out_required_generated_capacity = staged_required_generated_capacity;
    *out_required_random_capacity = staged_required_random_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-849 diagnostics-only no-write companion for top-k-default parity helper.
//
// Contract:
// - stage commit-only-parity diagnostics tuple without caller writes,
// - recompute canonical tuple from immutable generation snapshots,
// - enforce exact tuple parity and immutable-argument invariants,
// - atomically publish diagnostics only on full success.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityPreflightOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 parity_required_step_logits_cells;
    I64 parity_required_generated_capacity;
    I64 parity_required_random_capacity;

    I64 canonical_required_step_logits_cells;
    I64 canonical_required_history_capacity;
    I64 canonical_required_stage_logits_capacity;
    I64 canonical_required_topk_capacity;
    I64 canonical_required_random_capacity;
    I64 canonical_required_generated_capacity;

    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    parity_required_step_logits_cells = *out_required_step_logits_cells;
    parity_required_generated_capacity = *out_required_generated_capacity;
    parity_required_random_capacity = *out_required_random_capacity;

    canonical_required_step_logits_cells = 0;
    canonical_required_history_capacity = 0;
    canonical_required_stage_logits_capacity = 0;
    canonical_required_topk_capacity = 0;
    canonical_required_random_capacity = 0;
    canonical_required_generated_capacity = 0;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &parity_required_step_logits_cells,
        &parity_required_generated_capacity,
        &parity_required_random_capacity);
    if (status)
        return status;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &canonical_required_step_logits_cells,
        &canonical_required_history_capacity,
        &canonical_required_stage_logits_capacity,
        &canonical_required_topk_capacity,
        &canonical_required_random_capacity,
        &canonical_required_generated_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (parity_required_step_logits_cells != canonical_required_step_logits_cells)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (parity_required_generated_capacity != canonical_required_generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (parity_required_random_capacity != canonical_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = parity_required_step_logits_cells;
    *out_required_generated_capacity = parity_required_generated_capacity;
    *out_required_random_capacity = parity_required_random_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-850 commit-only diagnostics wrapper over top-k-default parity gate.
//
// Contract:
// - stage selected diagnostics tuple from `...CommitOnlyParity`;
// - enforce immutable generation/capacity snapshots across composed call;
// - publish staged tuple atomically after exact parity has already succeeded.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;

    I64 staged_required_step_logits_cells;
    I64 staged_required_generated_capacity;
    I64 staged_required_random_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    staged_required_step_logits_cells = *out_required_step_logits_cells;
    staged_required_generated_capacity = *out_required_generated_capacity;
    staged_required_random_capacity = *out_required_random_capacity;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &staged_required_step_logits_cells,
        &staged_required_generated_capacity,
        &staged_required_random_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_required_step_logits_cells;
    *out_required_generated_capacity = staged_required_generated_capacity;
    *out_required_random_capacity = staged_required_random_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-850 commit-only diagnostics wrapper over IQ-843 parity gate.
//
// Contract:
// 1) Stage diagnostics tuple through `...Parity` into locals only.
// 2) Revalidate immutable generation snapshots including derived top-k.
// 3) Atomically publish outputs only when full staged parity succeeds.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 staged_required_step_logits_cells;
    I64 staged_required_generated_capacity;
    I64 staged_required_random_capacity;

    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_top_k;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_top_k = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    staged_required_step_logits_cells = *out_required_step_logits_cells;
    staged_required_generated_capacity = *out_required_generated_capacity;
    staged_required_random_capacity = *out_required_random_capacity;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParity(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &staged_required_step_logits_cells,
        &staged_required_generated_capacity,
        &staged_required_random_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_top_k != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_required_step_logits_cells;
    *out_required_generated_capacity = staged_required_generated_capacity;
    *out_required_random_capacity = staged_required_random_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-851 strict parity gate composing IQ-850 and IQ-849 variants.
//
// Contract:
// 1) Stage diagnostics through commit-only parity wrapper into locals.
// 2) Stage diagnostics through preflight-only parity wrapper into locals.
// 3) Revalidate immutable generation/capacity snapshots.
// 4) Publish diagnostics only when staged tuples are exactly equal.
I32 InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnlyParity(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 top_p_q16,
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
    I64 *out_required_step_logits_cells,
    I64 *out_required_generated_capacity,
    I64 *out_required_random_capacity)
{
    I64 commit_required_step_logits_cells;
    I64 commit_required_generated_capacity;
    I64 commit_required_random_capacity;

    I64 preflight_required_step_logits_cells;
    I64 preflight_required_generated_capacity;
    I64 preflight_required_random_capacity;

    I64 snapshot_step_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_max_new_tokens;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_top_p_q16;
    I64 snapshot_random_q16_capacity;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_generated_capacity;
    I32 status;

    if (!out_required_step_logits_cells ||
        !out_required_generated_capacity ||
        !out_required_random_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_generated_capacity ||
        out_required_step_logits_cells == out_required_random_capacity ||
        out_required_generated_capacity == out_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !out_generated_tokens)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_step_logits_capacity = step_logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_max_new_tokens = max_new_tokens;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_random_q16_capacity = random_q16_capacity;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_generated_capacity = generated_capacity;

    commit_required_step_logits_cells = *out_required_step_logits_cells;
    commit_required_generated_capacity = *out_required_generated_capacity;
    commit_required_random_capacity = *out_required_random_capacity;

    preflight_required_step_logits_cells = *out_required_step_logits_cells;
    preflight_required_generated_capacity = *out_required_generated_capacity;
    preflight_required_random_capacity = *out_required_random_capacity;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityCommitOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &commit_required_step_logits_cells,
        &commit_required_generated_capacity,
        &commit_required_random_capacity);
    if (status)
        return status;

    status = InferenceGenerateTokensCheckedTopKDefaultPreflightOnlyCommitOnlyParityPreflightOnly(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        random_q16_capacity,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        out_generated_tokens,
        generated_capacity,
        &preflight_required_step_logits_cells,
        &preflight_required_generated_capacity,
        &preflight_required_random_capacity);
    if (status)
        return status;

    if (snapshot_step_logits_capacity != step_logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_max_new_tokens != max_new_tokens ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_random_q16_capacity != random_q16_capacity ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (commit_required_step_logits_cells != preflight_required_step_logits_cells)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (commit_required_generated_capacity != preflight_required_generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (commit_required_random_capacity != preflight_required_random_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = commit_required_step_logits_cells;
    *out_required_generated_capacity = commit_required_generated_capacity;
    *out_required_random_capacity = commit_required_random_capacity;
    return SAMPLING_Q16_OK;
}

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

    if (required_step_logits_cells != commit_required_logits)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_generated_capacity != commit_required_tokens)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_history_capacity != commit_final_token_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_logits = commit_required_logits;
    *out_required_tokens = commit_required_tokens;
    *out_final_token_count = commit_final_token_count;
    return SAMPLING_Q16_OK;
}

// WS7 execution wrapper: top-k-default generation with derived capacities.
//
// Fixed sampling controls are provided by
// `InferenceGenerateTokensCheckedTopKDefault`:
// - temperature_q16 = SAMPLING_Q16_ONE
// - top_k = vocab_size
// - repetition_penalty_q16 = SAMPLING_Q16_ONE
//
// Derived capacity contract (checked integer math only):
// - step_logits_capacity = vocab_size * max_new_tokens
// - token_history_capacity = token_history_count + max_new_tokens
// - workspace stage/top-k capacities = vocab_size
// - random/generated capacities = max_new_tokens
//
// The wrapper preserves exact checked/no-partial semantics by guarding all
// additive/multiplicative derivations against overflow, then delegating to
// canonical top-k-default checked generation.
I32 InferenceGenerateTokensCheckedTopKDefaultDefaultCapacity(
    I64 *step_logits_q16,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_count,
    I64 top_p_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 *workspace_topk_logits_q16,
    I64 *workspace_topk_indices,
    I64 *out_generated_tokens,
    I64 *out_generated_count)
{
    I64 step_logits_capacity;
    I64 token_history_capacity;

    if (vocab_size < 0 || max_new_tokens < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    token_history_capacity = token_history_count + max_new_tokens;
    if (token_history_capacity < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;

    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        step_logits_capacity = vocab_size * max_new_tokens;
    }
    else
    {
        step_logits_capacity = 0;
    }

    return InferenceGenerateTokensCheckedTopKDefault(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        top_p_q16,
        random_q16_values,
        workspace_stage_logits_q16,
        vocab_size,
        workspace_topk_logits_q16,
        vocab_size,
        workspace_topk_indices,
        vocab_size,
        out_generated_tokens,
        out_generated_count);
}

// WS7 API sugar wrapper: deterministic greedy generation defaults.
//
// Enforces the canonical greedy sampling configuration:
// - temperature_q16 = 1.0 (no temperature scaling)
// - top_k = 1 (single most likely token)
// - top_p_q16 = 1.0 (no nucleus truncation)
//
// Random lanes remain required by the shared checked contract, but with
// `top_k=1` they do not affect token choice. The wrapper preserves exact
// status/no-partial semantics of `InferenceGenerateTokensChecked`.
I32 InferenceGenerateTokensCheckedGreedyDefault(
    I64 *step_logits_q16,
    I64 step_logits_capacity,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_capacity,
    I64 token_history_count,
    I64 repetition_penalty_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 workspace_stage_logits_capacity,
    I64 *workspace_topk_logits_q16,
    I64 workspace_topk_logits_capacity,
    I64 *workspace_topk_indices,
    I64 workspace_topk_index_capacity,
    I64 *out_generated_tokens,
    I64 *out_generated_count)
{
    return InferenceGenerateTokensChecked(step_logits_q16,
                                          step_logits_capacity,
                                          vocab_size,
                                          max_new_tokens,
                                          token_history,
                                          token_history_capacity,
                                          token_history_count,
                                          SAMPLING_Q16_ONE,
                                          1,
                                          SAMPLING_Q16_ONE,
                                          repetition_penalty_q16,
                                          random_q16_values,
                                          workspace_stage_logits_q16,
                                          workspace_stage_logits_capacity,
                                          workspace_topk_logits_q16,
                                          workspace_topk_logits_capacity,
                                          workspace_topk_indices,
                                          workspace_topk_index_capacity,
                                          out_generated_tokens,
                                          out_generated_count);
}

// WS7 execution wrapper: greedy default-top-k with fully derived capacities.
//
// Derived capacity contract (checked integer math only):
// - step_logits_capacity = vocab_size * max_new_tokens
// - token_history_capacity = token_history_count + max_new_tokens
// - workspace stage/top-k capacities = vocab_size
// - random/generated capacities = max_new_tokens
//
// The wrapper preserves explicit composition behavior by computing all derived
// capacities locally with overflow guards, then delegating to canonical greedy
// checked generation.
I32 InferenceGenerateTokensCheckedGreedyDefaultTopKDefaultCapacity(
    I64 *step_logits_q16,
    I64 vocab_size,
    I64 max_new_tokens,
    I64 *token_history,
    I64 token_history_count,
    I64 repetition_penalty_q16,
    I64 *random_q16_values,
    I64 *workspace_stage_logits_q16,
    I64 *workspace_topk_logits_q16,
    I64 *workspace_topk_indices,
    I64 *out_generated_tokens,
    I64 *out_generated_count)
{
    I64 step_logits_capacity;
    I64 token_history_capacity;

    if (vocab_size < 0 || max_new_tokens < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    token_history_capacity = token_history_count + max_new_tokens;
    if (token_history_capacity < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;

    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        step_logits_capacity = vocab_size * max_new_tokens;
    }
    else
    {
        step_logits_capacity = 0;
    }

    return InferenceGenerateTokensCheckedGreedyDefault(
        step_logits_q16,
        step_logits_capacity,
        vocab_size,
        max_new_tokens,
        token_history,
        token_history_capacity,
        token_history_count,
        repetition_penalty_q16,
        random_q16_values,
        workspace_stage_logits_q16,
        vocab_size,
        workspace_topk_logits_q16,
        vocab_size,
        workspace_topk_indices,
        vocab_size,
        out_generated_tokens,
        out_generated_count);
}

// WS7 preflight wrapper for deterministic greedy-default generation.
//
// Mirrors `InferenceGenerateTokensCheckedGreedyDefault` fixed sampling knobs:
// - temperature_q16 = SAMPLING_Q16_ONE
// - top_k = 1
// - top_p_q16 = SAMPLING_Q16_ONE
//
// Capacity diagnostics are identical to the base checked preflight contract.
// This wrapper keeps exact BAD_PARAM/OVERFLOW classification and strict
// no-partial diagnostic writes by forwarding to canonical preflight.
I32 InferenceGenerateTokensPreflightCheckedGreedyDefault(
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
    I64 *out_required_generated_capacity)
{
    // Greedy defaults are fixed by the generation wrapper:
    // temperature_q16 = SAMPLING_Q16_ONE, top_k = 1, top_p_q16 = SAMPLING_Q16_ONE.
    // Preflight capacity math is sampling-parameter invariant, so this helper
    // delegates to the canonical checked preflight to preserve exact error
    // classification and strict no-partial diagnostic writes.
    return InferenceGenerateTokensPreflightChecked(
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
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 preflight wrapper for greedy top-k diagnostics API sugar.
//
// Mirrors greedy generation defaults where top-k is fixed to one candidate.
// Preflight capacity math is independent of sampling knobs, so this wrapper
// forwards to canonical checked preflight to preserve exact status surfaces
// and strict no-partial diagnostics writes.
I32 InferenceGenerateTokensPreflightCheckedGreedyDefaultTopK(
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
    I64 *out_required_generated_capacity)
{
    return InferenceGenerateTokensPreflightChecked(
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
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 diagnostics wrapper for greedy-top-k + default capacities.
//
// Derived capacity contract mirrors `InferenceGenerateTokensChecked`:
// - step logits cells        = vocab_size * max_new_tokens
// - token history capacity   = token_history_count + max_new_tokens
// - stage/top-k capacities   = vocab_size
// - random/generated buffers = max_new_tokens
//
// All size derivations are checked before dispatch so additive/multiplicative
// overflow remains deterministically classified as OVERFLOW.
I32 InferenceGenerateTokensPreflightCheckedGreedyDefaultTopKDefaultCapacity(
    I64 vocab_size,
    I64 max_new_tokens,
    I64 token_history_count,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity,
    I64 *out_required_random_capacity,
    I64 *out_required_generated_capacity)
{
    I64 derived_step_logits_capacity;
    I64 derived_token_history_capacity;
    I64 derived_workspace_stage_logits_capacity;
    I64 derived_workspace_topk_logits_capacity;
    I64 derived_workspace_topk_index_capacity;
    I64 derived_random_q16_capacity;
    I64 derived_generated_capacity;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity ||
        !out_required_random_capacity || !out_required_generated_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (vocab_size < 0 || max_new_tokens < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // history growth = existing tokens + generated tokens
    derived_token_history_capacity = token_history_count + max_new_tokens;
    if (derived_token_history_capacity < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;

    // step-logits workspace is a dense [max_new_tokens, vocab_size] slab.
    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        derived_step_logits_capacity = vocab_size * max_new_tokens;
    }
    else
    {
        derived_step_logits_capacity = 0;
    }

    derived_workspace_stage_logits_capacity = vocab_size;
    derived_workspace_topk_logits_capacity = vocab_size;
    derived_workspace_topk_index_capacity = vocab_size;
    derived_random_q16_capacity = max_new_tokens;
    derived_generated_capacity = max_new_tokens;

    return InferenceGenerateTokensPreflightCheckedGreedyDefaultTopK(
        derived_step_logits_capacity,
        vocab_size,
        max_new_tokens,
        derived_token_history_capacity,
        token_history_count,
        derived_workspace_stage_logits_capacity,
        derived_workspace_topk_logits_capacity,
        derived_workspace_topk_index_capacity,
        derived_random_q16_capacity,
        derived_generated_capacity,
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 preflight wrapper for the default-top-k generation API.
//
// `InferenceGenerateTokensCheckedDefaultTopK` fixes sampling top-k to
// `vocab_size`. Capacity diagnostics are otherwise identical to the base
// checked preflight contract, so this helper forwards inputs/outputs directly
// to `InferenceGenerateTokensPreflightChecked` to preserve exact status
// classification and no-partial output semantics.
I32 InferenceGenerateTokensPreflightCheckedDefaultTopK(
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
    I64 *out_required_generated_capacity)
{
    return InferenceGenerateTokensPreflightChecked(
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
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 preflight wrapper for the default-top-p generation API.
//
// `InferenceGenerateTokensCheckedDefaultTopP` fixes sampling top-p to
// `SAMPLING_Q16_ONE`. Capacity diagnostics are sampling-parameter invariant,
// so this helper forwards to canonical preflight and preserves exact
// BAD_PARAM/OVERFLOW classification with strict no-partial output writes.
I32 InferenceGenerateTokensPreflightCheckedDefaultTopP(
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
    I64 *out_required_generated_capacity)
{
    return InferenceGenerateTokensPreflightChecked(
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
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 preflight wrapper for deterministic top-k-default generation API.
//
// Runtime top-k-default generation fixes:
// - temperature_q16 = SAMPLING_Q16_ONE
// - repetition_penalty_q16 = SAMPLING_Q16_ONE
//
// Capacity diagnostics are independent of those sampling constants, so this
// helper forwards directly to canonical checked preflight to preserve exact
// BAD_PARAM/OVERFLOW surfaces and strict no-partial output semantics.
I32 InferenceGenerateTokensPreflightCheckedTopKDefault(
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
    I64 *out_required_generated_capacity)
{
    return InferenceGenerateTokensPreflightChecked(
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
        out_required_step_logits_cells,
        out_required_history_capacity,
        out_required_stage_logits_capacity,
        out_required_topk_capacity,
        out_required_random_capacity,
        out_required_generated_capacity);
}

// WS7 generation preflight diagnostics for default-capacity API wrappers.
//
// Exposes required capacities for `InferenceGenerateTokensChecked` contracts
// without touching caller generation/history buffers:
// - required_step_logits_cells: vocab_size * max_new_tokens
// - required_history_capacity: token_history_count + max_new_tokens
// - required_stage_logits_capacity: vocab_size
// - required_topk_capacity: vocab_size (logits + indices)
// - required_random_capacity: max_new_tokens
// - required_generated_capacity: max_new_tokens
//
// Failure classification is deterministic:
// - arithmetic overflow/underflow in checked size math => OVERFLOW
// - shape/capacity contract violations => BAD_PARAM
//
// No-partial diagnostics: outputs are written only after all checks pass.
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
    I64 *out_required_generated_capacity)
{
    I64 required_step_logits_cells;
    I64 required_history_capacity;
    I64 required_stage_logits_capacity;
    I64 required_topk_capacity;
    I64 required_random_capacity;
    I64 required_generated_capacity;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity ||
        !out_required_random_capacity || !out_required_generated_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (step_logits_capacity < 0 || vocab_size < 0 || max_new_tokens < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_capacity < 0 || token_history_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (workspace_stage_logits_capacity < 0 || workspace_topk_logits_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (workspace_topk_index_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (random_q16_capacity < 0 || generated_capacity < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (vocab_size <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (token_history_count > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    required_history_capacity = token_history_count + max_new_tokens;
    if (required_history_capacity < token_history_count)
        return SAMPLING_Q16_ERR_OVERFLOW;

    required_stage_logits_capacity = vocab_size;
    required_topk_capacity = vocab_size;
    required_random_capacity = max_new_tokens;
    required_generated_capacity = max_new_tokens;

    if (max_new_tokens)
    {
        if (vocab_size > 0x7FFFFFFFFFFFFFFF / max_new_tokens)
            return SAMPLING_Q16_ERR_OVERFLOW;
        required_step_logits_cells = vocab_size * max_new_tokens;
    }
    else
    {
        required_step_logits_cells = 0;
    }

    if (required_history_capacity > token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_stage_logits_capacity > workspace_stage_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_topk_capacity > workspace_topk_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_topk_capacity > workspace_topk_index_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_random_capacity > random_q16_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_generated_capacity > generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_step_logits_cells > step_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = required_step_logits_cells;
    *out_required_history_capacity = required_history_capacity;
    *out_required_stage_logits_capacity = required_stage_logits_capacity;
    *out_required_topk_capacity = required_topk_capacity;
    *out_required_random_capacity = required_random_capacity;
    *out_required_generated_capacity = required_generated_capacity;
    return SAMPLING_Q16_OK;
}
