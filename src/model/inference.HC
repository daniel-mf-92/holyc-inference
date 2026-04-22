// inference.HC
// Integer-only inference generation diagnostics wrappers for TempleOS runtime.

#define SAMPLING_Q16_OK            0
#define SAMPLING_Q16_ERR_NULL_PTR  1
#define SAMPLING_Q16_ERR_BAD_PARAM 2
#define SAMPLING_Q16_ERR_OVERFLOW  4

#define SAMPLING_Q16_SHIFT 16
#define SAMPLING_Q16_ONE   (1 << SAMPLING_Q16_SHIFT)

#define DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE "general.architecture"
#define DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN 20

#define DISPATCH_ARCH_Q16_ID_LLAMA   1
#define DISPATCH_ARCH_Q16_ID_MISTRAL 2
#define DISPATCH_ARCH_Q16_ID_QWEN2   3
#define DISPATCH_ARCH_Q16_ID_PHI3    4

#define DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT 4

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
    I64 lane_count);

I32 InferenceDispatchTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return SAMPLING_Q16_OK;
    }

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > (0x7FFFFFFFFFFFFFFF / rhs))
                return SAMPLING_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (rhs < (-0x8000000000000000 / lhs))
                return SAMPLING_Q16_ERR_OVERFLOW;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < (-0x8000000000000000 / rhs))
                return SAMPLING_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (lhs != 0 && rhs < (0x7FFFFFFFFFFFFFFF / lhs))
                return SAMPLING_Q16_ERR_OVERFLOW;
        }
    }

    *out_prod = lhs * rhs;
    return SAMPLING_Q16_OK;
}

I32 InferenceDispatchByteSpanEquals(U8 *lhs,
                                    I64 lhs_len,
                                    U8 *rhs,
                                    I64 rhs_len)
{
    I64 index;

    if (!lhs || !rhs)
        return FALSE;
    if (lhs_len < 0 || rhs_len < 0)
        return FALSE;
    if (lhs_len != rhs_len)
        return FALSE;

    for (index = 0; index < lhs_len; index++)
    {
        if (lhs[index] != rhs[index])
            return FALSE;
    }

    return TRUE;
}

I32 InferenceDispatchParseArchitectureIdChecked(U8 *meta_key_bytes,
                                                I64 meta_key_len,
                                                U8 *arch_value_bytes,
                                                I64 arch_value_len,
                                                I64 *out_arch_id)
{
    U8 *canonical_key;

    if (!meta_key_bytes || !arch_value_bytes || !out_arch_id)
        return SAMPLING_Q16_ERR_NULL_PTR;
    if (meta_key_len <= 0 || arch_value_len <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    canonical_key = (U8 *)DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE;
    if (meta_key_len != DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!InferenceDispatchByteSpanEquals(meta_key_bytes,
                                         meta_key_len,
                                         canonical_key,
                                         DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN))
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (InferenceDispatchByteSpanEquals(arch_value_bytes,
                                        arch_value_len,
                                        (U8 *)"llama",
                                        5))
    {
        *out_arch_id = DISPATCH_ARCH_Q16_ID_LLAMA;
        return SAMPLING_Q16_OK;
    }

    if (InferenceDispatchByteSpanEquals(arch_value_bytes,
                                        arch_value_len,
                                        (U8 *)"mistral",
                                        7))
    {
        *out_arch_id = DISPATCH_ARCH_Q16_ID_MISTRAL;
        return SAMPLING_Q16_OK;
    }

    // GGUF models usually publish `qwen2`; accept `qwen` for compatibility
    // with hand-authored metadata while staying integer-only and explicit.
    if (InferenceDispatchByteSpanEquals(arch_value_bytes,
                                        arch_value_len,
                                        (U8 *)"qwen2",
                                        5) ||
        InferenceDispatchByteSpanEquals(arch_value_bytes,
                                        arch_value_len,
                                        (U8 *)"qwen",
                                        4))
    {
        *out_arch_id = DISPATCH_ARCH_Q16_ID_QWEN2;
        return SAMPLING_Q16_OK;
    }

    if (InferenceDispatchByteSpanEquals(arch_value_bytes,
                                        arch_value_len,
                                        (U8 *)"phi3",
                                        4))
    {
        *out_arch_id = DISPATCH_ARCH_Q16_ID_PHI3;
        return SAMPLING_Q16_OK;
    }

    return SAMPLING_Q16_ERR_BAD_PARAM;
}

// IQ-1098 explicit architecture registry initialization for WS10.
//
// Contract:
// - Materialize auditable, deterministic dispatch slots:
//   {"llama","mistral","qwen2","phi3"}.
// - Stage all outputs privately, then atomically publish on success.
// - Revalidate immutable slot-count/output-pointer snapshots before publish.
// - Reject duplicate tags, duplicate arch IDs, or unsupported tag->ID pairs.
I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartial(
    U8 **out_registry_tags,
    I64 *out_registry_tag_lens,
    I64 *out_registry_arch_ids,
    I64 registry_slot_capacity,
    I64 *out_registry_count,
    I64 *out_default_arch_id)
{
    U8 *staged_tags[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 staged_tag_lens[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 staged_arch_ids[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 staged_registry_count;
    I64 staged_default_arch_id;

    U8 *canonical_key;
    I64 canonical_key_len;

    I64 slot_index;
    I64 compare_index;
    I64 parsed_arch_id;
    I64 snapshot_registry_slot_capacity;
    U8 **snapshot_out_registry_tags;
    I64 *snapshot_out_registry_tag_lens;
    I64 *snapshot_out_registry_arch_ids;
    I64 *snapshot_out_registry_count;
    I64 *snapshot_out_default_arch_id;
    I32 status;

    if (!out_registry_tags || !out_registry_tag_lens || !out_registry_arch_ids ||
        !out_registry_count || !out_default_arch_id)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (out_registry_count == out_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_registry_slot_capacity = registry_slot_capacity;
    snapshot_out_registry_tags = out_registry_tags;
    snapshot_out_registry_tag_lens = out_registry_tag_lens;
    snapshot_out_registry_arch_ids = out_registry_arch_ids;
    snapshot_out_registry_count = out_registry_count;
    snapshot_out_default_arch_id = out_default_arch_id;

    staged_tags[0] = (U8 *)"llama";
    staged_tags[1] = (U8 *)"mistral";
    staged_tags[2] = (U8 *)"qwen2";
    staged_tags[3] = (U8 *)"phi3";

    staged_tag_lens[0] = 5;
    staged_tag_lens[1] = 7;
    staged_tag_lens[2] = 5;
    staged_tag_lens[3] = 4;

    staged_arch_ids[0] = DISPATCH_ARCH_Q16_ID_LLAMA;
    staged_arch_ids[1] = DISPATCH_ARCH_Q16_ID_MISTRAL;
    staged_arch_ids[2] = DISPATCH_ARCH_Q16_ID_QWEN2;
    staged_arch_ids[3] = DISPATCH_ARCH_Q16_ID_PHI3;

    staged_registry_count = DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
    staged_default_arch_id = DISPATCH_ARCH_Q16_ID_LLAMA;

    canonical_key = (U8 *)DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE;
    canonical_key_len = DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN;

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        if (!staged_tags[slot_index] || staged_tag_lens[slot_index] <= 0)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        status = InferenceDispatchParseArchitectureIdChecked(
            canonical_key,
            canonical_key_len,
            staged_tags[slot_index],
            staged_tag_lens[slot_index],
            &parsed_arch_id);
        if (status)
            return status;

        if (parsed_arch_id != staged_arch_ids[slot_index])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        for (compare_index = slot_index + 1;
             compare_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
             compare_index++)
        {
            if (staged_arch_ids[slot_index] == staged_arch_ids[compare_index])
                return SAMPLING_Q16_ERR_BAD_PARAM;

            if (InferenceDispatchByteSpanEquals(staged_tags[slot_index],
                                                staged_tag_lens[slot_index],
                                                staged_tags[compare_index],
                                                staged_tag_lens[compare_index]))
                return SAMPLING_Q16_ERR_BAD_PARAM;
        }
    }

    if (snapshot_registry_slot_capacity != registry_slot_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_out_registry_tags != out_registry_tags ||
        snapshot_out_registry_tag_lens != out_registry_tag_lens ||
        snapshot_out_registry_arch_ids != out_registry_arch_ids ||
        snapshot_out_registry_count != out_registry_count ||
        snapshot_out_default_arch_id != out_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        out_registry_tags[slot_index] = staged_tags[slot_index];
        out_registry_tag_lens[slot_index] = staged_tag_lens[slot_index];
        out_registry_arch_ids[slot_index] = staged_arch_ids[slot_index];
    }

    *out_registry_count = staged_registry_count;
    *out_default_arch_id = staged_default_arch_id;
    return SAMPLING_Q16_OK;
}

// IQ-1094 explicit architecture dispatch for HolyC integer-only forward path.
//
// Dispatch contract:
// - Parse GGUF metadata key/value bytes for `general.architecture`.
// - Route through explicit dispatch table (`llama`, `mistral`, `qwen2`, `phi3`).
// - Stage diagnostics `{rows_processed,logits_written}` and publish atomically.
// - Enforce no-partial behavior: output diagnostics stay unchanged on failure.
I32 InferenceForwardDispatchModelArchQ16CheckedNoPartial(
    U8 *meta_key_bytes,
    I64 meta_key_len,
    U8 *arch_value_bytes,
    I64 arch_value_len,
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
    I64 lane_count,
    I64 *out_rows_processed,
    I64 *out_logits_written)
{
    I64 arch_id;
    I64 staged_rows_processed;
    I64 staged_logits_written;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;
    U8 *snapshot_meta_key_bytes;
    I64 snapshot_meta_key_len;
    U8 *snapshot_arch_value_bytes;
    I64 snapshot_arch_value_len;
    I64 *snapshot_out_rows_processed;
    I64 *snapshot_out_logits_written;
    I32 status;

    if (!meta_key_bytes || !arch_value_bytes ||
        !out_rows_processed || !out_logits_written)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_rows_processed == out_logits_written)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || lane_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_meta_key_bytes = meta_key_bytes;
    snapshot_meta_key_len = meta_key_len;
    snapshot_arch_value_bytes = arch_value_bytes;
    snapshot_arch_value_len = arch_value_len;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_out_rows_processed = out_rows_processed;
    snapshot_out_logits_written = out_logits_written;

    status = InferenceDispatchParseArchitectureIdChecked(meta_key_bytes,
                                                         meta_key_len,
                                                         arch_value_bytes,
                                                         arch_value_len,
                                                         &arch_id);
    if (status)
        return status;

    switch (arch_id)
    {
    case DISPATCH_ARCH_Q16_ID_LLAMA:
        status = InferenceForwardQ16CheckedNoPartial(embed_q16,
                                                     embed_capacity,
                                                     block_attn_q16,
                                                     block_attn_capacity,
                                                     block_ffn_q16,
                                                     block_ffn_capacity,
                                                     block_count,
                                                     gamma_attn_q16,
                                                     gamma_attn_capacity,
                                                     gamma_ffn_q16,
                                                     gamma_ffn_capacity,
                                                     block_eps_q16,
                                                     final_norm_gamma_q16,
                                                     final_norm_gamma_capacity,
                                                     final_norm_eps_q16,
                                                     workspace_q16,
                                                     workspace_capacity,
                                                     block_workspace_q16,
                                                     block_workspace_capacity,
                                                     logits_q16,
                                                     logits_capacity,
                                                     row_count,
                                                     lane_count);
        break;
    case DISPATCH_ARCH_Q16_ID_MISTRAL:
        status = InferenceForwardQ16CheckedNoPartial(embed_q16,
                                                     embed_capacity,
                                                     block_attn_q16,
                                                     block_attn_capacity,
                                                     block_ffn_q16,
                                                     block_ffn_capacity,
                                                     block_count,
                                                     gamma_attn_q16,
                                                     gamma_attn_capacity,
                                                     gamma_ffn_q16,
                                                     gamma_ffn_capacity,
                                                     block_eps_q16,
                                                     final_norm_gamma_q16,
                                                     final_norm_gamma_capacity,
                                                     final_norm_eps_q16,
                                                     workspace_q16,
                                                     workspace_capacity,
                                                     block_workspace_q16,
                                                     block_workspace_capacity,
                                                     logits_q16,
                                                     logits_capacity,
                                                     row_count,
                                                     lane_count);
        break;
    case DISPATCH_ARCH_Q16_ID_QWEN2:
        status = InferenceForwardQ16CheckedNoPartial(embed_q16,
                                                     embed_capacity,
                                                     block_attn_q16,
                                                     block_attn_capacity,
                                                     block_ffn_q16,
                                                     block_ffn_capacity,
                                                     block_count,
                                                     gamma_attn_q16,
                                                     gamma_attn_capacity,
                                                     gamma_ffn_q16,
                                                     gamma_ffn_capacity,
                                                     block_eps_q16,
                                                     final_norm_gamma_q16,
                                                     final_norm_gamma_capacity,
                                                     final_norm_eps_q16,
                                                     workspace_q16,
                                                     workspace_capacity,
                                                     block_workspace_q16,
                                                     block_workspace_capacity,
                                                     logits_q16,
                                                     logits_capacity,
                                                     row_count,
                                                     lane_count);
        break;
    case DISPATCH_ARCH_Q16_ID_PHI3:
        status = InferenceForwardQ16CheckedNoPartial(embed_q16,
                                                     embed_capacity,
                                                     block_attn_q16,
                                                     block_attn_capacity,
                                                     block_ffn_q16,
                                                     block_ffn_capacity,
                                                     block_count,
                                                     gamma_attn_q16,
                                                     gamma_attn_capacity,
                                                     gamma_ffn_q16,
                                                     gamma_ffn_capacity,
                                                     block_eps_q16,
                                                     final_norm_gamma_q16,
                                                     final_norm_gamma_capacity,
                                                     final_norm_eps_q16,
                                                     workspace_q16,
                                                     workspace_capacity,
                                                     block_workspace_q16,
                                                     block_workspace_capacity,
                                                     logits_q16,
                                                     logits_capacity,
                                                     row_count,
                                                     lane_count);
        break;
    default:
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (status)
        return status;

    staged_rows_processed = row_count;
    status = InferenceDispatchTryMulI64Checked(row_count,
                                               lane_count,
                                               &staged_logits_written);
    if (status)
        return status;

    if (snapshot_meta_key_bytes != meta_key_bytes ||
        snapshot_meta_key_len != meta_key_len ||
        snapshot_arch_value_bytes != arch_value_bytes ||
        snapshot_arch_value_len != arch_value_len ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_out_rows_processed != out_rows_processed ||
        snapshot_out_logits_written != out_logits_written)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_rows_processed = staged_rows_processed;
    *out_logits_written = staged_logits_written;
    return SAMPLING_Q16_OK;
}

// IQ-1108 diagnostics-only no-write companion over architecture dispatch.
//
// Compose:
//   1) staged diagnostics tuple from
//      `InferenceForwardDispatchModelArchQ16CheckedNoPartial`.
//   2) canonical diagnostics tuple recomputed from immutable metadata/geometry
//      snapshots: `{rows_processed=row_count, logits_written=row_count*lane_count}`.
//
// Publish contract:
//   - outputs are committed only when staged and canonical tuples match exactly;
//   - immutable snapshots for metadata bytes/lengths, geometry, and output
//     pointer identities must remain unchanged before publish;
//   - no partial writes on every failure path.
I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly(
    U8 *meta_key_bytes,
    I64 meta_key_len,
    U8 *arch_value_bytes,
    I64 arch_value_len,
    I64 row_count,
    I64 lane_count,
    I64 *out_rows_processed,
    I64 *out_logits_written)
{
    I64 staged_arch_id;
    I64 staged_rows_processed;
    I64 staged_logits_written;

    I64 canonical_arch_id;
    I64 canonical_rows_processed;
    I64 canonical_logits_written;

    U8 *snapshot_meta_key_bytes;
    I64 snapshot_meta_key_len;
    U8 *snapshot_arch_value_bytes;
    I64 snapshot_arch_value_len;
    I64 snapshot_row_count;
    I64 snapshot_lane_count;

    I64 *snapshot_out_rows_processed;
    I64 *snapshot_out_logits_written;
    I32 status;

    if (!meta_key_bytes || !arch_value_bytes ||
        !out_rows_processed || !out_logits_written)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_rows_processed == out_logits_written)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (row_count < 0 || lane_count < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_meta_key_bytes = meta_key_bytes;
    snapshot_meta_key_len = meta_key_len;
    snapshot_arch_value_bytes = arch_value_bytes;
    snapshot_arch_value_len = arch_value_len;
    snapshot_row_count = row_count;
    snapshot_lane_count = lane_count;
    snapshot_out_rows_processed = out_rows_processed;
    snapshot_out_logits_written = out_logits_written;

    status = InferenceDispatchParseArchitectureIdChecked(meta_key_bytes,
                                                         meta_key_len,
                                                         arch_value_bytes,
                                                         arch_value_len,
                                                         &staged_arch_id);
    if (status)
        return status;

    switch (staged_arch_id)
    {
    case DISPATCH_ARCH_Q16_ID_LLAMA:
    case DISPATCH_ARCH_Q16_ID_MISTRAL:
    case DISPATCH_ARCH_Q16_ID_QWEN2:
    case DISPATCH_ARCH_Q16_ID_PHI3:
        break;
    default:
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    staged_rows_processed = row_count;
    status = InferenceDispatchTryMulI64Checked(row_count,
                                               lane_count,
                                               &staged_logits_written);
    if (status)
        return status;

    status = InferenceDispatchParseArchitectureIdChecked(snapshot_meta_key_bytes,
                                                         snapshot_meta_key_len,
                                                         snapshot_arch_value_bytes,
                                                         snapshot_arch_value_len,
                                                         &canonical_arch_id);
    if (status)
        return status;

    switch (canonical_arch_id)
    {
    case DISPATCH_ARCH_Q16_ID_LLAMA:
    case DISPATCH_ARCH_Q16_ID_MISTRAL:
    case DISPATCH_ARCH_Q16_ID_QWEN2:
    case DISPATCH_ARCH_Q16_ID_PHI3:
        break;
    default:
        return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    canonical_rows_processed = snapshot_row_count;
    status = InferenceDispatchTryMulI64Checked(snapshot_row_count,
                                               snapshot_lane_count,
                                               &canonical_logits_written);
    if (status)
        return status;

    if (snapshot_meta_key_bytes != meta_key_bytes ||
        snapshot_meta_key_len != meta_key_len ||
        snapshot_arch_value_bytes != arch_value_bytes ||
        snapshot_arch_value_len != arch_value_len ||
        snapshot_row_count != row_count ||
        snapshot_lane_count != lane_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_out_rows_processed != out_rows_processed ||
        snapshot_out_logits_written != out_logits_written)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_rows_processed != canonical_rows_processed ||
        staged_logits_written != canonical_logits_written)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_rows_processed = staged_rows_processed;
    *out_logits_written = staged_logits_written;
    return SAMPLING_Q16_OK;
}


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
