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

#define INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION    1
#define INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS 5
#define INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE      0

#define INFERENCE_BOT_EVENT_TUPLE_CELLS 6
#define INFERENCE_BOT_PROFILE_SECURE 1
#define INFERENCE_BOT_PROFILE_DEV 2
#define INFERENCE_BOT_STATUS_BLOCKED 0
#define INFERENCE_BOT_STATUS_EMITTED 1
#define INFERENCE_BOT_DIGEST_OFFSET 1469598103934665603
#define INFERENCE_BOT_DIGEST_PRIME 1099511628211

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
                          I64 *out_token_id);

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

// IQ-1115 diagnostics-only preflight companion for registry initialization.
//
// Contract:
// - Stage full registry outputs privately through canonical initializer.
// - Recompute canonical diagnostics tuple `{registry_count, default_arch_id}`
//   from immutable `registry_slot_capacity` snapshot and fixed architecture set.
// - Revalidate immutable output-pointer identity before publish.
// - Publish staged registry table atomically only on exact parity success.
I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly(
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

    U8 *canonical_tags[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_tag_lens[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_arch_ids[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_registry_count;
    I64 canonical_default_arch_id;

    I64 snapshot_registry_slot_capacity;
    U8 **snapshot_out_registry_tags;
    I64 *snapshot_out_registry_tag_lens;
    I64 *snapshot_out_registry_arch_ids;
    I64 *snapshot_out_registry_count;
    I64 *snapshot_out_default_arch_id;

    U8 *canonical_key;
    I64 canonical_key_len;

    I64 slot_index;
    I64 compare_index;
    I64 staged_parsed_arch_id;
    I64 canonical_parsed_arch_id;
    I64 staged_default_matches;
    I64 canonical_default_matches;
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

    canonical_key = (U8 *)DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE;
    canonical_key_len = DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN;
    staged_default_matches = 0;
    canonical_default_matches = 0;

    status = InferenceForwardDispatchRegistryInitQ16CheckedNoPartial(
        staged_tags,
        staged_tag_lens,
        staged_arch_ids,
        registry_slot_capacity,
        &staged_registry_count,
        &staged_default_arch_id);
    if (status)
        return status;

    canonical_tags[0] = (U8 *)"llama";
    canonical_tags[1] = (U8 *)"mistral";
    canonical_tags[2] = (U8 *)"qwen2";
    canonical_tags[3] = (U8 *)"phi3";

    canonical_tag_lens[0] = 5;
    canonical_tag_lens[1] = 7;
    canonical_tag_lens[2] = 5;
    canonical_tag_lens[3] = 4;

    canonical_arch_ids[0] = DISPATCH_ARCH_Q16_ID_LLAMA;
    canonical_arch_ids[1] = DISPATCH_ARCH_Q16_ID_MISTRAL;
    canonical_arch_ids[2] = DISPATCH_ARCH_Q16_ID_QWEN2;
    canonical_arch_ids[3] = DISPATCH_ARCH_Q16_ID_PHI3;

    canonical_registry_count = DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
    canonical_default_arch_id = DISPATCH_ARCH_Q16_ID_LLAMA;

    if (snapshot_registry_slot_capacity != registry_slot_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_registry_slot_capacity < canonical_registry_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_out_registry_tags != out_registry_tags ||
        snapshot_out_registry_tag_lens != out_registry_tag_lens ||
        snapshot_out_registry_arch_ids != out_registry_arch_ids ||
        snapshot_out_registry_count != out_registry_count ||
        snapshot_out_default_arch_id != out_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_registry_count != canonical_registry_count ||
        staged_default_arch_id != canonical_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Keep tuple contract fully explicit and auditable: both staged and
    // canonical paths must resolve to the exact fixed registry cardinality and
    // default architecture ID.
    if (staged_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT ||
        canonical_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA ||
        canonical_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        if (!staged_tags[slot_index] || staged_tag_lens[slot_index] <= 0)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (staged_tag_lens[slot_index] != canonical_tag_lens[slot_index] ||
            staged_arch_ids[slot_index] != canonical_arch_ids[slot_index])
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (!InferenceDispatchByteSpanEquals(staged_tags[slot_index],
                                             staged_tag_lens[slot_index],
                                             canonical_tags[slot_index],
                                             canonical_tag_lens[slot_index]))
            return SAMPLING_Q16_ERR_BAD_PARAM;

        status = InferenceDispatchParseArchitectureIdChecked(
            canonical_key,
            canonical_key_len,
            staged_tags[slot_index],
            staged_tag_lens[slot_index],
            &staged_parsed_arch_id);
        if (status)
            return status;

        status = InferenceDispatchParseArchitectureIdChecked(
            canonical_key,
            canonical_key_len,
            canonical_tags[slot_index],
            canonical_tag_lens[slot_index],
            &canonical_parsed_arch_id);
        if (status)
            return status;

        if (staged_parsed_arch_id != staged_arch_ids[slot_index] ||
            canonical_parsed_arch_id != canonical_arch_ids[slot_index])
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (staged_arch_ids[slot_index] == staged_default_arch_id)
            staged_default_matches++;
        if (canonical_arch_ids[slot_index] == canonical_default_arch_id)
            canonical_default_matches++;
    }

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        for (compare_index = slot_index + 1;
             compare_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
             compare_index++)
        {
            if (staged_arch_ids[slot_index] == staged_arch_ids[compare_index] ||
                canonical_arch_ids[slot_index] == canonical_arch_ids[compare_index])
                return SAMPLING_Q16_ERR_BAD_PARAM;

            if (InferenceDispatchByteSpanEquals(staged_tags[slot_index],
                                                staged_tag_lens[slot_index],
                                                staged_tags[compare_index],
                                                staged_tag_lens[compare_index]))
                return SAMPLING_Q16_ERR_BAD_PARAM;

            if (InferenceDispatchByteSpanEquals(canonical_tags[slot_index],
                                                canonical_tag_lens[slot_index],
                                                canonical_tags[compare_index],
                                                canonical_tag_lens[compare_index]))
                return SAMPLING_Q16_ERR_BAD_PARAM;
        }
    }

    if (staged_default_matches != 1 || canonical_default_matches != 1)
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

// IQ-1130 diagnostics parity gate over preflight+canonical registry init.
//
// Contract:
// - Stage preflight diagnostics table through
//   `InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly`.
// - Stage canonical diagnostics table through
//   `InferenceForwardDispatchRegistryInitQ16CheckedNoPartial`.
// - Revalidate immutable `{registry_slot_capacity}` and output-pointer snapshots
//   before publish.
// - Enforce exact tuple parity `{registry_count,default_arch_id}` and slot parity
//   `{tag_bytes,tag_len,arch_id}` before atomic publish.
// - No partial writes on every error path.
I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
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

    U8 *canonical_tags[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_tag_lens[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_arch_ids[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_registry_count;
    I64 canonical_default_arch_id;

    I64 snapshot_registry_slot_capacity;
    U8 **snapshot_out_registry_tags;
    I64 *snapshot_out_registry_tag_lens;
    I64 *snapshot_out_registry_arch_ids;
    I64 *snapshot_out_registry_count;
    I64 *snapshot_out_default_arch_id;

    U8 *canonical_key;
    I64 canonical_key_len;

    I64 slot_index;
    I64 compare_index;
    I64 staged_parsed_arch_id;
    I64 canonical_parsed_arch_id;
    I64 staged_default_matches;
    I64 canonical_default_matches;
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

    canonical_key = (U8 *)DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE;
    canonical_key_len = DISPATCH_ARCH_Q16_KEY_GENERAL_ARCHITECTURE_LEN;
    staged_default_matches = 0;
    canonical_default_matches = 0;

    status = InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly(
        staged_tags,
        staged_tag_lens,
        staged_arch_ids,
        registry_slot_capacity,
        &staged_registry_count,
        &staged_default_arch_id);
    if (status)
        return status;

    status = InferenceForwardDispatchRegistryInitQ16CheckedNoPartial(
        canonical_tags,
        canonical_tag_lens,
        canonical_arch_ids,
        registry_slot_capacity,
        &canonical_registry_count,
        &canonical_default_arch_id);
    if (status)
        return status;

    if (snapshot_registry_slot_capacity != registry_slot_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_out_registry_tags != out_registry_tags ||
        snapshot_out_registry_tag_lens != out_registry_tag_lens ||
        snapshot_out_registry_arch_ids != out_registry_arch_ids ||
        snapshot_out_registry_count != out_registry_count ||
        snapshot_out_default_arch_id != out_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_registry_count != canonical_registry_count ||
        staged_default_arch_id != canonical_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT ||
        canonical_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA ||
        canonical_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        if (!staged_tags[slot_index] || !canonical_tags[slot_index] ||
            staged_tag_lens[slot_index] <= 0 || canonical_tag_lens[slot_index] <= 0)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (staged_tag_lens[slot_index] != canonical_tag_lens[slot_index] ||
            staged_arch_ids[slot_index] != canonical_arch_ids[slot_index])
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (!InferenceDispatchByteSpanEquals(staged_tags[slot_index],
                                             staged_tag_lens[slot_index],
                                             canonical_tags[slot_index],
                                             canonical_tag_lens[slot_index]))
            return SAMPLING_Q16_ERR_BAD_PARAM;

        status = InferenceDispatchParseArchitectureIdChecked(
            canonical_key,
            canonical_key_len,
            staged_tags[slot_index],
            staged_tag_lens[slot_index],
            &staged_parsed_arch_id);
        if (status)
            return status;

        status = InferenceDispatchParseArchitectureIdChecked(
            canonical_key,
            canonical_key_len,
            canonical_tags[slot_index],
            canonical_tag_lens[slot_index],
            &canonical_parsed_arch_id);
        if (status)
            return status;

        if (staged_parsed_arch_id != staged_arch_ids[slot_index] ||
            canonical_parsed_arch_id != canonical_arch_ids[slot_index])
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (staged_arch_ids[slot_index] == staged_default_arch_id)
            staged_default_matches++;
        if (canonical_arch_ids[slot_index] == canonical_default_arch_id)
            canonical_default_matches++;
    }

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        for (compare_index = slot_index + 1;
             compare_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
             compare_index++)
        {
            if (staged_arch_ids[slot_index] == staged_arch_ids[compare_index] ||
                canonical_arch_ids[slot_index] == canonical_arch_ids[compare_index])
                return SAMPLING_Q16_ERR_BAD_PARAM;

            if (InferenceDispatchByteSpanEquals(staged_tags[slot_index],
                                                staged_tag_lens[slot_index],
                                                staged_tags[compare_index],
                                                staged_tag_lens[compare_index]))
                return SAMPLING_Q16_ERR_BAD_PARAM;

            if (InferenceDispatchByteSpanEquals(canonical_tags[slot_index],
                                                canonical_tag_lens[slot_index],
                                                canonical_tags[compare_index],
                                                canonical_tag_lens[compare_index]))
                return SAMPLING_Q16_ERR_BAD_PARAM;
        }
    }

    if (staged_default_matches != 1 || canonical_default_matches != 1)
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

// IQ-1129 commit-only diagnostics wrapper for registry-init parity.
//
// Compose:
//   1) stage parity-validated tuple via
//      `InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParity`;
//   2) stage canonical tuple via
//      `InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly`;
//   3) revalidate immutable `{registry_slot_capacity}` and output-pointer
//      snapshots;
//   4) atomically publish only on strict tuple + slot parity success.
I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
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

    U8 *canonical_tags[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_tag_lens[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_arch_ids[DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT];
    I64 canonical_registry_count;
    I64 canonical_default_arch_id;

    I64 snapshot_registry_slot_capacity;
    U8 **snapshot_out_registry_tags;
    I64 *snapshot_out_registry_tag_lens;
    I64 *snapshot_out_registry_arch_ids;
    I64 *snapshot_out_registry_count;
    I64 *snapshot_out_default_arch_id;

    I64 slot_index;
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

    status = InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
        staged_tags,
        staged_tag_lens,
        staged_arch_ids,
        registry_slot_capacity,
        &staged_registry_count,
        &staged_default_arch_id);
    if (status)
        return status;

    status = InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly(
        canonical_tags,
        canonical_tag_lens,
        canonical_arch_ids,
        registry_slot_capacity,
        &canonical_registry_count,
        &canonical_default_arch_id);
    if (status)
        return status;

    if (snapshot_registry_slot_capacity != registry_slot_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_out_registry_tags != out_registry_tags ||
        snapshot_out_registry_tag_lens != out_registry_tag_lens ||
        snapshot_out_registry_arch_ids != out_registry_arch_ids ||
        snapshot_out_registry_count != out_registry_count ||
        snapshot_out_default_arch_id != out_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_registry_count != canonical_registry_count ||
        staged_default_arch_id != canonical_default_arch_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT ||
        canonical_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA ||
        canonical_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (slot_index = 0;
         slot_index < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT;
         slot_index++)
    {
        if (!staged_tags[slot_index] || !canonical_tags[slot_index] ||
            staged_tag_lens[slot_index] <= 0 || canonical_tag_lens[slot_index] <= 0)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (staged_tag_lens[slot_index] != canonical_tag_lens[slot_index] ||
            staged_arch_ids[slot_index] != canonical_arch_ids[slot_index])
            return SAMPLING_Q16_ERR_BAD_PARAM;

        if (!InferenceDispatchByteSpanEquals(staged_tags[slot_index],
                                             staged_tag_lens[slot_index],
                                             canonical_tags[slot_index],
                                             canonical_tag_lens[slot_index]))
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

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

// IQ-1111 diagnostics parity gate for architecture dispatch tuple publication.
//
// Contract:
// - Compose staged diagnostics from
//   `InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly`.
// - Recompute canonical tuple using architecture parse + checked multiply.
// - Publish `{rows_processed, logits_written}` only after strict immutable
//   snapshot revalidation and exact tuple parity.
I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
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

    status = InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly(
        meta_key_bytes,
        meta_key_len,
        arch_value_bytes,
        arch_value_len,
        row_count,
        lane_count,
        &staged_rows_processed,
        &staged_logits_written);
    if (status)
        return status;

    if (staged_rows_processed < 0 || staged_logits_written < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

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

    if (staged_arch_id != canonical_arch_id ||
        staged_rows_processed != canonical_rows_processed ||
        staged_logits_written != canonical_logits_written)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_rows_processed = staged_rows_processed;
    *out_logits_written = staged_logits_written;
    return SAMPLING_Q16_OK;
}

// IQ-1113 commit-only diagnostics wrapper for architecture dispatch parity.
//
// Compose:
//   1) stage tuple via
//      `InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParity`;
//   2) stage canonical tuple via
//      `InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly`;
//   3) revalidate immutable metadata/geometry/output-pointer snapshots;
//   4) atomically publish only on strict tuple parity success.
I32 InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    U8 *meta_key_bytes,
    I64 meta_key_len,
    U8 *arch_value_bytes,
    I64 arch_value_len,
    I64 row_count,
    I64 lane_count,
    I64 *out_rows_processed,
    I64 *out_logits_written)
{
    I64 staged_rows_processed;
    I64 staged_logits_written;

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

    staged_rows_processed = 0;
    staged_logits_written = 0;
    canonical_rows_processed = 0;
    canonical_logits_written = 0;

    status = InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(
        meta_key_bytes,
        meta_key_len,
        arch_value_bytes,
        arch_value_len,
        row_count,
        lane_count,
        &staged_rows_processed,
        &staged_logits_written);
    if (status)
        return status;

    status = InferenceForwardDispatchModelArchQ16CheckedNoPartialCommitOnlyPreflightOnly(
        meta_key_bytes,
        meta_key_len,
        arch_value_bytes,
        arch_value_len,
        row_count,
        lane_count,
        &canonical_rows_processed,
        &canonical_logits_written);
    if (status)
        return status;

    if (staged_rows_processed < 0 || staged_logits_written < 0 ||
        canonical_rows_processed < 0 || canonical_logits_written < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

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

// IQ-1170 checked per-token callback dispatcher for Book-of-Truth token stream.
//
// Callback ABI tuple (all I64, integer-only):
//   tuple[0] = ABI version (must equal INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION)
//   tuple[1] = callback flags (currently only 0 is accepted)
//   tuple[2] = token index in generated stream
//   tuple[3] = token id
//   tuple[4] = token probability/logit in Q16 fixed-point domain
//
// Fail-closed semantics:
// - Any precondition failure returns an error and leaves caller outputs untouched.
// - Callback return code must be SAMPLING_Q16_OK; any non-zero is rejected.
// - ABI tuple is staged locally and only committed to caller buffer on success.
I32 InferenceTokenStreamCallbackDispatchCheckedNoPartial(
    I32 (*token_callback)(I64 *abi_tuple,
                          I64 abi_tuple_cells,
                          I64 user_ctx),
    I64 *abi_tuple_buffer,
    I64 abi_tuple_buffer_capacity,
    I64 token_index,
    I64 token_id,
    I64 token_prob_q16,
    I64 callback_flags,
    I64 user_ctx,
    I64 *out_dispatch_count,
    I64 *out_callback_status)
{
    I64 staged_tuple[INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS];
    I64 staged_dispatch_count;
    I64 staged_callback_status;
    I64 snapshot_token_index;
    I64 snapshot_token_id;
    I64 snapshot_token_prob_q16;
    I64 snapshot_callback_flags;
    I64 snapshot_user_ctx;
    I64 *snapshot_abi_tuple_buffer;
    I64 snapshot_abi_tuple_buffer_capacity;
    I64 *snapshot_out_dispatch_count;
    I64 *snapshot_out_callback_status;
    I32 callback_status;
    I64 lane;

    if (!token_callback || !abi_tuple_buffer ||
        !out_dispatch_count || !out_callback_status)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_dispatch_count == out_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (token_index < 0 || token_id < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_token_index = token_index;
    snapshot_token_id = token_id;
    snapshot_token_prob_q16 = token_prob_q16;
    snapshot_callback_flags = callback_flags;
    snapshot_user_ctx = user_ctx;
    snapshot_abi_tuple_buffer = abi_tuple_buffer;
    snapshot_abi_tuple_buffer_capacity = abi_tuple_buffer_capacity;
    snapshot_out_dispatch_count = out_dispatch_count;
    snapshot_out_callback_status = out_callback_status;

    staged_tuple[0] = INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION;
    staged_tuple[1] = callback_flags;
    staged_tuple[2] = token_index;
    staged_tuple[3] = token_id;
    staged_tuple[4] = token_prob_q16;

    if (snapshot_token_index != token_index ||
        snapshot_token_id != token_id ||
        snapshot_token_prob_q16 != token_prob_q16 ||
        snapshot_callback_flags != callback_flags ||
        snapshot_user_ctx != user_ctx ||
        snapshot_abi_tuple_buffer != abi_tuple_buffer ||
        snapshot_abi_tuple_buffer_capacity != abi_tuple_buffer_capacity ||
        snapshot_out_dispatch_count != out_dispatch_count ||
        snapshot_out_callback_status != out_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    callback_status = token_callback(staged_tuple,
                                     INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
                                     user_ctx);
    if (callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_tuple[0] != INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_tuple[1] != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_tuple[2] != snapshot_token_index)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_tuple[3] != snapshot_token_id)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_tuple[4] != snapshot_token_prob_q16)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS; lane++)
        abi_tuple_buffer[lane] = staged_tuple[lane];

    staged_dispatch_count = 1;
    staged_callback_status = callback_status;
    *out_dispatch_count = staged_dispatch_count;
    *out_callback_status = staged_callback_status;
    return SAMPLING_Q16_OK;
}

// IQ-1172 diagnostics no-write callback-dispatch companion.
//
// Contract:
// - Execute canonical callback dispatch into staged-local outputs only.
// - Stage canonical ABI tuple `{abi_version,flags,index,id,prob_q16}` from
//   immutable input snapshots.
// - Revalidate callback/output pointer snapshots before publish.
// - Publish caller tuple + diagnostics atomically only on exact parity.
I32 InferenceTokenStreamCallbackDispatchCheckedNoPartialCommitOnlyPreflightOnly(
    I32 (*token_callback)(I64 *abi_tuple,
                          I64 abi_tuple_cells,
                          I64 user_ctx),
    I64 *abi_tuple_buffer,
    I64 abi_tuple_buffer_capacity,
    I64 token_index,
    I64 token_id,
    I64 token_prob_q16,
    I64 callback_flags,
    I64 user_ctx,
    I64 *out_dispatch_count,
    I64 *out_callback_status)
{
    I64 staged_tuple[INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS];
    I64 staged_dispatch_count;
    I64 staged_callback_status;
    I64 canonical_tuple[INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS];
    I64 canonical_dispatch_count;
    I64 canonical_callback_status;

    I32 (*snapshot_token_callback)(I64 *abi_tuple,
                                   I64 abi_tuple_cells,
                                   I64 user_ctx);
    I64 snapshot_token_index;
    I64 snapshot_token_id;
    I64 snapshot_token_prob_q16;
    I64 snapshot_callback_flags;
    I64 snapshot_user_ctx;
    I64 *snapshot_abi_tuple_buffer;
    I64 snapshot_abi_tuple_buffer_capacity;
    I64 *snapshot_out_dispatch_count;
    I64 *snapshot_out_callback_status;
    I32 status;
    I64 lane;

    if (!token_callback || !abi_tuple_buffer ||
        !out_dispatch_count || !out_callback_status)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_dispatch_count == out_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (token_index < 0 || token_id < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_token_callback = token_callback;
    snapshot_token_index = token_index;
    snapshot_token_id = token_id;
    snapshot_token_prob_q16 = token_prob_q16;
    snapshot_callback_flags = callback_flags;
    snapshot_user_ctx = user_ctx;
    snapshot_abi_tuple_buffer = abi_tuple_buffer;
    snapshot_abi_tuple_buffer_capacity = abi_tuple_buffer_capacity;
    snapshot_out_dispatch_count = out_dispatch_count;
    snapshot_out_callback_status = out_callback_status;

    status = InferenceTokenStreamCallbackDispatchCheckedNoPartial(
        token_callback,
        staged_tuple,
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
        token_index,
        token_id,
        token_prob_q16,
        callback_flags,
        user_ctx,
        &staged_dispatch_count,
        &staged_callback_status);
    if (status)
        return status;

    canonical_tuple[0] = INFERENCE_TOKEN_STREAM_CALLBACK_ABI_VERSION;
    canonical_tuple[1] = INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE;
    canonical_tuple[2] = token_index;
    canonical_tuple[3] = token_id;
    canonical_tuple[4] = token_prob_q16;
    canonical_dispatch_count = 1;
    canonical_callback_status = SAMPLING_Q16_OK;

    if (snapshot_token_callback != token_callback ||
        snapshot_token_index != token_index ||
        snapshot_token_id != token_id ||
        snapshot_token_prob_q16 != token_prob_q16 ||
        snapshot_callback_flags != callback_flags ||
        snapshot_user_ctx != user_ctx ||
        snapshot_abi_tuple_buffer != abi_tuple_buffer ||
        snapshot_abi_tuple_buffer_capacity != abi_tuple_buffer_capacity ||
        snapshot_out_dispatch_count != out_dispatch_count ||
        snapshot_out_callback_status != out_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS; lane++)
    {
        if (staged_tuple[lane] != canonical_tuple[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (staged_dispatch_count != canonical_dispatch_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_callback_status != canonical_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS; lane++)
        abi_tuple_buffer[lane] = staged_tuple[lane];

    *out_dispatch_count = staged_dispatch_count;
    *out_callback_status = staged_callback_status;
    return SAMPLING_Q16_OK;
}

// IQ-1173 strict diagnostics parity gate for callback-dispatch publication.
//
// Contract:
// - Stage diagnostics tuple via
//   `InferenceTokenStreamCallbackDispatchCheckedNoPartialCommitOnlyPreflightOnly`.
// - Stage canonical tuple via
//   `InferenceTokenStreamCallbackDispatchCheckedNoPartial`.
// - Revalidate immutable callback/input/output snapshots before publish.
// - Require exact parity for
//   `{abi_version,flags,token_index,token_id,token_prob_q16,dispatch_count,callback_status}`.
// - Atomically publish tuple + diagnostics only on strict parity success.
I32 InferenceTokenStreamCallbackDispatchCheckedNoPartialCommitOnlyPreflightOnlyParity(
    I32 (*token_callback)(I64 *abi_tuple,
                          I64 abi_tuple_cells,
                          I64 user_ctx),
    I64 *abi_tuple_buffer,
    I64 abi_tuple_buffer_capacity,
    I64 token_index,
    I64 token_id,
    I64 token_prob_q16,
    I64 callback_flags,
    I64 user_ctx,
    I64 *out_dispatch_count,
    I64 *out_callback_status)
{
    I64 staged_tuple[INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS];
    I64 staged_dispatch_count;
    I64 staged_callback_status;
    I64 canonical_tuple[INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS];
    I64 canonical_dispatch_count;
    I64 canonical_callback_status;

    I32 (*snapshot_token_callback)(I64 *abi_tuple,
                                   I64 abi_tuple_cells,
                                   I64 user_ctx);
    I64 snapshot_token_index;
    I64 snapshot_token_id;
    I64 snapshot_token_prob_q16;
    I64 snapshot_callback_flags;
    I64 snapshot_user_ctx;
    I64 *snapshot_abi_tuple_buffer;
    I64 snapshot_abi_tuple_buffer_capacity;
    I64 *snapshot_out_dispatch_count;
    I64 *snapshot_out_callback_status;
    I64 lane;
    I32 status;

    if (!token_callback || !abi_tuple_buffer ||
        !out_dispatch_count || !out_callback_status)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_dispatch_count == out_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (token_index < 0 || token_id < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (callback_flags != INFERENCE_TOKEN_STREAM_CALLBACK_FLAG_NONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_token_callback = token_callback;
    snapshot_token_index = token_index;
    snapshot_token_id = token_id;
    snapshot_token_prob_q16 = token_prob_q16;
    snapshot_callback_flags = callback_flags;
    snapshot_user_ctx = user_ctx;
    snapshot_abi_tuple_buffer = abi_tuple_buffer;
    snapshot_abi_tuple_buffer_capacity = abi_tuple_buffer_capacity;
    snapshot_out_dispatch_count = out_dispatch_count;
    snapshot_out_callback_status = out_callback_status;

    status = InferenceTokenStreamCallbackDispatchCheckedNoPartialCommitOnlyPreflightOnly(
        token_callback,
        staged_tuple,
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
        token_index,
        token_id,
        token_prob_q16,
        callback_flags,
        user_ctx,
        &staged_dispatch_count,
        &staged_callback_status);
    if (status)
        return status;

    status = InferenceTokenStreamCallbackDispatchCheckedNoPartial(
        token_callback,
        canonical_tuple,
        INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS,
        token_index,
        token_id,
        token_prob_q16,
        callback_flags,
        user_ctx,
        &canonical_dispatch_count,
        &canonical_callback_status);
    if (status)
        return status;

    if (snapshot_token_callback != token_callback ||
        snapshot_token_index != token_index ||
        snapshot_token_id != token_id ||
        snapshot_token_prob_q16 != token_prob_q16 ||
        snapshot_callback_flags != callback_flags ||
        snapshot_user_ctx != user_ctx ||
        snapshot_abi_tuple_buffer != abi_tuple_buffer ||
        snapshot_abi_tuple_buffer_capacity != abi_tuple_buffer_capacity ||
        snapshot_out_dispatch_count != out_dispatch_count ||
        snapshot_out_callback_status != out_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_abi_tuple_buffer_capacity < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS; lane++)
    {
        if (staged_tuple[lane] != canonical_tuple[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (staged_dispatch_count != canonical_dispatch_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_callback_status != canonical_callback_status)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_TOKEN_STREAM_CALLBACK_ABI_TUPLE_CELLS; lane++)
        abi_tuple_buffer[lane] = staged_tuple[lane];

    *out_dispatch_count = staged_dispatch_count;
    *out_callback_status = staged_callback_status;
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

// IQ-1225 one-step no-partial decode primitive for WS7 integration.
//
// Contract:
// - Decode exactly one token from one logits row using GenerationStepChecked.
// - Stage caller history + sampled token in private workspace first.
// - Atomically publish {history_tail, out_generated_token, out_next_history_count}
//   only after successful decode and immutable snapshot revalidation.
I32 InferenceGenerateStepCheckedTopKTopPNoPartial(
    I64 *logits_q16,
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
    I64 *workspace_history_stage,
    I64 workspace_history_stage_capacity,
    I64 *out_generated_token,
    I64 *out_next_history_count)
{
    I64 required_step_logits_cells;
    I64 required_history_capacity;
    I64 required_stage_logits_capacity;
    I64 required_topk_capacity;
    I64 required_random_capacity;
    I64 required_generated_capacity;

    I64 snapshot_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_temperature_q16;
    I64 snapshot_top_k;
    I64 snapshot_top_p_q16;
    I64 snapshot_repetition_penalty_q16;
    I64 snapshot_random_q16;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_workspace_history_stage_capacity;

    I64 *snapshot_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_workspace_history_stage;
    I64 *snapshot_out_generated_token;
    I64 *snapshot_out_next_history_count;

    I64 history_index;
    I64 staged_generated_token;
    I64 staged_next_history_count;
    I32 status;

    if (!logits_q16 || !token_history || !workspace_stage_logits_q16 ||
        !workspace_topk_logits_q16 || !workspace_topk_indices ||
        !workspace_history_stage || !out_generated_token ||
        !out_next_history_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_generated_token == out_next_history_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_logits_capacity = logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_temperature_q16 = temperature_q16;
    snapshot_top_k = top_k;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_repetition_penalty_q16 = repetition_penalty_q16;
    snapshot_random_q16 = random_q16;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity;

    snapshot_logits_q16 = logits_q16;
    snapshot_token_history = token_history;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_workspace_history_stage = workspace_history_stage;
    snapshot_out_generated_token = out_generated_token;
    snapshot_out_next_history_count = out_next_history_count;

    status = InferenceGenerateTokensPreflightChecked(
        logits_capacity,
        vocab_size,
        1,
        token_history_capacity,
        token_history_count,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        1,
        1,
        &required_step_logits_cells,
        &required_history_capacity,
        &required_stage_logits_capacity,
        &required_topk_capacity,
        &required_random_capacity,
        &required_generated_capacity);
    if (status)
        return status;

    if (required_history_capacity > workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_capacity != logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_temperature_q16 != temperature_q16 ||
        snapshot_top_k != top_k ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_repetition_penalty_q16 != repetition_penalty_q16 ||
        snapshot_random_q16 != random_q16 ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_workspace_history_stage != workspace_history_stage ||
        snapshot_out_generated_token != out_generated_token ||
        snapshot_out_next_history_count != out_next_history_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Stage history into private workspace so failed decode never mutates
    // caller-visible token history.
    for (history_index = 0; history_index < token_history_count; history_index++)
        workspace_history_stage[history_index] = token_history[history_index];

    status = GenerationStepChecked(
        logits_q16,
        required_step_logits_cells,
        vocab_size,
        workspace_history_stage,
        required_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_q16,
        required_stage_logits_capacity,
        workspace_topk_logits_q16,
        required_topk_capacity,
        workspace_topk_indices,
        required_topk_capacity,
        &staged_generated_token);
    if (status)
        return status;

    workspace_history_stage[token_history_count] = staged_generated_token;

    if (token_history_count == 0x7FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    staged_next_history_count = token_history_count + 1;

    if (snapshot_token_history_count != token_history_count ||
        snapshot_token_history_capacity != token_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (snapshot_token_history != token_history ||
        snapshot_out_generated_token != out_generated_token ||
        snapshot_out_next_history_count != out_next_history_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    token_history[token_history_count] = workspace_history_stage[token_history_count];
    *out_generated_token = staged_generated_token;
    *out_next_history_count = staged_next_history_count;
    return SAMPLING_Q16_OK;
}

// IQ-1227 diagnostics preflight-only companion for one-step decode staging.
//
// Contract:
// - Derive exact one-step staging requirements from canonical tokens preflight.
// - Enforce immutable snapshot parity across all step-call inputs.
// - Publish only `{required_step_logits_cells,required_history_capacity,`
//   `required_stage_logits_capacity,required_topk_capacity}`.
// - Zero-write publish semantics: no caller history/output token mutation.
I32 InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnly(
    I64 *logits_q16,
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
    I64 *workspace_history_stage,
    I64 workspace_history_stage_capacity,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity)
{
    I64 required_step_logits_cells;
    I64 required_history_capacity;
    I64 required_stage_logits_capacity;
    I64 required_topk_capacity;
    I64 required_random_capacity;
    I64 required_generated_capacity;
    I64 canonical_required_history_capacity;

    I64 snapshot_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_temperature_q16;
    I64 snapshot_top_k;
    I64 snapshot_top_p_q16;
    I64 snapshot_repetition_penalty_q16;
    I64 snapshot_random_q16;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_workspace_history_stage_capacity;

    I64 *snapshot_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_workspace_history_stage;
    I64 *snapshot_out_required_step_logits_cells;
    I64 *snapshot_out_required_history_capacity;
    I64 *snapshot_out_required_stage_logits_capacity;
    I64 *snapshot_out_required_topk_capacity;

    I32 status;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_history_capacity ||
        out_required_step_logits_cells == out_required_stage_logits_capacity ||
        out_required_step_logits_cells == out_required_topk_capacity ||
        out_required_history_capacity == out_required_stage_logits_capacity ||
        out_required_history_capacity == out_required_topk_capacity ||
        out_required_stage_logits_capacity == out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!logits_q16 || !token_history || !workspace_stage_logits_q16 ||
        !workspace_topk_logits_q16 || !workspace_topk_indices ||
        !workspace_history_stage)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_logits_capacity = logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_temperature_q16 = temperature_q16;
    snapshot_top_k = top_k;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_repetition_penalty_q16 = repetition_penalty_q16;
    snapshot_random_q16 = random_q16;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity;

    snapshot_logits_q16 = logits_q16;
    snapshot_token_history = token_history;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_workspace_history_stage = workspace_history_stage;
    snapshot_out_required_step_logits_cells = out_required_step_logits_cells;
    snapshot_out_required_history_capacity = out_required_history_capacity;
    snapshot_out_required_stage_logits_capacity = out_required_stage_logits_capacity;
    snapshot_out_required_topk_capacity = out_required_topk_capacity;

    status = InferenceGenerateTokensPreflightChecked(
        logits_capacity,
        vocab_size,
        1,
        token_history_capacity,
        token_history_count,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        1,
        1,
        &required_step_logits_cells,
        &required_history_capacity,
        &required_stage_logits_capacity,
        &required_topk_capacity,
        &required_random_capacity,
        &required_generated_capacity);
    if (status)
        return status;

    if (required_history_capacity > workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (required_random_capacity != 1 || required_generated_capacity != 1)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (required_step_logits_cells != vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_stage_logits_capacity != vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_topk_capacity != vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (token_history_count == 0x7FFFFFFFFFFFFFFF)
        return SAMPLING_Q16_ERR_OVERFLOW;
    canonical_required_history_capacity = token_history_count + 1;
    if (required_history_capacity != canonical_required_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_capacity != logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_temperature_q16 != temperature_q16 ||
        snapshot_top_k != top_k ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_repetition_penalty_q16 != repetition_penalty_q16 ||
        snapshot_random_q16 != random_q16 ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_workspace_history_stage != workspace_history_stage ||
        snapshot_out_required_step_logits_cells != out_required_step_logits_cells ||
        snapshot_out_required_history_capacity != out_required_history_capacity ||
        snapshot_out_required_stage_logits_capacity != out_required_stage_logits_capacity ||
        snapshot_out_required_topk_capacity != out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = required_step_logits_cells;
    *out_required_history_capacity = required_history_capacity;
    *out_required_stage_logits_capacity = required_stage_logits_capacity;
    *out_required_topk_capacity = required_topk_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-1229 strict diagnostics parity gate for one-step decode preflight staging.
//
// Contract:
// - Stage one-step diagnostics tuple via
//   `InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnly`.
// - Independently recompute canonical one-step requirements via
//   `InferenceGenerateTokensPreflightChecked(..., max_new_tokens=1, ...)`.
// - Enforce immutable snapshot parity across step inputs and publish pointers.
// - Require exact tuple parity for
//   `{required_step_logits_cells,required_history_capacity,`
//   `required_stage_logits_capacity,required_topk_capacity}` before publish.
// - Atomically publish diagnostics tuple only on strict parity success.
I32 InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity(
    I64 *logits_q16,
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
    I64 *workspace_history_stage,
    I64 workspace_history_stage_capacity,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity)
{
    I64 staged_required_step_logits_cells;
    I64 staged_required_history_capacity;
    I64 staged_required_stage_logits_capacity;
    I64 staged_required_topk_capacity;

    I64 canonical_required_step_logits_cells;
    I64 canonical_required_history_capacity;
    I64 canonical_required_stage_logits_capacity;
    I64 canonical_required_topk_capacity;
    I64 canonical_required_random_capacity;
    I64 canonical_required_generated_capacity;

    I64 snapshot_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_temperature_q16;
    I64 snapshot_top_k;
    I64 snapshot_top_p_q16;
    I64 snapshot_repetition_penalty_q16;
    I64 snapshot_random_q16;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_workspace_history_stage_capacity;

    I64 *snapshot_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_workspace_history_stage;
    I64 *snapshot_out_required_step_logits_cells;
    I64 *snapshot_out_required_history_capacity;
    I64 *snapshot_out_required_stage_logits_capacity;
    I64 *snapshot_out_required_topk_capacity;

    I32 status;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_history_capacity ||
        out_required_step_logits_cells == out_required_stage_logits_capacity ||
        out_required_step_logits_cells == out_required_topk_capacity ||
        out_required_history_capacity == out_required_stage_logits_capacity ||
        out_required_history_capacity == out_required_topk_capacity ||
        out_required_stage_logits_capacity == out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!logits_q16 || !token_history || !workspace_stage_logits_q16 ||
        !workspace_topk_logits_q16 || !workspace_topk_indices ||
        !workspace_history_stage)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_logits_capacity = logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_temperature_q16 = temperature_q16;
    snapshot_top_k = top_k;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_repetition_penalty_q16 = repetition_penalty_q16;
    snapshot_random_q16 = random_q16;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity;

    snapshot_logits_q16 = logits_q16;
    snapshot_token_history = token_history;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_workspace_history_stage = workspace_history_stage;
    snapshot_out_required_step_logits_cells = out_required_step_logits_cells;
    snapshot_out_required_history_capacity = out_required_history_capacity;
    snapshot_out_required_stage_logits_capacity = out_required_stage_logits_capacity;
    snapshot_out_required_topk_capacity = out_required_topk_capacity;

    status = InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        workspace_history_stage,
        workspace_history_stage_capacity,
        &staged_required_step_logits_cells,
        &staged_required_history_capacity,
        &staged_required_stage_logits_capacity,
        &staged_required_topk_capacity);
    if (status)
        return status;

    status = InferenceGenerateTokensPreflightChecked(
        logits_capacity,
        vocab_size,
        1,
        token_history_capacity,
        token_history_count,
        workspace_stage_logits_capacity,
        workspace_topk_logits_capacity,
        workspace_topk_index_capacity,
        1,
        1,
        &canonical_required_step_logits_cells,
        &canonical_required_history_capacity,
        &canonical_required_stage_logits_capacity,
        &canonical_required_topk_capacity,
        &canonical_required_random_capacity,
        &canonical_required_generated_capacity);
    if (status)
        return status;

    if (canonical_required_random_capacity != 1 ||
        canonical_required_generated_capacity != 1)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_capacity != logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_temperature_q16 != temperature_q16 ||
        snapshot_top_k != top_k ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_repetition_penalty_q16 != repetition_penalty_q16 ||
        snapshot_random_q16 != random_q16 ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_workspace_history_stage != workspace_history_stage ||
        snapshot_out_required_step_logits_cells != out_required_step_logits_cells ||
        snapshot_out_required_history_capacity != out_required_history_capacity ||
        snapshot_out_required_stage_logits_capacity != out_required_stage_logits_capacity ||
        snapshot_out_required_topk_capacity != out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_required_step_logits_cells != canonical_required_step_logits_cells)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_required_history_capacity != canonical_required_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_required_stage_logits_capacity != canonical_required_stage_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_required_topk_capacity != canonical_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_required_step_logits_cells;
    *out_required_history_capacity = staged_required_history_capacity;
    *out_required_stage_logits_capacity = staged_required_stage_logits_capacity;
    *out_required_topk_capacity = staged_required_topk_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-1231 commit-only diagnostics wrapper for one-step preflight parity gate.
//
// Contract:
// - Stage parity-validated diagnostics tuple via
//   `InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity`.
// - Stage canonical diagnostics tuple via
//   `InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnly`.
// - Revalidate immutable step snapshots
//   `{logits_capacity,vocab_size,token_history_capacity,token_history_count,`
//   `temperature_q16,top_k,top_p_q16,repetition_penalty_q16,random_q16,`
//   `workspace_stage_logits_capacity,workspace_topk_logits_capacity,`
//   `workspace_topk_index_capacity,workspace_history_stage_capacity}`.
// - Atomically publish
//   `{required_step_logits_cells,required_history_capacity,`
//   `required_stage_logits_capacity,required_topk_capacity}` only on strict
//   parity success.
I32 InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    I64 *logits_q16,
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
    I64 *workspace_history_stage,
    I64 workspace_history_stage_capacity,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity)
{
    I64 staged_required_step_logits_cells;
    I64 staged_required_history_capacity;
    I64 staged_required_stage_logits_capacity;
    I64 staged_required_topk_capacity;

    I64 canonical_required_step_logits_cells;
    I64 canonical_required_history_capacity;
    I64 canonical_required_stage_logits_capacity;
    I64 canonical_required_topk_capacity;

    I64 snapshot_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_temperature_q16;
    I64 snapshot_top_k;
    I64 snapshot_top_p_q16;
    I64 snapshot_repetition_penalty_q16;
    I64 snapshot_random_q16;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_workspace_history_stage_capacity;

    I64 *snapshot_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_workspace_history_stage;
    I64 *snapshot_out_required_step_logits_cells;
    I64 *snapshot_out_required_history_capacity;
    I64 *snapshot_out_required_stage_logits_capacity;
    I64 *snapshot_out_required_topk_capacity;

    I32 status;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_history_capacity ||
        out_required_step_logits_cells == out_required_stage_logits_capacity ||
        out_required_step_logits_cells == out_required_topk_capacity ||
        out_required_history_capacity == out_required_stage_logits_capacity ||
        out_required_history_capacity == out_required_topk_capacity ||
        out_required_stage_logits_capacity == out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!logits_q16 || !token_history || !workspace_stage_logits_q16 ||
        !workspace_topk_logits_q16 || !workspace_topk_indices ||
        !workspace_history_stage)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_logits_capacity = logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_temperature_q16 = temperature_q16;
    snapshot_top_k = top_k;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_repetition_penalty_q16 = repetition_penalty_q16;
    snapshot_random_q16 = random_q16;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity;

    snapshot_logits_q16 = logits_q16;
    snapshot_token_history = token_history;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_workspace_history_stage = workspace_history_stage;
    snapshot_out_required_step_logits_cells = out_required_step_logits_cells;
    snapshot_out_required_history_capacity = out_required_history_capacity;
    snapshot_out_required_stage_logits_capacity = out_required_stage_logits_capacity;
    snapshot_out_required_topk_capacity = out_required_topk_capacity;

    status = InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        workspace_history_stage,
        workspace_history_stage_capacity,
        &staged_required_step_logits_cells,
        &staged_required_history_capacity,
        &staged_required_stage_logits_capacity,
        &staged_required_topk_capacity);
    if (status)
        return status;

    status = InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnly(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        workspace_history_stage,
        workspace_history_stage_capacity,
        &canonical_required_step_logits_cells,
        &canonical_required_history_capacity,
        &canonical_required_stage_logits_capacity,
        &canonical_required_topk_capacity);
    if (status)
        return status;

    if (snapshot_logits_capacity != logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_temperature_q16 != temperature_q16 ||
        snapshot_top_k != top_k ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_repetition_penalty_q16 != repetition_penalty_q16 ||
        snapshot_random_q16 != random_q16 ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_workspace_history_stage != workspace_history_stage ||
        snapshot_out_required_step_logits_cells != out_required_step_logits_cells ||
        snapshot_out_required_history_capacity != out_required_history_capacity ||
        snapshot_out_required_stage_logits_capacity != out_required_stage_logits_capacity ||
        snapshot_out_required_topk_capacity != out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_required_step_logits_cells != canonical_required_step_logits_cells)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_required_history_capacity != canonical_required_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_required_stage_logits_capacity != canonical_required_stage_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_required_topk_capacity != canonical_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_required_step_logits_cells;
    *out_required_history_capacity = staged_required_history_capacity;
    *out_required_stage_logits_capacity = staged_required_stage_logits_capacity;
    *out_required_topk_capacity = staged_required_topk_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-1233 diagnostics zero-write companion over one-step commit/parity wrappers.
//
// Contract:
// - Stage diagnostics tuple via
//   `InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParityCommitOnly`.
// - Independently stage diagnostics tuple via
//   `InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity`.
// - Enforce immutable step snapshots
//   `{logits_capacity,vocab_size,token_history_capacity,token_history_count,`
//   `temperature_q16,top_k,top_p_q16,repetition_penalty_q16,random_q16,`
//   `workspace_stage_logits_capacity,workspace_topk_logits_capacity,`
//   `workspace_topk_index_capacity,workspace_history_stage_capacity}`.
// - Require strict tuple parity for
//   `{required_step_logits_cells,required_history_capacity,`
//   `required_stage_logits_capacity,required_topk_capacity}` before publish.
// - Zero-write publish semantics: outputs mutate only after all checks succeed.
I32 InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 *logits_q16,
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
    I64 *workspace_history_stage,
    I64 workspace_history_stage_capacity,
    I64 *out_required_step_logits_cells,
    I64 *out_required_history_capacity,
    I64 *out_required_stage_logits_capacity,
    I64 *out_required_topk_capacity)
{
    I64 staged_commit_required_step_logits_cells;
    I64 staged_commit_required_history_capacity;
    I64 staged_commit_required_stage_logits_capacity;
    I64 staged_commit_required_topk_capacity;

    I64 staged_parity_required_step_logits_cells;
    I64 staged_parity_required_history_capacity;
    I64 staged_parity_required_stage_logits_capacity;
    I64 staged_parity_required_topk_capacity;

    I64 snapshot_logits_capacity;
    I64 snapshot_vocab_size;
    I64 snapshot_token_history_capacity;
    I64 snapshot_token_history_count;
    I64 snapshot_temperature_q16;
    I64 snapshot_top_k;
    I64 snapshot_top_p_q16;
    I64 snapshot_repetition_penalty_q16;
    I64 snapshot_random_q16;
    I64 snapshot_workspace_stage_logits_capacity;
    I64 snapshot_workspace_topk_logits_capacity;
    I64 snapshot_workspace_topk_index_capacity;
    I64 snapshot_workspace_history_stage_capacity;

    I64 *snapshot_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_workspace_history_stage;
    I64 *snapshot_out_required_step_logits_cells;
    I64 *snapshot_out_required_history_capacity;
    I64 *snapshot_out_required_stage_logits_capacity;
    I64 *snapshot_out_required_topk_capacity;

    I32 status;

    if (!out_required_step_logits_cells || !out_required_history_capacity ||
        !out_required_stage_logits_capacity || !out_required_topk_capacity)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_required_step_logits_cells == out_required_history_capacity ||
        out_required_step_logits_cells == out_required_stage_logits_capacity ||
        out_required_step_logits_cells == out_required_topk_capacity ||
        out_required_history_capacity == out_required_stage_logits_capacity ||
        out_required_history_capacity == out_required_topk_capacity ||
        out_required_stage_logits_capacity == out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (!logits_q16 || !token_history || !workspace_stage_logits_q16 ||
        !workspace_topk_logits_q16 || !workspace_topk_indices ||
        !workspace_history_stage)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (temperature_q16 <= 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (repetition_penalty_q16 < SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_k <= 0 || top_k > vocab_size)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (top_p_q16 <= 0 || top_p_q16 > SAMPLING_Q16_ONE)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_logits_capacity = logits_capacity;
    snapshot_vocab_size = vocab_size;
    snapshot_token_history_capacity = token_history_capacity;
    snapshot_token_history_count = token_history_count;
    snapshot_temperature_q16 = temperature_q16;
    snapshot_top_k = top_k;
    snapshot_top_p_q16 = top_p_q16;
    snapshot_repetition_penalty_q16 = repetition_penalty_q16;
    snapshot_random_q16 = random_q16;
    snapshot_workspace_stage_logits_capacity = workspace_stage_logits_capacity;
    snapshot_workspace_topk_logits_capacity = workspace_topk_logits_capacity;
    snapshot_workspace_topk_index_capacity = workspace_topk_index_capacity;
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity;

    snapshot_logits_q16 = logits_q16;
    snapshot_token_history = token_history;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_workspace_history_stage = workspace_history_stage;
    snapshot_out_required_step_logits_cells = out_required_step_logits_cells;
    snapshot_out_required_history_capacity = out_required_history_capacity;
    snapshot_out_required_stage_logits_capacity = out_required_stage_logits_capacity;
    snapshot_out_required_topk_capacity = out_required_topk_capacity;

    status = InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        workspace_history_stage,
        workspace_history_stage_capacity,
        &staged_commit_required_step_logits_cells,
        &staged_commit_required_history_capacity,
        &staged_commit_required_stage_logits_capacity,
        &staged_commit_required_topk_capacity);
    if (status)
        return status;

    status = InferenceGenerateStepCheckedTopKTopPNoPartialCommitOnlyPreflightOnlyParity(
        logits_q16,
        logits_capacity,
        vocab_size,
        token_history,
        token_history_capacity,
        token_history_count,
        temperature_q16,
        top_k,
        top_p_q16,
        repetition_penalty_q16,
        random_q16,
        workspace_stage_logits_q16,
        workspace_stage_logits_capacity,
        workspace_topk_logits_q16,
        workspace_topk_logits_capacity,
        workspace_topk_indices,
        workspace_topk_index_capacity,
        workspace_history_stage,
        workspace_history_stage_capacity,
        &staged_parity_required_step_logits_cells,
        &staged_parity_required_history_capacity,
        &staged_parity_required_stage_logits_capacity,
        &staged_parity_required_topk_capacity);
    if (status)
        return status;

    if (snapshot_logits_capacity != logits_capacity ||
        snapshot_vocab_size != vocab_size ||
        snapshot_token_history_capacity != token_history_capacity ||
        snapshot_token_history_count != token_history_count ||
        snapshot_temperature_q16 != temperature_q16 ||
        snapshot_top_k != top_k ||
        snapshot_top_p_q16 != top_p_q16 ||
        snapshot_repetition_penalty_q16 != repetition_penalty_q16 ||
        snapshot_random_q16 != random_q16 ||
        snapshot_workspace_stage_logits_capacity != workspace_stage_logits_capacity ||
        snapshot_workspace_topk_logits_capacity != workspace_topk_logits_capacity ||
        snapshot_workspace_topk_index_capacity != workspace_topk_index_capacity ||
        snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_logits_q16 != logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_workspace_history_stage != workspace_history_stage ||
        snapshot_out_required_step_logits_cells != out_required_step_logits_cells ||
        snapshot_out_required_history_capacity != out_required_history_capacity ||
        snapshot_out_required_stage_logits_capacity != out_required_stage_logits_capacity ||
        snapshot_out_required_topk_capacity != out_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_commit_required_step_logits_cells != staged_parity_required_step_logits_cells)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_commit_required_history_capacity != staged_parity_required_history_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_commit_required_stage_logits_capacity != staged_parity_required_stage_logits_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (staged_commit_required_topk_capacity != staged_parity_required_topk_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_required_step_logits_cells = staged_commit_required_step_logits_cells;
    *out_required_history_capacity = staged_commit_required_history_capacity;
    *out_required_stage_logits_capacity = staged_commit_required_stage_logits_capacity;
    *out_required_topk_capacity = staged_commit_required_topk_capacity;
    return SAMPLING_Q16_OK;
}

// IQ-1222 no-partial top-k/top-p generation loop wrapper for WS7.
//
// Contract:
// - Stage caller history into a private workspace before decode starts.
// - For each step, decode from immutable per-step logits row and stage token id
//   into private generated/history workspaces.
// - Commit exactly one complete step atomically to caller history/output after
//   successful decode of that step.
// - On failure, the current step has no partial publish. Prior committed steps
//   remain visible by design (step-atomic, not whole-loop atomic).
I32 InferenceGenerateTokensCheckedTopKTopPNoPartial(
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
    I64 *workspace_history_stage,
    I64 workspace_history_stage_capacity,
    I64 *workspace_generated_stage,
    I64 workspace_generated_stage_capacity,
    I64 *out_generated_tokens,
    I64 generated_capacity,
    I64 *out_generated_count)
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
    I64 snapshot_workspace_history_stage_capacity;
    I64 snapshot_workspace_generated_stage_capacity;
    I64 snapshot_generated_capacity;

    I64 *snapshot_step_logits_q16;
    I64 *snapshot_token_history;
    I64 *snapshot_random_q16_values;
    I64 *snapshot_workspace_stage_logits_q16;
    I64 *snapshot_workspace_topk_logits_q16;
    I64 *snapshot_workspace_topk_indices;
    I64 *snapshot_workspace_history_stage;
    I64 *snapshot_workspace_generated_stage;
    I64 *snapshot_out_generated_tokens;
    I64 *snapshot_out_generated_count;

    I64 history_index;
    I64 step_index;
    I64 logits_row_offset;
    I64 logits_row_end;
    I64 sampled_token_id;
    I32 status;

    if (!step_logits_q16 || !token_history || !random_q16_values ||
        !workspace_stage_logits_q16 || !workspace_topk_logits_q16 ||
        !workspace_topk_indices || !workspace_history_stage ||
        !workspace_generated_stage || !out_generated_tokens ||
        !out_generated_count)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_generated_count == out_generated_tokens)
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
    snapshot_workspace_history_stage_capacity = workspace_history_stage_capacity;
    snapshot_workspace_generated_stage_capacity = workspace_generated_stage_capacity;
    snapshot_generated_capacity = generated_capacity;

    snapshot_step_logits_q16 = step_logits_q16;
    snapshot_token_history = token_history;
    snapshot_random_q16_values = random_q16_values;
    snapshot_workspace_stage_logits_q16 = workspace_stage_logits_q16;
    snapshot_workspace_topk_logits_q16 = workspace_topk_logits_q16;
    snapshot_workspace_topk_indices = workspace_topk_indices;
    snapshot_workspace_history_stage = workspace_history_stage;
    snapshot_workspace_generated_stage = workspace_generated_stage;
    snapshot_out_generated_tokens = out_generated_tokens;
    snapshot_out_generated_count = out_generated_count;

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

    if (required_history_capacity > workspace_history_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;
    if (required_generated_capacity > workspace_generated_stage_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

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
        snapshot_workspace_history_stage_capacity != workspace_history_stage_capacity ||
        snapshot_workspace_generated_stage_capacity != workspace_generated_stage_capacity ||
        snapshot_generated_capacity != generated_capacity)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_step_logits_q16 != step_logits_q16 ||
        snapshot_token_history != token_history ||
        snapshot_random_q16_values != random_q16_values ||
        snapshot_workspace_stage_logits_q16 != workspace_stage_logits_q16 ||
        snapshot_workspace_topk_logits_q16 != workspace_topk_logits_q16 ||
        snapshot_workspace_topk_indices != workspace_topk_indices ||
        snapshot_workspace_history_stage != workspace_history_stage ||
        snapshot_workspace_generated_stage != workspace_generated_stage ||
        snapshot_out_generated_tokens != out_generated_tokens ||
        snapshot_out_generated_count != out_generated_count)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    // Stage current history first so decode never mutates caller history before
    // a full step succeeds.
    for (history_index = 0; history_index < token_history_count; history_index++)
        workspace_history_stage[history_index] = token_history[history_index];

    for (step_index = 0; step_index < max_new_tokens; step_index++)
    {
        status = InferenceDispatchTryMulI64Checked(step_index,
                                                   vocab_size,
                                                   &logits_row_offset);
        if (status)
            return status;

        status = InferenceDispatchTryMulI64Checked(step_index + 1,
                                                   vocab_size,
                                                   &logits_row_end);
        if (status)
            return status;

        if (logits_row_end > step_logits_capacity)
            return SAMPLING_Q16_ERR_BAD_PARAM;

        status = GenerationStepChecked(
            step_logits_q16 + logits_row_offset,
            vocab_size,
            vocab_size,
            workspace_history_stage,
            required_history_capacity,
            token_history_count + step_index,
            temperature_q16,
            top_k,
            top_p_q16,
            repetition_penalty_q16,
            random_q16_values[step_index],
            workspace_stage_logits_q16,
            required_stage_logits_capacity,
            workspace_topk_logits_q16,
            required_topk_capacity,
            workspace_topk_indices,
            required_topk_capacity,
            &sampled_token_id);
        if (status)
            return status;

        workspace_generated_stage[step_index] = sampled_token_id;
        workspace_history_stage[token_history_count + step_index] = sampled_token_id;

        // Per-step atomic commit: publish history tail + generated token + count
        // only after the whole decode stage completed.
        token_history[token_history_count + step_index] =
            workspace_history_stage[token_history_count + step_index];
        out_generated_tokens[step_index] = workspace_generated_stage[step_index];
        *out_generated_count = step_index + 1;
    }

    return SAMPLING_Q16_OK;
}

// IQ-1791 per-token Book-of-Truth emission gate for secure-local replay logs.
//
// Contract:
// - Snapshot immutable event tuple
//   `{session_id,step_index,token_id,logit_q16,policy_digest_q64,profile_mode}`.
// - Fail-closed policy: secure-local + digest match required for emit.
// - Validate status domain exactly `{blocked=0,emitted=1}`.
// - No partial writes: caller event/output buffers mutate only after all checks.
I32 BotTokenEmitChecked(
    I64 *event_buffer,
    I64 event_buffer_capacity,
    I64 session_id,
    I64 step_index,
    I64 token_id,
    I64 logit_q16,
    I64 policy_digest_q64,
    I64 expected_policy_digest_q64,
    I64 profile_mode,
    I64 *out_event_status,
    I64 *out_event_count,
    I64 *out_event_digest_q64)
{
    I64 staged_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_event_status;
    I64 staged_event_count;
    I64 staged_event_digest_q64;
    I64 digest_q64;
    I64 lane;

    I64 snapshot_event_buffer_capacity;
    I64 snapshot_session_id;
    I64 snapshot_step_index;
    I64 snapshot_token_id;
    I64 snapshot_logit_q16;
    I64 snapshot_policy_digest_q64;
    I64 snapshot_expected_policy_digest_q64;
    I64 snapshot_profile_mode;
    I64 *snapshot_event_buffer;
    I64 *snapshot_out_event_status;
    I64 *snapshot_out_event_count;
    I64 *snapshot_out_event_digest_q64;

    if (!event_buffer || !out_event_status || !out_event_count ||
        !out_event_digest_q64)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_event_status == out_event_count ||
        out_event_status == out_event_digest_q64 ||
        out_event_count == out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (session_id < 0 || step_index < 0 || token_id < 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (profile_mode != INFERENCE_BOT_PROFILE_SECURE &&
        profile_mode != INFERENCE_BOT_PROFILE_DEV)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_event_buffer_capacity = event_buffer_capacity;
    snapshot_session_id = session_id;
    snapshot_step_index = step_index;
    snapshot_token_id = token_id;
    snapshot_logit_q16 = logit_q16;
    snapshot_policy_digest_q64 = policy_digest_q64;
    snapshot_expected_policy_digest_q64 = expected_policy_digest_q64;
    snapshot_profile_mode = profile_mode;
    snapshot_event_buffer = event_buffer;
    snapshot_out_event_status = out_event_status;
    snapshot_out_event_count = out_event_count;
    snapshot_out_event_digest_q64 = out_event_digest_q64;

    staged_event[0] = session_id;
    staged_event[1] = step_index;
    staged_event[2] = token_id;
    staged_event[3] = logit_q16;
    staged_event[4] = policy_digest_q64;
    staged_event[5] = profile_mode;

    if (profile_mode == INFERENCE_BOT_PROFILE_SECURE &&
        policy_digest_q64 == expected_policy_digest_q64)
    {
        staged_event_status = INFERENCE_BOT_STATUS_EMITTED;
        staged_event_count = 1;
    }
    else
    {
        staged_event_status = INFERENCE_BOT_STATUS_BLOCKED;
        staged_event_count = 0;
    }

    if (staged_event_status != INFERENCE_BOT_STATUS_BLOCKED &&
        staged_event_status != INFERENCE_BOT_STATUS_EMITTED)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_event_status == INFERENCE_BOT_STATUS_EMITTED &&
        staged_event_count != 1)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_event_status == INFERENCE_BOT_STATUS_BLOCKED &&
        staged_event_count != 0)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    digest_q64 = INFERENCE_BOT_DIGEST_OFFSET;
    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        digest_q64 = digest_q64 ^ staged_event[lane];
        digest_q64 = digest_q64 * INFERENCE_BOT_DIGEST_PRIME;
    }
    digest_q64 = digest_q64 ^ staged_event_status;
    digest_q64 = digest_q64 * INFERENCE_BOT_DIGEST_PRIME;
    digest_q64 = digest_q64 ^ staged_event_count;
    digest_q64 = digest_q64 * INFERENCE_BOT_DIGEST_PRIME;
    staged_event_digest_q64 = digest_q64;

    if (snapshot_event_buffer_capacity != event_buffer_capacity ||
        snapshot_session_id != session_id ||
        snapshot_step_index != step_index ||
        snapshot_token_id != token_id ||
        snapshot_logit_q16 != logit_q16 ||
        snapshot_policy_digest_q64 != policy_digest_q64 ||
        snapshot_expected_policy_digest_q64 != expected_policy_digest_q64 ||
        snapshot_profile_mode != profile_mode)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer != event_buffer ||
        snapshot_out_event_status != out_event_status ||
        snapshot_out_event_count != out_event_count ||
        snapshot_out_event_digest_q64 != out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_event_status == INFERENCE_BOT_STATUS_EMITTED)
    {
        for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
            event_buffer[lane] = staged_event[lane];
    }

    *out_event_status = staged_event_status;
    *out_event_count = staged_event_count;
    *out_event_digest_q64 = staged_event_digest_q64;
    return SAMPLING_Q16_OK;
}

// IQ-1792 commit-only wrapper over per-token Book-of-Truth event emission.
//
// Contract:
// - Snapshot immutable event tuple and output-pointer identities.
// - Re-run canonical emission twice into private staging buffers.
// - Publish caller-visible `{event_status,event_count,event_digest_q64}` and
//   emitted event tuple only when strict replay parity holds.
I32 BotTokenEmitCommitOnly(
    I64 *event_buffer,
    I64 event_buffer_capacity,
    I64 session_id,
    I64 step_index,
    I64 token_id,
    I64 logit_q16,
    I64 policy_digest_q64,
    I64 expected_policy_digest_q64,
    I64 profile_mode,
    I64 *out_event_status,
    I64 *out_event_count,
    I64 *out_event_digest_q64)
{
    I64 staged_event_primary[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_event_replay[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_event_status_primary;
    I64 staged_event_count_primary;
    I64 staged_event_digest_primary;
    I64 staged_event_status_replay;
    I64 staged_event_count_replay;
    I64 staged_event_digest_replay;
    I64 snapshot_event_buffer_capacity;
    I64 snapshot_session_id;
    I64 snapshot_step_index;
    I64 snapshot_token_id;
    I64 snapshot_logit_q16;
    I64 snapshot_policy_digest_q64;
    I64 snapshot_expected_policy_digest_q64;
    I64 snapshot_profile_mode;
    I64 *snapshot_event_buffer;
    I64 *snapshot_out_event_status;
    I64 *snapshot_out_event_count;
    I64 *snapshot_out_event_digest_q64;
    I32 status_primary;
    I32 status_replay;
    I64 lane;

    if (!event_buffer || !out_event_status || !out_event_count ||
        !out_event_digest_q64)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_event_status == out_event_count ||
        out_event_status == out_event_digest_q64 ||
        out_event_count == out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_event_buffer_capacity = event_buffer_capacity;
    snapshot_session_id = session_id;
    snapshot_step_index = step_index;
    snapshot_token_id = token_id;
    snapshot_logit_q16 = logit_q16;
    snapshot_policy_digest_q64 = policy_digest_q64;
    snapshot_expected_policy_digest_q64 = expected_policy_digest_q64;
    snapshot_profile_mode = profile_mode;
    snapshot_event_buffer = event_buffer;
    snapshot_out_event_status = out_event_status;
    snapshot_out_event_count = out_event_count;
    snapshot_out_event_digest_q64 = out_event_digest_q64;

    status_primary = BotTokenEmitChecked(
        staged_event_primary,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_event_status_primary,
        &staged_event_count_primary,
        &staged_event_digest_primary);
    if (status_primary)
        return status_primary;

    status_replay = BotTokenEmitChecked(
        staged_event_replay,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_event_status_replay,
        &staged_event_count_replay,
        &staged_event_digest_replay);
    if (status_replay)
        return status_replay;

    if (snapshot_event_buffer_capacity != event_buffer_capacity ||
        snapshot_session_id != session_id ||
        snapshot_step_index != step_index ||
        snapshot_token_id != token_id ||
        snapshot_logit_q16 != logit_q16 ||
        snapshot_policy_digest_q64 != policy_digest_q64 ||
        snapshot_expected_policy_digest_q64 != expected_policy_digest_q64 ||
        snapshot_profile_mode != profile_mode)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer != event_buffer ||
        snapshot_out_event_status != out_event_status ||
        snapshot_out_event_count != out_event_count ||
        snapshot_out_event_digest_q64 != out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_event_status_primary != staged_event_status_replay ||
        staged_event_count_primary != staged_event_count_replay ||
        staged_event_digest_primary != staged_event_digest_replay)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (staged_event_primary[lane] != staged_event_replay[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (staged_event_status_primary == INFERENCE_BOT_STATUS_EMITTED)
    {
        for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
            event_buffer[lane] = staged_event_primary[lane];
    }

    *out_event_status = staged_event_status_primary;
    *out_event_count = staged_event_count_primary;
    *out_event_digest_q64 = staged_event_digest_primary;
    return SAMPLING_Q16_OK;
}

// IQ-1793 zero-write diagnostics companion for per-token Book-of-Truth event
// emission parity between commit-only and canonical checked paths.
//
// Contract:
// - Snapshot immutable event tuple inputs and output-pointer identities.
// - Evaluate both commit-only and canonical checked paths into private staging.
// - Require strict tuple parity `{event_status,event_count,event_digest_q64}` and
//   staged event-lane parity.
// - Preserve caller output slots and caller event buffer (no writes on success).
I32 BotTokenEmitPreflightOnly(
    I64 *event_buffer,
    I64 event_buffer_capacity,
    I64 session_id,
    I64 step_index,
    I64 token_id,
    I64 logit_q16,
    I64 policy_digest_q64,
    I64 expected_policy_digest_q64,
    I64 profile_mode,
    I64 *out_event_status,
    I64 *out_event_count,
    I64 *out_event_digest_q64)
{
    I64 staged_commit_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_canonical_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_commit_status;
    I64 staged_commit_count;
    I64 staged_commit_digest_q64;
    I64 staged_canonical_status;
    I64 staged_canonical_count;
    I64 staged_canonical_digest_q64;
    I64 snapshot_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 snapshot_event_status;
    I64 snapshot_event_count;
    I64 snapshot_event_digest_q64;
    I64 snapshot_event_buffer_capacity;
    I64 snapshot_session_id;
    I64 snapshot_step_index;
    I64 snapshot_token_id;
    I64 snapshot_logit_q16;
    I64 snapshot_policy_digest_q64;
    I64 snapshot_expected_policy_digest_q64;
    I64 snapshot_profile_mode;
    I64 *snapshot_event_buffer;
    I64 *snapshot_out_event_status;
    I64 *snapshot_out_event_count;
    I64 *snapshot_out_event_digest_q64;
    I32 status_commit;
    I32 status_canonical;
    I64 lane;

    if (!event_buffer || !out_event_status || !out_event_count ||
        !out_event_digest_q64)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_event_status == out_event_count ||
        out_event_status == out_event_digest_q64 ||
        out_event_count == out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_event_buffer_capacity = event_buffer_capacity;
    snapshot_session_id = session_id;
    snapshot_step_index = step_index;
    snapshot_token_id = token_id;
    snapshot_logit_q16 = logit_q16;
    snapshot_policy_digest_q64 = policy_digest_q64;
    snapshot_expected_policy_digest_q64 = expected_policy_digest_q64;
    snapshot_profile_mode = profile_mode;
    snapshot_event_buffer = event_buffer;
    snapshot_out_event_status = out_event_status;
    snapshot_out_event_count = out_event_count;
    snapshot_out_event_digest_q64 = out_event_digest_q64;
    snapshot_event_status = *out_event_status;
    snapshot_event_count = *out_event_count;
    snapshot_event_digest_q64 = *out_event_digest_q64;
    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
        snapshot_event[lane] = event_buffer[lane];

    status_commit = BotTokenEmitCommitOnly(
        staged_commit_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_commit_status,
        &staged_commit_count,
        &staged_commit_digest_q64);
    if (status_commit)
        return status_commit;

    status_canonical = BotTokenEmitChecked(
        staged_canonical_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_canonical_status,
        &staged_canonical_count,
        &staged_canonical_digest_q64);
    if (status_canonical)
        return status_canonical;

    if (snapshot_event_buffer_capacity != event_buffer_capacity ||
        snapshot_session_id != session_id ||
        snapshot_step_index != step_index ||
        snapshot_token_id != token_id ||
        snapshot_logit_q16 != logit_q16 ||
        snapshot_policy_digest_q64 != policy_digest_q64 ||
        snapshot_expected_policy_digest_q64 != expected_policy_digest_q64 ||
        snapshot_profile_mode != profile_mode)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer != event_buffer ||
        snapshot_out_event_status != out_event_status ||
        snapshot_out_event_count != out_event_count ||
        snapshot_out_event_digest_q64 != out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_commit_status != staged_canonical_status ||
        staged_commit_count != staged_canonical_count ||
        staged_commit_digest_q64 != staged_canonical_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (staged_commit_event[lane] != staged_canonical_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (*out_event_status != snapshot_event_status ||
        *out_event_count != snapshot_event_count ||
        *out_event_digest_q64 != snapshot_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (event_buffer[lane] != snapshot_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    return SAMPLING_Q16_OK;
}

// IQ-1794 strict diagnostics parity gate over preflight-only + commit-only
// Book-of-Truth token event emission paths.
//
// Contract:
// - Snapshot immutable event tuple inputs and output-pointer identities.
// - Evaluate preflight-only and commit-only paths into private staging outputs.
// - Enforce exact tuple parity `{event_status,event_count,event_digest_q64}` with
//   status-domain checks and no partial writes.
// - Preserve caller output slots and caller event buffer on success.
I32 BotTokenEmitParity(
    I64 *event_buffer,
    I64 event_buffer_capacity,
    I64 session_id,
    I64 step_index,
    I64 token_id,
    I64 logit_q16,
    I64 policy_digest_q64,
    I64 expected_policy_digest_q64,
    I64 profile_mode,
    I64 *out_event_status,
    I64 *out_event_count,
    I64 *out_event_digest_q64)
{
    I64 staged_preflight_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_commit_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_checked_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_preflight_status;
    I64 staged_preflight_count;
    I64 staged_preflight_digest_q64;
    I64 staged_commit_status;
    I64 staged_commit_count;
    I64 staged_commit_digest_q64;
    I64 staged_checked_status;
    I64 staged_checked_count;
    I64 staged_checked_digest_q64;
    I64 snapshot_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 snapshot_event_status;
    I64 snapshot_event_count;
    I64 snapshot_event_digest_q64;
    I64 snapshot_event_buffer_capacity;
    I64 snapshot_session_id;
    I64 snapshot_step_index;
    I64 snapshot_token_id;
    I64 snapshot_logit_q16;
    I64 snapshot_policy_digest_q64;
    I64 snapshot_expected_policy_digest_q64;
    I64 snapshot_profile_mode;
    I64 *snapshot_event_buffer;
    I64 *snapshot_out_event_status;
    I64 *snapshot_out_event_count;
    I64 *snapshot_out_event_digest_q64;
    I32 status_preflight;
    I32 status_commit;
    I64 lane;

    if (!event_buffer || !out_event_status || !out_event_count ||
        !out_event_digest_q64)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_event_status == out_event_count ||
        out_event_status == out_event_digest_q64 ||
        out_event_count == out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_event_buffer_capacity = event_buffer_capacity;
    snapshot_session_id = session_id;
    snapshot_step_index = step_index;
    snapshot_token_id = token_id;
    snapshot_logit_q16 = logit_q16;
    snapshot_policy_digest_q64 = policy_digest_q64;
    snapshot_expected_policy_digest_q64 = expected_policy_digest_q64;
    snapshot_profile_mode = profile_mode;
    snapshot_event_buffer = event_buffer;
    snapshot_out_event_status = out_event_status;
    snapshot_out_event_count = out_event_count;
    snapshot_out_event_digest_q64 = out_event_digest_q64;
    snapshot_event_status = *out_event_status;
    snapshot_event_count = *out_event_count;
    snapshot_event_digest_q64 = *out_event_digest_q64;
    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        snapshot_event[lane] = event_buffer[lane];
        staged_preflight_event[lane] = 0;
        staged_commit_event[lane] = 0;
        staged_checked_event[lane] = 0;
    }
    staged_preflight_status = 0x1212;
    staged_preflight_count = 0x3434;
    staged_preflight_digest_q64 = 0x5656;

    status_preflight = BotTokenEmitPreflightOnly(
        staged_preflight_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_preflight_status,
        &staged_preflight_count,
        &staged_preflight_digest_q64);
    if (status_preflight)
        return status_preflight;

    if (staged_preflight_status != 0x1212 ||
        staged_preflight_count != 0x3434 ||
        staged_preflight_digest_q64 != 0x5656)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    status_commit = BotTokenEmitCommitOnly(
        staged_commit_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_commit_status,
        &staged_commit_count,
        &staged_commit_digest_q64);
    if (status_commit)
        return status_commit;

    status_preflight = BotTokenEmitChecked(
        staged_checked_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_checked_status,
        &staged_checked_count,
        &staged_checked_digest_q64);
    if (status_preflight)
        return status_preflight;

    if (snapshot_event_buffer_capacity != event_buffer_capacity ||
        snapshot_session_id != session_id ||
        snapshot_step_index != step_index ||
        snapshot_token_id != token_id ||
        snapshot_logit_q16 != logit_q16 ||
        snapshot_policy_digest_q64 != policy_digest_q64 ||
        snapshot_expected_policy_digest_q64 != expected_policy_digest_q64 ||
        snapshot_profile_mode != profile_mode)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer != event_buffer ||
        snapshot_out_event_status != out_event_status ||
        snapshot_out_event_count != out_event_count ||
        snapshot_out_event_digest_q64 != out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if ((staged_commit_status != INFERENCE_BOT_STATUS_BLOCKED &&
         staged_commit_status != INFERENCE_BOT_STATUS_EMITTED) ||
        (staged_checked_status != INFERENCE_BOT_STATUS_BLOCKED &&
         staged_checked_status != INFERENCE_BOT_STATUS_EMITTED))
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_commit_status != staged_checked_status ||
        staged_commit_count != staged_checked_count ||
        staged_commit_digest_q64 != staged_checked_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (staged_commit_event[lane] != staged_checked_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (*out_event_status != snapshot_event_status ||
        *out_event_count != snapshot_event_count ||
        *out_event_digest_q64 != snapshot_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (event_buffer[lane] != snapshot_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    return SAMPLING_Q16_OK;
}

// IQ-1795 commit-only hardening wrapper over parity + preflight diagnostics.
//
// Contract:
// - Snapshot immutable event tuple inputs and output-pointer identities.
// - Require parity and preflight diagnostics to succeed in zero-write mode.
// - Run commit-only emission twice into private staging buffers and require
//   strict replay parity before any publish.
// - Publish `{event_status,event_count,event_digest_q64}` and event tuple
//   atomically (only on emitted status), with no partial writes on failure.
I32 BotTokenEmitParityCommit(
    I64 *event_buffer,
    I64 event_buffer_capacity,
    I64 session_id,
    I64 step_index,
    I64 token_id,
    I64 logit_q16,
    I64 policy_digest_q64,
    I64 expected_policy_digest_q64,
    I64 profile_mode,
    I64 *out_event_status,
    I64 *out_event_count,
    I64 *out_event_digest_q64)
{
    I64 staged_parity_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_preflight_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_primary_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_replay_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_parity_status;
    I64 staged_parity_count;
    I64 staged_parity_digest_q64;
    I64 staged_preflight_status;
    I64 staged_preflight_count;
    I64 staged_preflight_digest_q64;
    I64 staged_primary_status;
    I64 staged_primary_count;
    I64 staged_primary_digest_q64;
    I64 staged_replay_status;
    I64 staged_replay_count;
    I64 staged_replay_digest_q64;
    I64 snapshot_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 snapshot_event_status;
    I64 snapshot_event_count;
    I64 snapshot_event_digest_q64;
    I64 snapshot_event_buffer_capacity;
    I64 snapshot_session_id;
    I64 snapshot_step_index;
    I64 snapshot_token_id;
    I64 snapshot_logit_q16;
    I64 snapshot_policy_digest_q64;
    I64 snapshot_expected_policy_digest_q64;
    I64 snapshot_profile_mode;
    I64 *snapshot_event_buffer;
    I64 *snapshot_out_event_status;
    I64 *snapshot_out_event_count;
    I64 *snapshot_out_event_digest_q64;
    I32 status_parity;
    I32 status_preflight;
    I32 status_primary;
    I32 status_replay;
    I64 lane;

    if (!event_buffer || !out_event_status || !out_event_count ||
        !out_event_digest_q64)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_event_status == out_event_count ||
        out_event_status == out_event_digest_q64 ||
        out_event_count == out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_event_buffer_capacity = event_buffer_capacity;
    snapshot_session_id = session_id;
    snapshot_step_index = step_index;
    snapshot_token_id = token_id;
    snapshot_logit_q16 = logit_q16;
    snapshot_policy_digest_q64 = policy_digest_q64;
    snapshot_expected_policy_digest_q64 = expected_policy_digest_q64;
    snapshot_profile_mode = profile_mode;
    snapshot_event_buffer = event_buffer;
    snapshot_out_event_status = out_event_status;
    snapshot_out_event_count = out_event_count;
    snapshot_out_event_digest_q64 = out_event_digest_q64;
    snapshot_event_status = *out_event_status;
    snapshot_event_count = *out_event_count;
    snapshot_event_digest_q64 = *out_event_digest_q64;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        snapshot_event[lane] = event_buffer[lane];
        staged_parity_event[lane] = 0;
        staged_preflight_event[lane] = 0;
        staged_primary_event[lane] = 0;
        staged_replay_event[lane] = 0;
    }

    staged_parity_status = 0x1111;
    staged_parity_count = 0x2222;
    staged_parity_digest_q64 = 0x3333;
    staged_preflight_status = 0x4444;
    staged_preflight_count = 0x5555;
    staged_preflight_digest_q64 = 0x6666;

    status_parity = BotTokenEmitParity(
        staged_parity_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_parity_status,
        &staged_parity_count,
        &staged_parity_digest_q64);
    if (status_parity)
        return status_parity;

    if (staged_parity_status != 0x1111 ||
        staged_parity_count != 0x2222 ||
        staged_parity_digest_q64 != 0x3333)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    status_preflight = BotTokenEmitPreflightOnly(
        staged_preflight_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_preflight_status,
        &staged_preflight_count,
        &staged_preflight_digest_q64);
    if (status_preflight)
        return status_preflight;

    if (staged_preflight_status != 0x4444 ||
        staged_preflight_count != 0x5555 ||
        staged_preflight_digest_q64 != 0x6666)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    status_primary = BotTokenEmitCommitOnly(
        staged_primary_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_primary_status,
        &staged_primary_count,
        &staged_primary_digest_q64);
    if (status_primary)
        return status_primary;

    status_replay = BotTokenEmitCommitOnly(
        staged_replay_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_replay_status,
        &staged_replay_count,
        &staged_replay_digest_q64);
    if (status_replay)
        return status_replay;

    if (snapshot_event_buffer_capacity != event_buffer_capacity ||
        snapshot_session_id != session_id ||
        snapshot_step_index != step_index ||
        snapshot_token_id != token_id ||
        snapshot_logit_q16 != logit_q16 ||
        snapshot_policy_digest_q64 != policy_digest_q64 ||
        snapshot_expected_policy_digest_q64 != expected_policy_digest_q64 ||
        snapshot_profile_mode != profile_mode)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer != event_buffer ||
        snapshot_out_event_status != out_event_status ||
        snapshot_out_event_count != out_event_count ||
        snapshot_out_event_digest_q64 != out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_primary_status != staged_replay_status ||
        staged_primary_count != staged_replay_count ||
        staged_primary_digest_q64 != staged_replay_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (staged_primary_event[lane] != staged_replay_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (staged_primary_status == INFERENCE_BOT_STATUS_EMITTED)
    {
        for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
            event_buffer[lane] = staged_primary_event[lane];
    }
    else if (staged_primary_status != INFERENCE_BOT_STATUS_BLOCKED)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    *out_event_status = staged_primary_status;
    *out_event_count = staged_primary_count;
    *out_event_digest_q64 = staged_primary_digest_q64;
    return SAMPLING_Q16_OK;
}

// IQ-1796 zero-write diagnostics companion over parity-commit and parity paths.
//
// Contract:
// - Snapshot immutable event tuple inputs and output-pointer identities.
// - Evaluate parity-commit and parity paths into private staging buffers.
// - Require strict tuple parity `{event_status,event_count,event_digest_q64}`
//   and staged event-lane parity.
// - Preserve caller output slots and caller event buffer on success.
I32 BotTokenEmitParityCommitPreflight(
    I64 *event_buffer,
    I64 event_buffer_capacity,
    I64 session_id,
    I64 step_index,
    I64 token_id,
    I64 logit_q16,
    I64 policy_digest_q64,
    I64 expected_policy_digest_q64,
    I64 profile_mode,
    I64 *out_event_status,
    I64 *out_event_count,
    I64 *out_event_digest_q64)
{
    I64 staged_commit_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_parity_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 staged_commit_status;
    I64 staged_commit_count;
    I64 staged_commit_digest_q64;
    I64 staged_parity_status;
    I64 staged_parity_count;
    I64 staged_parity_digest_q64;
    I64 snapshot_event[INFERENCE_BOT_EVENT_TUPLE_CELLS];
    I64 snapshot_event_status;
    I64 snapshot_event_count;
    I64 snapshot_event_digest_q64;
    I64 snapshot_event_buffer_capacity;
    I64 snapshot_session_id;
    I64 snapshot_step_index;
    I64 snapshot_token_id;
    I64 snapshot_logit_q16;
    I64 snapshot_policy_digest_q64;
    I64 snapshot_expected_policy_digest_q64;
    I64 snapshot_profile_mode;
    I64 *snapshot_event_buffer;
    I64 *snapshot_out_event_status;
    I64 *snapshot_out_event_count;
    I64 *snapshot_out_event_digest_q64;
    I32 status_commit;
    I32 status_parity;
    I64 lane;

    if (!event_buffer || !out_event_status || !out_event_count ||
        !out_event_digest_q64)
        return SAMPLING_Q16_ERR_NULL_PTR;

    if (out_event_status == out_event_count ||
        out_event_status == out_event_digest_q64 ||
        out_event_count == out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (event_buffer_capacity < INFERENCE_BOT_EVENT_TUPLE_CELLS)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    snapshot_event_buffer_capacity = event_buffer_capacity;
    snapshot_session_id = session_id;
    snapshot_step_index = step_index;
    snapshot_token_id = token_id;
    snapshot_logit_q16 = logit_q16;
    snapshot_policy_digest_q64 = policy_digest_q64;
    snapshot_expected_policy_digest_q64 = expected_policy_digest_q64;
    snapshot_profile_mode = profile_mode;
    snapshot_event_buffer = event_buffer;
    snapshot_out_event_status = out_event_status;
    snapshot_out_event_count = out_event_count;
    snapshot_out_event_digest_q64 = out_event_digest_q64;
    snapshot_event_status = *out_event_status;
    snapshot_event_count = *out_event_count;
    snapshot_event_digest_q64 = *out_event_digest_q64;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        snapshot_event[lane] = event_buffer[lane];
        staged_commit_event[lane] = 0;
        staged_parity_event[lane] = 0;
    }

    status_commit = BotTokenEmitParityCommit(
        staged_commit_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_commit_status,
        &staged_commit_count,
        &staged_commit_digest_q64);
    if (status_commit)
        return status_commit;

    status_parity = BotTokenEmitParity(
        staged_parity_event,
        INFERENCE_BOT_EVENT_TUPLE_CELLS,
        session_id,
        step_index,
        token_id,
        logit_q16,
        policy_digest_q64,
        expected_policy_digest_q64,
        profile_mode,
        &staged_parity_status,
        &staged_parity_count,
        &staged_parity_digest_q64);
    if (status_parity)
        return status_parity;

    if (snapshot_event_buffer_capacity != event_buffer_capacity ||
        snapshot_session_id != session_id ||
        snapshot_step_index != step_index ||
        snapshot_token_id != token_id ||
        snapshot_logit_q16 != logit_q16 ||
        snapshot_policy_digest_q64 != policy_digest_q64 ||
        snapshot_expected_policy_digest_q64 != expected_policy_digest_q64 ||
        snapshot_profile_mode != profile_mode)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (snapshot_event_buffer != event_buffer ||
        snapshot_out_event_status != out_event_status ||
        snapshot_out_event_count != out_event_count ||
        snapshot_out_event_digest_q64 != out_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    if (staged_commit_status != staged_parity_status ||
        staged_commit_count != staged_parity_count ||
        staged_commit_digest_q64 != staged_parity_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (staged_commit_event[lane] != staged_parity_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    if (*out_event_status != snapshot_event_status ||
        *out_event_count != snapshot_event_count ||
        *out_event_digest_q64 != snapshot_event_digest_q64)
        return SAMPLING_Q16_ERR_BAD_PARAM;

    for (lane = 0; lane < INFERENCE_BOT_EVENT_TUPLE_CELLS; lane++)
    {
        if (event_buffer[lane] != snapshot_event[lane])
            return SAMPLING_Q16_ERR_BAD_PARAM;
    }

    return SAMPLING_Q16_OK;
}
