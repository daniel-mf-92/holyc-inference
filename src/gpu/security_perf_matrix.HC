// security_perf_matrix.HC
// Secure-on GPU throughput matrix runner (WS9-21/WS9-31/WS9-32).
//
// Security contract:
// - Matrix execution is fail-closed unless secure-local policy gates pass.
// - Requires IOMMU and Book-of-Truth DMA/MMIO/dispatch hooks.
// - Integer-only throughput and audit-overhead metrics (Q16 fixed point).

#define GPU_SEC_PERF_OK                 0
#define GPU_SEC_PERF_ERR_NULL_PTR       1
#define GPU_SEC_PERF_ERR_BAD_PARAM      2
#define GPU_SEC_PERF_ERR_POLICY_GUARD   3
#define GPU_SEC_PERF_ERR_CAPACITY       4
#define GPU_SEC_PERF_ERR_OVERFLOW       5

#define GPU_SEC_PERF_PROFILE_SECURE_LOCAL 1

#define GPU_SEC_PERF_QUANT_Q4_0 40
#define GPU_SEC_PERF_QUANT_Q8_0 80

#define GPU_SEC_PERF_ROW_GATE_REASON_ALLOW                  0
#define GPU_SEC_PERF_ROW_GATE_REASON_PROFILE_GUARD          1
#define GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD            2
#define GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD             3
#define GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH 4
#define GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY       5
#define GPU_SEC_PERF_ROW_GATE_REASON_BAD_QUANT_PROFILE      6

#define GPU_SEC_PERF_Q16_ONE    65536
#define GPU_SEC_PERF_CLOCK_HZ   1000000000
#define GPU_SEC_PERF_I64_MAX    0x7FFFFFFFFFFFFFFF

class GPUSecurityPerfMatrixRowInput
{
    I64 prompt_tokens;
    I64 batch_size;
    I64 quant_level;
    I64 tokens_generated;
    I64 secure_cycles;
    I64 relaxed_cycles;
};

class GPUSecurityPerfMatrixRowOutput
{
    I64 tok_per_sec_q16;
    I64 audit_overhead_delta_q16;
    I64 secure_cycles_per_token_q16;
};

I32 GPUPolicyAllowDispatchChecked(I64 iommu_enabled,
                                  I64 bot_dma_log_enabled,
                                  I64 bot_mmio_log_enabled,
                                  I64 bot_dispatch_log_enabled,
                                  I64 *out_allow_dispatch,
                                  I64 *out_reason_code,
                                  I64 *out_profile_id);

I32 GPUSecurityPerfMatrixAuditOverheadBudgetChecked(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_enabled,
    I64 bot_dma_log_enabled,
    I64 bot_mmio_log_enabled,
    I64 bot_dispatch_log_enabled,
    I64 max_audit_overhead_delta_q16,
    I64 *out_checked_rows,
    I64 *out_first_violation_index,
    I64 *out_max_observed_overhead_q16);

I32 GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 percentile_num,
    I64 percentile_den,
    I64 *out_value_q16)
{
    I64 scaled_rank_num;
    I64 rank;
    I64 i;
    I64 best_value;
    Bool found;

    if (!rows || !out_value_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_value_q16 = 0;

    if (row_count <= 0 ||
        percentile_num <= 0 ||
        percentile_den <= 0 ||
        percentile_num > percentile_den)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfMulChecked(percentile_num,
                                   row_count,
                                   &scaled_rank_num))
        return GPU_SEC_PERF_ERR_OVERFLOW;

    rank = scaled_rank_num / percentile_den;
    if ((scaled_rank_num % percentile_den) != 0)
    {
        if (!GPUSecurityPerfAddChecked(rank, 1, &rank))
            return GPU_SEC_PERF_ERR_OVERFLOW;
    }

    if (rank <= 0)
        rank = 1;
    if (rank > row_count)
        rank = row_count;

    found = FALSE;
    best_value = 0;

    for (i = 0; i < row_count; ++i)
    {
        I64 candidate;
        I64 less_count;
        I64 less_equal_count;
        I64 j;

        candidate = rows[i].secure_cycles_per_token_q16;
        if (candidate < 0)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        less_count = 0;
        less_equal_count = 0;

        for (j = 0; j < row_count; ++j)
        {
            I64 observed;

            observed = rows[j].secure_cycles_per_token_q16;
            if (observed < 0)
                return GPU_SEC_PERF_ERR_BAD_PARAM;

            if (observed < candidate)
                ++less_count;
            if (observed <= candidate)
                ++less_equal_count;
        }

        if (less_count < rank && rank <= less_equal_count)
        {
            if (!found || candidate < best_value)
            {
                best_value = candidate;
                found = TRUE;
            }
        }
    }

    if (!found)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_value_q16 = best_value;
    return GPU_SEC_PERF_OK;
}

Bool GPUSecurityPerfAddChecked(I64 lhs,
                               I64 rhs,
                               I64 *out_sum)
{
    if (!out_sum)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (lhs > (GPU_SEC_PERF_I64_MAX - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

Bool GPUSecurityPerfMulChecked(I64 lhs,
                               I64 rhs,
                               I64 *out_product)
{
    if (!out_product)
        return FALSE;

    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (lhs == 0 || rhs == 0)
    {
        *out_product = 0;
        return TRUE;
    }

    if (lhs > (GPU_SEC_PERF_I64_MAX / rhs))
        return FALSE;

    *out_product = lhs * rhs;
    return TRUE;
}

Bool GPUSecurityPerfFlagIsBinary(I64 value)
{
    return (value == 0 || value == 1);
}

Bool GPUSecurityPerfIsSupportedQuant(I64 quant_level)
{
    return (quant_level == GPU_SEC_PERF_QUANT_Q4_0 ||
            quant_level == GPU_SEC_PERF_QUANT_Q8_0);
}

I32 GPUSecurityPerfMatrixRowGateChecked(I64 secure_local_mode,
                                        I64 iommu_active,
                                        I64 book_of_truth_gpu_hooks,
                                        I64 policy_digest_parity,
                                        I64 row_prompt_tokens,
                                        I64 row_batch_size,
                                        I64 row_quant_profile,
                                        I64 *out_gate_reason_code,
                                        I64 *out_row_allowed)
{
    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY;
    *out_row_allowed = 0;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
    {
        *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_PROFILE_GUARD;
        return GPU_SEC_PERF_ERR_POLICY_GUARD;
    }

    if (row_prompt_tokens < 0 || row_batch_size <= 0)
    {
        *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY;
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    }

    if (!GPUSecurityPerfIsSupportedQuant(row_quant_profile))
    {
        *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BAD_QUANT_PROFILE;
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    }

    if (!iommu_active)
    {
        *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_IOMMU_GUARD;
        return GPU_SEC_PERF_ERR_POLICY_GUARD;
    }

    if (!book_of_truth_gpu_hooks)
    {
        *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BOOK_GUARD;
        return GPU_SEC_PERF_ERR_POLICY_GUARD;
    }

    if (!policy_digest_parity)
    {
        *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_POLICY_DIGEST_MISMATCH;
        return GPU_SEC_PERF_ERR_POLICY_GUARD;
    }

    *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_ALLOW;
    *out_row_allowed = 1;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedCommitOnly(I64 secure_local_mode,
                                                  I64 iommu_active,
                                                  I64 book_of_truth_gpu_hooks,
                                                  I64 policy_digest_parity,
                                                  I64 row_prompt_tokens,
                                                  I64 row_batch_size,
                                                  I64 row_quant_profile,
                                                  I64 *out_gate_reason_code,
                                                  I64 *out_row_allowed)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_gate_reason_code_ptr;
    I64 *snapshot_out_row_allowed_ptr;
    I64 staged_gate_reason_code;
    I64 staged_row_allowed;
    I64 parity_gate_reason_code;
    I64 parity_row_allowed;
    I32 status_primary;
    I32 status_parity;

    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY;
    *out_row_allowed = 0;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_gate_reason_code_ptr = out_gate_reason_code;
    snapshot_out_row_allowed_ptr = out_row_allowed;

    // IQ-1467 commit-only row gate hardening contract:
    // 1) stage canonical row gate tuple from immutable argument snapshots;
    // 2) recompute canonical tuple over the same immutable snapshots;
    // 3) require exact status + tuple parity and unchanged output pointers;
    // 4) publish `{gate_reason_code,row_allowed}` atomically only on parity.
    status_primary = GPUSecurityPerfMatrixRowGateChecked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_gate_reason_code,
        &staged_row_allowed);

    status_parity = GPUSecurityPerfMatrixRowGateChecked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &parity_gate_reason_code,
        &parity_row_allowed);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_gate_reason_code_ptr != out_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_row_allowed_ptr != out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != status_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_gate_reason_code != parity_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_row_allowed != parity_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != GPU_SEC_PERF_OK)
        return status_primary;

    *out_gate_reason_code = staged_gate_reason_code;
    *out_row_allowed = staged_row_allowed;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedNoPartial(I64 secure_local_mode,
                                                 I64 iommu_active,
                                                 I64 book_of_truth_gpu_hooks,
                                                 I64 policy_digest_parity,
                                                 I64 row_prompt_tokens,
                                                 I64 row_batch_size,
                                                 I64 row_quant_profile,
                                                 I64 *out_gate_reason_code,
                                                 I64 *out_row_allowed)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_gate_reason_code_ptr;
    I64 *snapshot_out_row_allowed_ptr;
    I64 staged_gate_reason_code;
    I64 staged_row_allowed;
    I64 commit_gate_reason_code;
    I64 commit_row_allowed;
    I32 status_preflight;
    I32 status_commit;

    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_gate_reason_code == out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_gate_reason_code_ptr = out_gate_reason_code;
    snapshot_out_row_allowed_ptr = out_row_allowed;

    // IQ-1468 no-partial row gate wrapper:
    // 1) snapshot immutable argument tuple + output pointers;
    // 2) run commit-only row gate preflight tuple derivation;
    // 3) re-run commit-only tuple derivation and require exact parity;
    // 4) publish `{gate_reason_code,row_allowed}` atomically only on success.
    status_preflight = GPUSecurityPerfMatrixRowGateCheckedCommitOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_gate_reason_code,
        &staged_row_allowed);

    status_commit = GPUSecurityPerfMatrixRowGateCheckedCommitOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &commit_gate_reason_code,
        &commit_row_allowed);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_gate_reason_code_ptr != out_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_row_allowed_ptr != out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != status_commit)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_gate_reason_code != commit_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_row_allowed != commit_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != GPU_SEC_PERF_OK)
        return status_preflight;

    *out_gate_reason_code = staged_gate_reason_code;
    *out_row_allowed = staged_row_allowed;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedCommitOnlyNoPartialBridge(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_gate_reason_code,
    I64 *out_row_allowed)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_gate_reason_code_ptr;
    I64 *snapshot_out_row_allowed_ptr;
    I64 bridge_gate_reason_code;
    I64 bridge_row_allowed;
    I64 commit_gate_reason_code;
    I64 commit_row_allowed;
    I32 status_bridge_preflight;
    I32 status_bridge_commit;

    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_gate_reason_code == out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_gate_reason_code_ptr = out_gate_reason_code;
    snapshot_out_row_allowed_ptr = out_row_allowed;

    // IQ-1481 no-partial bridge over commit-only row gate:
    // 1) snapshot immutable gate tuple + output pointers;
    // 2) run commit-only row gate twice from the same immutable snapshots;
    // 3) require exact status + tuple parity before publishing outputs;
    // 4) atomically publish `{gate_reason_code,row_allowed}` only on success.
    status_bridge_preflight = GPUSecurityPerfMatrixRowGateCheckedCommitOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &bridge_gate_reason_code,
        &bridge_row_allowed);

    status_bridge_commit = GPUSecurityPerfMatrixRowGateCheckedCommitOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &commit_gate_reason_code,
        &commit_row_allowed);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_gate_reason_code_ptr != out_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_row_allowed_ptr != out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_bridge_preflight != status_bridge_commit)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (bridge_gate_reason_code != commit_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (bridge_row_allowed != commit_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_bridge_preflight != GPU_SEC_PERF_OK)
        return status_bridge_preflight;

    *out_gate_reason_code = bridge_gate_reason_code;
    *out_row_allowed = bridge_row_allowed;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnly(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_gate_reason_code,
    I64 *out_row_allowed)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_gate_reason_code_ptr;
    I64 *snapshot_out_row_allowed_ptr;
    I64 saved_gate_reason_code;
    I64 saved_row_allowed;
    I64 staged_gate_reason_code;
    I64 staged_row_allowed;
    I64 canonical_gate_reason_code;
    I64 canonical_row_allowed;
    I32 status_nopartial;
    I32 status_canonical;

    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_gate_reason_code == out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_gate_reason_code_ptr = out_gate_reason_code;
    snapshot_out_row_allowed_ptr = out_row_allowed;
    saved_gate_reason_code = *out_gate_reason_code;
    saved_row_allowed = *out_row_allowed;

    // IQ-1469 preflight-only row gate companion:
    // 1) snapshot immutable gate tuple + caller output slots;
    // 2) derive staged tuple through no-partial wrapper;
    // 3) recompute canonical tuple from base row gate checker;
    // 4) require exact status + tuple parity and unchanged snapshots;
    // 5) preserve caller outputs (zero-write diagnostics semantics).
    status_nopartial = GPUSecurityPerfMatrixRowGateCheckedNoPartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_gate_reason_code,
        &staged_row_allowed);

    status_canonical = GPUSecurityPerfMatrixRowGateChecked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &canonical_gate_reason_code,
        &canonical_row_allowed);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_gate_reason_code_ptr != out_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_row_allowed_ptr != out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_nopartial != status_canonical)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_gate_reason_code != canonical_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_row_allowed != canonical_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_gate_reason_code != saved_gate_reason_code ||
        *out_row_allowed != saved_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_gate_reason_code = saved_gate_reason_code;
    *out_row_allowed = saved_row_allowed;
    return status_nopartial;
}

I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyParity(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_gate_reason_code,
    I64 *out_row_allowed)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_gate_reason_code_ptr;
    I64 *snapshot_out_row_allowed_ptr;
    I64 saved_gate_reason_code;
    I64 saved_row_allowed;
    I64 staged_gate_reason_code;
    I64 staged_row_allowed;
    I64 parity_gate_reason_code;
    I64 parity_row_allowed;
    I32 status_preflight_only;
    I32 status_nopartial;

    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_gate_reason_code == out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    saved_gate_reason_code = *out_gate_reason_code;
    saved_row_allowed = *out_row_allowed;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_gate_reason_code_ptr = out_gate_reason_code;
    snapshot_out_row_allowed_ptr = out_row_allowed;

    // IQ-1470 preflight-only parity gate:
    // 1) snapshot immutable gate tuple + caller output slots;
    // 2) derive parity tuple through no-partial wrapper;
    // 3) replay preflight-only wrapper with parity tuple as immutable seed;
    // 4) enforce immutable snapshots and exact status/tuple parity;
    // 5) preserve zero-write diagnostics semantics for caller outputs.
    parity_gate_reason_code = GPU_SEC_PERF_ROW_GATE_REASON_BAD_ROW_GEOMETRY;
    parity_row_allowed = 0;
    status_nopartial = GPUSecurityPerfMatrixRowGateCheckedNoPartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &parity_gate_reason_code,
        &parity_row_allowed);

    staged_gate_reason_code = parity_gate_reason_code;
    staged_row_allowed = parity_row_allowed;
    status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_gate_reason_code,
        &staged_row_allowed);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_gate_reason_code_ptr != out_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_row_allowed_ptr != out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight_only != status_nopartial)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_gate_reason_code != parity_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_row_allowed != parity_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_gate_reason_code != saved_gate_reason_code ||
        *out_row_allowed != saved_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_gate_reason_code = saved_gate_reason_code;
    *out_row_allowed = saved_row_allowed;
    return status_preflight_only;
}

I32 GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnlyCommitOnly(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_gate_reason_code,
    I64 *out_row_allowed)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_gate_reason_code_ptr;
    I64 *snapshot_out_row_allowed_ptr;
    I64 saved_gate_reason_code;
    I64 saved_row_allowed;
    I64 staged_gate_reason_code;
    I64 staged_row_allowed;
    I64 parity_gate_reason_code;
    I64 parity_row_allowed;
    I32 status_nopartial;
    I32 status_preflight_only;

    if (!out_gate_reason_code || !out_row_allowed)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_gate_reason_code == out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_gate_reason_code_ptr = out_gate_reason_code;
    snapshot_out_row_allowed_ptr = out_row_allowed;
    saved_gate_reason_code = *out_gate_reason_code;
    saved_row_allowed = *out_row_allowed;

    // IQ-1483 commit-only hardening wrapper:
    // 1) snapshot immutable gate tuple + caller output slots;
    // 2) derive canonical `{gate_reason_code,row_allowed}` via no-partial wrapper;
    // 3) replay preflight-only wrapper with canonical tuple as immutable seed;
    // 4) enforce immutable snapshots and strict status/tuple parity;
    // 5) publish outputs atomically only on strict parity and success.
    status_nopartial = GPUSecurityPerfMatrixRowGateCheckedNoPartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &parity_gate_reason_code,
        &parity_row_allowed);

    staged_gate_reason_code = parity_gate_reason_code;
    staged_row_allowed = parity_row_allowed;
    status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedNoPartialPreflightOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_gate_reason_code,
        &staged_row_allowed);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_gate_reason_code_ptr != out_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_row_allowed_ptr != out_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_nopartial != status_preflight_only)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (parity_gate_reason_code != staged_gate_reason_code)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (parity_row_allowed != staged_row_allowed)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_nopartial != GPU_SEC_PERF_OK)
    {
        if (*out_gate_reason_code != saved_gate_reason_code ||
            *out_row_allowed != saved_row_allowed)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        *out_gate_reason_code = saved_gate_reason_code;
        *out_row_allowed = saved_row_allowed;
        return status_nopartial;
    }

    *out_gate_reason_code = parity_gate_reason_code;
    *out_row_allowed = parity_row_allowed;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_snapshot_digest_q64)
{
    I64 tuple_values[9];
    I64 tuple_count;
    I64 i;
    I64 gate_reason_code;
    I64 row_allowed;
    I64 digest_q64;
    I64 stride;
    I64 offset_value;
    I64 mixed_term;
    I32 status_gate;

    if (!out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_snapshot_digest_q64 = 0;

    status_gate = GPUSecurityPerfMatrixRowGateChecked(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &gate_reason_code,
        &row_allowed);
    if (status_gate != GPU_SEC_PERF_OK)
        return status_gate;

    tuple_values[0] = secure_local_mode;
    tuple_values[1] = iommu_active;
    tuple_values[2] = book_of_truth_gpu_hooks;
    tuple_values[3] = policy_digest_parity;
    tuple_values[4] = row_prompt_tokens;
    tuple_values[5] = row_batch_size;
    tuple_values[6] = row_quant_profile;
    tuple_values[7] = gate_reason_code;
    tuple_values[8] = row_allowed;
    tuple_count = 9;

    // IQ-1471 row-gate policy snapshot digest:
    // - fold immutable gate snapshot tuple into a deterministic integer digest;
    // - use position-dependent strides so bit flips in any field change output;
    // - fail closed on any arithmetic overflow.
    digest_q64 = 1469598103934665603;
    stride = 104729;

    for (i = 0; i < tuple_count; ++i)
    {
        if (!GPUSecurityPerfAddChecked(tuple_values[i], i + 1, &offset_value))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfMulChecked(offset_value, stride, &mixed_term))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfAddChecked(digest_q64, mixed_term, &digest_q64))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfAddChecked(stride, 104729, &stride))
            return GPU_SEC_PERF_ERR_OVERFLOW;
    }

    *out_snapshot_digest_q64 = digest_q64;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnly(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_snapshot_digest_q64)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_snapshot_digest_q64_ptr;
    I64 staged_snapshot_digest_q64;
    I64 parity_snapshot_digest_q64;
    I32 status_primary;
    I32 status_parity;

    if (!out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_snapshot_digest_q64 = 0;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_snapshot_digest_q64_ptr = out_snapshot_digest_q64;

    // IQ-1472 commit-only policy snapshot digest wrapper:
    // 1) stage canonical digest from immutable row-gate snapshots;
    // 2) recompute canonical digest over the same immutable snapshots;
    // 3) require exact status + digest parity with unchanged output pointer;
    // 4) publish `{snapshot_digest_q64}` atomically only on strict parity.
    status_primary = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_snapshot_digest_q64);

    status_parity = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &parity_snapshot_digest_q64);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_snapshot_digest_q64_ptr != out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != status_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_snapshot_digest_q64 != parity_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != GPU_SEC_PERF_OK)
        return status_primary;

    *out_snapshot_digest_q64 = staged_snapshot_digest_q64;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartial(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_snapshot_digest_q64)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_snapshot_digest_q64_ptr;
    I64 saved_snapshot_digest_q64;
    I64 staged_snapshot_digest_q64;
    I64 commit_snapshot_digest_q64;
    I32 status_preflight;
    I32 status_commit;

    if (!out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_snapshot_digest_q64_ptr = out_snapshot_digest_q64;
    saved_snapshot_digest_q64 = *out_snapshot_digest_q64;

    // IQ-1473 no-partial policy snapshot digest wrapper:
    // 1) snapshot immutable row-gate tuple + caller output slot;
    // 2) derive staged digest via commit-only wrapper pass;
    // 3) rerun commit-only wrapper and require exact status/digest parity;
    // 4) preserve caller output on failure and publish atomically on success.
    status_preflight = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_snapshot_digest_q64);

    status_commit = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64CommitOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &commit_snapshot_digest_q64);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_snapshot_digest_q64_ptr != out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != status_commit)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_snapshot_digest_q64 != commit_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != GPU_SEC_PERF_OK)
    {
        if (*out_snapshot_digest_q64 != saved_snapshot_digest_q64)
            return GPU_SEC_PERF_ERR_BAD_PARAM;
        *out_snapshot_digest_q64 = saved_snapshot_digest_q64;
        return status_preflight;
    }

    *out_snapshot_digest_q64 = staged_snapshot_digest_q64;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartialPreflightOnly(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_snapshot_digest_q64)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_snapshot_digest_q64_ptr;
    I64 saved_snapshot_digest_q64;
    I64 staged_snapshot_digest_q64;
    I64 canonical_snapshot_digest_q64;
    I32 status_nopartial;
    I32 status_canonical;

    if (!out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_snapshot_digest_q64_ptr = out_snapshot_digest_q64;
    saved_snapshot_digest_q64 = *out_snapshot_digest_q64;

    // IQ-1475 preflight-only policy snapshot digest companion:
    // 1) snapshot immutable row-gate tuple + caller digest slot;
    // 2) derive staged digest via no-partial wrapper;
    // 3) recompute canonical digest from base helper over same tuple;
    // 4) enforce immutable snapshots and exact status/digest parity;
    // 5) preserve zero-write diagnostics semantics for caller output.
    status_nopartial = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_snapshot_digest_q64);

    status_canonical = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &canonical_snapshot_digest_q64);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_snapshot_digest_q64_ptr != out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_nopartial != status_canonical)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_snapshot_digest_q64 != canonical_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_snapshot_digest_q64 != saved_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_snapshot_digest_q64 = saved_snapshot_digest_q64;
    return status_nopartial;
}

I32 GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartialPreflightOnlyParity(
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 row_prompt_tokens,
    I64 row_batch_size,
    I64 row_quant_profile,
    I64 *out_snapshot_digest_q64)
{
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    I64 snapshot_row_prompt_tokens;
    I64 snapshot_row_batch_size;
    I64 snapshot_row_quant_profile;
    I64 *snapshot_out_snapshot_digest_q64_ptr;
    I64 saved_snapshot_digest_q64;
    I64 staged_snapshot_digest_q64;
    I64 parity_snapshot_digest_q64;
    I32 status_preflight_only;
    I32 status_nopartial;

    if (!out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    saved_snapshot_digest_q64 = *out_snapshot_digest_q64;

    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_row_prompt_tokens = row_prompt_tokens;
    snapshot_row_batch_size = row_batch_size;
    snapshot_row_quant_profile = row_quant_profile;
    snapshot_out_snapshot_digest_q64_ptr = out_snapshot_digest_q64;

    // IQ-1482 preflight-only parity gate for policy snapshot digest:
    // 1) snapshot immutable row-gate tuple + caller digest slot;
    // 2) derive parity digest through no-partial wrapper pass;
    // 3) replay preflight-only wrapper with parity digest as immutable seed;
    // 4) enforce immutable snapshots and strict status/digest parity;
    // 5) preserve zero-write diagnostics semantics for caller output.
    parity_snapshot_digest_q64 = 0;
    status_nopartial = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartial(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &parity_snapshot_digest_q64);

    staged_snapshot_digest_q64 = parity_snapshot_digest_q64;
    status_preflight_only = GPUSecurityPerfMatrixRowGateCheckedPolicySnapshotDigestQ64NoPartialPreflightOnly(
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        row_prompt_tokens,
        row_batch_size,
        row_quant_profile,
        &staged_snapshot_digest_q64);

    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_prompt_tokens != row_prompt_tokens)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_batch_size != row_batch_size)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_row_quant_profile != row_quant_profile)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_snapshot_digest_q64_ptr != out_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight_only != status_nopartial)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_snapshot_digest_q64 != parity_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_snapshot_digest_q64 != saved_snapshot_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_snapshot_digest_q64 = saved_snapshot_digest_q64;
    return status_preflight_only;
}

I32 GPUSecurityPerfMatrixRunChecked(const GPUSecurityPerfMatrixRowInput *rows,
                                    I64 row_count,
                                    I64 profile_id,
                                    I64 iommu_enabled,
                                    I64 bot_dma_log_enabled,
                                    I64 bot_mmio_log_enabled,
                                    I64 bot_dispatch_log_enabled,
                                    GPUSecurityPerfMatrixRowOutput *out_rows,
                                    I64 out_capacity,
                                    I64 *out_written_rows,
                                    I64 *out_tok_per_sec_q16_total,
                                    I64 *out_audit_overhead_delta_q16_total)
{
    I64 i;
    I64 allow_dispatch;
    I64 reason_code;
    I64 gated_profile_id;
    I32 policy_status;
    I64 tok_total;
    I64 overhead_total;

    if (!rows || !out_rows || !out_written_rows ||
        !out_tok_per_sec_q16_total || !out_audit_overhead_delta_q16_total)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_written_rows = 0;
    *out_tok_per_sec_q16_total = 0;
    *out_audit_overhead_delta_q16_total = 0;

    if (row_count <= 0)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < row_count)
        return GPU_SEC_PERF_ERR_CAPACITY;

    // Secure-on matrix must always execute under secure-local profile.
    if (profile_id != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    allow_dispatch = 0;
    reason_code = 0;
    gated_profile_id = 0;

    policy_status = GPUPolicyAllowDispatchChecked(iommu_enabled,
                                                  bot_dma_log_enabled,
                                                  bot_mmio_log_enabled,
                                                  bot_dispatch_log_enabled,
                                                  &allow_dispatch,
                                                  &reason_code,
                                                  &gated_profile_id);
    if (policy_status != 0 || !allow_dispatch ||
        gated_profile_id != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    tok_total = 0;
    overhead_total = 0;

    for (i = 0; i < row_count; ++i)
    {
        I64 tok_num;
        I64 tok_num_q16;
        I64 tok_per_sec_q16;
        I64 overhead_cycles;
        I64 overhead_num_q16;
        I64 overhead_delta_q16;
        I64 cycles_per_token_num_q16;
        I64 cycles_per_token_q16;
        I64 new_tok_total;
        I64 new_overhead_total;

        if (rows[i].prompt_tokens < 0 ||
            rows[i].batch_size <= 0 ||
            rows[i].tokens_generated <= 0 ||
            rows[i].secure_cycles <= 0 ||
            rows[i].relaxed_cycles <= 0)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        if (!GPUSecurityPerfIsSupportedQuant(rows[i].quant_level))
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        // Audit hooks should never make secure run faster in deterministic replay.
        if (rows[i].secure_cycles < rows[i].relaxed_cycles)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        if (!GPUSecurityPerfMulChecked(rows[i].tokens_generated,
                                       GPU_SEC_PERF_CLOCK_HZ,
                                       &tok_num))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfMulChecked(tok_num,
                                       GPU_SEC_PERF_Q16_ONE,
                                       &tok_num_q16))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        tok_per_sec_q16 = tok_num_q16 / rows[i].secure_cycles;

        overhead_cycles = rows[i].secure_cycles - rows[i].relaxed_cycles;

        if (!GPUSecurityPerfMulChecked(overhead_cycles,
                                       GPU_SEC_PERF_Q16_ONE,
                                       &overhead_num_q16))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        overhead_delta_q16 = overhead_num_q16 / rows[i].relaxed_cycles;

        if (!GPUSecurityPerfMulChecked(rows[i].secure_cycles,
                                       GPU_SEC_PERF_Q16_ONE,
                                       &cycles_per_token_num_q16))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        cycles_per_token_q16 = cycles_per_token_num_q16 / rows[i].tokens_generated;

        if (!GPUSecurityPerfAddChecked(tok_total,
                                       tok_per_sec_q16,
                                       &new_tok_total))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfAddChecked(overhead_total,
                                       overhead_delta_q16,
                                       &new_overhead_total))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        // Commit row after all checked math is complete.
        out_rows[i].tok_per_sec_q16 = tok_per_sec_q16;
        out_rows[i].audit_overhead_delta_q16 = overhead_delta_q16;
        out_rows[i].secure_cycles_per_token_q16 = cycles_per_token_q16;

        tok_total = new_tok_total;
        overhead_total = new_overhead_total;
    }

    *out_written_rows = row_count;
    *out_tok_per_sec_q16_total = tok_total;
    *out_audit_overhead_delta_q16_total = overhead_total;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixAuditOverheadBudgetChecked(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_enabled,
    I64 bot_dma_log_enabled,
    I64 bot_mmio_log_enabled,
    I64 bot_dispatch_log_enabled,
    I64 max_audit_overhead_delta_q16,
    I64 *out_checked_rows,
    I64 *out_first_violation_index,
    I64 *out_max_observed_overhead_q16)
{
    I64 allow_dispatch;
    I64 reason_code;
    I64 gated_profile_id;
    I32 policy_status;
    I64 i;
    I64 max_observed_q16;

    if (!rows || !out_checked_rows || !out_first_violation_index ||
        !out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_checked_rows = 0;
    *out_first_violation_index = -1;
    *out_max_observed_overhead_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (max_audit_overhead_delta_q16 < 0)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    // Budget checks are part of secure-local policy enforcement.
    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    allow_dispatch = 0;
    reason_code = 0;
    gated_profile_id = 0;

    policy_status = GPUPolicyAllowDispatchChecked(iommu_enabled,
                                                  bot_dma_log_enabled,
                                                  bot_mmio_log_enabled,
                                                  bot_dispatch_log_enabled,
                                                  &allow_dispatch,
                                                  &reason_code,
                                                  &gated_profile_id);
    if (policy_status != 0 || !allow_dispatch ||
        gated_profile_id != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    max_observed_q16 = 0;

    for (i = 0; i < row_count; ++i)
    {
        I64 row_overhead_q16;

        if (rows[i].tok_per_sec_q16 < 0 ||
            rows[i].audit_overhead_delta_q16 < 0 ||
            rows[i].secure_cycles_per_token_q16 < 0)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        row_overhead_q16 = rows[i].audit_overhead_delta_q16;
        if (i == 0 || row_overhead_q16 > max_observed_q16)
            max_observed_q16 = row_overhead_q16;

        *out_checked_rows = i + 1;
        *out_max_observed_overhead_q16 = max_observed_q16;

        // Fail closed on first over-budget row before summary publish.
        if (row_overhead_q16 > max_audit_overhead_delta_q16)
        {
            *out_first_violation_index = i;
            return GPU_SEC_PERF_ERR_POLICY_GUARD;
        }
    }

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedCommitOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_enabled,
    I64 book_hooks,
    I64 max_audit_overhead_delta_q16,
    I64 *out_checked_rows,
    I64 *out_first_violation_index,
    I64 *out_max_observed_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_enabled;
    I64 snapshot_book_hooks;
    I64 snapshot_max_audit_overhead_delta_q16;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_checked_rows_ptr;
    I64 *snapshot_out_first_violation_index_ptr;
    I64 *snapshot_out_max_observed_overhead_q16_ptr;
    I64 staged_checked_rows;
    I64 staged_first_violation_index;
    I64 staged_max_observed_overhead_q16;
    I64 parity_checked_rows;
    I64 parity_first_violation_index;
    I64 parity_max_observed_overhead_q16;
    I32 status_primary;
    I32 status_parity;

    if (!rows || !out_checked_rows || !out_first_violation_index ||
        !out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_checked_rows = 0;
    *out_first_violation_index = -1;
    *out_max_observed_overhead_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (max_audit_overhead_delta_q16 < 0)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfFlagIsBinary(iommu_enabled) ||
        !GPUSecurityPerfFlagIsBinary(book_hooks))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_enabled = iommu_enabled;
    snapshot_book_hooks = book_hooks;
    snapshot_max_audit_overhead_delta_q16 = max_audit_overhead_delta_q16;
    snapshot_rows_ptr = rows;
    snapshot_out_checked_rows_ptr = out_checked_rows;
    snapshot_out_first_violation_index_ptr = out_first_violation_index;
    snapshot_out_max_observed_overhead_q16_ptr = out_max_observed_overhead_q16;

    // IQ-1463 commit-only audit-overhead hardening contract:
    // 1) stage canonical budget tuple from immutable snapshot;
    // 2) recompute tuple from the same immutable snapshot;
    // 3) require exact status + tuple parity before publish;
    // 4) atomically publish tuple to caller slots.
    status_primary = GPUSecurityPerfMatrixAuditOverheadBudgetChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        book_hooks,
        book_hooks,
        max_audit_overhead_delta_q16,
        &staged_checked_rows,
        &staged_first_violation_index,
        &staged_max_observed_overhead_q16);

    status_parity = GPUSecurityPerfMatrixAuditOverheadBudgetChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        book_hooks,
        book_hooks,
        max_audit_overhead_delta_q16,
        &parity_checked_rows,
        &parity_first_violation_index,
        &parity_max_observed_overhead_q16);

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_enabled != iommu_enabled)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_hooks != book_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_max_audit_overhead_delta_q16 != max_audit_overhead_delta_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_checked_rows_ptr != out_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_first_violation_index_ptr != out_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_observed_overhead_q16_ptr != out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != status_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_checked_rows != parity_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_first_violation_index != parity_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_observed_overhead_q16 != parity_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_checked_rows = staged_checked_rows;
    *out_first_violation_index = staged_first_violation_index;
    *out_max_observed_overhead_q16 = staged_max_observed_overhead_q16;

    return status_primary;
}

I32 GPUSecurityPerfMatrixSummaryQ16Checked(const GPUSecurityPerfMatrixRowOutput *rows,
                                           I64 row_count,
                                           I64 rows_capacity,
                                           I64 secure_local_mode,
                                           I64 iommu_enabled,
                                           I64 bot_dma_log_enabled,
                                           I64 bot_mmio_log_enabled,
                                           I64 bot_dispatch_log_enabled,
                                           I64 *out_tok_per_sec_q16,
                                           I64 *out_audit_overhead_q16,
                                           I64 *out_p50_q16,
                                           I64 *out_p95_q16)
{
    I64 allow_dispatch;
    I64 reason_code;
    I64 gated_profile_id;
    I32 policy_status;
    I64 i;
    I64 tok_total;
    I64 overhead_total;
    I64 p50_q16;
    I64 p95_q16;
    I64 row_count_q16;
    I64 overhead_total_q16;
    I32 percentile_status;

    if (!rows || !out_tok_per_sec_q16 || !out_audit_overhead_q16 ||
        !out_p50_q16 || !out_p95_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_tok_per_sec_q16 = 0;
    *out_audit_overhead_q16 = 0;
    *out_p50_q16 = 0;
    *out_p95_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    allow_dispatch = 0;
    reason_code = 0;
    gated_profile_id = 0;
    policy_status = GPUPolicyAllowDispatchChecked(iommu_enabled,
                                                  bot_dma_log_enabled,
                                                  bot_mmio_log_enabled,
                                                  bot_dispatch_log_enabled,
                                                  &allow_dispatch,
                                                  &reason_code,
                                                  &gated_profile_id);
    if (policy_status != 0 || !allow_dispatch ||
        gated_profile_id != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    tok_total = 0;
    overhead_total = 0;

    for (i = 0; i < row_count; ++i)
    {
        I64 next_tok_total;
        I64 next_overhead_total;

        if (rows[i].tok_per_sec_q16 < 0 ||
            rows[i].audit_overhead_delta_q16 < 0 ||
            rows[i].secure_cycles_per_token_q16 < 0)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        if (!GPUSecurityPerfAddChecked(tok_total,
                                       rows[i].tok_per_sec_q16,
                                       &next_tok_total))
            return GPU_SEC_PERF_ERR_OVERFLOW;
        if (!GPUSecurityPerfAddChecked(overhead_total,
                                       rows[i].audit_overhead_delta_q16,
                                       &next_overhead_total))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        tok_total = next_tok_total;
        overhead_total = next_overhead_total;
    }

    if (!GPUSecurityPerfMulChecked(row_count,
                                   GPU_SEC_PERF_Q16_ONE,
                                   &row_count_q16))
        return GPU_SEC_PERF_ERR_OVERFLOW;

    if (row_count_q16 <= 0)
        return GPU_SEC_PERF_ERR_OVERFLOW;

    percentile_status = GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(rows,
                                                                                    row_count,
                                                                                    50,
                                                                                    100,
                                                                                    &p50_q16);
    if (percentile_status != GPU_SEC_PERF_OK)
        return percentile_status;

    percentile_status = GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(rows,
                                                                                    row_count,
                                                                                    95,
                                                                                    100,
                                                                                    &p95_q16);
    if (percentile_status != GPU_SEC_PERF_OK)
        return percentile_status;

    if (!GPUSecurityPerfMulChecked(overhead_total,
                                   GPU_SEC_PERF_Q16_ONE,
                                   &overhead_total_q16))
        return GPU_SEC_PERF_ERR_OVERFLOW;

    *out_tok_per_sec_q16 = tok_total;
    *out_audit_overhead_q16 = overhead_total_q16 / row_count_q16;
    *out_p50_q16 = p50_q16;
    *out_p95_q16 = p95_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedCommitOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_tok_per_sec_q16,
    I64 *out_audit_overhead_q16,
    I64 *out_p50_q16,
    I64 *out_p95_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_tok_ptr;
    I64 *snapshot_out_overhead_ptr;
    I64 *snapshot_out_p50_ptr;
    I64 *snapshot_out_p95_ptr;
    I64 staged_tok_per_sec_q16;
    I64 staged_audit_overhead_q16;
    I64 staged_p50_q16;
    I64 staged_p95_q16;
    I64 parity_tok_per_sec_q16;
    I64 parity_audit_overhead_q16;
    I64 parity_p50_q16;
    I64 parity_p95_q16;
    I32 status_primary;
    I32 status_parity;

    if (!rows || !out_tok_per_sec_q16 || !out_audit_overhead_q16 ||
        !out_p50_q16 || !out_p95_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_tok_per_sec_q16 = 0;
    *out_audit_overhead_q16 = 0;
    *out_p50_q16 = 0;
    *out_p95_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 4)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_tok_ptr = out_tok_per_sec_q16;
    snapshot_out_overhead_ptr = out_audit_overhead_q16;
    snapshot_out_p50_ptr = out_p50_q16;
    snapshot_out_p95_ptr = out_p95_q16;

    // IQ-1457 commit-only summary hardening contract:
    // 1) stage canonical secure-on summary tuple;
    // 2) recompute canonical tuple from immutable snapshots;
    // 3) enforce immutable snapshots and exact tuple parity;
    // 4) atomically publish tuple only after parity succeeds.
    status_primary = GPUSecurityPerfMatrixSummaryQ16Checked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        &staged_tok_per_sec_q16,
        &staged_audit_overhead_q16,
        &staged_p50_q16,
        &staged_p95_q16);
    if (status_primary != GPU_SEC_PERF_OK)
        return status_primary;

    status_parity = GPUSecurityPerfMatrixSummaryQ16Checked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        &parity_tok_per_sec_q16,
        &parity_audit_overhead_q16,
        &parity_p50_q16,
        &parity_p95_q16);
    if (status_parity != GPU_SEC_PERF_OK)
        return status_parity;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_tok_ptr != out_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_overhead_ptr != out_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p50_ptr != out_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p95_ptr != out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (staged_tok_per_sec_q16 != parity_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_audit_overhead_q16 != parity_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p50_q16 != parity_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p95_q16 != parity_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_tok_per_sec_q16 = staged_tok_per_sec_q16;
    *out_audit_overhead_q16 = staged_audit_overhead_q16;
    *out_p50_q16 = staged_p50_q16;
    *out_p95_q16 = staged_p95_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 *out_digest_q64)
{
    I64 i;
    I64 digest_q64;
    I64 factor;
    I64 term;

    if (!rows || !out_digest_q64)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_digest_q64 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    digest_q64 = 1469598103934665603;

    for (i = 0; i < row_count; ++i)
    {
        if (rows[i].tok_per_sec_q16 < 0 ||
            rows[i].audit_overhead_delta_q16 < 0 ||
            rows[i].secure_cycles_per_token_q16 < 0)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        factor = i + 1;
        if (!GPUSecurityPerfMulChecked(rows[i].tok_per_sec_q16,
                                       factor,
                                       &term))
            return GPU_SEC_PERF_ERR_OVERFLOW;
        if (!GPUSecurityPerfAddChecked(digest_q64,
                                       term,
                                       &digest_q64))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfMulChecked(rows[i].audit_overhead_delta_q16,
                                       factor + row_count,
                                       &term))
            return GPU_SEC_PERF_ERR_OVERFLOW;
        if (!GPUSecurityPerfAddChecked(digest_q64,
                                       term,
                                       &digest_q64))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        if (!GPUSecurityPerfMulChecked(rows[i].secure_cycles_per_token_q16,
                                       factor + (row_count << 1),
                                       &term))
            return GPU_SEC_PERF_ERR_OVERFLOW;
        if (!GPUSecurityPerfAddChecked(digest_q64,
                                       term,
                                       &digest_q64))
            return GPU_SEC_PERF_ERR_OVERFLOW;
    }

    *out_digest_q64 = digest_q64;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartial(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_enabled,
    I64 book_hooks,
    I64 max_audit_overhead_delta_q16,
    I64 *out_checked_rows,
    I64 *out_first_violation_index,
    I64 *out_max_observed_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_enabled;
    I64 snapshot_book_hooks;
    I64 snapshot_max_audit_overhead_delta_q16;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_checked_rows_ptr;
    I64 *snapshot_out_first_violation_index_ptr;
    I64 *snapshot_out_max_observed_overhead_q16_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_mid_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 staged_checked_rows;
    I64 staged_first_violation_index;
    I64 staged_max_observed_overhead_q16;
    I64 commit_checked_rows;
    I64 commit_first_violation_index;
    I64 commit_max_observed_overhead_q16;
    I32 status_snapshot;
    I32 status_preflight;
    I32 status_commit;

    if (!rows || !out_checked_rows || !out_first_violation_index ||
        !out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_checked_rows == out_first_violation_index ||
        out_checked_rows == out_max_observed_overhead_q16 ||
        out_first_violation_index == out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (max_audit_overhead_delta_q16 < 0)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfFlagIsBinary(iommu_enabled) ||
        !GPUSecurityPerfFlagIsBinary(book_hooks))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_enabled = iommu_enabled;
    snapshot_book_hooks = book_hooks;
    snapshot_max_audit_overhead_delta_q16 = max_audit_overhead_delta_q16;
    snapshot_rows_ptr = rows;
    snapshot_out_checked_rows_ptr = out_checked_rows;
    snapshot_out_first_violation_index_ptr = out_first_violation_index;
    snapshot_out_max_observed_overhead_q16_ptr = out_max_observed_overhead_q16;

    // IQ-1464 no-partial audit-overhead wrapper:
    // 1) snapshot immutable row digest and argument tuple;
    // 2) stage tuple via commit-only wrapper in preflight pass;
    // 3) enforce unchanged digest + immutable snapshots;
    // 4) recompute via commit-only wrapper and require exact parity;
    // 5) publish outputs atomically only after parity succeeds.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    status_preflight = GPUSecurityPerfMatrixAuditOverheadBudgetCheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
        &staged_checked_rows,
        &staged_first_violation_index,
        &staged_max_observed_overhead_q16);

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_mid_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_mid_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    status_commit = GPUSecurityPerfMatrixAuditOverheadBudgetCheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
        &commit_checked_rows,
        &commit_first_violation_index,
        &commit_max_observed_overhead_q16);

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_enabled != iommu_enabled)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_hooks != book_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_max_audit_overhead_delta_q16 != max_audit_overhead_delta_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_checked_rows_ptr != out_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_first_violation_index_ptr != out_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_observed_overhead_q16_ptr != out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != status_commit)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_checked_rows != commit_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_first_violation_index != commit_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_observed_overhead_q16 != commit_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != GPU_SEC_PERF_OK)
        return status_preflight;

    *out_checked_rows = staged_checked_rows;
    *out_first_violation_index = staged_first_violation_index;
    *out_max_observed_overhead_q16 = staged_max_observed_overhead_q16;

    return status_preflight;
}

I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartialPreflightOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_enabled,
    I64 book_hooks,
    I64 max_audit_overhead_delta_q16,
    I64 *out_checked_rows,
    I64 *out_first_violation_index,
    I64 *out_max_observed_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_enabled;
    I64 snapshot_book_hooks;
    I64 snapshot_max_audit_overhead_delta_q16;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_checked_rows_ptr;
    I64 *snapshot_out_first_violation_index_ptr;
    I64 *snapshot_out_max_observed_overhead_q16_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 saved_checked_rows;
    I64 saved_first_violation_index;
    I64 saved_max_observed_overhead_q16;
    I64 staged_checked_rows;
    I64 staged_first_violation_index;
    I64 staged_max_observed_overhead_q16;
    I64 canonical_checked_rows;
    I64 canonical_first_violation_index;
    I64 canonical_max_observed_overhead_q16;
    I32 status_snapshot;
    I32 status_nopartial;
    I32 status_canonical;

    if (!rows || !out_checked_rows || !out_first_violation_index ||
        !out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_checked_rows == out_first_violation_index ||
        out_checked_rows == out_max_observed_overhead_q16 ||
        out_first_violation_index == out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (max_audit_overhead_delta_q16 < 0)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfFlagIsBinary(iommu_enabled) ||
        !GPUSecurityPerfFlagIsBinary(book_hooks))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    saved_checked_rows = *out_checked_rows;
    saved_first_violation_index = *out_first_violation_index;
    saved_max_observed_overhead_q16 = *out_max_observed_overhead_q16;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_enabled = iommu_enabled;
    snapshot_book_hooks = book_hooks;
    snapshot_max_audit_overhead_delta_q16 = max_audit_overhead_delta_q16;
    snapshot_rows_ptr = rows;
    snapshot_out_checked_rows_ptr = out_checked_rows;
    snapshot_out_first_violation_index_ptr = out_first_violation_index;
    snapshot_out_max_observed_overhead_q16_ptr = out_max_observed_overhead_q16;

    // IQ-1465 preflight-only audit-overhead wrapper:
    // 1) snapshot immutable row digest and caller output slots;
    // 2) compute staged tuple through no-partial wrapper;
    // 3) recompute canonical tuple from audit budget checker;
    // 4) enforce immutable snapshots and exact status + tuple parity;
    // 5) preserve zero-write diagnostics semantics by restoring outputs.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    status_nopartial = GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartial(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
        &staged_checked_rows,
        &staged_first_violation_index,
        &staged_max_observed_overhead_q16);

    status_canonical = GPUSecurityPerfMatrixAuditOverheadBudgetChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        book_hooks,
        book_hooks,
        max_audit_overhead_delta_q16,
        &canonical_checked_rows,
        &canonical_first_violation_index,
        &canonical_max_observed_overhead_q16);

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_enabled != iommu_enabled)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_hooks != book_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_max_audit_overhead_delta_q16 != max_audit_overhead_delta_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_checked_rows_ptr != out_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_first_violation_index_ptr != out_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_observed_overhead_q16_ptr != out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_nopartial != status_canonical)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_checked_rows != canonical_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_first_violation_index != canonical_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_observed_overhead_q16 != canonical_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_checked_rows != saved_checked_rows ||
        *out_first_violation_index != saved_first_violation_index ||
        *out_max_observed_overhead_q16 != saved_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_checked_rows = saved_checked_rows;
    *out_first_violation_index = saved_first_violation_index;
    *out_max_observed_overhead_q16 = saved_max_observed_overhead_q16;

    return status_nopartial;
}

I32 GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartialPreflightOnlyParity(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_enabled,
    I64 book_hooks,
    I64 max_audit_overhead_delta_q16,
    I64 *out_checked_rows,
    I64 *out_first_violation_index,
    I64 *out_max_observed_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_enabled;
    I64 snapshot_book_hooks;
    I64 snapshot_max_audit_overhead_delta_q16;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_checked_rows_ptr;
    I64 *snapshot_out_first_violation_index_ptr;
    I64 *snapshot_out_max_observed_overhead_q16_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 saved_checked_rows;
    I64 saved_first_violation_index;
    I64 saved_max_observed_overhead_q16;
    I64 staged_checked_rows;
    I64 staged_first_violation_index;
    I64 staged_max_observed_overhead_q16;
    I64 parity_checked_rows;
    I64 parity_first_violation_index;
    I64 parity_max_observed_overhead_q16;
    I32 status_snapshot;
    I32 status_preflight_only;
    I32 status_nopartial;

    if (!rows || !out_checked_rows || !out_first_violation_index ||
        !out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_checked_rows == out_first_violation_index ||
        out_checked_rows == out_max_observed_overhead_q16 ||
        out_first_violation_index == out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (max_audit_overhead_delta_q16 < 0)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfFlagIsBinary(iommu_enabled) ||
        !GPUSecurityPerfFlagIsBinary(book_hooks))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    saved_checked_rows = *out_checked_rows;
    saved_first_violation_index = *out_first_violation_index;
    saved_max_observed_overhead_q16 = *out_max_observed_overhead_q16;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_enabled = iommu_enabled;
    snapshot_book_hooks = book_hooks;
    snapshot_max_audit_overhead_delta_q16 = max_audit_overhead_delta_q16;
    snapshot_rows_ptr = rows;
    snapshot_out_checked_rows_ptr = out_checked_rows;
    snapshot_out_first_violation_index_ptr = out_first_violation_index;
    snapshot_out_max_observed_overhead_q16_ptr = out_max_observed_overhead_q16;

    // IQ-1466 preflight-only parity gate:
    // 1) snapshot immutable row digest + caller output slots;
    // 2) compute staged tuple via preflight-only diagnostics wrapper;
    // 3) recompute tuple via no-partial wrapper for strict parity;
    // 4) enforce immutable snapshots + exact status/tuple parity;
    // 5) preserve zero-write diagnostics semantics for caller outputs.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    staged_checked_rows = 0;
    staged_first_violation_index = -1;
    staged_max_observed_overhead_q16 = 0;
    status_preflight_only =
        GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartialPreflightOnly(
            rows,
            row_count,
            rows_capacity,
            secure_local_mode,
            iommu_enabled,
            book_hooks,
            max_audit_overhead_delta_q16,
            &staged_checked_rows,
            &staged_first_violation_index,
            &staged_max_observed_overhead_q16);

    parity_checked_rows = 0;
    parity_first_violation_index = -1;
    parity_max_observed_overhead_q16 = 0;
    status_nopartial = GPUSecurityPerfMatrixAuditOverheadBudgetCheckedNoPartial(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_enabled,
        book_hooks,
        max_audit_overhead_delta_q16,
        &parity_checked_rows,
        &parity_first_violation_index,
        &parity_max_observed_overhead_q16);

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_enabled != iommu_enabled)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_hooks != book_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_max_audit_overhead_delta_q16 != max_audit_overhead_delta_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_checked_rows_ptr != out_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_first_violation_index_ptr != out_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_observed_overhead_q16_ptr != out_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight_only != status_nopartial)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_checked_rows != parity_checked_rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_first_violation_index != parity_first_violation_index)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_observed_overhead_q16 != parity_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_checked_rows != saved_checked_rows ||
        *out_first_violation_index != saved_first_violation_index ||
        *out_max_observed_overhead_q16 != saved_max_observed_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_checked_rows = saved_checked_rows;
    *out_first_violation_index = saved_first_violation_index;
    *out_max_observed_overhead_q16 = saved_max_observed_overhead_q16;

    return status_preflight_only;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedNoPartial(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_tok_per_sec_q16,
    I64 *out_audit_overhead_q16,
    I64 *out_p50_q16,
    I64 *out_p95_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_tok_ptr;
    I64 *snapshot_out_overhead_ptr;
    I64 *snapshot_out_p50_ptr;
    I64 *snapshot_out_p95_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_mid_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 staged_tok_per_sec_q16;
    I64 staged_audit_overhead_q16;
    I64 staged_p50_q16;
    I64 staged_p95_q16;
    I64 commit_tok_per_sec_q16;
    I64 commit_audit_overhead_q16;
    I64 commit_p50_q16;
    I64 commit_p95_q16;
    I32 status_snapshot;
    I32 status_preflight;
    I32 status_commit;

    if (!rows || !out_tok_per_sec_q16 || !out_audit_overhead_q16 ||
        !out_p50_q16 || !out_p95_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_tok_per_sec_q16 == out_audit_overhead_q16 ||
        out_tok_per_sec_q16 == out_p50_q16 ||
        out_tok_per_sec_q16 == out_p95_q16 ||
        out_audit_overhead_q16 == out_p50_q16 ||
        out_audit_overhead_q16 == out_p95_q16 ||
        out_p50_q16 == out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 4)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_tok_ptr = out_tok_per_sec_q16;
    snapshot_out_overhead_ptr = out_audit_overhead_q16;
    snapshot_out_p50_ptr = out_p50_q16;
    snapshot_out_p95_ptr = out_p95_q16;

    // IQ-1458 no-partial summary wrapper:
    // 1) snapshot immutable row digest;
    // 2) run preflight tuple derivation;
    // 3) require unchanged row digest and immutable args;
    // 4) run commit tuple derivation and require exact parity;
    // 5) publish outputs atomically after parity succeeds.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    status_preflight = GPUSecurityPerfMatrixSummaryQ16CheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_tok_per_sec_q16,
        &staged_audit_overhead_q16,
        &staged_p50_q16,
        &staged_p95_q16);
    if (status_preflight != GPU_SEC_PERF_OK)
        return status_preflight;

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_mid_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_mid_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    status_commit = GPUSecurityPerfMatrixSummaryQ16CheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &commit_tok_per_sec_q16,
        &commit_audit_overhead_q16,
        &commit_p50_q16,
        &commit_p95_q16);
    if (status_commit != GPU_SEC_PERF_OK)
        return status_commit;

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_tok_ptr != out_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_overhead_ptr != out_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p50_ptr != out_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p95_ptr != out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (staged_tok_per_sec_q16 != commit_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_audit_overhead_q16 != commit_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p50_q16 != commit_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p95_q16 != commit_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_tok_per_sec_q16 = staged_tok_per_sec_q16;
    *out_audit_overhead_q16 = staged_audit_overhead_q16;
    *out_p50_q16 = staged_p50_q16;
    *out_p95_q16 = staged_p95_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedNoPartialPreflightOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_tok_per_sec_q16,
    I64 *out_audit_overhead_q16,
    I64 *out_p50_q16,
    I64 *out_p95_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_tok_ptr;
    I64 *snapshot_out_overhead_ptr;
    I64 *snapshot_out_p50_ptr;
    I64 *snapshot_out_p95_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 saved_tok_per_sec_q16;
    I64 saved_audit_overhead_q16;
    I64 saved_p50_q16;
    I64 saved_p95_q16;
    I64 staged_tok_per_sec_q16;
    I64 staged_audit_overhead_q16;
    I64 staged_p50_q16;
    I64 staged_p95_q16;
    I64 canonical_tok_per_sec_q16;
    I64 canonical_audit_overhead_q16;
    I64 canonical_p50_q16;
    I64 canonical_p95_q16;
    I32 status_snapshot;
    I32 status_nopartial;
    I32 status_canonical;

    if (!rows || !out_tok_per_sec_q16 || !out_audit_overhead_q16 ||
        !out_p50_q16 || !out_p95_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_tok_per_sec_q16 == out_audit_overhead_q16 ||
        out_tok_per_sec_q16 == out_p50_q16 ||
        out_tok_per_sec_q16 == out_p95_q16 ||
        out_audit_overhead_q16 == out_p50_q16 ||
        out_audit_overhead_q16 == out_p95_q16 ||
        out_p50_q16 == out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 4)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    saved_tok_per_sec_q16 = *out_tok_per_sec_q16;
    saved_audit_overhead_q16 = *out_audit_overhead_q16;
    saved_p50_q16 = *out_p50_q16;
    saved_p95_q16 = *out_p95_q16;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_tok_ptr = out_tok_per_sec_q16;
    snapshot_out_overhead_ptr = out_audit_overhead_q16;
    snapshot_out_p50_ptr = out_p50_q16;
    snapshot_out_p95_ptr = out_p95_q16;

    // IQ-1459 preflight-only summary wrapper:
    // 1) snapshot immutable row digest and output slot values;
    // 2) compute staged tuple via no-partial wrapper and canonical summary;
    // 3) require immutable snapshots plus exact tuple parity;
    // 4) restore caller outputs to preserve zero-write diagnostics semantics.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    status_nopartial = GPUSecurityPerfMatrixSummaryQ16CheckedNoPartial(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_tok_per_sec_q16,
        &staged_audit_overhead_q16,
        &staged_p50_q16,
        &staged_p95_q16);
    if (status_nopartial != GPU_SEC_PERF_OK)
        return status_nopartial;

    status_canonical = GPUSecurityPerfMatrixSummaryQ16Checked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        &canonical_tok_per_sec_q16,
        &canonical_audit_overhead_q16,
        &canonical_p50_q16,
        &canonical_p95_q16);
    if (status_canonical != GPU_SEC_PERF_OK)
        return status_canonical;

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_tok_ptr != out_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_overhead_ptr != out_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p50_ptr != out_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p95_ptr != out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (staged_tok_per_sec_q16 != canonical_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_audit_overhead_q16 != canonical_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p50_q16 != canonical_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p95_q16 != canonical_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_tok_per_sec_q16 != saved_tok_per_sec_q16 ||
        *out_audit_overhead_q16 != saved_audit_overhead_q16 ||
        *out_p50_q16 != saved_p50_q16 ||
        *out_p95_q16 != saved_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_tok_per_sec_q16 = saved_tok_per_sec_q16;
    *out_audit_overhead_q16 = saved_audit_overhead_q16;
    *out_p50_q16 = saved_p50_q16;
    *out_p95_q16 = saved_p95_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedNoPartialPreflightOnlyParity(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_tok_per_sec_q16,
    I64 *out_audit_overhead_q16,
    I64 *out_p50_q16,
    I64 *out_p95_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_tok_ptr;
    I64 *snapshot_out_overhead_ptr;
    I64 *snapshot_out_p50_ptr;
    I64 *snapshot_out_p95_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 saved_tok_per_sec_q16;
    I64 saved_audit_overhead_q16;
    I64 saved_p50_q16;
    I64 saved_p95_q16;
    I64 staged_tok_per_sec_q16;
    I64 staged_audit_overhead_q16;
    I64 staged_p50_q16;
    I64 staged_p95_q16;
    I64 parity_tok_per_sec_q16;
    I64 parity_audit_overhead_q16;
    I64 parity_p50_q16;
    I64 parity_p95_q16;
    I32 status_snapshot;
    I32 status_preflight_only;
    I32 status_nopartial;

    if (!rows || !out_tok_per_sec_q16 || !out_audit_overhead_q16 ||
        !out_p50_q16 || !out_p95_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_tok_per_sec_q16 == out_audit_overhead_q16 ||
        out_tok_per_sec_q16 == out_p50_q16 ||
        out_tok_per_sec_q16 == out_p95_q16 ||
        out_audit_overhead_q16 == out_p50_q16 ||
        out_audit_overhead_q16 == out_p95_q16 ||
        out_p50_q16 == out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 4)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    saved_tok_per_sec_q16 = *out_tok_per_sec_q16;
    saved_audit_overhead_q16 = *out_audit_overhead_q16;
    saved_p50_q16 = *out_p50_q16;
    saved_p95_q16 = *out_p95_q16;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_tok_ptr = out_tok_per_sec_q16;
    snapshot_out_overhead_ptr = out_audit_overhead_q16;
    snapshot_out_p50_ptr = out_p50_q16;
    snapshot_out_p95_ptr = out_p95_q16;

    // IQ-1460 preflight-only parity gate:
    // 1) snapshot immutable row digest and caller outputs;
    // 2) compute staged tuple via preflight-only diagnostics wrapper;
    // 3) independently compute tuple via no-partial wrapper;
    // 4) require immutable snapshots + exact tuple parity;
    // 5) preserve zero-write diagnostics semantics for caller outputs.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    staged_tok_per_sec_q16 = 0;
    staged_audit_overhead_q16 = 0;
    staged_p50_q16 = 0;
    staged_p95_q16 = 0;
    status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedNoPartialPreflightOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_tok_per_sec_q16,
        &staged_audit_overhead_q16,
        &staged_p50_q16,
        &staged_p95_q16);
    if (status_preflight_only != GPU_SEC_PERF_OK)
        return status_preflight_only;

    status_nopartial = GPUSecurityPerfMatrixSummaryQ16CheckedNoPartial(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &parity_tok_per_sec_q16,
        &parity_audit_overhead_q16,
        &parity_p50_q16,
        &parity_p95_q16);
    if (status_nopartial != GPU_SEC_PERF_OK)
        return status_nopartial;

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_tok_ptr != out_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_overhead_ptr != out_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p50_ptr != out_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p95_ptr != out_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (staged_tok_per_sec_q16 != parity_tok_per_sec_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_audit_overhead_q16 != parity_audit_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p50_q16 != parity_p50_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p95_q16 != parity_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_tok_per_sec_q16 != saved_tok_per_sec_q16 ||
        *out_audit_overhead_q16 != saved_audit_overhead_q16 ||
        *out_p50_q16 != saved_p50_q16 ||
        *out_p95_q16 != saved_p95_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_tok_per_sec_q16 = saved_tok_per_sec_q16;
    *out_audit_overhead_q16 = saved_audit_overhead_q16;
    *out_p50_q16 = saved_p50_q16;
    *out_p95_q16 = saved_p95_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_p05_q16,
    I64 *out_p25_q16,
    I64 *out_p75_q16,
    I64 *out_p99_q16)
{
    I64 baseline_tok_per_sec_q16;
    I64 baseline_audit_overhead_q16;
    I64 baseline_p50_q16;
    I64 baseline_p95_q16;
    I32 status_baseline;
    I32 status_percentile;
    I64 staged_p05_q16;
    I64 staged_p25_q16;
    I64 staged_p75_q16;
    I64 staged_p99_q16;

    if (!rows || !out_p05_q16 || !out_p25_q16 ||
        !out_p75_q16 || !out_p99_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_p05_q16 = 0;
    *out_p25_q16 = 0;
    *out_p75_q16 = 0;
    *out_p99_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    // Reuse canonical summary guard path so latency-window collection
    // remains fail-closed on the same secure-local dispatch prerequisites.
    status_baseline = GPUSecurityPerfMatrixSummaryQ16Checked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        &baseline_tok_per_sec_q16,
        &baseline_audit_overhead_q16,
        &baseline_p50_q16,
        &baseline_p95_q16);
    if (status_baseline != GPU_SEC_PERF_OK)
        return status_baseline;

    status_percentile = GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(
        rows,
        row_count,
        5,
        100,
        &staged_p05_q16);
    if (status_percentile != GPU_SEC_PERF_OK)
        return status_percentile;

    status_percentile = GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(
        rows,
        row_count,
        25,
        100,
        &staged_p25_q16);
    if (status_percentile != GPU_SEC_PERF_OK)
        return status_percentile;

    status_percentile = GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(
        rows,
        row_count,
        75,
        100,
        &staged_p75_q16);
    if (status_percentile != GPU_SEC_PERF_OK)
        return status_percentile;

    status_percentile = GPUSecurityPerfMatrixSelectPercentileNearestRankQ16Checked(
        rows,
        row_count,
        99,
        100,
        &staged_p99_q16);
    if (status_percentile != GPU_SEC_PERF_OK)
        return status_percentile;

    *out_p05_q16 = staged_p05_q16;
    *out_p25_q16 = staged_p25_q16;
    *out_p75_q16 = staged_p75_q16;
    *out_p99_q16 = staged_p99_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_p05_q16,
    I64 *out_p25_q16,
    I64 *out_p75_q16,
    I64 *out_p99_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_p05_ptr;
    I64 *snapshot_out_p25_ptr;
    I64 *snapshot_out_p75_ptr;
    I64 *snapshot_out_p99_ptr;
    I64 staged_p05_q16;
    I64 staged_p25_q16;
    I64 staged_p75_q16;
    I64 staged_p99_q16;
    I64 parity_p05_q16;
    I64 parity_p25_q16;
    I64 parity_p75_q16;
    I64 parity_p99_q16;
    I32 status_primary;
    I32 status_parity;

    if (!rows || !out_p05_q16 || !out_p25_q16 || !out_p75_q16 || !out_p99_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_p05_q16 = 0;
    *out_p25_q16 = 0;
    *out_p75_q16 = 0;
    *out_p99_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 4)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_p05_ptr = out_p05_q16;
    snapshot_out_p25_ptr = out_p25_q16;
    snapshot_out_p75_ptr = out_p75_q16;
    snapshot_out_p99_ptr = out_p99_q16;

    // IQ-1474 commit-only latency-window hardening contract:
    // 1) snapshot immutable inputs + output pointer tuple;
    // 2) stage canonical latency-window tuple from immutable snapshots;
    // 3) recompute canonical tuple over same snapshots and require parity;
    // 4) atomically publish tuple only after status + tuple parity.
    status_primary = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_p05_q16,
        &staged_p25_q16,
        &staged_p75_q16,
        &staged_p99_q16);

    status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &parity_p05_q16,
        &parity_p25_q16,
        &parity_p75_q16,
        &parity_p99_q16);

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p05_ptr != out_p05_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p25_ptr != out_p25_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p75_ptr != out_p75_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p99_ptr != out_p99_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != status_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p05_q16 != parity_p05_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p25_q16 != parity_p25_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p75_q16 != parity_p75_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p99_q16 != parity_p99_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != GPU_SEC_PERF_OK)
        return status_primary;

    *out_p05_q16 = staged_p05_q16;
    *out_p25_q16 = staged_p25_q16;
    *out_p75_q16 = staged_p75_q16;
    *out_p99_q16 = staged_p99_q16;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedNoPartial(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_p05_q16,
    I64 *out_p25_q16,
    I64 *out_p75_q16,
    I64 *out_p99_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_p05_ptr;
    I64 *snapshot_out_p25_ptr;
    I64 *snapshot_out_p75_ptr;
    I64 *snapshot_out_p99_ptr;
    I64 staged_p05_q16;
    I64 staged_p25_q16;
    I64 staged_p75_q16;
    I64 staged_p99_q16;
    I64 commit_p05_q16;
    I64 commit_p25_q16;
    I64 commit_p75_q16;
    I64 commit_p99_q16;
    I32 status_preflight;
    I32 status_commit;

    if (!rows || !out_p05_q16 || !out_p25_q16 ||
        !out_p75_q16 || !out_p99_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_p05_q16 == out_p25_q16 ||
        out_p05_q16 == out_p75_q16 ||
        out_p05_q16 == out_p99_q16 ||
        out_p25_q16 == out_p75_q16 ||
        out_p25_q16 == out_p99_q16 ||
        out_p75_q16 == out_p99_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 4)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_p05_ptr = out_p05_q16;
    snapshot_out_p25_ptr = out_p25_q16;
    snapshot_out_p75_ptr = out_p75_q16;
    snapshot_out_p99_ptr = out_p99_q16;

    // IQ-1476 no-partial latency-window wrapper:
    // 1) snapshot immutable inputs + output pointer tuple;
    // 2) run commit-only latency-window preflight staging pass;
    // 3) replay commit-only latency-window pass and enforce parity;
    // 4) atomically publish `{p05_q16,p25_q16,p75_q16,p99_q16}` only on success.
    status_preflight = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_p05_q16,
        &staged_p25_q16,
        &staged_p75_q16,
        &staged_p99_q16);

    status_commit = GPUSecurityPerfMatrixSummaryQ16CheckedLatencyWindowCheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &commit_p05_q16,
        &commit_p25_q16,
        &commit_p75_q16,
        &commit_p99_q16);

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p05_ptr != out_p05_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p25_ptr != out_p25_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p75_ptr != out_p75_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_p99_ptr != out_p99_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != status_commit)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p05_q16 != commit_p05_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p25_q16 != commit_p25_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p75_q16 != commit_p75_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_p99_q16 != commit_p99_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != GPU_SEC_PERF_OK)
        return status_preflight;

    *out_p05_q16 = staged_p05_q16;
    *out_p25_q16 = staged_p25_q16;
    *out_p75_q16 = staged_p75_q16;
    *out_p99_q16 = staged_p99_q16;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeChecked(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_min_overhead_q16,
    I64 *out_max_overhead_q16,
    I64 *out_mean_overhead_q16)
{
    I64 baseline_tok_per_sec_q16;
    I64 baseline_audit_overhead_q16;
    I64 baseline_p50_q16;
    I64 baseline_p95_q16;
    I32 status_baseline;
    I64 i;
    I64 staged_min_overhead_q16;
    I64 staged_max_overhead_q16;
    I64 staged_mean_overhead_q16;
    I64 overhead_sum_q16;
    I64 row_overhead_q16;
    I64 row_secure_cycles_per_token_q16;
    I64 prev_overhead_q16;
    I64 prev_secure_cycles_per_token_q16;

    if (!rows || !out_min_overhead_q16 ||
        !out_max_overhead_q16 || !out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_min_overhead_q16 = 0;
    *out_max_overhead_q16 = 0;
    *out_mean_overhead_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    // Reuse canonical secure-on summary gate path before envelope math.
    status_baseline = GPUSecurityPerfMatrixSummaryQ16Checked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        book_of_truth_gpu_hooks,
        &baseline_tok_per_sec_q16,
        &baseline_audit_overhead_q16,
        &baseline_p50_q16,
        &baseline_p95_q16);
    if (status_baseline != GPU_SEC_PERF_OK)
        return status_baseline;

    overhead_sum_q16 = 0;
    staged_min_overhead_q16 = 0;
    staged_max_overhead_q16 = 0;
    prev_overhead_q16 = 0;
    prev_secure_cycles_per_token_q16 = 0;

    for (i = 0; i < row_count; ++i)
    {
        row_overhead_q16 = rows[i].audit_overhead_delta_q16;
        row_secure_cycles_per_token_q16 = rows[i].secure_cycles_per_token_q16;

        if (rows[i].tok_per_sec_q16 < 0 ||
            row_overhead_q16 < 0 ||
            row_secure_cycles_per_token_q16 < 0)
            return GPU_SEC_PERF_ERR_BAD_PARAM;

        // Monotonic secure-vs-relaxed invariant:
        // when secure cycle cost rises, audit overhead may not shrink.
        if (i > 0)
        {
            if (row_secure_cycles_per_token_q16 > prev_secure_cycles_per_token_q16 &&
                row_overhead_q16 < prev_overhead_q16)
                return GPU_SEC_PERF_ERR_POLICY_GUARD;
        }

        if (i == 0 || row_overhead_q16 < staged_min_overhead_q16)
            staged_min_overhead_q16 = row_overhead_q16;
        if (i == 0 || row_overhead_q16 > staged_max_overhead_q16)
            staged_max_overhead_q16 = row_overhead_q16;

        if (!GPUSecurityPerfAddChecked(overhead_sum_q16,
                                       row_overhead_q16,
                                       &overhead_sum_q16))
            return GPU_SEC_PERF_ERR_OVERFLOW;

        prev_overhead_q16 = row_overhead_q16;
        prev_secure_cycles_per_token_q16 = row_secure_cycles_per_token_q16;
    }

    staged_mean_overhead_q16 = overhead_sum_q16 / row_count;

    *out_min_overhead_q16 = staged_min_overhead_q16;
    *out_max_overhead_q16 = staged_max_overhead_q16;
    *out_mean_overhead_q16 = staged_mean_overhead_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_min_overhead_q16,
    I64 *out_max_overhead_q16,
    I64 *out_mean_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_min_ptr;
    I64 *snapshot_out_max_ptr;
    I64 *snapshot_out_mean_ptr;
    I64 staged_min_overhead_q16;
    I64 staged_max_overhead_q16;
    I64 staged_mean_overhead_q16;
    I64 parity_min_overhead_q16;
    I64 parity_max_overhead_q16;
    I64 parity_mean_overhead_q16;
    I32 status_primary;
    I32 status_parity;

    if (!rows || !out_min_overhead_q16 ||
        !out_max_overhead_q16 || !out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    *out_min_overhead_q16 = 0;
    *out_max_overhead_q16 = 0;
    *out_mean_overhead_q16 = 0;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 3)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_min_ptr = out_min_overhead_q16;
    snapshot_out_max_ptr = out_max_overhead_q16;
    snapshot_out_mean_ptr = out_mean_overhead_q16;

    // IQ-1477 commit-only overhead-envelope hardening contract:
    // 1) snapshot immutable inputs + output pointer tuple;
    // 2) stage canonical envelope tuple from immutable snapshots;
    // 3) recompute canonical envelope tuple and require strict parity;
    // 4) atomically publish `{min,max,mean}` only after status + tuple parity.
    status_primary = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_min_overhead_q16,
        &staged_max_overhead_q16,
        &staged_mean_overhead_q16);

    status_parity = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &parity_min_overhead_q16,
        &parity_max_overhead_q16,
        &parity_mean_overhead_q16);

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_min_ptr != out_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_ptr != out_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_mean_ptr != out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != status_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_min_overhead_q16 != parity_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_overhead_q16 != parity_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_mean_overhead_q16 != parity_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_primary != GPU_SEC_PERF_OK)
        return status_primary;

    *out_min_overhead_q16 = staged_min_overhead_q16;
    *out_max_overhead_q16 = staged_max_overhead_q16;
    *out_mean_overhead_q16 = staged_mean_overhead_q16;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartial(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_min_overhead_q16,
    I64 *out_max_overhead_q16,
    I64 *out_mean_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_min_ptr;
    I64 *snapshot_out_max_ptr;
    I64 *snapshot_out_mean_ptr;
    I64 staged_min_overhead_q16;
    I64 staged_max_overhead_q16;
    I64 staged_mean_overhead_q16;
    I64 commit_min_overhead_q16;
    I64 commit_max_overhead_q16;
    I64 commit_mean_overhead_q16;
    I32 status_preflight;
    I32 status_commit;

    if (!rows || !out_min_overhead_q16 ||
        !out_max_overhead_q16 || !out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_min_overhead_q16 == out_max_overhead_q16 ||
        out_min_overhead_q16 == out_mean_overhead_q16 ||
        out_max_overhead_q16 == out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_min_ptr = out_min_overhead_q16;
    snapshot_out_max_ptr = out_max_overhead_q16;
    snapshot_out_mean_ptr = out_mean_overhead_q16;

    // IQ-1478 no-partial overhead-envelope wrapper:
    // 1) snapshot immutable inputs + output pointer tuple;
    // 2) derive staged tuple via commit-only wrapper;
    // 3) re-run commit-only wrapper and require strict status + tuple parity;
    // 4) publish `{min,max,mean}` atomically only after full parity.
    status_preflight = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_min_overhead_q16,
        &staged_max_overhead_q16,
        &staged_mean_overhead_q16);

    status_commit = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedCommitOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &commit_min_overhead_q16,
        &commit_max_overhead_q16,
        &commit_mean_overhead_q16);

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_min_ptr != out_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_ptr != out_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_mean_ptr != out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != status_commit)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_min_overhead_q16 != commit_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_overhead_q16 != commit_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_mean_overhead_q16 != commit_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (status_preflight != GPU_SEC_PERF_OK)
        return status_preflight;

    *out_min_overhead_q16 = staged_min_overhead_q16;
    *out_max_overhead_q16 = staged_max_overhead_q16;
    *out_mean_overhead_q16 = staged_mean_overhead_q16;
    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialPreflightOnly(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_min_overhead_q16,
    I64 *out_max_overhead_q16,
    I64 *out_mean_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_min_ptr;
    I64 *snapshot_out_max_ptr;
    I64 *snapshot_out_mean_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 saved_min_overhead_q16;
    I64 saved_max_overhead_q16;
    I64 saved_mean_overhead_q16;
    I64 staged_min_overhead_q16;
    I64 staged_max_overhead_q16;
    I64 staged_mean_overhead_q16;
    I64 canonical_min_overhead_q16;
    I64 canonical_max_overhead_q16;
    I64 canonical_mean_overhead_q16;
    I32 status_snapshot;
    I32 status_nopartial;
    I32 status_canonical;

    if (!rows || !out_min_overhead_q16 ||
        !out_max_overhead_q16 || !out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_min_overhead_q16 == out_max_overhead_q16 ||
        out_min_overhead_q16 == out_mean_overhead_q16 ||
        out_max_overhead_q16 == out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 3)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    saved_min_overhead_q16 = *out_min_overhead_q16;
    saved_max_overhead_q16 = *out_max_overhead_q16;
    saved_mean_overhead_q16 = *out_mean_overhead_q16;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_min_ptr = out_min_overhead_q16;
    snapshot_out_max_ptr = out_max_overhead_q16;
    snapshot_out_mean_ptr = out_mean_overhead_q16;

    // IQ-1479 preflight-only overhead-envelope wrapper:
    // 1) snapshot immutable row digest and caller output slots;
    // 2) compute staged tuple via no-partial wrapper and canonical envelope;
    // 3) require immutable snapshots and strict tuple parity;
    // 4) restore caller outputs to preserve zero-write diagnostics semantics.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    status_nopartial = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartial(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_min_overhead_q16,
        &staged_max_overhead_q16,
        &staged_mean_overhead_q16);
    if (status_nopartial != GPU_SEC_PERF_OK)
        return status_nopartial;

    status_canonical = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeChecked(
        rows,
        row_count,
        rows_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &canonical_min_overhead_q16,
        &canonical_max_overhead_q16,
        &canonical_mean_overhead_q16);
    if (status_canonical != GPU_SEC_PERF_OK)
        return status_canonical;

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_min_ptr != out_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_ptr != out_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_mean_ptr != out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (staged_min_overhead_q16 != canonical_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_overhead_q16 != canonical_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_mean_overhead_q16 != canonical_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_min_overhead_q16 != saved_min_overhead_q16 ||
        *out_max_overhead_q16 != saved_max_overhead_q16 ||
        *out_mean_overhead_q16 != saved_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_min_overhead_q16 = saved_min_overhead_q16;
    *out_max_overhead_q16 = saved_max_overhead_q16;
    *out_mean_overhead_q16 = saved_mean_overhead_q16;

    return GPU_SEC_PERF_OK;
}

I32 GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialPreflightOnlyParity(
    const GPUSecurityPerfMatrixRowOutput *rows,
    I64 row_count,
    I64 rows_capacity,
    I64 out_capacity,
    I64 secure_local_mode,
    I64 iommu_active,
    I64 book_of_truth_gpu_hooks,
    I64 policy_digest_parity,
    I64 *out_min_overhead_q16,
    I64 *out_max_overhead_q16,
    I64 *out_mean_overhead_q16)
{
    I64 snapshot_row_count;
    I64 snapshot_rows_capacity;
    I64 snapshot_out_capacity;
    I64 snapshot_secure_local_mode;
    I64 snapshot_iommu_active;
    I64 snapshot_book_of_truth_gpu_hooks;
    I64 snapshot_policy_digest_parity;
    const GPUSecurityPerfMatrixRowOutput *snapshot_rows_ptr;
    I64 *snapshot_out_min_ptr;
    I64 *snapshot_out_max_ptr;
    I64 *snapshot_out_mean_ptr;
    I64 snapshot_before_digest_q64;
    I64 snapshot_after_digest_q64;
    I64 saved_min_overhead_q16;
    I64 saved_max_overhead_q16;
    I64 saved_mean_overhead_q16;
    I64 staged_min_overhead_q16;
    I64 staged_max_overhead_q16;
    I64 staged_mean_overhead_q16;
    I64 parity_min_overhead_q16;
    I64 parity_max_overhead_q16;
    I64 parity_mean_overhead_q16;
    I32 status_snapshot;
    I32 status_preflight_only;
    I32 status_nopartial;

    if (!rows || !out_min_overhead_q16 ||
        !out_max_overhead_q16 || !out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_NULL_PTR;

    if (out_min_overhead_q16 == out_max_overhead_q16 ||
        out_min_overhead_q16 == out_mean_overhead_q16 ||
        out_max_overhead_q16 == out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (row_count <= 0 || rows_capacity < row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (out_capacity < 3)
        return GPU_SEC_PERF_ERR_CAPACITY;

    if (!GPUSecurityPerfFlagIsBinary(iommu_active) ||
        !GPUSecurityPerfFlagIsBinary(book_of_truth_gpu_hooks) ||
        !GPUSecurityPerfFlagIsBinary(policy_digest_parity))
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (secure_local_mode != GPU_SEC_PERF_PROFILE_SECURE_LOCAL)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    if (!policy_digest_parity)
        return GPU_SEC_PERF_ERR_POLICY_GUARD;

    saved_min_overhead_q16 = *out_min_overhead_q16;
    saved_max_overhead_q16 = *out_max_overhead_q16;
    saved_mean_overhead_q16 = *out_mean_overhead_q16;

    snapshot_row_count = row_count;
    snapshot_rows_capacity = rows_capacity;
    snapshot_out_capacity = out_capacity;
    snapshot_secure_local_mode = secure_local_mode;
    snapshot_iommu_active = iommu_active;
    snapshot_book_of_truth_gpu_hooks = book_of_truth_gpu_hooks;
    snapshot_policy_digest_parity = policy_digest_parity;
    snapshot_rows_ptr = rows;
    snapshot_out_min_ptr = out_min_overhead_q16;
    snapshot_out_max_ptr = out_max_overhead_q16;
    snapshot_out_mean_ptr = out_mean_overhead_q16;

    // IQ-1480 strict preflight-only parity gate:
    // 1) snapshot immutable row digest and caller output slots;
    // 2) compute staged tuple via no-partial preflight-only diagnostics wrapper;
    // 3) independently compute tuple via no-partial wrapper;
    // 4) require immutable snapshots + exact tuple parity;
    // 5) preserve zero-write diagnostics semantics for caller outputs.
    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_before_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;

    status_preflight_only = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartialPreflightOnly(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &staged_min_overhead_q16,
        &staged_max_overhead_q16,
        &staged_mean_overhead_q16);
    if (status_preflight_only != GPU_SEC_PERF_OK)
        return status_preflight_only;

    status_nopartial = GPUSecurityPerfMatrixSummaryQ16CheckedOverheadEnvelopeCheckedNoPartial(
        rows,
        row_count,
        rows_capacity,
        out_capacity,
        secure_local_mode,
        iommu_active,
        book_of_truth_gpu_hooks,
        policy_digest_parity,
        &parity_min_overhead_q16,
        &parity_max_overhead_q16,
        &parity_mean_overhead_q16);
    if (status_nopartial != GPU_SEC_PERF_OK)
        return status_nopartial;

    status_snapshot = GPUSecurityPerfMatrixRowsSnapshotDigestQ64Checked(
        rows,
        row_count,
        rows_capacity,
        &snapshot_after_digest_q64);
    if (status_snapshot != GPU_SEC_PERF_OK)
        return status_snapshot;
    if (snapshot_after_digest_q64 != snapshot_before_digest_q64)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (snapshot_row_count != row_count)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_capacity != rows_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_capacity != out_capacity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_secure_local_mode != secure_local_mode)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_iommu_active != iommu_active)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_book_of_truth_gpu_hooks != book_of_truth_gpu_hooks)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_policy_digest_parity != policy_digest_parity)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_rows_ptr != rows)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_min_ptr != out_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_max_ptr != out_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (snapshot_out_mean_ptr != out_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (staged_min_overhead_q16 != parity_min_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_max_overhead_q16 != parity_max_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;
    if (staged_mean_overhead_q16 != parity_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    if (*out_min_overhead_q16 != saved_min_overhead_q16 ||
        *out_max_overhead_q16 != saved_max_overhead_q16 ||
        *out_mean_overhead_q16 != saved_mean_overhead_q16)
        return GPU_SEC_PERF_ERR_BAD_PARAM;

    *out_min_overhead_q16 = saved_min_overhead_q16;
    *out_max_overhead_q16 = saved_max_overhead_q16;
    *out_mean_overhead_q16 = saved_mean_overhead_q16;

    return GPU_SEC_PERF_OK;
}
