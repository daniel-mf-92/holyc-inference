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
