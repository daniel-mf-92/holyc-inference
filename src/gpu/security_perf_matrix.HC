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

Bool GPUSecurityPerfIsSupportedQuant(I64 quant_level)
{
    return (quant_level == GPU_SEC_PERF_QUANT_Q4_0 ||
            quant_level == GPU_SEC_PERF_QUANT_Q8_0);
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
