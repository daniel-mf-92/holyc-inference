// model_info.HC
// GGUF model-info validation helpers for HolyC parser tooling.

#define GGUF_INFO_OK 0
#define GGUF_INFO_ERR_NULL_PTR 1
#define GGUF_INFO_ERR_BAD_PARAM 2
#define GGUF_INFO_ERR_OVERFLOW 3
#define GGUF_INFO_ERR_OUT_OF_BOUNDS 4
#define GGUF_INFO_ERR_OVERLAP 5

class GGUFInfoRow
{
    U64 rel_offset;
    U64 byte_count;
    U64 abs_start;
    U64 abs_end;
};

Bool GGUFInfoTryAddU64(U64 lhs, U64 rhs, U64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (lhs > (0xFFFFFFFFFFFFFFFF - rhs))
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

// Build validated tensor rows suitable for a human-readable GGUF summary tool.
//
// Contract:
// - input offsets are relative to tensor_data_base
// - tensors must be sorted and non-overlapping in payload space
// - no-partial writes: rows/summary outputs are committed only after preflight
I64 GGUFModelInfoBuildCheckedNoPartial(
    U64 tensor_data_base,
    U64 file_nbytes,
    U64 tensor_count,
    U64 *tensor_offsets,
    U64 *tensor_sizes,
    GGUFInfoRow *out_rows,
    U64 out_row_capacity,
    U64 *out_total_payload,
    U64 *out_last_end)
{
    U64 tensor_i;
    U64 rel_offset;
    U64 byte_count;
    U64 abs_start;
    U64 abs_end;
    U64 prev_end;
    U64 staged_total_payload;
    U64 staged_last_end;

    if (!tensor_offsets || !tensor_sizes || !out_rows ||
        !out_total_payload || !out_last_end)
        return GGUF_INFO_ERR_NULL_PTR;

    if (out_row_capacity < tensor_count)
        return GGUF_INFO_ERR_BAD_PARAM;

    if (tensor_data_base > file_nbytes)
        return GGUF_INFO_ERR_BAD_PARAM;

    prev_end = tensor_data_base;
    staged_total_payload = 0;
    staged_last_end = tensor_data_base;

    // Full preflight pass before any output writes.
    for (tensor_i = 0; tensor_i < tensor_count; tensor_i++)
    {
        rel_offset = tensor_offsets[tensor_i];
        byte_count = tensor_sizes[tensor_i];

        if (!GGUFInfoTryAddU64(tensor_data_base,
                               rel_offset,
                               &abs_start))
            return GGUF_INFO_ERR_OVERFLOW;

        if (abs_start < prev_end)
            return GGUF_INFO_ERR_OVERLAP;

        if (!GGUFInfoTryAddU64(abs_start,
                               byte_count,
                               &abs_end))
            return GGUF_INFO_ERR_OVERFLOW;

        if (abs_end > file_nbytes)
            return GGUF_INFO_ERR_OUT_OF_BOUNDS;

        if (!GGUFInfoTryAddU64(staged_total_payload,
                               byte_count,
                               &staged_total_payload))
            return GGUF_INFO_ERR_OVERFLOW;

        prev_end = abs_end;
        staged_last_end = abs_end;
    }

    // Commit rows only after all validation passes.
    for (tensor_i = 0; tensor_i < tensor_count; tensor_i++)
    {
        rel_offset = tensor_offsets[tensor_i];
        byte_count = tensor_sizes[tensor_i];

        GGUFInfoTryAddU64(tensor_data_base,
                          rel_offset,
                          &abs_start);
        GGUFInfoTryAddU64(abs_start,
                          byte_count,
                          &abs_end);

        out_rows[tensor_i].rel_offset = rel_offset;
        out_rows[tensor_i].byte_count = byte_count;
        out_rows[tensor_i].abs_start = abs_start;
        out_rows[tensor_i].abs_end = abs_end;
    }

    *out_total_payload = staged_total_payload;
    *out_last_end = staged_last_end;
    return GGUF_INFO_OK;
}
