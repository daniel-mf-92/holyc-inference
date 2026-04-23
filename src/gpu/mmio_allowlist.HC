// mmio_allowlist.HC
// GPU BAR/MMIO write allowlist with deny-by-default policy.
//
// Security contract:
// - Writes are denied unless a BAR+register range+width tuple matches.
// - Table validation rejects malformed/overlapping windows.
// - Integer-only checked math; no implicit wrap or partial publish.

#define GPU_MMIO_ALLOW_OK                    0
#define GPU_MMIO_ALLOW_ERR_NULL_PTR          1
#define GPU_MMIO_ALLOW_ERR_BAD_PARAM         2
#define GPU_MMIO_ALLOW_ERR_BAD_TABLE         3
#define GPU_MMIO_ALLOW_ERR_OVERLAP           4

#define GPU_MMIO_ALLOW_REASON_ALLOW                 0
#define GPU_MMIO_ALLOW_REASON_DENY_DEFAULT          1
#define GPU_MMIO_ALLOW_REASON_BAD_BAR               2
#define GPU_MMIO_ALLOW_REASON_BAD_OFFSET            3
#define GPU_MMIO_ALLOW_REASON_BAD_WIDTH             4
#define GPU_MMIO_ALLOW_REASON_BAD_TABLE             5
#define GPU_MMIO_ALLOW_REASON_TABLE_OVERLAP         6

#define GPU_MMIO_WIDTH_MASK_1B  1
#define GPU_MMIO_WIDTH_MASK_2B  2
#define GPU_MMIO_WIDTH_MASK_4B  4
#define GPU_MMIO_WIDTH_MASK_8B  8

class GPUMMIOAllowEntry
{
    I64 bar_index;
    I64 reg_start;
    I64 reg_end;
    I64 width_mask;
};

Bool GPUMMIOWidthBytesIsValid(I64 width_bytes)
{
    return (width_bytes == 1 || width_bytes == 2 || width_bytes == 4 || width_bytes == 8);
}

I64 GPUMMIOWidthMaskFromBytes(I64 width_bytes)
{
    if (width_bytes == 1)
        return GPU_MMIO_WIDTH_MASK_1B;
    if (width_bytes == 2)
        return GPU_MMIO_WIDTH_MASK_2B;
    if (width_bytes == 4)
        return GPU_MMIO_WIDTH_MASK_4B;
    if (width_bytes == 8)
        return GPU_MMIO_WIDTH_MASK_8B;
    return 0;
}

Bool GPUMMIOWidthMaskIsValid(I64 width_mask)
{
    I64 known_mask;

    known_mask = GPU_MMIO_WIDTH_MASK_1B |
                 GPU_MMIO_WIDTH_MASK_2B |
                 GPU_MMIO_WIDTH_MASK_4B |
                 GPU_MMIO_WIDTH_MASK_8B;

    if (width_mask <= 0)
        return FALSE;

    return ((width_mask & ~known_mask) == 0);
}

Bool GPUMMIOEntryOverlap(const GPUMMIOAllowEntry *lhs,
                         const GPUMMIOAllowEntry *rhs)
{
    if (lhs->bar_index != rhs->bar_index)
        return FALSE;

    // Closed interval overlap check:
    // [lhs.start, lhs.end] intersects [rhs.start, rhs.end]
    // iff lhs.start <= rhs.end && rhs.start <= lhs.end.
    if (lhs->reg_start <= rhs->reg_end && rhs->reg_start <= lhs->reg_end)
        return TRUE;

    return FALSE;
}

I32 GPUMMIOAllowlistValidateTableChecked(const GPUMMIOAllowEntry *table,
                                         I64 table_count)
{
    I64 i;
    I64 j;

    if (table_count < 0)
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM;

    if (table_count == 0)
        return GPU_MMIO_ALLOW_OK;

    if (!table)
        return GPU_MMIO_ALLOW_ERR_NULL_PTR;

    for (i = 0; i < table_count; ++i)
    {
        if (table[i].bar_index < 0)
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE;

        if (table[i].reg_start < 0 || table[i].reg_end < 0)
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE;

        if (table[i].reg_end < table[i].reg_start)
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE;

        if (!GPUMMIOWidthMaskIsValid(table[i].width_mask))
            return GPU_MMIO_ALLOW_ERR_BAD_TABLE;
    }

    for (i = 0; i < table_count; ++i)
    {
        for (j = i + 1; j < table_count; ++j)
        {
            if (GPUMMIOEntryOverlap(&table[i], &table[j]))
                return GPU_MMIO_ALLOW_ERR_OVERLAP;
        }
    }

    return GPU_MMIO_ALLOW_OK;
}

I32 GPUMMIOAllowlistCheckWriteChecked(const GPUMMIOAllowEntry *table,
                                      I64 table_count,
                                      I64 bar_index,
                                      I64 reg_offset,
                                      I64 width_bytes,
                                      I64 *out_allow,
                                      I64 *out_reason,
                                      I64 *out_match_index)
{
    I32 table_status;
    I64 i;
    I64 width_mask;
    I64 bar_seen;

    if (!out_allow || !out_reason || !out_match_index)
        return GPU_MMIO_ALLOW_ERR_NULL_PTR;

    *out_allow = 0;
    *out_reason = GPU_MMIO_ALLOW_REASON_DENY_DEFAULT;
    *out_match_index = -1;

    if (bar_index < 0)
    {
        *out_reason = GPU_MMIO_ALLOW_REASON_BAD_BAR;
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM;
    }

    if (reg_offset < 0)
    {
        *out_reason = GPU_MMIO_ALLOW_REASON_BAD_OFFSET;
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM;
    }

    if (!GPUMMIOWidthBytesIsValid(width_bytes))
    {
        *out_reason = GPU_MMIO_ALLOW_REASON_BAD_WIDTH;
        return GPU_MMIO_ALLOW_ERR_BAD_PARAM;
    }

    table_status = GPUMMIOAllowlistValidateTableChecked(table, table_count);
    if (table_status == GPU_MMIO_ALLOW_ERR_OVERLAP)
    {
        *out_reason = GPU_MMIO_ALLOW_REASON_TABLE_OVERLAP;
        return table_status;
    }
    if (table_status != GPU_MMIO_ALLOW_OK)
    {
        *out_reason = GPU_MMIO_ALLOW_REASON_BAD_TABLE;
        return table_status;
    }

    width_mask = GPUMMIOWidthMaskFromBytes(width_bytes);
    bar_seen = 0;

    for (i = 0; i < table_count; ++i)
    {
        if (table[i].bar_index != bar_index)
            continue;

        bar_seen = 1;

        if (reg_offset < table[i].reg_start || reg_offset > table[i].reg_end)
            continue;

        if ((table[i].width_mask & width_mask) == 0)
        {
            *out_reason = GPU_MMIO_ALLOW_REASON_BAD_WIDTH;
            return GPU_MMIO_ALLOW_OK;
        }

        *out_allow = 1;
        *out_reason = GPU_MMIO_ALLOW_REASON_ALLOW;
        *out_match_index = i;
        return GPU_MMIO_ALLOW_OK;
    }

    if (!bar_seen)
        *out_reason = GPU_MMIO_ALLOW_REASON_BAD_BAR;

    return GPU_MMIO_ALLOW_OK;
}

