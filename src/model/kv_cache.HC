// kv_cache.HC
// Integer-only KV-cache addressing helpers for TempleOS inference.
//
// IQ-871: checked layer/token base/span index derivation for Q16 KV slabs.

#define KV_Q16_OK            0
#define KV_Q16_ERR_NULL_PTR  1
#define KV_Q16_ERR_BAD_PARAM 2
#define KV_Q16_ERR_OVERFLOW  4

#define KV_I64_MAX_VALUE  0x7FFFFFFFFFFFFFFF
#define KV_I64_MIN_VALUE -0x8000000000000000

I32 KVTryAddI64Checked(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return KV_Q16_ERR_NULL_PTR;

    if (rhs > 0 && lhs > KV_I64_MAX_VALUE - rhs)
        return KV_Q16_ERR_OVERFLOW;
    if (rhs < 0 && lhs < KV_I64_MIN_VALUE - rhs)
        return KV_Q16_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return KV_Q16_OK;
}

I32 KVTryMulI64Checked(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return KV_Q16_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return KV_Q16_OK;
    }

    if (lhs > 0)
    {
        if (rhs > 0)
        {
            if (lhs > (KV_I64_MAX_VALUE / rhs))
                return KV_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (rhs < (KV_I64_MIN_VALUE / lhs))
                return KV_Q16_ERR_OVERFLOW;
        }
    }
    else
    {
        if (rhs > 0)
        {
            if (lhs < (KV_I64_MIN_VALUE / rhs))
                return KV_Q16_ERR_OVERFLOW;
        }
        else
        {
            if (lhs != 0 && rhs < (KV_I64_MAX_VALUE / lhs))
                return KV_Q16_ERR_OVERFLOW;
        }
    }

    *out_prod = lhs * rhs;
    return KV_Q16_OK;
}

// Span in Q16 cells for one token in one layer:
// span_cells = kv_heads * head_dim
I32 KVCacheQ16ComputeLayerTokenSpanCellsChecked(
    I64 kv_heads,
    I64 head_dim,
    I64 *out_span_cells)
{
    if (!out_span_cells)
        return KV_Q16_ERR_NULL_PTR;

    if (kv_heads < 0 || head_dim < 0)
        return KV_Q16_ERR_BAD_PARAM;

    return KVTryMulI64Checked(kv_heads,
                              head_dim,
                              out_span_cells);
}

// Total Q16 cells needed for one cache slab (K or V):
// total_cells = layer_count * token_capacity * kv_heads * head_dim
// The computation is staged through checked helpers to keep overflow
// classification deterministic and auditable.
I32 KVCacheQ16ComputeTotalCellsChecked(
    I64 layer_count,
    I64 token_capacity,
    I64 kv_heads,
    I64 head_dim,
    I64 *out_total_cells)
{
    I64 token_span_cells;
    I64 layer_span_cells;
    I32 status;

    if (!out_total_cells)
        return KV_Q16_ERR_NULL_PTR;

    if (layer_count < 0 || token_capacity < 0 ||
        kv_heads < 0 || head_dim < 0)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16ComputeLayerTokenSpanCellsChecked(kv_heads,
                                                         head_dim,
                                                         &token_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(token_capacity,
                                token_span_cells,
                                &layer_span_cells);
    if (status != KV_Q16_OK)
        return status;

    return KVTryMulI64Checked(layer_count,
                              layer_span_cells,
                              out_total_cells);
}

// Base index in Q16 cells for one token inside one layer slab:
// layer_span_cells = token_capacity * (kv_heads * head_dim)
// token_base_in_layer = token_idx * (kv_heads * head_dim)
// base_index = layer_idx * layer_span_cells + token_base_in_layer
I32 KVCacheQ16ComputeLayerTokenBaseIndexChecked(
    I64 layer_idx,
    I64 token_idx,
    I64 layer_count,
    I64 token_capacity,
    I64 kv_heads,
    I64 head_dim,
    I64 *out_base_index,
    I64 *out_span_cells)
{
    I64 span_cells;
    I64 layer_span_cells;
    I64 layer_base_cells;
    I64 token_base_cells;
    I32 status;

    if (!out_base_index || !out_span_cells)
        return KV_Q16_ERR_NULL_PTR;
    if (out_base_index == out_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    if (layer_idx < 0 || token_idx < 0)
        return KV_Q16_ERR_BAD_PARAM;
    if (layer_count < 0 || token_capacity < 0)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16ComputeLayerTokenSpanCellsChecked(kv_heads,
                                                         head_dim,
                                                         &span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (layer_idx >= layer_count)
        return KV_Q16_ERR_BAD_PARAM;
    if (token_idx >= token_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryMulI64Checked(token_capacity,
                                span_cells,
                                &layer_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(layer_idx,
                                layer_span_cells,
                                &layer_base_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(token_idx,
                                span_cells,
                                &token_base_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(layer_base_cells,
                                token_base_cells,
                                out_base_index);
    if (status != KV_Q16_OK)
        return status;

    *out_span_cells = span_cells;
    return KV_Q16_OK;
}

// Copy one token's K/V slices into cache slabs with strict no-partial writes.
// Layout per layer:
//   [token0 span][token1 span]...[tokenN span]
// where span = kv_heads * head_dim Q16 cells.
I32 KVCacheQ16WriteTokenCheckedNoPartial(
    I64 *k_cache_q16,
    I64 k_cache_capacity,
    I64 *v_cache_q16,
    I64 v_cache_capacity,
    I64 layer_idx,
    I64 token_idx,
    I64 layer_count,
    I64 token_capacity,
    I64 kv_heads,
    I64 head_dim,
    I64 *k_token_src_q16,
    I64 k_token_src_capacity,
    I64 *v_token_src_q16,
    I64 v_token_src_capacity)
{
    I64 k_base_index;
    I64 k_span_cells;
    I64 v_base_index;
    I64 v_span_cells;
    I64 k_end_index;
    I64 v_end_index;
    I64 cell_idx;
    I32 status;

    if (!k_cache_q16 || !v_cache_q16 || !k_token_src_q16 || !v_token_src_q16)
        return KV_Q16_ERR_NULL_PTR;

    if (k_cache_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_cache_capacity < 0 || v_cache_capacity < 0 ||
        k_token_src_capacity < 0 || v_token_src_capacity < 0)
        return KV_Q16_ERR_BAD_PARAM;

    if (layer_idx < 0 || token_idx < 0 ||
        layer_count < 0 || token_capacity < 0 ||
        kv_heads < 0 || head_dim < 0)
        return KV_Q16_ERR_BAD_PARAM;

    // Keep read-only source windows distinct from cache slabs so commit phase is
    // deterministic and cannot self-overwrite under aliasing.
    if (k_token_src_q16 == k_cache_q16 || k_token_src_q16 == v_cache_q16 ||
        v_token_src_q16 == k_cache_q16 || v_token_src_q16 == v_cache_q16 ||
        k_token_src_q16 == v_token_src_q16)
        return KV_Q16_ERR_BAD_PARAM;

    // Resolve destination geometry through canonical checked helpers.
    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &k_base_index,
                                                         &k_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &v_base_index,
                                                         &v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (k_base_index != v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (k_span_cells != v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(k_base_index, k_span_cells, &k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(v_base_index, v_span_cells, &v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (k_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (v_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_span_cells > k_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (v_span_cells > v_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    // All checks are complete above; copy is now an atomic commit region with
    // no remaining failure branches and therefore no partial-write exits.
    cell_idx = 0;
    while (cell_idx < k_span_cells)
    {
        k_cache_q16[k_base_index + cell_idx] = k_token_src_q16[cell_idx];
        v_cache_q16[v_base_index + cell_idx] = v_token_src_q16[cell_idx];

        cell_idx++;
        if (!cell_idx)
            return KV_Q16_ERR_OVERFLOW;
    }

    return KV_Q16_OK;
}

// Copy one token's K/V slices out of cache slabs with strict no-partial writes.
// Layout per layer:
//   [token0 span][token1 span]...[tokenN span]
// where span = kv_heads * head_dim Q16 cells.
I32 KVCacheQ16ReadTokenCheckedNoPartial(
    I64 *k_cache_q16,
    I64 k_cache_capacity,
    I64 *v_cache_q16,
    I64 v_cache_capacity,
    I64 layer_idx,
    I64 token_idx,
    I64 layer_count,
    I64 token_capacity,
    I64 kv_heads,
    I64 head_dim,
    I64 *k_token_out_q16,
    I64 k_token_out_capacity,
    I64 *v_token_out_q16,
    I64 v_token_out_capacity)
{
    I64 k_base_index;
    I64 k_span_cells;
    I64 v_base_index;
    I64 v_span_cells;
    I64 k_end_index;
    I64 v_end_index;
    I64 cell_idx;
    I32 status;

    if (!k_cache_q16 || !v_cache_q16 || !k_token_out_q16 || !v_token_out_q16)
        return KV_Q16_ERR_NULL_PTR;

    if (k_cache_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;
    if (k_token_out_q16 == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_cache_capacity < 0 || v_cache_capacity < 0 ||
        k_token_out_capacity < 0 || v_token_out_capacity < 0)
        return KV_Q16_ERR_BAD_PARAM;

    if (layer_idx < 0 || token_idx < 0 ||
        layer_count < 0 || token_capacity < 0 ||
        kv_heads < 0 || head_dim < 0)
        return KV_Q16_ERR_BAD_PARAM;

    // Keep read destination windows distinct from cache slabs so commit phase is
    // deterministic and cannot self-overwrite source cells under aliasing.
    if (k_token_out_q16 == k_cache_q16 || k_token_out_q16 == v_cache_q16 ||
        v_token_out_q16 == k_cache_q16 || v_token_out_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    // Resolve source geometry through canonical checked helpers.
    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &k_base_index,
                                                         &k_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &v_base_index,
                                                         &v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (k_base_index != v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (k_span_cells != v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(k_base_index, k_span_cells, &k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(v_base_index, v_span_cells, &v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (k_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (v_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_span_cells > k_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (v_span_cells > v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    // All checks are complete above; copy is now an atomic commit region with
    // no remaining failure branches and therefore no partial-write exits.
    cell_idx = 0;
    while (cell_idx < k_span_cells)
    {
        k_token_out_q16[cell_idx] = k_cache_q16[k_base_index + cell_idx];
        v_token_out_q16[cell_idx] = v_cache_q16[v_base_index + cell_idx];

        cell_idx++;
        if (!cell_idx)
            return KV_Q16_ERR_OVERFLOW;
    }

    return KV_Q16_OK;
}

// Zero one token's K/V slices inside cache slabs with strict no-partial writes.
// Layout per layer:
//   [token0 span][token1 span]...[tokenN span]
// where span = kv_heads * head_dim Q16 cells.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartial(
    I64 *k_cache_q16,
    I64 k_cache_capacity,
    I64 *v_cache_q16,
    I64 v_cache_capacity,
    I64 layer_idx,
    I64 token_idx,
    I64 layer_count,
    I64 token_capacity,
    I64 kv_heads,
    I64 head_dim)
{
    I64 k_base_index;
    I64 k_span_cells;
    I64 v_base_index;
    I64 v_span_cells;
    I64 k_end_index;
    I64 v_end_index;
    I64 cell_idx;
    I32 status;

    if (!k_cache_q16 || !v_cache_q16)
        return KV_Q16_ERR_NULL_PTR;

    if (k_cache_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_cache_capacity < 0 || v_cache_capacity < 0)
        return KV_Q16_ERR_BAD_PARAM;

    if (layer_idx < 0 || token_idx < 0 ||
        layer_count < 0 || token_capacity < 0 ||
        kv_heads < 0 || head_dim < 0)
        return KV_Q16_ERR_BAD_PARAM;

    // Resolve canonical token geometry via checked base/span helper.
    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &k_base_index,
                                                         &k_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &v_base_index,
                                                         &v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (k_base_index != v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (k_span_cells != v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(k_base_index, k_span_cells, &k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(v_base_index, v_span_cells, &v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (k_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (v_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    // All checks are complete above; zero phase is atomic commit region with
    // no remaining failure branches and therefore no partial-write exits.
    cell_idx = 0;
    while (cell_idx < k_span_cells)
    {
        k_cache_q16[k_base_index + cell_idx] = 0;
        v_cache_q16[v_base_index + cell_idx] = 0;

        cell_idx++;
        if (!cell_idx)
            return KV_Q16_ERR_OVERFLOW;
    }

    return KV_Q16_OK;
}
