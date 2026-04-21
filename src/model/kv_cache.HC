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

// Commit-only diagnostics wrapper over token read helper.
// Stages canonical base/span/end geometry first, revalidates immutable inputs,
// executes canonical read commit, then atomically publishes diagnostics.
I32 KVCacheQ16ReadTokenCheckedNoPartialCommitOnly(
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
    I64 v_token_out_capacity,
    I64 *out_base_index,
    I64 *out_span_cells,
    I64 *out_end_index)
{
    I64 staged_k_base_index;
    I64 staged_k_span_cells;
    I64 staged_v_base_index;
    I64 staged_v_span_cells;
    I64 staged_end_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I32 status;

    if (!out_base_index || !out_span_cells || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_base_index == out_span_cells ||
        out_base_index == out_end_index ||
        out_span_cells == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

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

    if (k_token_out_q16 == k_cache_q16 || k_token_out_q16 == v_cache_q16 ||
        v_token_out_q16 == k_cache_q16 || v_token_out_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_k_base_index,
                                                         &staged_k_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_v_base_index,
                                                         &staged_v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_base_index != staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_k_span_cells != staged_v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_base_index < 0 || staged_v_base_index < 0 ||
        staged_k_span_cells < 0 || staged_v_span_cells < 0)
        return KV_Q16_ERR_BAD_PARAM;

    // Defensive no-wrap invariants: checked geometry for token spans must stay
    // non-negative and monotonic before any diagnostics are published.
    if (staged_k_base_index < 0 || staged_v_base_index < 0 ||
        staged_k_span_cells < 0 || staged_v_span_cells < 0)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(staged_k_base_index,
                                staged_k_span_cells,
                                &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_span_cells > k_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_k_span_cells > v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16ReadTokenCheckedNoPartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity);
    if (status != KV_Q16_OK)
        return status;

    *out_base_index = staged_k_base_index;
    *out_span_cells = staged_k_span_cells;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Diagnostics-only no-write companion over commit-only read wrapper.
// Stages canonical read geometry, revalidates immutable snapshots, and
// atomically publishes diagnostics with strict zero-write/no-partial semantics.
I32 KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnly(
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
    I64 v_token_out_capacity,
    I64 *out_required_span_cells,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_k_base_index;
    I64 staged_k_span_cells;
    I64 staged_v_base_index;
    I64 staged_v_span_cells;
    I64 staged_k_end_index;
    I64 staged_v_end_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I32 status;

    if (!out_required_span_cells || !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

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

    if (k_token_out_q16 == k_cache_q16 || k_token_out_q16 == v_cache_q16 ||
        v_token_out_q16 == k_cache_q16 || v_token_out_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_out_q16 ||
        out_required_span_cells == v_token_out_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_out_q16 ||
        out_k_base_index == v_token_out_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_out_q16 ||
        out_v_base_index == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_k_base_index,
                                                         &staged_k_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_v_base_index,
                                                         &staged_v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_base_index != staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_k_span_cells != staged_v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_base_index < 0 || staged_v_base_index < 0 ||
        staged_k_span_cells < 0 || staged_v_span_cells < 0)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(staged_k_base_index,
                                staged_k_span_cells,
                                &staged_k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_index,
                                staged_v_span_cells,
                                &staged_v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_end_index != staged_v_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_end_index < staged_k_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index < staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_span_cells > k_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_span_cells > v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_k_span_cells;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Diagnostics-only helper over read commit-only preflight that derives
// required token bytes and publishes staged geometry atomically.
I32 KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
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
    I64 v_token_out_capacity,
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_required_span_cells;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_required_token_bytes;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_out_q16 ||
        out_required_span_cells == v_token_out_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_required_token_bytes == k_token_out_q16 ||
        out_required_token_bytes == v_token_out_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_out_q16 ||
        out_k_base_index == v_token_out_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_out_q16 ||
        out_v_base_index == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
        &staged_required_span_cells,
        &staged_k_base_index,
        &staged_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_required_span_cells < 0)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryMulI64Checked(staged_required_span_cells,
                                sizeof(I64),
                                &staged_required_token_bytes);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
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

// Commit-only diagnostics wrapper over token-span zero helper.
// Stages canonical required-span/base geometry first, revalidates immutable
// inputs, then executes canonical zero commit and atomically publishes
// diagnostics.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnly(
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
    I64 *out_required_span_cells,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_k_base_index;
    I64 staged_required_span_cells;
    I64 staged_v_base_index;
    I64 staged_v_span_cells;
    I64 staged_k_end_index;
    I64 staged_v_end_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

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

    if (out_required_span_cells == k_cache_q16 || out_required_span_cells == v_cache_q16 ||
        out_k_base_index == k_cache_q16 || out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 || out_v_base_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_k_base_index,
                                                         &staged_required_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_v_base_index,
                                                         &staged_v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_base_index != staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_required_span_cells != staged_v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_base_index < 0 || staged_v_base_index < 0 ||
        staged_required_span_cells < 0 || staged_v_span_cells < 0)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(staged_k_base_index,
                                staged_required_span_cells,
                                &staged_k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_index,
                                staged_v_span_cells,
                                &staged_v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_end_index != staged_v_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim);
    if (status != KV_Q16_OK)
        return status;

    *out_required_span_cells = staged_required_span_cells;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Diagnostics-only no-write companion over zero-span commit-only wrapper.
// Stages canonical zero geometry (base/span/end), revalidates immutable
// snapshots, and atomically publishes diagnostics with strict zero-write/
// no-partial semantics.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnly(
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
    I64 *out_required_span_cells,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_k_base_index;
    I64 staged_required_span_cells;
    I64 staged_v_base_index;
    I64 staged_v_span_cells;
    I64 staged_k_end_index;
    I64 staged_v_end_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_k_base_index ||
        !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

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

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_k_base_index,
                                                         &staged_required_span_cells);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_v_base_index,
                                                         &staged_v_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_base_index != staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_required_span_cells != staged_v_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryAddI64Checked(staged_k_base_index,
                                staged_required_span_cells,
                                &staged_k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_index,
                                staged_v_span_cells,
                                &staged_v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_end_index != staged_v_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_end_index < staged_k_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index < staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_k_end_index;
    return KV_Q16_OK;
}

// Diagnostics-only required-bytes companion over zero-span preflight-only
// helper. Derives required_token_bytes = required_span_cells * sizeof(I64)
// with checked integer math, revalidates immutable geometry/capacity
// snapshots, and atomically publishes the diagnostics tuple.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
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
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_required_token_bytes_recomputed;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_end_index;
    I64 staged_k_end_index;
    I64 staged_v_end_index;
    I64 staged_k_end_index;
    I64 staged_v_end_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_required_token_bytes == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(staged_required_span_cells,
                                sizeof(I64),
                                &staged_required_token_bytes);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_k_base_index,
                                staged_required_span_cells,
                                &staged_k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_index,
                                staged_required_span_cells,
                                &staged_v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_k_end_index != staged_end_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index != staged_end_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index < staged_k_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index < staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Commit-only diagnostics wrapper over required-bytes preflight helper.
// Stages canonical tuple
// {required_span_cells,required_token_bytes,k_base_index,v_base_index,end_index},
// revalidates immutable geometry/capacity snapshots, then atomically publishes
// diagnostics without touching K/V cache slabs.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnly(
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
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_end_index;
    I64 staged_required_token_bytes_recomputed;
    I64 staged_k_end_index;
    I64 staged_v_end_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_required_token_bytes == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

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

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells,
        &staged_required_token_bytes,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    // Recompute byte + end diagnostics so publish phase commits only if the
    // staged tuple remains internally consistent.
    status = KVTryMulI64Checked(staged_required_span_cells,
                                sizeof(I64),
                                &staged_required_token_bytes_recomputed);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_k_base_index,
                                staged_required_span_cells,
                                &staged_k_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_index,
                                staged_required_span_cells,
                                &staged_v_end_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_required_span_cells < 0)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_required_token_bytes < 0)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_required_token_bytes_recomputed != staged_required_token_bytes)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_base_index < 0 || staged_v_base_index < 0)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_k_base_index != staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_k_end_index != staged_end_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_end_index != staged_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_end_index < staged_k_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index < staged_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index > k_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Diagnostics-only no-write companion over commit-only required-bytes helper.
// Stages canonical tuple outputs, revalidates immutable geometry snapshots,
// and atomically publishes outputs only when all checked invariants pass.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyPreflightOnly(
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
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_end_index;
    I64 staged_required_span_cells_canonical;
    I64 staged_required_token_bytes_canonical;
    I64 staged_k_base_index_canonical;
    I64 staged_v_base_index_canonical;
    I64 staged_end_index_canonical;
    I64 staged_span_cells_recomputed;
    I64 staged_k_base_recomputed;
    I64 staged_v_base_recomputed;
    I64 staged_required_token_bytes_recomputed;
    I64 staged_end_recomputed_k;
    I64 staged_end_recomputed_v;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_required_token_bytes == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells,
        &staged_required_token_bytes,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells_canonical,
        &staged_required_token_bytes_canonical,
        &staged_k_base_index_canonical,
        &staged_v_base_index_canonical,
        &staged_end_index_canonical);
    if (status != KV_Q16_OK)
        return status;

    // Recompute canonical geometry directly from base/index primitives so this
    // helper does not trust wrapper composition alone.
    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_k_base_recomputed,
                                                         &staged_span_cells_recomputed);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_v_base_recomputed,
                                                         &staged_span_cells_recomputed);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(staged_span_cells_recomputed,
                                (I64)sizeof(I64),
                                &staged_required_token_bytes_recomputed);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_k_base_recomputed,
                                staged_span_cells_recomputed,
                                &staged_end_recomputed_k);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_recomputed,
                                staged_span_cells_recomputed,
                                &staged_end_recomputed_v);
    if (status != KV_Q16_OK)
        return status;

    if (staged_required_span_cells != staged_required_span_cells_canonical ||
        staged_required_token_bytes != staged_required_token_bytes_canonical ||
        staged_k_base_index != staged_k_base_index_canonical ||
        staged_v_base_index != staged_v_base_index_canonical ||
        staged_end_index != staged_end_index_canonical)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_required_span_cells != staged_span_cells_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_required_token_bytes != staged_required_token_bytes_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_k_base_index != staged_k_base_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_base_index != staged_v_base_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index != staged_end_recomputed_k)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index != staged_end_recomputed_v)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Diagnostics-only parity gate that composes commit-only required-bytes
// staging with canonical required-bytes derivation, revalidates immutable
// geometry/capacity snapshots, and atomically publishes exact tuple parity
// under strict zero-write/no-partial semantics.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyParity(
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
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_end_index;
    I64 staged_required_span_cells_canonical;
    I64 staged_required_token_bytes_canonical;
    I64 staged_k_base_index_canonical;
    I64 staged_v_base_index_canonical;
    I64 staged_end_index_canonical;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_required_token_bytes == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells,
        &staged_required_token_bytes,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells_canonical,
        &staged_required_token_bytes_canonical,
        &staged_k_base_index_canonical,
        &staged_v_base_index_canonical,
        &staged_end_index_canonical);
    if (status != KV_Q16_OK)
        return status;

    if (staged_required_span_cells != staged_required_span_cells_canonical ||
        staged_required_token_bytes != staged_required_token_bytes_canonical ||
        staged_k_base_index != staged_k_base_index_canonical ||
        staged_v_base_index != staged_v_base_index_canonical ||
        staged_end_index != staged_end_index_canonical)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Diagnostics-only no-write companion over commit-only parity helper.
// Stages canonical tuple outputs from parity+preflight diagnostics,
// revalidates immutable geometry snapshots, and atomically publishes only
// after all checks pass with strict zero-write/no-partial semantics.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyParityPreflightOnly(
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
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_end_index;
    I64 staged_required_span_cells_canonical;
    I64 staged_required_token_bytes_canonical;
    I64 staged_k_base_index_canonical;
    I64 staged_v_base_index_canonical;
    I64 staged_end_index_canonical;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_required_token_bytes == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyParity(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells,
        &staged_required_token_bytes,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells_canonical,
        &staged_required_token_bytes_canonical,
        &staged_k_base_index_canonical,
        &staged_v_base_index_canonical,
        &staged_end_index_canonical);
    if (status != KV_Q16_OK)
        return status;

    if (staged_required_span_cells != staged_required_span_cells_canonical ||
        staged_required_token_bytes != staged_required_token_bytes_canonical ||
        staged_k_base_index != staged_k_base_index_canonical ||
        staged_v_base_index != staged_v_base_index_canonical ||
        staged_end_index != staged_end_index_canonical)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Diagnostics-only no-write companion over commit-only parity helper.
// Stages parity-checked tuple outputs, recomputes canonical geometry/bytes
// directly from checked base-index math, revalidates immutable snapshots, and
// atomically publishes diagnostics only when every invariant matches.
I32 KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyParityPreflightOnly(
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
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_end_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_end_index;
    I64 staged_span_cells_recomputed_k;
    I64 staged_span_cells_recomputed_v;
    I64 staged_k_base_recomputed;
    I64 staged_v_base_recomputed;
    I64 staged_required_token_bytes_recomputed;
    I64 staged_end_recomputed_k;
    I64 staged_end_recomputed_v;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index || !out_end_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_span_cells == out_end_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_required_token_bytes == out_end_index ||
        out_k_base_index == out_v_base_index ||
        out_k_base_index == out_end_index ||
        out_v_base_index == out_end_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_end_index == k_cache_q16 ||
        out_end_index == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;

    status = KVCacheQ16ZeroTokenSpanCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytesCommitOnlyParity(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        &staged_required_span_cells,
        &staged_required_token_bytes,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_end_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_k_base_recomputed,
                                                         &staged_span_cells_recomputed_k);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ComputeLayerTokenBaseIndexChecked(layer_idx,
                                                         token_idx,
                                                         layer_count,
                                                         token_capacity,
                                                         kv_heads,
                                                         head_dim,
                                                         &staged_v_base_recomputed,
                                                         &staged_span_cells_recomputed_v);
    if (status != KV_Q16_OK)
        return status;

    if (staged_span_cells_recomputed_k != staged_span_cells_recomputed_v)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVTryMulI64Checked(staged_span_cells_recomputed_k,
                                sizeof(I64),
                                &staged_required_token_bytes_recomputed);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_k_base_recomputed,
                                staged_span_cells_recomputed_k,
                                &staged_end_recomputed_k);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryAddI64Checked(staged_v_base_recomputed,
                                staged_span_cells_recomputed_v,
                                &staged_end_recomputed_v);
    if (status != KV_Q16_OK)
        return status;

    if (staged_required_span_cells != staged_span_cells_recomputed_k)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_required_token_bytes != staged_required_token_bytes_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_k_base_index != staged_k_base_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_v_base_index != staged_v_base_recomputed)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index != staged_end_recomputed_k)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_end_index != staged_end_recomputed_v)
        return KV_Q16_ERR_BAD_PARAM;

    if (staged_end_index > k_cache_capacity || staged_end_index > v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    *out_end_index = staged_end_index;
    return KV_Q16_OK;
}

// Checked diagnostics preflight for one-token K/V write.
// Resolves destination base/span for both slabs, validates all capacities,
// and publishes write geometry without touching destination cells.
I32 KVCacheQ16WriteTokenCheckedNoPartialPreflight(
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
    I64 v_token_src_capacity,
    I64 *out_k_base_index,
    I64 *out_v_base_index,
    I64 *out_span_cells)
{
    I64 k_base_index;
    I64 k_span_cells;
    I64 v_base_index;
    I64 v_span_cells;
    I64 k_end_index;
    I64 v_end_index;
    I32 status;

    if (!out_k_base_index || !out_v_base_index || !out_span_cells)
        return KV_Q16_ERR_NULL_PTR;

    if (out_k_base_index == out_v_base_index ||
        out_k_base_index == out_span_cells ||
        out_v_base_index == out_span_cells)
        return KV_Q16_ERR_BAD_PARAM;

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

    // Keep source windows distinct from both destination slabs.
    if (k_token_src_q16 == k_cache_q16 || k_token_src_q16 == v_cache_q16 ||
        v_token_src_q16 == k_cache_q16 || v_token_src_q16 == v_cache_q16 ||
        k_token_src_q16 == v_token_src_q16)
        return KV_Q16_ERR_BAD_PARAM;

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

    *out_k_base_index = k_base_index;
    *out_v_base_index = v_base_index;
    *out_span_cells = k_span_cells;
    return KV_Q16_OK;
}

// Diagnostics-only companion over commit-only write wrapper.
// Stages canonical write geometry through checked preflight, revalidates
// immutable geometry/source-capacity snapshots, and atomically publishes
// diagnostics with strict zero-write/no-partial semantics.
I32 KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyPreflightOnly(
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
    I64 v_token_src_capacity,
    I64 *out_required_span_cells,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_span_cells;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I32 status;

    if (!out_required_span_cells || !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_src_q16 ||
        out_required_span_cells == v_token_src_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_src_q16 ||
        out_k_base_index == v_token_src_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_src_q16 ||
        out_v_base_index == v_token_src_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;

    status = KVCacheQ16WriteTokenCheckedNoPartialPreflight(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_span_cells;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Diagnostics-only required-bytes companion over write preflight-only helper.
// Computes required_token_bytes = required_span_cells * sizeof(I64) with
// checked overflow handling, revalidates immutable snapshots, and atomically
// publishes outputs with strict zero-write/no-partial semantics.
I32 KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
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
    I64 v_token_src_capacity,
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_src_q16 ||
        out_required_span_cells == v_token_src_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_required_token_bytes == k_token_src_q16 ||
        out_required_token_bytes == v_token_src_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_src_q16 ||
        out_k_base_index == v_token_src_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_src_q16 ||
        out_v_base_index == v_token_src_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;

    status = KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        &staged_required_span_cells,
        &staged_k_base_index,
        &staged_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(staged_required_span_cells,
                                sizeof(I64),
                                &staged_required_token_bytes);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Commit-only required-bytes helper over canonical write commit-only path.
// Stages diagnostics from commit execution, derives required token bytes with
// checked multiplication, and atomically publishes outputs only after immutable
// snapshot revalidation succeeds.
I32 KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyRequiredBytes(
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
    I64 v_token_src_capacity,
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_cache_capacity;
    I64 snapshot_v_cache_capacity;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_src_q16 ||
        out_required_span_cells == v_token_src_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_required_token_bytes == k_token_src_q16 ||
        out_required_token_bytes == v_token_src_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_src_q16 ||
        out_k_base_index == v_token_src_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_src_q16 ||
        out_v_base_index == v_token_src_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_cache_capacity = k_cache_capacity;
    snapshot_v_cache_capacity = v_cache_capacity;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;

    status = KVCacheQ16WriteTokenCheckedNoPartialCommitOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        &staged_required_span_cells,
        &staged_k_base_index,
        &staged_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(staged_required_span_cells,
                                sizeof(I64),
                                &staged_required_token_bytes);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_cache_capacity != k_cache_capacity ||
        snapshot_v_cache_capacity != v_cache_capacity ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Commit-only wrapper: stage diagnostics via checked preflight, then atomically
// write K/V source cells into destination cache slabs with no partial writes.
I32 KVCacheQ16WriteTokenCheckedNoPartialCommitOnly(
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
    I64 v_token_src_capacity,
    I64 *out_required_span_cells,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_span_cells;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I32 status;

    if (!out_required_span_cells || !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;

    status = KVCacheQ16WriteTokenCheckedNoPartialPreflight(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        &staged_k_base_index,
        &staged_v_base_index,
        &staged_span_cells);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    // Canonical commit implementation already has no remaining failure branches
    // after validation and guarantees strict no-partial destination writes.
    status = KVCacheQ16WriteTokenCheckedNoPartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity);
    if (status != KV_Q16_OK)
        return status;

    *out_required_span_cells = staged_span_cells;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Compose checked write+read paths into one deterministic round-trip validator.
// The function guarantees strict no-partial behavior by delegating all memory
// mutation to canonical no-partial helpers and publishing success only after a
// full element-wise equality pass between source and readback token windows.
I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial(
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
    I64 v_token_src_capacity,
    I64 *k_token_out_q16,
    I64 k_token_out_capacity,
    I64 *v_token_out_q16,
    I64 v_token_out_capacity)
{
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I64 span_cells;
    I64 cell_idx;
    I32 status;

    if (!k_cache_q16 || !v_cache_q16 || !k_token_src_q16 || !v_token_src_q16 ||
        !k_token_out_q16 || !v_token_out_q16)
        return KV_Q16_ERR_NULL_PTR;

    if (k_cache_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_token_src_q16 == v_token_src_q16)
        return KV_Q16_ERR_BAD_PARAM;
    if (k_token_out_q16 == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_token_out_q16 == k_cache_q16 || k_token_out_q16 == v_cache_q16 ||
        v_token_out_q16 == k_cache_q16 || v_token_out_q16 == v_cache_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_token_src_q16 == k_token_out_q16 || k_token_src_q16 == v_token_out_q16 ||
        v_token_src_q16 == k_token_out_q16 || v_token_src_q16 == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    if (k_cache_capacity < 0 || v_cache_capacity < 0 ||
        k_token_src_capacity < 0 || v_token_src_capacity < 0 ||
        k_token_out_capacity < 0 || v_token_out_capacity < 0)
        return KV_Q16_ERR_BAD_PARAM;

    if (layer_idx < 0 || token_idx < 0 ||
        layer_count < 0 || token_capacity < 0 ||
        kv_heads < 0 || head_dim < 0)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16ComputeLayerTokenSpanCellsChecked(kv_heads,
                                                         head_dim,
                                                         &span_cells);
    if (status != KV_Q16_OK)
        return status;

    // Validate both source and readback buffers up front so failure cannot
    // occur after write commit. This preserves strict no-partial semantics for
    // both cache slabs and caller output token windows.
    if (span_cells > k_token_src_capacity || span_cells > v_token_src_capacity)
        return KV_Q16_ERR_BAD_PARAM;
    if (span_cells > k_token_out_capacity || span_cells > v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16WriteTokenCheckedNoPartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16ReadTokenCheckedNoPartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity);
    if (status != KV_Q16_OK)
        return status;

    cell_idx = 0;
    while (cell_idx < span_cells)
    {
        if (k_token_out_q16[cell_idx] != k_token_src_q16[cell_idx])
            return KV_Q16_ERR_BAD_PARAM;
        if (v_token_out_q16[cell_idx] != v_token_src_q16[cell_idx])
            return KV_Q16_ERR_BAD_PARAM;

        cell_idx++;
        if (!cell_idx)
            return KV_Q16_ERR_OVERFLOW;
    }

    return KV_Q16_OK;
}

// Commit-only diagnostics wrapper over round-trip validator.
// Stages token geometry diagnostics via commit-only write helper, revalidates
// immutable inputs, executes canonical round-trip, then atomically publishes.
I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnly(
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
    I64 v_token_src_capacity,
    I64 *k_token_out_q16,
    I64 k_token_out_capacity,
    I64 *v_token_out_q16,
    I64 v_token_out_capacity,
    I64 *out_token_cells,
    I64 *out_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_token_cells;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 staged_token_bytes;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I32 status;

    if (!out_token_cells || !out_token_bytes ||
        !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_token_cells == out_token_bytes ||
        out_token_cells == out_k_base_index ||
        out_token_cells == out_v_base_index ||
        out_token_bytes == out_k_base_index ||
        out_token_bytes == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_token_cells == k_cache_q16 || out_token_cells == v_cache_q16 ||
        out_token_cells == k_token_src_q16 || out_token_cells == v_token_src_q16 ||
        out_token_cells == k_token_out_q16 || out_token_cells == v_token_out_q16 ||
        out_token_bytes == k_cache_q16 || out_token_bytes == v_cache_q16 ||
        out_token_bytes == k_token_src_q16 || out_token_bytes == v_token_src_q16 ||
        out_token_bytes == k_token_out_q16 || out_token_bytes == v_token_out_q16 ||
        out_k_base_index == k_cache_q16 || out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_src_q16 || out_k_base_index == v_token_src_q16 ||
        out_k_base_index == k_token_out_q16 || out_k_base_index == v_token_out_q16 ||
        out_v_base_index == k_cache_q16 || out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_src_q16 || out_v_base_index == v_token_src_q16 ||
        out_v_base_index == k_token_out_q16 || out_v_base_index == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16WriteTokenCheckedNoPartialCommitOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        &staged_token_cells,
        &staged_k_base_index,
        &staged_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(staged_token_cells,
                                sizeof(I64),
                                &staged_token_bytes);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    status = KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartial(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity);
    if (status != KV_Q16_OK)
        return status;

    *out_token_cells = staged_token_cells;
    *out_token_bytes = staged_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}

// Diagnostics-only no-write helper for round-trip geometry checks.
// Composes write/read preflight-only helpers, enforces exact tuple parity,
// revalidates immutable geometry snapshots, and atomically publishes.
I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnly(
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
    I64 v_token_src_capacity,
    I64 *k_token_out_q16,
    I64 k_token_out_capacity,
    I64 *v_token_out_q16,
    I64 v_token_out_capacity,
    I64 *out_required_span_cells,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_write_span_cells;
    I64 staged_write_k_base_index;
    I64 staged_write_v_base_index;
    I64 staged_read_span_cells;
    I64 staged_read_k_base_index;
    I64 staged_read_v_base_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I32 status;

    if (!out_required_span_cells || !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_src_q16 ||
        out_required_span_cells == v_token_src_q16 ||
        out_required_span_cells == k_token_out_q16 ||
        out_required_span_cells == v_token_out_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_src_q16 ||
        out_k_base_index == v_token_src_q16 ||
        out_k_base_index == k_token_out_q16 ||
        out_k_base_index == v_token_out_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_src_q16 ||
        out_v_base_index == v_token_src_q16 ||
        out_v_base_index == k_token_out_q16 ||
        out_v_base_index == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16WriteTokenCheckedNoPartialCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        &staged_write_span_cells,
        &staged_write_k_base_index,
        &staged_write_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVCacheQ16ReadTokenCheckedNoPartialCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
        &staged_read_span_cells,
        &staged_read_k_base_index,
        &staged_read_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    if (staged_write_span_cells != staged_read_span_cells)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_write_k_base_index != staged_read_k_base_index)
        return KV_Q16_ERR_BAD_PARAM;
    if (staged_write_v_base_index != staged_read_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_write_span_cells;
    *out_k_base_index = staged_write_k_base_index;
    *out_v_base_index = staged_write_v_base_index;
    return KV_Q16_OK;
}

// Diagnostics-only required-bytes companion for round-trip preflight helper.
// Derives checked token-byte diagnostics from staged span cells and publishes
// atomically after immutable snapshot validation.
I32 KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnlyRequiredBytes(
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
    I64 v_token_src_capacity,
    I64 *k_token_out_q16,
    I64 k_token_out_capacity,
    I64 *v_token_out_q16,
    I64 v_token_out_capacity,
    I64 *out_required_span_cells,
    I64 *out_required_token_bytes,
    I64 *out_k_base_index,
    I64 *out_v_base_index)
{
    I64 staged_required_span_cells;
    I64 staged_required_token_bytes;
    I64 staged_k_base_index;
    I64 staged_v_base_index;
    I64 snapshot_layer_idx;
    I64 snapshot_token_idx;
    I64 snapshot_layer_count;
    I64 snapshot_token_capacity;
    I64 snapshot_kv_heads;
    I64 snapshot_head_dim;
    I64 snapshot_k_src_capacity;
    I64 snapshot_v_src_capacity;
    I64 snapshot_k_out_capacity;
    I64 snapshot_v_out_capacity;
    I32 status;

    if (!out_required_span_cells || !out_required_token_bytes ||
        !out_k_base_index || !out_v_base_index)
        return KV_Q16_ERR_NULL_PTR;

    if (out_required_span_cells == out_required_token_bytes ||
        out_required_span_cells == out_k_base_index ||
        out_required_span_cells == out_v_base_index ||
        out_required_token_bytes == out_k_base_index ||
        out_required_token_bytes == out_v_base_index ||
        out_k_base_index == out_v_base_index)
        return KV_Q16_ERR_BAD_PARAM;

    if (out_required_span_cells == k_cache_q16 ||
        out_required_span_cells == v_cache_q16 ||
        out_required_span_cells == k_token_src_q16 ||
        out_required_span_cells == v_token_src_q16 ||
        out_required_span_cells == k_token_out_q16 ||
        out_required_span_cells == v_token_out_q16 ||
        out_required_token_bytes == k_cache_q16 ||
        out_required_token_bytes == v_cache_q16 ||
        out_required_token_bytes == k_token_src_q16 ||
        out_required_token_bytes == v_token_src_q16 ||
        out_required_token_bytes == k_token_out_q16 ||
        out_required_token_bytes == v_token_out_q16 ||
        out_k_base_index == k_cache_q16 ||
        out_k_base_index == v_cache_q16 ||
        out_k_base_index == k_token_src_q16 ||
        out_k_base_index == v_token_src_q16 ||
        out_k_base_index == k_token_out_q16 ||
        out_k_base_index == v_token_out_q16 ||
        out_v_base_index == k_cache_q16 ||
        out_v_base_index == v_cache_q16 ||
        out_v_base_index == k_token_src_q16 ||
        out_v_base_index == v_token_src_q16 ||
        out_v_base_index == k_token_out_q16 ||
        out_v_base_index == v_token_out_q16)
        return KV_Q16_ERR_BAD_PARAM;

    snapshot_layer_idx = layer_idx;
    snapshot_token_idx = token_idx;
    snapshot_layer_count = layer_count;
    snapshot_token_capacity = token_capacity;
    snapshot_kv_heads = kv_heads;
    snapshot_head_dim = head_dim;
    snapshot_k_src_capacity = k_token_src_capacity;
    snapshot_v_src_capacity = v_token_src_capacity;
    snapshot_k_out_capacity = k_token_out_capacity;
    snapshot_v_out_capacity = v_token_out_capacity;

    status = KVCacheQ16ReadWriteTokenRoundTripCheckedNoPartialCommitOnlyPreflightOnly(
        k_cache_q16,
        k_cache_capacity,
        v_cache_q16,
        v_cache_capacity,
        layer_idx,
        token_idx,
        layer_count,
        token_capacity,
        kv_heads,
        head_dim,
        k_token_src_q16,
        k_token_src_capacity,
        v_token_src_q16,
        v_token_src_capacity,
        k_token_out_q16,
        k_token_out_capacity,
        v_token_out_q16,
        v_token_out_capacity,
        &staged_required_span_cells,
        &staged_k_base_index,
        &staged_v_base_index);
    if (status != KV_Q16_OK)
        return status;

    status = KVTryMulI64Checked(staged_required_span_cells,
                                sizeof(I64),
                                &staged_required_token_bytes);
    if (status != KV_Q16_OK)
        return status;

    if (snapshot_layer_idx != layer_idx ||
        snapshot_token_idx != token_idx ||
        snapshot_layer_count != layer_count ||
        snapshot_token_capacity != token_capacity ||
        snapshot_kv_heads != kv_heads ||
        snapshot_head_dim != head_dim ||
        snapshot_k_src_capacity != k_token_src_capacity ||
        snapshot_v_src_capacity != v_token_src_capacity ||
        snapshot_k_out_capacity != k_token_out_capacity ||
        snapshot_v_out_capacity != v_token_out_capacity)
        return KV_Q16_ERR_BAD_PARAM;

    *out_required_span_cells = staged_required_span_cells;
    *out_required_token_bytes = staged_required_token_bytes;
    *out_k_base_index = staged_k_base_index;
    *out_v_base_index = staged_v_base_index;
    return KV_Q16_OK;
}
