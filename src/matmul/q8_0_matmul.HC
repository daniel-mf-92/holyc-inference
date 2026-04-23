// q8_0_matmul.HC
// Checked tiled Q8_0 matmul kernel for TempleOS.
//
// Runtime contract:
//   - lhs is an MxK matrix in row-major block layout (Q8_0 blocks).
//   - rhs is an NxK matrix in row-major "column bank" layout where each output
//     column of logical B (KxN) is stored as one contiguous row of K blocks.
//     This keeps every output cell as an explicit row-dot between two block rows.
//   - out is MxN Q16 matrix (I64 lanes) with explicit output row stride.

#define Q8_0_MATMUL_VALUES_PER_BLOCK 32
#define Q8_0_MATMUL_PACKED_BYTES     32

#define Q8_0_MATMUL_OK               0
#define Q8_0_MATMUL_ERR_NULL_PTR     1
#define Q8_0_MATMUL_ERR_BAD_DST_LEN  2
#define Q8_0_MATMUL_ERR_OVERFLOW     3
#define Q8_0_MATMUL_I64_MAX          0x7FFFFFFFFFFFFFFF

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q8_0_MATMUL_PACKED_BYTES];
};

// Declared in src/quant/q8_0_dot.HC.
I32 Q8_0DotProductBlocksQ16AccumulateChecked(Q8_0Block *lhs, Q8_0Block *rhs, I64 block_count, I64 *io_accum_q16);
// Declared in src/quant/q8_0_avx2.HC.
I32 Q8_0DotBlocksAVX2Q32Checked(Q8_0Block *lhs, Q8_0Block *rhs, I64 block_count, I64 *out_dot_q32);
I32 Q8_0DotBlocksAVX2Q32ToQ16Checked(Q8_0Block *lhs, Q8_0Block *rhs, I64 block_count, I64 *out_dot_q16);

Bool Q8_0MatMulTryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return FALSE;
    if (lhs < 0 || rhs < 0)
        return FALSE;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return TRUE;
    }

    if (lhs > Q8_0_MATMUL_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q8_0MatMulTryAddI64NonNeg(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (lhs < 0 || rhs < 0)
        return FALSE;
    if (lhs > Q8_0_MATMUL_I64_MAX - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

I64 Q8_0MatMulMinI64(I64 lhs, I64 rhs)
{
    if (lhs < rhs)
        return lhs;
    return rhs;
}

// Shared checked span-bounds math for tiled row/column traversals:
//   span_end = min(span_start + span_len, axis_len)
// The helper keeps overflow and BAD_LEN behavior identical across
// scalar Q16, AVX2 Q32, and AVX2 Q16 entrypoints.
I32 Q8_0MatMulTiledComputeSpanBoundsChecked(
    I64 span_start,
    I64 span_len,
    I64 axis_len,
    I64 *out_span_end)
{
    I64 span_end;

    if (!out_span_end)
        return Q8_0_MATMUL_ERR_NULL_PTR;
    if (span_start < 0 || span_len < 0 || axis_len < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (span_start > axis_len)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q8_0MatMulTryAddI64NonNeg(span_start,
                                   span_len,
                                   &span_end))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    *out_span_end = Q8_0MatMulMinI64(span_end,
                                     axis_len);
    return Q8_0_MATMUL_OK;
}

// Shared tile-shape contract for tiled Q8_0 matmul call-sites.
// Both tile extents must be strictly positive so tiled traversal always
// makes forward progress and never loops forever on zero-sized tiles.
I32 Q8_0MatMulTiledValidateTileShapeChecked(I64 tile_rows,
                                            I64 tile_cols)
{
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    return Q8_0_MATMUL_OK;
}

// Deterministic tile scheduler shared by scalar and AVX2 tiled matmul paths.
// For the current tile starts, compute bounded tile ends for both axes in one
// checked call so overflow/BAD_LEN behavior stays identical across kernels.
I32 Q8_0MatMulTileSchedulerChecked(
    I64 row_tile_start,
    I64 col_tile_start,
    I64 row_count,
    I64 col_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_row_tile_end,
    I64 *out_col_tile_end)
{
    I32 status;

    if (!out_row_tile_end || !out_col_tile_end)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    status = Q8_0MatMulTiledValidateTileShapeChecked(tile_rows,
                                                     tile_cols);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTiledComputeSpanBoundsChecked(row_tile_start,
                                                     tile_rows,
                                                     row_count,
                                                     out_row_tile_end);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTiledComputeSpanBoundsChecked(col_tile_start,
                                                     tile_cols,
                                                     col_count,
                                                     out_col_tile_end);
    if (status != Q8_0_MATMUL_OK)
        return status;

    return Q8_0_MATMUL_OK;
}

// No-partial tile-table scheduler for tiled Q8_0 matmul traversal.
//
// Contract:
// - Build the full deterministic tile span table in private stage arrays.
// - Publish span starts/ends and span_count only after full schedule succeeds.
// - On any failure, caller span arrays and `out_span_count` stay unchanged.
//
// Span layout (one entry per tile in row-major tile order):
//   out_row_tile_starts[i], out_row_tile_ends[i],
//   out_col_tile_starts[i], out_col_tile_ends[i]
I32 Q8_0MatMulTileSchedulerCheckedNoPartial(
    I64 row_count,
    I64 col_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_row_tile_starts,
    I64 *out_row_tile_ends,
    I64 *out_col_tile_starts,
    I64 *out_col_tile_ends,
    I64 span_capacity,
    I64 *out_span_count)
{
    I64 row_numer;
    I64 col_numer;
    I64 row_tile_count;
    I64 col_tile_count;
    I64 tile_total_count;
    I64 stage_bytes;
    I64 row_tile_start;
    I64 row_tile_end;
    I64 col_tile_start;
    I64 col_tile_end;
    I64 span_index;
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_tile_rows;
    I64 snapshot_tile_cols;
    I64 snapshot_span_capacity;
    I64 *staged_row_tile_starts;
    I64 *staged_row_tile_ends;
    I64 *staged_col_tile_starts;
    I64 *staged_col_tile_ends;
    I32 status;

    if (!out_span_count)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (span_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (span_capacity > 0 &&
        (!out_row_tile_starts ||
         !out_row_tile_ends ||
         !out_col_tile_starts ||
         !out_col_tile_ends))
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (row_count < 0 || col_count < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    status = Q8_0MatMulTiledValidateTileShapeChecked(tile_rows,
                                                     tile_cols);
    if (status != Q8_0_MATMUL_OK)
        return status;

    // Checked ceil division for tile counts:
    //   tile_count = (axis_len + tile_len - 1) / tile_len
    if (!Q8_0MatMulTryAddI64NonNeg(row_count,
                                   tile_rows - 1,
                                   &row_numer))
        return Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q8_0MatMulTryAddI64NonNeg(col_count,
                                   tile_cols - 1,
                                   &col_numer))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    row_tile_count = row_numer / tile_rows;
    col_tile_count = col_numer / tile_cols;

    if (!Q8_0MatMulTryMulI64NonNeg(row_tile_count,
                                   col_tile_count,
                                   &tile_total_count))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    if (tile_total_count > span_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!tile_total_count)
    {
        *out_span_count = 0;
        return Q8_0_MATMUL_OK;
    }

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_tile_rows = tile_rows;
    snapshot_tile_cols = tile_cols;
    snapshot_span_capacity = span_capacity;

    if (!Q8_0MatMulTryMulI64NonNeg(tile_total_count,
                                   sizeof(I64),
                                   &stage_bytes))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    staged_row_tile_starts = MAlloc(stage_bytes);
    if (!staged_row_tile_starts)
        return Q8_0_MATMUL_ERR_OVERFLOW;

    staged_row_tile_ends = MAlloc(stage_bytes);
    if (!staged_row_tile_ends)
    {
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_OVERFLOW;
    }

    staged_col_tile_starts = MAlloc(stage_bytes);
    if (!staged_col_tile_starts)
    {
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_OVERFLOW;
    }

    staged_col_tile_ends = MAlloc(stage_bytes);
    if (!staged_col_tile_ends)
    {
        Free(staged_col_tile_starts);
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_OVERFLOW;
    }

    span_index = 0;
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                    col_tile_start,
                                                    row_count,
                                                    col_count,
                                                    tile_rows,
                                                    tile_cols,
                                                    &row_tile_end,
                                                    &col_tile_end);
            if (status != Q8_0_MATMUL_OK)
            {
                Free(staged_col_tile_ends);
                Free(staged_col_tile_starts);
                Free(staged_row_tile_ends);
                Free(staged_row_tile_starts);
                return status;
            }

            if (span_index >= tile_total_count)
            {
                Free(staged_col_tile_ends);
                Free(staged_col_tile_starts);
                Free(staged_row_tile_ends);
                Free(staged_row_tile_starts);
                return Q8_0_MATMUL_ERR_BAD_DST_LEN;
            }

            staged_row_tile_starts[span_index] = row_tile_start;
            staged_row_tile_ends[span_index] = row_tile_end;
            staged_col_tile_starts[span_index] = col_tile_start;
            staged_col_tile_ends[span_index] = col_tile_end;
            ++span_index;

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    if (span_index != tile_total_count)
    {
        Free(staged_col_tile_ends);
        Free(staged_col_tile_starts);
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    }

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_tile_rows != tile_rows ||
        snapshot_tile_cols != tile_cols ||
        snapshot_span_capacity != span_capacity)
    {
        Free(staged_col_tile_ends);
        Free(staged_col_tile_starts);
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    }

    // Atomic publish region: no remaining checked failure paths.
    for (span_index = 0; span_index < tile_total_count; ++span_index)
    {
        out_row_tile_starts[span_index] = staged_row_tile_starts[span_index];
        out_row_tile_ends[span_index] = staged_row_tile_ends[span_index];
        out_col_tile_starts[span_index] = staged_col_tile_starts[span_index];
        out_col_tile_ends[span_index] = staged_col_tile_ends[span_index];
    }
    *out_span_count = tile_total_count;

    Free(staged_col_tile_ends);
    Free(staged_col_tile_starts);
    Free(staged_row_tile_ends);
    Free(staged_row_tile_starts);
    return Q8_0_MATMUL_OK;
}

// Commit-only diagnostics wrapper over no-partial tile scheduler.
//
// Contract:
// - Use canonical `Q8_0MatMulTileSchedulerCheckedNoPartial` to stage full
//   tile spans privately.
// - Keep immutable geometry snapshots across compute.
// - Atomically publish only `{tile_count,last_row_end,last_col_end}`.
// - On any error, caller diagnostics outputs remain unchanged.
I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnly(
    I64 row_count,
    I64 col_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 span_capacity,
    I64 *out_tile_count,
    I64 *out_last_row_end,
    I64 *out_last_col_end)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_tile_rows;
    I64 snapshot_tile_cols;
    I64 snapshot_span_capacity;
    I64 stage_bytes;
    I64 staged_tile_count;
    I64 staged_last_row_end;
    I64 staged_last_col_end;
    I64 *staged_row_tile_starts;
    I64 *staged_row_tile_ends;
    I64 *staged_col_tile_starts;
    I64 *staged_col_tile_ends;
    I32 status;

    if (!out_tile_count || !out_last_row_end || !out_last_col_end)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (row_count < 0 || col_count < 0 || span_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    status = Q8_0MatMulTiledValidateTileShapeChecked(tile_rows,
                                                     tile_cols);
    if (status != Q8_0_MATMUL_OK)
        return status;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_tile_rows = tile_rows;
    snapshot_tile_cols = tile_cols;
    snapshot_span_capacity = span_capacity;

    if (!span_capacity)
    {
        // With zero span capacity, canonical scheduler can only succeed for
        // empty axes; mirror that by running canonical call with null tables.
        status = Q8_0MatMulTileSchedulerCheckedNoPartial(row_count,
                                                         col_count,
                                                         tile_rows,
                                                         tile_cols,
                                                         NULL,
                                                         NULL,
                                                         NULL,
                                                         NULL,
                                                         0,
                                                         &staged_tile_count);
        if (status != Q8_0_MATMUL_OK)
            return status;

        if (snapshot_row_count != row_count ||
            snapshot_col_count != col_count ||
            snapshot_tile_rows != tile_rows ||
            snapshot_tile_cols != tile_cols ||
            snapshot_span_capacity != span_capacity)
            return Q8_0_MATMUL_ERR_BAD_DST_LEN;

        *out_tile_count = staged_tile_count;
        *out_last_row_end = 0;
        *out_last_col_end = 0;
        return Q8_0_MATMUL_OK;
    }

    if (!Q8_0MatMulTryMulI64NonNeg(span_capacity,
                                   sizeof(I64),
                                   &stage_bytes))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    staged_row_tile_starts = MAlloc(stage_bytes);
    if (!staged_row_tile_starts)
        return Q8_0_MATMUL_ERR_OVERFLOW;

    staged_row_tile_ends = MAlloc(stage_bytes);
    if (!staged_row_tile_ends)
    {
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_OVERFLOW;
    }

    staged_col_tile_starts = MAlloc(stage_bytes);
    if (!staged_col_tile_starts)
    {
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_OVERFLOW;
    }

    staged_col_tile_ends = MAlloc(stage_bytes);
    if (!staged_col_tile_ends)
    {
        Free(staged_col_tile_starts);
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_OVERFLOW;
    }

    status = Q8_0MatMulTileSchedulerCheckedNoPartial(row_count,
                                                     col_count,
                                                     tile_rows,
                                                     tile_cols,
                                                     staged_row_tile_starts,
                                                     staged_row_tile_ends,
                                                     staged_col_tile_starts,
                                                     staged_col_tile_ends,
                                                     span_capacity,
                                                     &staged_tile_count);
    if (status != Q8_0_MATMUL_OK)
    {
        Free(staged_col_tile_ends);
        Free(staged_col_tile_starts);
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return status;
    }

    staged_last_row_end = 0;
    staged_last_col_end = 0;
    if (staged_tile_count > 0)
    {
        staged_last_row_end = staged_row_tile_ends[staged_tile_count - 1];
        staged_last_col_end = staged_col_tile_ends[staged_tile_count - 1];
    }

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_tile_rows != tile_rows ||
        snapshot_tile_cols != tile_cols ||
        snapshot_span_capacity != span_capacity)
    {
        Free(staged_col_tile_ends);
        Free(staged_col_tile_starts);
        Free(staged_row_tile_ends);
        Free(staged_row_tile_starts);
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    }

    // Atomic publish region: diagnostics tuple only.
    *out_tile_count = staged_tile_count;
    *out_last_row_end = staged_last_row_end;
    *out_last_col_end = staged_last_col_end;

    Free(staged_col_tile_ends);
    Free(staged_col_tile_starts);
    Free(staged_row_tile_ends);
    Free(staged_row_tile_starts);
    return Q8_0_MATMUL_OK;
}

// Diagnostics-only preflight companion for commit-only tile scheduler tuple.
//
// Contract:
// - Reuse canonical commit-only wrapper to derive deterministic
//   `{tile_count,last_row_end,last_col_end}` from immutable scheduler shape.
// - Publish diagnostics tuple atomically only after strict snapshot parity.
// - Keep no-partial semantics: on any failure, output tuple is unchanged.
I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnly(
    I64 row_count,
    I64 col_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 span_capacity,
    I64 *out_tile_count,
    I64 *out_last_row_end,
    I64 *out_last_col_end)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_tile_rows;
    I64 snapshot_tile_cols;
    I64 snapshot_span_capacity;
    I64 staged_tile_count;
    I64 staged_last_row_end;
    I64 staged_last_col_end;
    I32 status;

    if (!out_tile_count || !out_last_row_end || !out_last_col_end)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (row_count < 0 || col_count < 0 || span_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_tile_rows = tile_rows;
    snapshot_tile_cols = tile_cols;
    snapshot_span_capacity = span_capacity;

    status = Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnly(row_count,
                                                               col_count,
                                                               tile_rows,
                                                               tile_cols,
                                                               span_capacity,
                                                               &staged_tile_count,
                                                               &staged_last_row_end,
                                                               &staged_last_col_end);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_tile_rows != tile_rows ||
        snapshot_tile_cols != tile_cols ||
        snapshot_span_capacity != span_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Atomic publish region: diagnostics tuple only.
    *out_tile_count = staged_tile_count;
    *out_last_row_end = staged_last_row_end;
    *out_last_col_end = staged_last_col_end;
    return Q8_0_MATMUL_OK;
}

// Strict parity gate over diagnostics preflight and commit-only wrappers.
//
// Contract:
// - Run both diagnostics preflight and commit-only wrappers from immutable
//   geometry/capacity snapshots.
// - Require exact tuple parity on
//   `{tile_count,last_row_end,last_col_end}`.
// - Publish caller tuple atomically only after strict parity success.
// - On any failure, caller tuple remains unchanged.
I32 Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnlyParity(
    I64 row_count,
    I64 col_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 span_capacity,
    I64 *out_tile_count,
    I64 *out_last_row_end,
    I64 *out_last_col_end)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_tile_rows;
    I64 snapshot_tile_cols;
    I64 snapshot_span_capacity;
    I64 staged_pre_tile_count;
    I64 staged_pre_last_row_end;
    I64 staged_pre_last_col_end;
    I64 staged_commit_tile_count;
    I64 staged_commit_last_row_end;
    I64 staged_commit_last_col_end;
    I32 status;

    if (!out_tile_count || !out_last_row_end || !out_last_col_end)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_tile_rows = tile_rows;
    snapshot_tile_cols = tile_cols;
    snapshot_span_capacity = span_capacity;

    status = Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnlyPreflightOnly(row_count,
                                                                             col_count,
                                                                             tile_rows,
                                                                             tile_cols,
                                                                             span_capacity,
                                                                             &staged_pre_tile_count,
                                                                             &staged_pre_last_row_end,
                                                                             &staged_pre_last_col_end);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTileSchedulerCheckedNoPartialCommitOnly(row_count,
                                                               col_count,
                                                               tile_rows,
                                                               tile_cols,
                                                               span_capacity,
                                                               &staged_commit_tile_count,
                                                               &staged_commit_last_row_end,
                                                               &staged_commit_last_col_end);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_tile_rows != tile_rows ||
        snapshot_tile_cols != tile_cols ||
        snapshot_span_capacity != span_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (staged_pre_tile_count != staged_commit_tile_count ||
        staged_pre_last_row_end != staged_commit_last_row_end ||
        staged_pre_last_col_end != staged_commit_last_col_end)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Atomic publish region: parity tuple only.
    *out_tile_count = staged_commit_tile_count;
    *out_last_row_end = staged_commit_last_row_end;
    *out_last_col_end = staged_commit_last_col_end;
    return Q8_0_MATMUL_OK;
}

// Shared stride/K-span invariants for tiled Q8_0 matmul call-sites.
// This centralizes the active-K and output-row-stride checks used by
// scalar and AVX2 entrypoints so BAD_LEN surfaces stay identical.
I32 Q8_0MatMulTiledValidateStridesAndKChecked(
    I64 row_count,
    I64 col_count,
    I64 lhs_row_stride_blocks,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 out_row_stride_cells)
{
    if (row_count < 0 || col_count < 0 || k_block_count < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (lhs_row_stride_blocks < 0 || rhs_col_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Active K span must fit inside each source row stride.
    if (k_block_count > lhs_row_stride_blocks || k_block_count > rhs_col_stride_blocks)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Output row stride must cover active output columns.
    if (row_count > 0 && out_row_stride_cells < col_count)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    return Q8_0_MATMUL_OK;
}

// Computes full row-strided capacities required for tiled matmul views:
//   lhs_required_blocks = row_count * lhs_row_stride_blocks
//   rhs_required_blocks = col_count * rhs_col_stride_blocks
//   out_required_cells  = row_count * out_row_stride_cells
// The helper is overflow-checked and shared by preflight and math entrypoints.
I32 Q8_0MatMulTiledComputeRequiredCapacitiesChecked(
    I64 row_count,
    I64 lhs_row_stride_blocks,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 out_row_stride_cells,
    I64 *out_lhs_required_blocks,
    I64 *out_rhs_required_blocks,
    I64 *out_required_cells)
{
    if (!out_lhs_required_blocks || !out_rhs_required_blocks || !out_required_cells)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (row_count < 0 || col_count < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (lhs_row_stride_blocks < 0 || rhs_col_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q8_0MatMulTryMulI64NonNeg(row_count, lhs_row_stride_blocks, out_lhs_required_blocks))
        return Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q8_0MatMulTryMulI64NonNeg(col_count, rhs_col_stride_blocks, out_rhs_required_blocks))
        return Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q8_0MatMulTryMulI64NonNeg(row_count, out_row_stride_cells, out_required_cells))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    return Q8_0_MATMUL_OK;
}

// Shared checked row-base math for tiled Q8_0 matmul inner loops:
//   lhs_row_base = row_index * lhs_row_stride_blocks
//   out_row_base = row_index * out_row_stride_cells
// Centralizing this keeps overflow/BAD_LEN behavior identical across
// scalar Q16, AVX2 Q32, and AVX2 Q16 tiled entrypoints.
I32 Q8_0MatMulTiledComputeRowBaseChecked(
    I64 row_index,
    I64 lhs_row_stride_blocks,
    I64 out_row_stride_cells,
    I64 *out_lhs_row_base,
    I64 *out_out_row_base)
{
    if (!out_lhs_row_base || !out_out_row_base)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (row_index < 0 || lhs_row_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q8_0MatMulTryMulI64NonNeg(row_index,
                                   lhs_row_stride_blocks,
                                   out_lhs_row_base))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q8_0MatMulTryMulI64NonNeg(row_index,
                                   out_row_stride_cells,
                                   out_out_row_base))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    return Q8_0_MATMUL_OK;
}

// Shared checked output-index math for tiled Q8_0 matmul inner loops:
//   out_index = out_row_base + col_index
// Centralizing this keeps overflow/BAD_LEN behavior identical across
// scalar Q16, AVX2 Q32, and AVX2 Q16 tiled entrypoints.
I32 Q8_0MatMulTiledComputeOutIndexChecked(
    I64 out_row_base,
    I64 col_index,
    I64 *out_out_index)
{
    if (!out_out_index)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (out_row_base < 0 || col_index < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q8_0MatMulTryAddI64NonNeg(out_row_base,
                                   col_index,
                                   out_out_index))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    return Q8_0_MATMUL_OK;
}

// Shared checked output-index capacity validation for tiled Q8_0 matmul writes:
//   out_index = out_row_base + col_index
//   require out_index < out_cell_capacity
// This centralizes BAD_LEN vs OVERFLOW behavior at store sites across
// scalar Q16, AVX2 Q32, and AVX2 Q16 tiled entrypoints.
I32 Q8_0MatMulTiledValidateOutIndexCapacityChecked(
    I64 out_row_base,
    I64 col_index,
    I64 out_cell_capacity,
    I64 *out_out_index)
{
    I32 status;

    if (!out_out_index)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (out_cell_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    status = Q8_0MatMulTiledComputeOutIndexChecked(out_row_base,
                                                   col_index,
                                                   out_out_index);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (*out_out_index >= out_cell_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    return Q8_0_MATMUL_OK;
}

// Shared checked RHS-column base math for tiled Q8_0 matmul inner loops:
//   rhs_col_base = col_index * rhs_col_stride_blocks
// Centralizing this keeps overflow/BAD_LEN behavior identical across
// scalar Q16, AVX2 Q32, and AVX2 Q16 tiled entrypoints.
I32 Q8_0MatMulTiledComputeRhsColBaseChecked(
    I64 col_index,
    I64 rhs_col_stride_blocks,
    I64 *out_rhs_col_base)
{
    if (!out_rhs_col_base)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (col_index < 0 || rhs_col_stride_blocks < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q8_0MatMulTryMulI64NonNeg(col_index,
                                   rhs_col_stride_blocks,
                                   out_rhs_col_base))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    return Q8_0_MATMUL_OK;
}

// Shared checked row-slice extent validation for tiled Q8_0 matmul loops:
//   row_slice_end = row_base + k_block_count
//   require row_slice_end <= block_capacity
// This centralizes BAD_LEN vs OVERFLOW behavior for row-window bounds across
// scalar Q16, AVX2 Q32, and AVX2 Q16 tiled entrypoints.
I32 Q8_0MatMulTiledValidateRowSliceChecked(
    I64 row_base,
    I64 k_block_count,
    I64 block_capacity,
    I64 *out_row_slice_end)
{
    if (!out_row_slice_end)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (row_base < 0 || k_block_count < 0 || block_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q8_0MatMulTryAddI64NonNeg(row_base,
                                   k_block_count,
                                   out_row_slice_end))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    if (*out_row_slice_end > block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    return Q8_0_MATMUL_OK;
}

I32 Q8_0MatMulTiledValidateArgsChecked(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_cells,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_lhs_required_blocks,
    I64 *out_rhs_required_blocks,
    I64 *out_required_cells)
{
    I32 status;

    if (!lhs_blocks || !rhs_col_blocks || !out_cells)
        return Q8_0_MATMUL_ERR_NULL_PTR;
    if (!out_lhs_required_blocks || !out_rhs_required_blocks || !out_required_cells)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_cell_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    status = Q8_0MatMulTiledValidateStridesAndKChecked(row_count,
                                                       col_count,
                                                       lhs_row_stride_blocks,
                                                       rhs_col_stride_blocks,
                                                       k_block_count,
                                                       out_row_stride_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTiledValidateTileShapeChecked(tile_rows,
                                                     tile_cols);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTiledComputeRequiredCapacitiesChecked(row_count,
                                                             lhs_row_stride_blocks,
                                                             col_count,
                                                             rhs_col_stride_blocks,
                                                             out_row_stride_cells,
                                                             out_lhs_required_blocks,
                                                             out_rhs_required_blocks,
                                                             out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (*out_lhs_required_blocks > lhs_block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (*out_rhs_required_blocks > rhs_block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (*out_required_cells > out_cell_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    return Q8_0_MATMUL_OK;
}

// Compatibility alias for existing callers during matmul preflight unification.
I32 Q8_0MatMulValidateTiledArgsChecked(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_cells,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_lhs_required_blocks,
    I64 *out_rhs_required_blocks,
    I64 *out_required_cells)
{
    return Q8_0MatMulTiledValidateArgsChecked(lhs_blocks,
                                              lhs_block_capacity,
                                              row_count,
                                              lhs_row_stride_blocks,
                                              rhs_col_blocks,
                                              rhs_block_capacity,
                                              col_count,
                                              rhs_col_stride_blocks,
                                              k_block_count,
                                              tile_rows,
                                              tile_cols,
                                              out_cells,
                                              out_cell_capacity,
                                              out_row_stride_cells,
                                              out_lhs_required_blocks,
                                              out_rhs_required_blocks,
                                              out_required_cells);
}

I32 Q8_0MatMulQ16TiledChecked(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_cells_q16,
    I64 out_cell_capacity,
    I64 out_row_stride_cells)
{
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 out_required_cells;
    I64 row_tile_start;
    I64 col_tile_start;
    I64 row_tile_end;
    I64 col_tile_end;
    I64 row_index;
    I64 col_index;
    I64 lhs_row_base;
    I64 lhs_row_slice_end;
    I64 rhs_col_base;
    I64 rhs_col_slice_end;
    I64 out_row_base;
    I64 out_index;
    I64 cell_dot_q16;
    Q8_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    status = Q8_0MatMulTiledValidateArgsChecked(lhs_blocks,
                                                lhs_block_capacity,
                                                row_count,
                                                lhs_row_stride_blocks,
                                                rhs_col_blocks,
                                                rhs_block_capacity,
                                                col_count,
                                                rhs_col_stride_blocks,
                                                k_block_count,
                                                tile_rows,
                                                tile_cols,
                                                out_cells_q16,
                                                out_cell_capacity,
                                                out_row_stride_cells,
                                                &lhs_required_blocks,
                                                &rhs_required_blocks,
                                                &out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    // Tiled traversal keeps working-set bounded while preserving direct math visibility.
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                0,
                                                row_count,
                                                col_count,
                                                tile_rows,
                                                tile_cols,
                                                &row_tile_end,
                                                &col_tile_end);
        if (status != Q8_0_MATMUL_OK)
            return status;

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                    col_tile_start,
                                                    row_count,
                                                    col_count,
                                                    tile_rows,
                                                    tile_cols,
                                                    &row_tile_end,
                                                    &col_tile_end);
            if (status != Q8_0_MATMUL_OK)
                return status;

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                status = Q8_0MatMulTiledComputeRowBaseChecked(row_index,
                                                              lhs_row_stride_blocks,
                                                              out_row_stride_cells,
                                                              &lhs_row_base,
                                                              &out_row_base);
                if (status != Q8_0_MATMUL_OK)
                    return status;

                status = Q8_0MatMulTiledValidateRowSliceChecked(lhs_row_base,
                                                                 k_block_count,
                                                                 lhs_block_capacity,
                                                                 &lhs_row_slice_end);
                if (status != Q8_0_MATMUL_OK)
                    return status;

                lhs_row_ptr = &lhs_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    status = Q8_0MatMulTiledComputeRhsColBaseChecked(col_index,
                                                                      rhs_col_stride_blocks,
                                                                      &rhs_col_base);
                    if (status != Q8_0_MATMUL_OK)
                        return status;
                    status = Q8_0MatMulTiledValidateOutIndexCapacityChecked(out_row_base,
                                                                             col_index,
                                                                             out_cell_capacity,
                                                                             &out_index);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    status = Q8_0MatMulTiledValidateRowSliceChecked(rhs_col_base,
                                                                     k_block_count,
                                                                     rhs_block_capacity,
                                                                     &rhs_col_slice_end);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    rhs_col_ptr = &rhs_col_blocks[rhs_col_base];
                    cell_dot_q16 = 0;

                    status = Q8_0DotProductBlocksQ16AccumulateChecked(lhs_row_ptr, rhs_col_ptr, k_block_count, &cell_dot_q16);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    out_cells_q16[out_index] = cell_dot_q16;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q8_0_MATMUL_OK;
}

I32 Q8_0MatMulQ32TiledAVX2Checked(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells)
{
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 out_required_cells;
    I64 row_tile_start;
    I64 col_tile_start;
    I64 row_tile_end;
    I64 col_tile_end;
    I64 row_index;
    I64 col_index;
    I64 lhs_row_base;
    I64 lhs_row_slice_end;
    I64 rhs_col_base;
    I64 rhs_col_slice_end;
    I64 out_row_base;
    I64 out_index;
    I64 cell_dot_q32;
    Q8_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    status = Q8_0MatMulTiledValidateArgsChecked(lhs_blocks,
                                                lhs_block_capacity,
                                                row_count,
                                                lhs_row_stride_blocks,
                                                rhs_col_blocks,
                                                rhs_block_capacity,
                                                col_count,
                                                rhs_col_stride_blocks,
                                                k_block_count,
                                                tile_rows,
                                                tile_cols,
                                                out_cells_q32,
                                                out_cell_capacity,
                                                out_row_stride_cells,
                                                &lhs_required_blocks,
                                                &rhs_required_blocks,
                                                &out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    // Tiled traversal mirrors Q16 path but preserves Q32 accumulator results.
    // Each output cell is one checked AVX2-shape block dot over K blocks.
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                0,
                                                row_count,
                                                col_count,
                                                tile_rows,
                                                tile_cols,
                                                &row_tile_end,
                                                &col_tile_end);
        if (status != Q8_0_MATMUL_OK)
            return status;

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                    col_tile_start,
                                                    row_count,
                                                    col_count,
                                                    tile_rows,
                                                    tile_cols,
                                                    &row_tile_end,
                                                    &col_tile_end);
            if (status != Q8_0_MATMUL_OK)
                return status;

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                status = Q8_0MatMulTiledComputeRowBaseChecked(row_index,
                                                              lhs_row_stride_blocks,
                                                              out_row_stride_cells,
                                                              &lhs_row_base,
                                                              &out_row_base);
                if (status != Q8_0_MATMUL_OK)
                    return status;

                status = Q8_0MatMulTiledValidateRowSliceChecked(lhs_row_base,
                                                                 k_block_count,
                                                                 lhs_block_capacity,
                                                                 &lhs_row_slice_end);
                if (status != Q8_0_MATMUL_OK)
                    return status;

                lhs_row_ptr = &lhs_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    status = Q8_0MatMulTiledComputeRhsColBaseChecked(col_index,
                                                                      rhs_col_stride_blocks,
                                                                      &rhs_col_base);
                    if (status != Q8_0_MATMUL_OK)
                        return status;
                    status = Q8_0MatMulTiledValidateOutIndexCapacityChecked(out_row_base,
                                                                             col_index,
                                                                             out_cell_capacity,
                                                                             &out_index);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    status = Q8_0MatMulTiledValidateRowSliceChecked(rhs_col_base,
                                                                     k_block_count,
                                                                     rhs_block_capacity,
                                                                     &rhs_col_slice_end);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    rhs_col_ptr = &rhs_col_blocks[rhs_col_base];
                    status = Q8_0DotBlocksAVX2Q32Checked(lhs_row_ptr,
                                                         rhs_col_ptr,
                                                         k_block_count,
                                                         &cell_dot_q32);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    out_cells_q32[out_index] = cell_dot_q32;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q8_0_MATMUL_OK;
}

I32 Q8_0MatMulQ16TiledAVX2Checked(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_cells_q16,
    I64 out_cell_capacity,
    I64 out_row_stride_cells)
{
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 out_required_cells;
    I64 row_tile_start;
    I64 col_tile_start;
    I64 row_tile_end;
    I64 col_tile_end;
    I64 row_index;
    I64 col_index;
    I64 lhs_row_base;
    I64 lhs_row_slice_end;
    I64 rhs_col_base;
    I64 rhs_col_slice_end;
    I64 out_row_base;
    I64 out_index;
    I64 cell_dot_q16;
    Q8_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    status = Q8_0MatMulTiledValidateArgsChecked(lhs_blocks,
                                                lhs_block_capacity,
                                                row_count,
                                                lhs_row_stride_blocks,
                                                rhs_col_blocks,
                                                rhs_block_capacity,
                                                col_count,
                                                rhs_col_stride_blocks,
                                                k_block_count,
                                                tile_rows,
                                                tile_cols,
                                                out_cells_q16,
                                                out_cell_capacity,
                                                out_row_stride_cells,
                                                &lhs_required_blocks,
                                                &rhs_required_blocks,
                                                &out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    // Tiled traversal mirrors Q32 AVX2 path but stores Q16 outputs.
    // Each cell runs one checked full-row Q32 dot and downshifts once to Q16.
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                0,
                                                row_count,
                                                col_count,
                                                tile_rows,
                                                tile_cols,
                                                &row_tile_end,
                                                &col_tile_end);
        if (status != Q8_0_MATMUL_OK)
            return status;

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            status = Q8_0MatMulTileSchedulerChecked(row_tile_start,
                                                    col_tile_start,
                                                    row_count,
                                                    col_count,
                                                    tile_rows,
                                                    tile_cols,
                                                    &row_tile_end,
                                                    &col_tile_end);
            if (status != Q8_0_MATMUL_OK)
                return status;

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                status = Q8_0MatMulTiledComputeRowBaseChecked(row_index,
                                                              lhs_row_stride_blocks,
                                                              out_row_stride_cells,
                                                              &lhs_row_base,
                                                              &out_row_base);
                if (status != Q8_0_MATMUL_OK)
                    return status;

                status = Q8_0MatMulTiledValidateRowSliceChecked(lhs_row_base,
                                                                 k_block_count,
                                                                 lhs_block_capacity,
                                                                 &lhs_row_slice_end);
                if (status != Q8_0_MATMUL_OK)
                    return status;

                lhs_row_ptr = &lhs_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    status = Q8_0MatMulTiledComputeRhsColBaseChecked(col_index,
                                                                      rhs_col_stride_blocks,
                                                                      &rhs_col_base);
                    if (status != Q8_0_MATMUL_OK)
                        return status;
                    status = Q8_0MatMulTiledValidateOutIndexCapacityChecked(out_row_base,
                                                                             col_index,
                                                                             out_cell_capacity,
                                                                             &out_index);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    status = Q8_0MatMulTiledValidateRowSliceChecked(rhs_col_base,
                                                                     k_block_count,
                                                                     rhs_block_capacity,
                                                                     &rhs_col_slice_end);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    rhs_col_ptr = &rhs_col_blocks[rhs_col_base];
                    status = Q8_0DotBlocksAVX2Q32ToQ16Checked(lhs_row_ptr,
                                                              rhs_col_ptr,
                                                              k_block_count,
                                                              &cell_dot_q16);
                    if (status != Q8_0_MATMUL_OK)
                        return status;

                    out_cells_q16[out_index] = cell_dot_q16;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q8_0_MATMUL_OK;
}

// Scalar checked baseline matmul for Q8_0 rows x Q8_0 column-bank rows.
//
// No-partial contract:
// - Compute every output cell into a private stage buffer first.
// - Publish to caller output only after full row/column pass succeeds.
// - Any error leaves caller `out_cells_q16` unchanged.
I32 Q8_0MatMulQ16NaiveCheckedNoPartial(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q16,
    I64 out_cell_capacity,
    I64 out_row_stride_cells)
{
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 out_required_cells;
    I64 stage_bytes;
    I64 row_index;
    I64 col_index;
    I64 out_index;
    I64 lhs_row_base;
    I64 rhs_col_base;
    I64 lhs_row_slice_end;
    I64 rhs_col_slice_end;
    I64 out_row_base;
    I64 cell_dot_q16;
    I64 *staged_cells_q16;
    Q8_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    if (!lhs_blocks || !rhs_col_blocks || !out_cells_q16)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_cell_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    status = Q8_0MatMulTiledValidateStridesAndKChecked(row_count,
                                                       col_count,
                                                       lhs_row_stride_blocks,
                                                       rhs_col_stride_blocks,
                                                       k_block_count,
                                                       out_row_stride_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTiledComputeRequiredCapacitiesChecked(row_count,
                                                             lhs_row_stride_blocks,
                                                             col_count,
                                                             rhs_col_stride_blocks,
                                                             out_row_stride_cells,
                                                             &lhs_required_blocks,
                                                             &rhs_required_blocks,
                                                             &out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (lhs_required_blocks > lhs_block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (rhs_required_blocks > rhs_block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (out_required_cells > out_cell_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!out_required_cells)
        return Q8_0_MATMUL_OK;

    if (!Q8_0MatMulTryMulI64NonNeg(out_required_cells,
                                   sizeof(I64),
                                   &stage_bytes))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    staged_cells_q16 = MAlloc(stage_bytes);
    if (!staged_cells_q16)
        return Q8_0_MATMUL_ERR_OVERFLOW;

    // Preserve padding semantics: lanes outside active `col_count` but inside
    // `out_row_stride_cells` keep caller values on success.
    for (out_index = 0; out_index < out_required_cells; ++out_index)
        staged_cells_q16[out_index] = out_cells_q16[out_index];

    // Scalar baseline math: explicit row x column traversal with one checked
    // Q8_0 row-dot per output cell.
    for (row_index = 0; row_index < row_count; ++row_index)
    {
        status = Q8_0MatMulTiledComputeRowBaseChecked(row_index,
                                                      lhs_row_stride_blocks,
                                                      out_row_stride_cells,
                                                      &lhs_row_base,
                                                      &out_row_base);
        if (status != Q8_0_MATMUL_OK)
        {
            Free(staged_cells_q16);
            return status;
        }

        status = Q8_0MatMulTiledValidateRowSliceChecked(lhs_row_base,
                                                         k_block_count,
                                                         lhs_block_capacity,
                                                         &lhs_row_slice_end);
        if (status != Q8_0_MATMUL_OK)
        {
            Free(staged_cells_q16);
            return status;
        }

        lhs_row_ptr = &lhs_blocks[lhs_row_base];

        for (col_index = 0; col_index < col_count; ++col_index)
        {
            status = Q8_0MatMulTiledComputeRhsColBaseChecked(col_index,
                                                              rhs_col_stride_blocks,
                                                              &rhs_col_base);
            if (status != Q8_0_MATMUL_OK)
            {
                Free(staged_cells_q16);
                return status;
            }

            status = Q8_0MatMulTiledValidateOutIndexCapacityChecked(out_row_base,
                                                                     col_index,
                                                                     out_required_cells,
                                                                     &out_index);
            if (status != Q8_0_MATMUL_OK)
            {
                Free(staged_cells_q16);
                return status;
            }

            status = Q8_0MatMulTiledValidateRowSliceChecked(rhs_col_base,
                                                             k_block_count,
                                                             rhs_block_capacity,
                                                             &rhs_col_slice_end);
            if (status != Q8_0_MATMUL_OK)
            {
                Free(staged_cells_q16);
                return status;
            }

            rhs_col_ptr = &rhs_col_blocks[rhs_col_base];
            cell_dot_q16 = 0;
            status = Q8_0DotProductBlocksQ16AccumulateChecked(lhs_row_ptr,
                                                              rhs_col_ptr,
                                                              k_block_count,
                                                              &cell_dot_q16);
            if (status != Q8_0_MATMUL_OK)
            {
                Free(staged_cells_q16);
                return status;
            }

            staged_cells_q16[out_index] = cell_dot_q16;
        }
    }

    for (out_index = 0; out_index < out_required_cells; ++out_index)
        out_cells_q16[out_index] = staged_cells_q16[out_index];

    Free(staged_cells_q16);
    return Q8_0_MATMUL_OK;
}

// Commit-only wrapper over no-partial scalar Q8_0 matmul.
//
// Contract:
// - Take immutable snapshots of matrix geometry/strides before compute.
// - Compute into a private stage buffer via canonical no-partial helper.
// - Publish staged rows to caller output only after strict snapshot parity.
// - On any error, caller output remains unchanged.
I32 Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnly(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q16,
    I64 out_cell_capacity,
    I64 out_row_stride_cells)
{
    I64 lhs_required_blocks;
    I64 rhs_required_blocks;
    I64 out_required_cells;
    I64 stage_bytes;
    I64 cell_index;
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_lhs_row_stride_blocks;
    I64 snapshot_rhs_col_stride_blocks;
    I64 snapshot_k_block_count;
    I64 snapshot_out_row_stride_cells;
    I64 *staged_cells_q16;
    I32 status;

    if (!lhs_blocks || !rhs_col_blocks || !out_cells_q16)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_cell_capacity < 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    status = Q8_0MatMulTiledValidateStridesAndKChecked(row_count,
                                                       col_count,
                                                       lhs_row_stride_blocks,
                                                       rhs_col_stride_blocks,
                                                       k_block_count,
                                                       out_row_stride_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    status = Q8_0MatMulTiledComputeRequiredCapacitiesChecked(row_count,
                                                             lhs_row_stride_blocks,
                                                             col_count,
                                                             rhs_col_stride_blocks,
                                                             out_row_stride_cells,
                                                             &lhs_required_blocks,
                                                             &rhs_required_blocks,
                                                             &out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (lhs_required_blocks > lhs_block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (rhs_required_blocks > rhs_block_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (out_required_cells > out_cell_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!out_required_cells)
        return Q8_0_MATMUL_OK;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks;
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks;
    snapshot_k_block_count = k_block_count;
    snapshot_out_row_stride_cells = out_row_stride_cells;

    if (!Q8_0MatMulTryMulI64NonNeg(out_required_cells,
                                   sizeof(I64),
                                   &stage_bytes))
        return Q8_0_MATMUL_ERR_OVERFLOW;

    staged_cells_q16 = MAlloc(stage_bytes);
    if (!staged_cells_q16)
        return Q8_0_MATMUL_ERR_OVERFLOW;

    // Preserve row padding lanes exactly as the canonical no-partial helper.
    for (cell_index = 0; cell_index < out_required_cells; ++cell_index)
        staged_cells_q16[cell_index] = out_cells_q16[cell_index];

    status = Q8_0MatMulQ16NaiveCheckedNoPartial(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        staged_cells_q16,
        out_required_cells,
        out_row_stride_cells);
    if (status != Q8_0_MATMUL_OK)
    {
        Free(staged_cells_q16);
        return status;
    }

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks ||
        snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks ||
        snapshot_k_block_count != k_block_count ||
        snapshot_out_row_stride_cells != out_row_stride_cells)
    {
        Free(staged_cells_q16);
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;
    }

    // Atomic publish region: no remaining error branches.
    for (cell_index = 0; cell_index < out_required_cells; ++cell_index)
        out_cells_q16[cell_index] = staged_cells_q16[cell_index];

    Free(staged_cells_q16);
    return Q8_0_MATMUL_OK;
}

// Diagnostics zero-write companion over commit-only and canonical no-partial
// scalar Q8_0 matmul capacity diagnostics.
//
// Contract:
// - Zero-write path: never publishes caller diagnostics outputs.
// - Recompute canonical 
//   from immutable geometry/capacity snapshots.
// - Enforce strict tuple parity against staged commit-only diagnostics before return.
I32 Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnly(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q16,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_lhs_required_blocks,
    I64 *out_rhs_required_blocks,
    I64 *out_out_required_cells)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_lhs_row_stride_blocks;
    I64 snapshot_rhs_col_stride_blocks;
    I64 snapshot_k_block_count;
    I64 snapshot_out_row_stride_cells;
    I64 snapshot_lhs_block_capacity;
    I64 snapshot_rhs_block_capacity;
    I64 snapshot_out_cell_capacity;
    I64 staged_lhs_required_blocks;
    I64 staged_rhs_required_blocks;
    I64 staged_out_required_cells;
    I64 canonical_lhs_required_blocks;
    I64 canonical_rhs_required_blocks;
    I64 canonical_out_required_cells;
    I32 status;

    if (!out_lhs_required_blocks ||
        !out_rhs_required_blocks ||
        !out_out_required_cells)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    if (out_lhs_required_blocks == out_rhs_required_blocks ||
        out_lhs_required_blocks == out_out_required_cells ||
        out_rhs_required_blocks == out_out_required_cells)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks;
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks;
    snapshot_k_block_count = k_block_count;
    snapshot_out_row_stride_cells = out_row_stride_cells;
    snapshot_lhs_block_capacity = lhs_block_capacity;
    snapshot_rhs_block_capacity = rhs_block_capacity;
    snapshot_out_cell_capacity = out_cell_capacity;

    status = Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnly(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        &staged_lhs_required_blocks,
        &staged_rhs_required_blocks,
        &staged_out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks ||
        snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks ||
        snapshot_k_block_count != k_block_count ||
        snapshot_out_row_stride_cells != out_row_stride_cells ||
        snapshot_lhs_block_capacity != lhs_block_capacity ||
        snapshot_rhs_block_capacity != rhs_block_capacity ||
        snapshot_out_cell_capacity != out_cell_capacity)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    status = Q8_0MatMulQ16NaiveCheckedNoPartial(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        &canonical_lhs_required_blocks,
        &canonical_rhs_required_blocks,
        &canonical_out_required_cells);
    if (status != Q8_0_MATMUL_OK)
        return status;

    if (canonical_lhs_required_blocks != staged_lhs_required_blocks ||
        canonical_rhs_required_blocks != staged_rhs_required_blocks ||
        canonical_out_required_cells != staged_out_required_cells)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    return Q8_0_MATMUL_OK;
}

// Diagnostics zero-write companion over commit-only and canonical no-partial
// scalar Q8_0 matmul paths.
//
// Contract:
// - Evaluate canonical commit-only status first (full checked validation path).
// - Run preflight-only capacity derivation into private staged diagnostics.
// - Enforce immutable geometry/capacity snapshots across both paths.
// - Publish diagnostics tuple only when statuses and tuple values match exactly.
// - On any mismatch/error, caller diagnostics outputs remain unchanged.
I32 Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnlyParity(
    Q8_0Block *lhs_blocks,
    I64 lhs_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q16,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_lhs_required_blocks,
    I64 *out_rhs_required_blocks,
    I64 *out_out_required_cells)
{
    I64 snapshot_lhs_block_capacity;
    I64 snapshot_row_count;
    I64 snapshot_lhs_row_stride_blocks;
    I64 snapshot_rhs_block_capacity;
    I64 snapshot_col_count;
    I64 snapshot_rhs_col_stride_blocks;
    I64 snapshot_k_block_count;
    I64 snapshot_out_cell_capacity;
    I64 snapshot_out_row_stride_cells;
    Q8_0Block *snapshot_lhs_blocks;
    Q8_0Block *snapshot_rhs_col_blocks;
    I64 *snapshot_out_cells_q16;
    I64 *snapshot_out_lhs_required_blocks;
    I64 *snapshot_out_rhs_required_blocks;
    I64 *snapshot_out_out_required_cells;
    I64 staged_lhs_required_blocks;
    I64 staged_rhs_required_blocks;
    I64 staged_out_required_cells;
    I64 canonical_lhs_required_blocks;
    I64 canonical_rhs_required_blocks;
    I64 canonical_out_required_cells;
    I32 canonical_status;
    I32 preflight_status;

    if (!out_lhs_required_blocks ||
        !out_rhs_required_blocks ||
        !out_out_required_cells)
        return Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_lhs_blocks = lhs_blocks;
    snapshot_lhs_block_capacity = lhs_block_capacity;
    snapshot_row_count = row_count;
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks;
    snapshot_rhs_col_blocks = rhs_col_blocks;
    snapshot_rhs_block_capacity = rhs_block_capacity;
    snapshot_col_count = col_count;
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks;
    snapshot_k_block_count = k_block_count;
    snapshot_out_cells_q16 = out_cells_q16;
    snapshot_out_cell_capacity = out_cell_capacity;
    snapshot_out_row_stride_cells = out_row_stride_cells;
    snapshot_out_lhs_required_blocks = out_lhs_required_blocks;
    snapshot_out_rhs_required_blocks = out_rhs_required_blocks;
    snapshot_out_out_required_cells = out_out_required_cells;

    canonical_status = Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnly(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells);

    preflight_status = Q8_0MatMulQ16NaiveCheckedNoPartialCommitOnlyPreflightOnly(
        lhs_blocks,
        lhs_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q16,
        out_cell_capacity,
        out_row_stride_cells,
        &staged_lhs_required_blocks,
        &staged_rhs_required_blocks,
        &staged_out_required_cells);

    if (snapshot_lhs_blocks != lhs_blocks ||
        snapshot_lhs_block_capacity != lhs_block_capacity ||
        snapshot_row_count != row_count ||
        snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks ||
        snapshot_rhs_col_blocks != rhs_col_blocks ||
        snapshot_rhs_block_capacity != rhs_block_capacity ||
        snapshot_col_count != col_count ||
        snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks ||
        snapshot_k_block_count != k_block_count ||
        snapshot_out_cells_q16 != out_cells_q16 ||
        snapshot_out_cell_capacity != out_cell_capacity ||
        snapshot_out_row_stride_cells != out_row_stride_cells ||
        snapshot_out_lhs_required_blocks != out_lhs_required_blocks ||
        snapshot_out_rhs_required_blocks != out_rhs_required_blocks ||
        snapshot_out_out_required_cells != out_out_required_cells)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (preflight_status != canonical_status)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (canonical_status != Q8_0_MATMUL_OK)
        return canonical_status;

    canonical_lhs_required_blocks = 0;
    canonical_rhs_required_blocks = 0;
    canonical_out_required_cells = 0;
    if (Q8_0MatMulTiledComputeRequiredCapacitiesChecked(row_count,
                                                        lhs_row_stride_blocks,
                                                        col_count,
                                                        rhs_col_stride_blocks,
                                                        out_row_stride_cells,
                                                        &canonical_lhs_required_blocks,
                                                        &canonical_rhs_required_blocks,
                                                        &canonical_out_required_cells) != Q8_0_MATMUL_OK)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (staged_lhs_required_blocks != canonical_lhs_required_blocks ||
        staged_rhs_required_blocks != canonical_rhs_required_blocks ||
        staged_out_required_cells != canonical_out_required_cells)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_lhs_required_blocks = staged_lhs_required_blocks;
    *out_rhs_required_blocks = staged_rhs_required_blocks;
    *out_out_required_cells = staged_out_required_cells;

    return Q8_0_MATMUL_OK;
}
