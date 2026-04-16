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
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q8_0_MATMUL_ERR_BAD_DST_LEN;

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
    I64 rhs_col_base;
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
        if (!Q8_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q8_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q8_0MatMulMinI64(row_tile_end, row_count);

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            if (!Q8_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q8_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q8_0MatMulMinI64(col_tile_end, col_count);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q8_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q8_0_MATMUL_ERR_OVERFLOW;
                if (!Q8_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cells, &out_row_base))
                    return Q8_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q8_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q8_0_MATMUL_ERR_OVERFLOW;
                    if (!Q8_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q8_0_MATMUL_ERR_OVERFLOW;

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
    I64 rhs_col_base;
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
        if (!Q8_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q8_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q8_0MatMulMinI64(row_tile_end, row_count);

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            if (!Q8_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q8_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q8_0MatMulMinI64(col_tile_end, col_count);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q8_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q8_0_MATMUL_ERR_OVERFLOW;
                if (!Q8_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cells, &out_row_base))
                    return Q8_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q8_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q8_0_MATMUL_ERR_OVERFLOW;
                    if (!Q8_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q8_0_MATMUL_ERR_OVERFLOW;

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
    I64 rhs_col_base;
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
        if (!Q8_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q8_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q8_0MatMulMinI64(row_tile_end, row_count);

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            if (!Q8_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q8_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q8_0MatMulMinI64(col_tile_end, col_count);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q8_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q8_0_MATMUL_ERR_OVERFLOW;
                if (!Q8_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cells, &out_row_base))
                    return Q8_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q8_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q8_0_MATMUL_ERR_OVERFLOW;
                    if (!Q8_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q8_0_MATMUL_ERR_OVERFLOW;

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
