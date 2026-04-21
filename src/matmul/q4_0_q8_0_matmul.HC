// q4_0_q8_0_matmul.HC
// Checked tiled mixed Q4_0 x Q8_0 matmul kernel for TempleOS.
//
// Runtime contract:
//   - lhs is an MxK matrix in row-major block layout (Q4_0 blocks).
//   - rhs is an NxK matrix in row-major "column bank" layout (Q8_0 blocks),
//     where each logical B-column is stored as one contiguous row of K blocks.
//   - out is MxN Q16 matrix (I64 lanes) with explicit output row stride.

#define Q4_0_Q8_0_MATMUL_VALUES_PER_BLOCK 32
#define Q4_0_Q8_0_MATMUL_Q4_PACKED_BYTES  16
#define Q4_0_Q8_0_MATMUL_Q8_PACKED_BYTES  32

#define Q4_0_Q8_0_MATMUL_OK               0
#define Q4_0_Q8_0_MATMUL_ERR_NULL_PTR     1
#define Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN  2
#define Q4_0_Q8_0_MATMUL_ERR_OVERFLOW     3
#define Q4_0_Q8_0_MATMUL_I64_MAX          0x7FFFFFFFFFFFFFFF

#define Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M   4
#define Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N   4
#define Q4_0_Q8_0_MATMUL_DEFAULT_TILE_K   8

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_Q8_0_MATMUL_Q4_PACKED_BYTES];
};

class Q8_0Block
{
    U16 d_fp16;
    I8 qs[Q4_0_Q8_0_MATMUL_Q8_PACKED_BYTES];
};

// Declared in src/quant/q4_0_q8_0_dot.HC.
I32 Q4_0_Q8_0DotProductBlocksQ16AccumulateChecked(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *io_accum_q16);
I32 Q4_0_Q8_0DotBlocksAVX2Q32Checked(Q4_0Block *lhs_q4, Q8_0Block *rhs_q8, I64 block_count, I64 *out_dot_q32);
I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_lhs_blocks,
    I64 *out_required_rhs_blocks,
    I64 *out_required_out_cells,
    I64 *out_tile_m,
    I64 *out_tile_n,
    I64 *out_tile_k_blocks);
I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesRequiredBytes(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_lhs_blocks,
    I64 *out_required_rhs_blocks,
    I64 *out_required_out_cells,
    I64 *out_tile_m,
    I64 *out_tile_n,
    I64 *out_tile_k_blocks);
I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRows(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_total_cells,
    I64 *out_total_block_dots,
    I64 *out_last_status);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyParity(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots);
Bool Q4_0Q8_0MatMulTryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
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

    if (lhs > Q4_0_Q8_0_MATMUL_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q4_0Q8_0MatMulTryAddI64NonNeg(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (lhs < 0 || rhs < 0)
        return FALSE;
    if (lhs > Q4_0_Q8_0_MATMUL_I64_MAX - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

I64 Q4_0Q8_0MatMulMinI64(I64 lhs, I64 rhs)
{
    if (lhs < rhs)
        return lhs;
    return rhs;
}

I32 Q4_0Q8_0MatMulQ16TiledChecked(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
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
    Q4_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    if (!lhs_q4_blocks || !rhs_q8_col_blocks || !out_cells_q16)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_q4_block_capacity < 0 || rhs_q8_block_capacity < 0 || out_cell_capacity < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (row_count < 0 || col_count < 0 || k_block_count < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (lhs_row_stride_blocks < 0 || rhs_col_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Active K span must fit inside each source row stride.
    if (k_block_count > lhs_row_stride_blocks || k_block_count > rhs_col_stride_blocks)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Output row stride must cover active output columns.
    if (row_count > 0 && out_row_stride_cells < col_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Capacity proofs for full row-strided views.
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, lhs_row_stride_blocks, &lhs_required_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_count, rhs_col_stride_blocks, &rhs_required_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, out_row_stride_cells, &out_required_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (lhs_required_blocks > lhs_q4_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (rhs_required_blocks > rhs_q8_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (out_required_cells > out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Tiled traversal keeps cache locality while leaving every row/col dot visible.
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        if (!Q4_0Q8_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q4_0Q8_0MatMulMinI64(row_tile_end, row_count);

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            if (!Q4_0Q8_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q4_0Q8_0MatMulMinI64(col_tile_end, col_count);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
                if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cells, &out_row_base))
                    return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_q4_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
                    if (!Q4_0Q8_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

                    rhs_col_ptr = &rhs_q8_col_blocks[rhs_col_base];
                    cell_dot_q16 = 0;

                    // Core mixed-kernel inner product stays fully integer and checked.
                    status = Q4_0_Q8_0DotProductBlocksQ16AccumulateChecked(lhs_row_ptr, rhs_col_ptr, k_block_count, &cell_dot_q16);
                    if (status != Q4_0_Q8_0_MATMUL_OK)
                        return status;

                    out_cells_q16[out_index] = cell_dot_q16;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2Checked(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
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
    Q4_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    if (!lhs_q4_blocks || !rhs_q8_col_blocks || !out_cells_q32)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_q4_block_capacity < 0 || rhs_q8_block_capacity < 0 || out_cell_capacity < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (row_count < 0 || col_count < 0 || k_block_count < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (lhs_row_stride_blocks < 0 || rhs_col_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Active K span must fit inside each source row stride.
    if (k_block_count > lhs_row_stride_blocks || k_block_count > rhs_col_stride_blocks)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Output row stride must cover active output columns.
    if (row_count > 0 && out_row_stride_cells < col_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Capacity proofs for full row-strided views.
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, lhs_row_stride_blocks, &lhs_required_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_count, rhs_col_stride_blocks, &rhs_required_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, out_row_stride_cells, &out_required_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (lhs_required_blocks > lhs_q4_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (rhs_required_blocks > rhs_q8_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (out_required_cells > out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // AVX2-shape tiled traversal keeps row/column math explicit.
    // Every output cell is one checked mixed Q4_0xQ8_0 block dot over K blocks.
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        if (!Q4_0Q8_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q4_0Q8_0MatMulMinI64(row_tile_end, row_count);

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            if (!Q4_0Q8_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q4_0Q8_0MatMulMinI64(col_tile_end, col_count);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
                if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cells, &out_row_base))
                    return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_q4_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
                    if (!Q4_0Q8_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

                    rhs_col_ptr = &rhs_q8_col_blocks[rhs_col_base];
                    status = Q4_0_Q8_0DotBlocksAVX2Q32Checked(lhs_row_ptr,
                                                              rhs_col_ptr,
                                                              k_block_count,
                                                              &cell_dot_q32);
                    if (status != Q4_0_Q8_0_MATMUL_OK)
                        return status;

                    out_cells_q32[out_index] = cell_dot_q32;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTiles(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells)
{
    I64 required_lhs_blocks;
    I64 required_rhs_blocks;
    I64 required_out_cells;
    I64 tile_m;
    I64 tile_n;
    I64 tile_k_blocks;
    I32 status;

    status = Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        &required_lhs_blocks,
        &required_rhs_blocks,
        &required_out_cells,
        &tile_m,
        &tile_n,
        &tile_k_blocks);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    return Q4_0Q8_0MatMulQ32TiledAVX2Checked(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M,
        Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells);
}

I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_lhs_blocks,
    I64 *out_required_rhs_blocks,
    I64 *out_required_out_cells,
    I64 *out_tile_m,
    I64 *out_tile_n,
    I64 *out_tile_k_blocks)
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
    I64 cell_dot_q32;
    Q4_0Block *lhs_row_ptr;
    Q8_0Block *rhs_col_ptr;
    I32 status;

    if (!out_required_lhs_blocks || !out_required_rhs_blocks ||
        !out_required_out_cells || !out_tile_m || !out_tile_n ||
        !out_tile_k_blocks)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (!lhs_q4_blocks || !rhs_q8_col_blocks || !out_cells_q32)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_q4_block_capacity < 0 || rhs_q8_block_capacity < 0 || out_cell_capacity < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (row_count < 0 || col_count < 0 || k_block_count < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (lhs_row_stride_blocks < 0 || rhs_col_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // IQ-687: default kernel tile contract is fixed at (M,N,K)=(4,4,8).
    // K tile is a scheduling default for the compute kernel family; this
    // wrapper accepts any checked K-block count the canonical kernel accepts.
    if (k_block_count > lhs_row_stride_blocks || k_block_count > rhs_col_stride_blocks)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (row_count > 0 && out_row_stride_cells < col_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, lhs_row_stride_blocks, &lhs_required_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_count, rhs_col_stride_blocks, &rhs_required_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, out_row_stride_cells, &out_required_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (lhs_required_blocks > lhs_q4_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (rhs_required_blocks > rhs_q8_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (out_required_cells > out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_required_lhs_blocks = lhs_required_blocks;
    *out_required_rhs_blocks = rhs_required_blocks;
    *out_required_out_cells = out_required_cells;
    *out_tile_m = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M;
    *out_tile_n = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N;
    *out_tile_k_blocks = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_K;

    // Preflight-only no-partial policy: validate every dot path and publish
    // deterministic required capacities + fixed default tiles with zero output
    // writes. Compute/commit remains in the canonical AVX2 kernel wrapper.
    row_tile_start = 0;
    while (row_tile_start < row_count)
    {
        if (!Q4_0Q8_0MatMulTryAddI64NonNeg(row_tile_start,
                                           Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M,
                                           &row_tile_end))
            return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q4_0Q8_0MatMulMinI64(row_tile_end, row_count);

        col_tile_start = 0;
        while (col_tile_start < col_count)
        {
            if (!Q4_0Q8_0MatMulTryAddI64NonNeg(col_tile_start,
                                               Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N,
                                               &col_tile_end))
                return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q4_0Q8_0MatMulMinI64(col_tile_end, col_count);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_index,
                                                   lhs_row_stride_blocks,
                                                   &lhs_row_base))
                    return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
                lhs_row_ptr = &lhs_q4_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_index,
                                                       rhs_col_stride_blocks,
                                                       &rhs_col_base))
                        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
                    rhs_col_ptr = &rhs_q8_col_blocks[rhs_col_base];

                    cell_dot_q32 = 0;
                    status = Q4_0_Q8_0DotBlocksAVX2Q32Checked(lhs_row_ptr,
                                                              rhs_col_ptr,
                                                              k_block_count,
                                                              &cell_dot_q32);
                    if (status != Q4_0_Q8_0_MATMUL_OK)
                        return status;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesRequiredBytes(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_lhs_blocks,
    I64 *out_required_rhs_blocks,
    I64 *out_required_out_cells,
    I64 *out_tile_m,
    I64 *out_tile_n,
    I64 *out_tile_k_blocks)
{
    I64 required_lhs_blocks;
    I64 required_rhs_blocks;
    I64 required_out_cells;
    I64 tile_m;
    I64 tile_n;
    I64 tile_k_blocks;
    I32 status;

    if (!out_required_lhs_blocks || !out_required_rhs_blocks ||
        !out_required_out_cells || !out_tile_m || !out_tile_n ||
        !out_tile_k_blocks)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    // IQ-690 required-bytes contract: route through canonical default-tiles
    // preflight with local diagnostics first, then publish atomically so error
    // paths cannot leak partially-updated required/tile outputs.
    status = Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        &required_lhs_blocks,
        &required_rhs_blocks,
        &required_out_cells,
        &tile_m,
        &tile_n,
        &tile_k_blocks);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    *out_required_lhs_blocks = required_lhs_blocks;
    *out_required_rhs_blocks = required_rhs_blocks;
    *out_required_out_cells = required_out_cells;
    *out_tile_m = tile_m;
    *out_tile_n = tile_n;
    *out_tile_k_blocks = tile_k_blocks;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRows(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_total_cells,
    I64 *out_total_block_dots,
    I64 *out_last_status)
{
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;
    I64 staged_last_status;
    I64 iter_index;
    I32 status;

    if (!out_total_cells || !out_total_block_dots || !out_last_status)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (iter_count < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // IQ-822: derive deterministic benchmark diagnostics with checked integer
    // math so the benchmark wrapper stays integer-only and overflow-safe.
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count, col_count, &staged_cells_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_cells_per_iter,
                                       k_block_count,
                                       &staged_block_dots_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_cells_per_iter,
                                       iter_count,
                                       &staged_total_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_block_dots_per_iter,
                                       iter_count,
                                       &staged_total_block_dots))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    staged_last_status = Q4_0_Q8_0_MATMUL_OK;

    // Execute full checked AVX2 matmul each loop and preserve strict no-
    // partial diagnostics publish semantics: required outputs commit only
    // after every iteration succeeds.
    for (iter_index = 0; iter_index < iter_count; ++iter_index)
    {
        status = Q4_0Q8_0MatMulQ32TiledAVX2Checked(
            lhs_q4_blocks,
            lhs_q4_block_capacity,
            row_count,
            lhs_row_stride_blocks,
            rhs_q8_col_blocks,
            rhs_q8_block_capacity,
            col_count,
            rhs_col_stride_blocks,
            k_block_count,
            Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M,
            Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N,
            out_cells_q32,
            out_cell_capacity,
            out_row_stride_cells);
        if (status != Q4_0_Q8_0_MATMUL_OK)
            return status;

        staged_last_status = status;
    }

    *out_total_cells = staged_total_cells;
    *out_total_block_dots = staged_total_block_dots;
    *out_last_status = staged_last_status;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;
    I64 staged_required_lhs_blocks;
    I64 staged_required_rhs_blocks;
    I64 staged_required_out_cells;
    I64 staged_tile_m;
    I64 staged_tile_n;
    I64 staged_tile_k_blocks;
    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (iter_count < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    // IQ-825 diagnostics-only bench preflight: validate checked geometry and
    // dot-kernel addressability without writing output cells.
    status = Q4_0Q8_0MatMulTiledQ32CheckedNoAllocDefaultTilesPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        &staged_required_lhs_blocks,
        &staged_required_rhs_blocks,
        &staged_required_out_cells,
        &staged_tile_m,
        &staged_tile_n,
        &staged_tile_k_blocks);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    // Immutable geometry contract: diagnostics-only preflight wrappers must
    // not mutate caller geometry parameters.
    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count,
                                       col_count,
                                       &staged_cells_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_cells_per_iter,
                                       k_block_count,
                                       &staged_block_dots_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_cells_per_iter,
                                       iter_count,
                                       &staged_total_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_block_dots_per_iter,
                                       iter_count,
                                       &staged_total_block_dots))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    *out_cells_per_iter = staged_cells_per_iter;
    *out_block_dots_per_iter = staged_block_dots_per_iter;
    *out_total_cells = staged_total_cells;
    *out_total_block_dots = staged_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;
    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    staged_cells_per_iter = *out_cells_per_iter;
    staged_block_dots_per_iter = *out_block_dots_per_iter;
    staged_total_cells = *out_total_cells;
    staged_total_block_dots = *out_total_block_dots;

    // IQ-826 commit-only diagnostics wrapper: stage benchmark geometry counts
    // via canonical preflight helper, then publish outputs atomically.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &staged_cells_per_iter,
        &staged_block_dots_per_iter,
        &staged_total_cells,
        &staged_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    // Immutable input geometry contract for commit-only diagnostics wrappers.
    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = staged_cells_per_iter;
    *out_block_dots_per_iter = staged_block_dots_per_iter;
    *out_total_cells = staged_total_cells;
    *out_total_block_dots = staged_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 commit_cells_per_iter;
    I64 commit_block_dots_per_iter;
    I64 commit_total_cells;
    I64 commit_total_block_dots;

    I64 canonical_cells_per_iter;
    I64 canonical_block_dots_per_iter;
    I64 canonical_total_cells;
    I64 canonical_total_block_dots;

    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;
    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    commit_cells_per_iter = *out_cells_per_iter;
    commit_block_dots_per_iter = *out_block_dots_per_iter;
    commit_total_cells = *out_total_cells;
    commit_total_block_dots = *out_total_block_dots;

    canonical_cells_per_iter = commit_cells_per_iter;
    canonical_block_dots_per_iter = commit_block_dots_per_iter;
    canonical_total_cells = commit_total_cells;
    canonical_total_block_dots = commit_total_block_dots;

    // IQ-828 diagnostics-only parity gate:
    // 1) stage commit-only diagnostics from IQ-827 helper,
    // 2) stage canonical preflight diagnostics independently,
    // 3) enforce immutable geometry and exact tuple parity,
    // 4) publish atomically only after full parity succeeds.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &commit_cells_per_iter,
        &commit_block_dots_per_iter,
        &commit_total_cells,
        &commit_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &canonical_cells_per_iter,
        &canonical_block_dots_per_iter,
        &canonical_total_cells,
        &canonical_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (commit_cells_per_iter != canonical_cells_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_block_dots_per_iter != canonical_block_dots_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_total_cells != canonical_total_cells)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_total_block_dots != canonical_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = commit_cells_per_iter;
    *out_block_dots_per_iter = commit_block_dots_per_iter;
    *out_total_cells = commit_total_cells;
    *out_total_block_dots = commit_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 parity_cells_per_iter;
    I64 parity_block_dots_per_iter;
    I64 parity_total_cells;
    I64 parity_total_block_dots;

    I64 canonical_cells_per_iter;
    I64 canonical_block_dots_per_iter;
    I64 canonical_total_cells;
    I64 canonical_total_block_dots;

    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;
    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    // Diagnostics-only staging must not depend on caller scratch contents.
    // Start from deterministic neutral values, then fill from the parity
    // helper and publish only after all snapshot/parity checks pass.
    parity_cells_per_iter = 0;
    parity_block_dots_per_iter = 0;
    parity_total_cells = 0;
    parity_total_block_dots = 0;

    canonical_cells_per_iter = 0;
    canonical_block_dots_per_iter = 0;
    canonical_total_cells = 0;
    canonical_total_block_dots = 0;

    // IQ-847 diagnostics-only companion:
    // - stage parity-checked commit-only tuple (no caller writes yet),
    // - recompute canonical tuple from immutable benchmark geometry,
    // - enforce immutable geometry and exact tuple equality,
    // - publish tuple atomically only after all checks pass.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &parity_cells_per_iter,
        &parity_block_dots_per_iter,
        &parity_total_cells,
        &parity_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count,
                                       snapshot_col_count,
                                       &canonical_cells_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_cells_per_iter,
                                       snapshot_k_block_count,
                                       &canonical_block_dots_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_cells_per_iter,
                                       snapshot_iter_count,
                                       &canonical_total_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_block_dots_per_iter,
                                       snapshot_iter_count,
                                       &canonical_total_block_dots))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (parity_cells_per_iter != canonical_cells_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (parity_block_dots_per_iter != canonical_block_dots_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (parity_total_cells != canonical_total_cells)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (parity_total_block_dots != canonical_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = parity_cells_per_iter;
    *out_block_dots_per_iter = parity_block_dots_per_iter;
    *out_total_cells = parity_total_cells;
    *out_total_block_dots = parity_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;

    I64 parity_cells_per_iter;
    I64 parity_block_dots_per_iter;
    I64 parity_total_cells;
    I64 parity_total_block_dots;

    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    parity_cells_per_iter = *out_cells_per_iter;
    parity_block_dots_per_iter = *out_block_dots_per_iter;
    parity_total_cells = *out_total_cells;
    parity_total_block_dots = *out_total_block_dots;

    // IQ-840 commit-only diagnostics wrapper:
    // - stage tuple from parity preflight-only helper,
    // - revalidate immutable benchmark geometry snapshots,
    // - atomically publish tuple only after full success.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &parity_cells_per_iter,
        &parity_block_dots_per_iter,
        &parity_total_cells,
        &parity_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = parity_cells_per_iter;
    *out_block_dots_per_iter = parity_block_dots_per_iter;
    *out_total_cells = parity_total_cells;
    *out_total_block_dots = parity_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyParity(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;

    I64 commit_cells_per_iter;
    I64 commit_block_dots_per_iter;
    I64 commit_total_cells;
    I64 commit_total_block_dots;

    I64 preflight_cells_per_iter;
    I64 preflight_block_dots_per_iter;
    I64 preflight_total_cells;
    I64 preflight_total_block_dots;

    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    commit_cells_per_iter = *out_cells_per_iter;
    commit_block_dots_per_iter = *out_block_dots_per_iter;
    commit_total_cells = *out_total_cells;
    commit_total_block_dots = *out_total_block_dots;

    preflight_cells_per_iter = commit_cells_per_iter;
    preflight_block_dots_per_iter = commit_block_dots_per_iter;
    preflight_total_cells = commit_total_cells;
    preflight_total_block_dots = commit_total_block_dots;

    // IQ-855 strict parity gate:
    // 1) stage tuple via commit-only wrapper over preflight-only parity,
    // 2) stage canonical preflight-only parity tuple independently,
    // 3) enforce immutable benchmark snapshots + exact tuple parity,
    // 4) publish atomically only after full agreement.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &commit_cells_per_iter,
        &commit_block_dots_per_iter,
        &commit_total_cells,
        &commit_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &preflight_cells_per_iter,
        &preflight_block_dots_per_iter,
        &preflight_total_cells,
        &preflight_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (commit_cells_per_iter != preflight_cells_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_block_dots_per_iter != preflight_block_dots_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_total_cells != preflight_total_cells)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_total_block_dots != preflight_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = commit_cells_per_iter;
    *out_block_dots_per_iter = commit_block_dots_per_iter;
    *out_total_cells = commit_total_cells;
    *out_total_block_dots = commit_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;
    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    staged_cells_per_iter = *out_cells_per_iter;
    staged_block_dots_per_iter = *out_block_dots_per_iter;
    staged_total_cells = *out_total_cells;
    staged_total_block_dots = *out_total_block_dots;

    // IQ-839 strict commit-only diagnostics wrapper:
    // 1) stage parity tuple from IQ-828 helper,
    // 2) enforce immutable geometry snapshots,
    // 3) atomically publish staged outputs (no partial writes on failure).
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParity(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &staged_cells_per_iter,
        &staged_block_dots_per_iter,
        &staged_total_cells,
        &staged_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = staged_cells_per_iter;
    *out_block_dots_per_iter = staged_block_dots_per_iter;
    *out_total_cells = staged_total_cells;
    *out_total_block_dots = staged_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 iter_count,
    I64 *out_cells_per_iter,
    I64 *out_block_dots_per_iter,
    I64 *out_total_cells,
    I64 *out_total_block_dots)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;

    I64 commit_cells_per_iter;
    I64 commit_block_dots_per_iter;
    I64 commit_total_cells;
    I64 commit_total_block_dots;

    I64 canonical_cells_per_iter;
    I64 canonical_block_dots_per_iter;
    I64 canonical_total_cells;
    I64 canonical_total_block_dots;

    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    commit_cells_per_iter = *out_cells_per_iter;
    commit_block_dots_per_iter = *out_block_dots_per_iter;
    commit_total_cells = *out_total_cells;
    commit_total_block_dots = *out_total_block_dots;

    // IQ-853 diagnostics-only companion over the commit-only parity wrapper:
    // stage tuple outputs, enforce immutable geometry snapshots, recompute
    // canonical benchmark counts, and publish atomically only on full success.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        iter_count,
        &commit_cells_per_iter,
        &commit_block_dots_per_iter,
        &commit_total_cells,
        &commit_total_block_dots);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_iter_count != iter_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count,
                                       snapshot_col_count,
                                       &canonical_cells_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_cells_per_iter,
                                       snapshot_k_block_count,
                                       &canonical_block_dots_per_iter))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_cells_per_iter,
                                       snapshot_iter_count,
                                       &canonical_total_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_block_dots_per_iter,
                                       snapshot_iter_count,
                                       &canonical_total_block_dots))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (commit_cells_per_iter != canonical_cells_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_block_dots_per_iter != canonical_block_dots_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_total_cells != canonical_total_cells)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (commit_total_block_dots != canonical_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = commit_cells_per_iter;
    *out_block_dots_per_iter = commit_block_dots_per_iter;
    *out_total_cells = commit_total_cells;
    *out_total_block_dots = commit_total_block_dots;
    return Q4_0_Q8_0_MATMUL_OK;
}
