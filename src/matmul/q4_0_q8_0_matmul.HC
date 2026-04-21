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
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartial(
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
    I64 out_row_stride_cells);
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols);
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTiles(
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
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols);
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnly(
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
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols);
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols);
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParity(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols);
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols);
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParity(
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnly(
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnly(
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnlyPreflightOnly(
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnlyParity(
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyPreflightOnly(
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
I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyParityCommitOnly(
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

Bool Q4_0Q8_0MatMulComputeTileGridChecked(
    I64 row_count,
    I64 col_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_tile_row_count,
    I64 *out_tile_col_count,
    I64 *out_tile_total_count)
{
    I64 row_plus_tile_minus_one;
    I64 col_plus_tile_minus_one;
    I64 tile_row_count;
    I64 tile_col_count;
    I64 tile_total_count;

    if (!out_tile_row_count || !out_tile_col_count || !out_tile_total_count)
        return FALSE;
    if (row_count < 0 || col_count < 0)
        return FALSE;
    if (tile_rows <= 0 || tile_cols <= 0)
        return FALSE;

    if (!Q4_0Q8_0MatMulTryAddI64NonNeg(row_count,
                                       tile_rows - 1,
                                       &row_plus_tile_minus_one))
        return FALSE;
    if (!Q4_0Q8_0MatMulTryAddI64NonNeg(col_count,
                                       tile_cols - 1,
                                       &col_plus_tile_minus_one))
        return FALSE;

    tile_row_count = row_plus_tile_minus_one / tile_rows;
    tile_col_count = col_plus_tile_minus_one / tile_cols;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(tile_row_count,
                                       tile_col_count,
                                       &tile_total_count))
        return FALSE;

    *out_tile_row_count = tile_row_count;
    *out_tile_col_count = tile_col_count;
    *out_tile_total_count = tile_total_count;
    return TRUE;
}

// Returns TRUE when a single I64 diagnostic slot overlaps any byte of the
// output matrix backing range [out_cells_base, out_cells_base + out_cell_count).
//
// This guard prevents commit-only diagnostics from aliasing matrix payload
// storage, which would make metadata publication overwrite model outputs.
Bool Q4_0Q8_0MatMulDiagPtrOverlapsOutCellsRange(I64 *diag_slot,
                                                I64 *out_cells_base,
                                                I64 out_cell_count)
{
    I64 out_bytes;
    U8 *out_begin;
    U8 *out_end;
    U8 *diag_begin;
    U8 *diag_end;

    if (!diag_slot || !out_cells_base)
        return FALSE;
    if (out_cell_count <= 0)
        return FALSE;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(out_cell_count, sizeof(I64), &out_bytes))
        return FALSE;

    out_begin = (U8 *)out_cells_base;
    out_end = out_begin + out_bytes;
    diag_begin = (U8 *)diag_slot;
    diag_end = diag_begin + sizeof(I64);

    if (diag_end <= out_begin)
        return FALSE;
    if (diag_begin >= out_end)
        return FALSE;
    return TRUE;
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

I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartial(
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
    I64 tile_row_count;
    I64 tile_col_count;
    I64 tile_total_count;
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
    I64 staged_cell_q32;
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

    // Tile-geometry preflight: prove tile enumeration is finite and checked
    // before any output commit path can execute.
    if (!Q4_0Q8_0MatMulComputeTileGridChecked(row_count,
                                              col_count,
                                              tile_rows,
                                              tile_cols,
                                              &tile_row_count,
                                              &tile_col_count,
                                              &tile_total_count))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (row_count && col_count && !tile_total_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Preflight pass: compute every destination cell and discard the value.
    // If any checked dot faults, no output cell has been written.
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
                                                              &staged_cell_q32);
                    if (status != Q4_0_Q8_0_MATMUL_OK)
                        return status;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    // Commit pass: preflight proved all indexed cells are valid.
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
                                                              &staged_cell_q32);
                    if (status != Q4_0_Q8_0_MATMUL_OK)
                        return status;

                    out_cells_q32[out_index] = staged_cell_q32;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q4_0_Q8_0_MATMUL_OK;
}

I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(
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
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_tile_rows;
    I64 snapshot_tile_cols;
    I64 snapshot_out_capacity;
    I64 staged_required_out_cells;
    I64 staged_required_out_bytes;
    I64 staged_tile_rows;
    I64 staged_tile_cols;
    I32 status;

    if (!out_required_out_cells || !out_required_out_bytes ||
        !out_tile_rows || !out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_tile_rows = tile_rows;
    snapshot_tile_cols = tile_cols;
    snapshot_out_capacity = out_cell_capacity;

    staged_required_out_cells = 0;
    staged_required_out_bytes = 0;
    staged_tile_rows = 0;
    staged_tile_cols = 0;

    // IQ-993 commit-only diagnostics wrapper:
    // 1) execute canonical no-partial tiled matmul (all output writes live here),
    // 2) stage deterministic diagnostics from immutable geometry snapshots,
    // 3) publish diagnostics atomically only after snapshot revalidation.
    status = MatMulQ4_0Q8_0Q32CheckedTiledNoPartial(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        tile_rows,
        tile_cols,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_tile_rows != tile_rows ||
        snapshot_tile_cols != tile_cols ||
        snapshot_out_capacity != out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count,
                                       out_row_stride_cells,
                                       &staged_required_out_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (staged_required_out_cells > snapshot_out_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_required_out_cells,
                                       sizeof(I64),
                                       &staged_required_out_bytes))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    staged_tile_rows = snapshot_tile_rows;
    staged_tile_cols = snapshot_tile_cols;

    *out_required_out_cells = staged_required_out_cells;
    *out_required_out_bytes = staged_required_out_bytes;
    *out_tile_rows = staged_tile_rows;
    *out_tile_cols = staged_tile_cols;
    return Q4_0_Q8_0_MATMUL_OK;
}

// IQ-1003 default-tile commit-only diagnostics wrapper.
//
// Uses canonical default M/N tile constants, runs commit-only matmul path, then
// recomputes deterministic diagnostics from immutable geometry snapshots and
// atomically publishes diagnostics only after parity checks succeed.
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTiles(
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
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_out_capacity;
    I64 staged_required_out_cells;
    I64 staged_required_out_bytes;
    I64 staged_tile_rows;
    I64 staged_tile_cols;
    I64 canonical_required_out_cells;
    I64 canonical_required_out_bytes;
    I64 canonical_tile_rows;
    I64 canonical_tile_cols;
    I32 status;

    if (!out_required_out_cells || !out_required_out_bytes ||
        !out_tile_rows || !out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (out_required_out_cells == out_required_out_bytes ||
        out_required_out_cells == out_tile_rows ||
        out_required_out_cells == out_tile_cols ||
        out_required_out_bytes == out_tile_rows ||
        out_required_out_bytes == out_tile_cols ||
        out_tile_rows == out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    // Diagnostics must remain logically disjoint from payload output cells.
    // When aliased, metadata publication could corrupt quantized matmul output.
    if (Q4_0Q8_0MatMulDiagPtrOverlapsOutCellsRange(out_required_out_cells,
                                                   out_cells_q32,
                                                   out_cell_capacity) ||
        Q4_0Q8_0MatMulDiagPtrOverlapsOutCellsRange(out_required_out_bytes,
                                                   out_cells_q32,
                                                   out_cell_capacity) ||
        Q4_0Q8_0MatMulDiagPtrOverlapsOutCellsRange(out_tile_rows,
                                                   out_cells_q32,
                                                   out_cell_capacity) ||
        Q4_0Q8_0MatMulDiagPtrOverlapsOutCellsRange(out_tile_cols,
                                                   out_cells_q32,
                                                   out_cell_capacity))
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (out_required_out_cells == (I64 *)lhs_q4_blocks ||
        out_required_out_cells == (I64 *)rhs_q8_col_blocks ||
        out_required_out_cells == out_cells_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4_blocks ||
        out_required_out_bytes == (I64 *)rhs_q8_col_blocks ||
        out_required_out_bytes == out_cells_q32 ||
        out_tile_rows == (I64 *)lhs_q4_blocks ||
        out_tile_rows == (I64 *)rhs_q8_col_blocks ||
        out_tile_rows == out_cells_q32 ||
        out_tile_cols == (I64 *)lhs_q4_blocks ||
        out_tile_cols == (I64 *)rhs_q8_col_blocks ||
        out_tile_cols == out_cells_q32)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_out_capacity = out_cell_capacity;

    staged_required_out_cells = 0;
    staged_required_out_bytes = 0;
    staged_tile_rows = 0;
    staged_tile_cols = 0;

    status = MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(
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
        out_row_stride_cells,
        &staged_required_out_cells,
        &staged_required_out_bytes,
        &staged_tile_rows,
        &staged_tile_cols);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_out_capacity != out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count,
                                       out_row_stride_cells,
                                       &canonical_required_out_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (canonical_required_out_cells > snapshot_out_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_required_out_cells,
                                       sizeof(I64),
                                       &canonical_required_out_bytes))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    canonical_tile_rows = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M;
    canonical_tile_cols = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N;

    if (staged_required_out_cells != canonical_required_out_cells ||
        staged_required_out_bytes != canonical_required_out_bytes ||
        staged_tile_rows != canonical_tile_rows ||
        staged_tile_cols != canonical_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_required_out_cells = staged_required_out_cells;
    *out_required_out_bytes = staged_required_out_bytes;
    *out_tile_rows = staged_tile_rows;
    *out_tile_cols = staged_tile_cols;
    return Q4_0_Q8_0_MATMUL_OK;
}

// IQ-1012 diagnostics-only no-write companion over default-tile commit-only path.
//
// Composes the canonical tiled preflight-only checker with default M/N tile
// constants, then enforces immutable snapshot + canonical tuple parity before
// publishing diagnostics so caller output cells are never modified.
I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyDefaultTilesPreflightOnly(
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
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_out_capacity;
    I64 staged_required_out_cells;
    I64 staged_required_out_bytes;
    I64 staged_tile_rows;
    I64 staged_tile_cols;
    I64 canonical_required_out_cells;
    I64 canonical_required_out_bytes;
    I64 canonical_tile_rows;
    I64 canonical_tile_cols;
    I32 status;

    if (!out_required_out_cells || !out_required_out_bytes ||
        !out_tile_rows || !out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (out_required_out_cells == out_required_out_bytes ||
        out_required_out_cells == out_tile_rows ||
        out_required_out_cells == out_tile_cols ||
        out_required_out_bytes == out_tile_rows ||
        out_required_out_bytes == out_tile_cols ||
        out_tile_rows == out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (out_required_out_cells == (I64 *)lhs_q4_blocks ||
        out_required_out_cells == (I64 *)rhs_q8_col_blocks ||
        out_required_out_cells == out_cells_q32 ||
        out_required_out_bytes == (I64 *)lhs_q4_blocks ||
        out_required_out_bytes == (I64 *)rhs_q8_col_blocks ||
        out_required_out_bytes == out_cells_q32 ||
        out_tile_rows == (I64 *)lhs_q4_blocks ||
        out_tile_rows == (I64 *)rhs_q8_col_blocks ||
        out_tile_rows == out_cells_q32 ||
        out_tile_cols == (I64 *)lhs_q4_blocks ||
        out_tile_cols == (I64 *)rhs_q8_col_blocks ||
        out_tile_cols == out_cells_q32)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_out_capacity = out_cell_capacity;

    staged_required_out_cells = 0;
    staged_required_out_bytes = 0;
    staged_tile_rows = 0;
    staged_tile_cols = 0;

    status = MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(
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
        out_row_stride_cells,
        &staged_required_out_cells,
        &staged_required_out_bytes,
        &staged_tile_rows,
        &staged_tile_cols);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_out_capacity != out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count,
                                       out_row_stride_cells,
                                       &canonical_required_out_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (canonical_required_out_cells > snapshot_out_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_required_out_cells,
                                       sizeof(I64),
                                       &canonical_required_out_bytes))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    canonical_tile_rows = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_M;
    canonical_tile_cols = Q4_0_Q8_0_MATMUL_DEFAULT_TILE_N;

    if (staged_required_out_cells != canonical_required_out_cells ||
        staged_required_out_bytes != canonical_required_out_bytes ||
        staged_tile_rows != canonical_tile_rows ||
        staged_tile_cols != canonical_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_required_out_cells = staged_required_out_cells;
    *out_required_out_bytes = staged_required_out_bytes;
    *out_tile_rows = staged_tile_rows;
    *out_tile_cols = staged_tile_cols;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_tile_row_span;
    I64 snapshot_tile_col_span;
    I64 snapshot_out_capacity;

    I64 required_lhs_blocks;
    I64 required_rhs_blocks;
    I64 staged_required_out_cells;
    I64 staged_required_out_bytes;
    I64 staged_tile_rows;
    I64 staged_tile_cols;

    I64 tile_row_count;
    I64 tile_col_count;
    I64 tile_total_count;

    // IQ-994 diagnostics-only no-write preflight:
    // validate canonical tiled matmul geometry/capacity contract without
    // touching caller output cells, then atomically publish diagnostics.
    if (!out_required_out_cells || !out_required_out_bytes ||
        !out_tile_rows || !out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (!lhs_q4_blocks || !rhs_q8_col_blocks || !out_cells_q32)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (lhs_q4_block_capacity < 0 || rhs_q8_block_capacity < 0 || out_cell_capacity < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (row_count < 0 || col_count < 0 || k_block_count < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (lhs_row_stride_blocks < 0 || rhs_col_stride_blocks < 0 || out_row_stride_cells < 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (tile_row_span <= 0 || tile_col_span <= 0)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (k_block_count > lhs_row_stride_blocks || k_block_count > rhs_col_stride_blocks)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (row_count > 0 && out_row_stride_cells < col_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_tile_row_span = tile_row_span;
    snapshot_tile_col_span = tile_col_span;
    snapshot_out_capacity = out_cell_capacity;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count,
                                       lhs_row_stride_blocks,
                                       &required_lhs_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(col_count,
                                       rhs_col_stride_blocks,
                                       &required_rhs_blocks))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(row_count,
                                       out_row_stride_cells,
                                       &staged_required_out_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (required_lhs_blocks > lhs_q4_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (required_rhs_blocks > rhs_q8_block_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (staged_required_out_cells > out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulComputeTileGridChecked(row_count,
                                              col_count,
                                              tile_row_span,
                                              tile_col_span,
                                              &tile_row_count,
                                              &tile_col_count,
                                              &tile_total_count))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;
    if (row_count && col_count && !tile_total_count)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(staged_required_out_cells,
                                       sizeof(I64),
                                       &staged_required_out_bytes))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    // Publish diagnostics from immutable snapshots only. This keeps the
    // no-partial contract explicit: staged tuple must be derived from the
    // same geometry we revalidate right before commit.
    staged_tile_rows = snapshot_tile_row_span;
    staged_tile_cols = snapshot_tile_col_span;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_tile_row_span != tile_row_span ||
        snapshot_tile_col_span != tile_col_span ||
        snapshot_out_capacity != out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_required_out_cells = staged_required_out_cells;
    *out_required_out_bytes = staged_required_out_bytes;
    *out_tile_rows = staged_tile_rows;
    *out_tile_cols = staged_tile_cols;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParity(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_tile_row_span;
    I64 snapshot_tile_col_span;
    I64 snapshot_out_capacity;

    I64 staged_pre_required_out_cells;
    I64 staged_pre_required_out_bytes;
    I64 staged_pre_tile_rows;
    I64 staged_pre_tile_cols;

    I64 staged_commit_required_out_cells;
    I64 staged_commit_required_out_bytes;
    I64 staged_commit_tile_rows;
    I64 staged_commit_tile_cols;
    I32 status;

    // IQ-995 diagnostics-only parity gate:
    // 1) no-write preflight wrapper computes canonical diagnostics tuple;
    // 2) canonical commit wrapper performs output writes and emits tuple;
    // 3) tuple parity + immutable snapshot checks gate one atomic diagnostics publish.
    if (!out_required_out_cells || !out_required_out_bytes ||
        !out_tile_rows || !out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_tile_row_span = tile_row_span;
    snapshot_tile_col_span = tile_col_span;
    snapshot_out_capacity = out_cell_capacity;

    staged_pre_required_out_cells = 0;
    staged_pre_required_out_bytes = 0;
    staged_pre_tile_rows = 0;
    staged_pre_tile_cols = 0;
    staged_commit_required_out_cells = 0;
    staged_commit_required_out_bytes = 0;
    staged_commit_tile_rows = 0;
    staged_commit_tile_cols = 0;

    status = MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        tile_row_span,
        tile_col_span,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        &staged_pre_required_out_cells,
        &staged_pre_required_out_bytes,
        &staged_pre_tile_rows,
        &staged_pre_tile_cols);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    status = MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnly(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        tile_row_span,
        tile_col_span,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        &staged_commit_required_out_cells,
        &staged_commit_required_out_bytes,
        &staged_commit_tile_rows,
        &staged_commit_tile_cols);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_tile_row_span != tile_row_span ||
        snapshot_tile_col_span != tile_col_span ||
        snapshot_out_capacity != out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (staged_pre_required_out_cells != staged_commit_required_out_cells ||
        staged_pre_required_out_bytes != staged_commit_required_out_bytes ||
        staged_pre_tile_rows != staged_commit_tile_rows ||
        staged_pre_tile_cols != staged_commit_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_required_out_cells = staged_pre_required_out_cells;
    *out_required_out_bytes = staged_pre_required_out_bytes;
    *out_tile_rows = staged_pre_tile_rows;
    *out_tile_cols = staged_pre_tile_cols;
    return Q4_0_Q8_0_MATMUL_OK;
}

I32 MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    Q4_0Block *lhs_q4_blocks,
    I64 lhs_q4_block_capacity,
    I64 row_count,
    I64 lhs_row_stride_blocks,
    Q8_0Block *rhs_q8_col_blocks,
    I64 rhs_q8_block_capacity,
    I64 col_count,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_row_span,
    I64 tile_col_span,
    I64 *out_cells_q32,
    I64 out_cell_capacity,
    I64 out_row_stride_cells,
    I64 *out_required_out_cells,
    I64 *out_required_out_bytes,
    I64 *out_tile_rows,
    I64 *out_tile_cols)
{
    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_tile_row_span;
    I64 snapshot_tile_col_span;
    I64 snapshot_out_capacity;

    I64 staged_required_out_cells;
    I64 staged_required_out_bytes;
    I64 staged_tile_rows;
    I64 staged_tile_cols;

    I64 canonical_required_out_cells;
    I64 canonical_required_out_bytes;
    I64 canonical_tile_rows;
    I64 canonical_tile_cols;
    I32 status;

    // IQ-1004 commit-only wrapper:
    // 1) execute parity-checked commit path (includes output writes),
    // 2) recompute canonical diagnostics tuple from immutable snapshots,
    // 3) atomically publish only after tuple parity and snapshot parity hold.
    if (!out_required_out_cells || !out_required_out_bytes ||
        !out_tile_rows || !out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    if (out_required_out_cells == out_required_out_bytes ||
        out_required_out_cells == out_tile_rows ||
        out_required_out_cells == out_tile_cols ||
        out_required_out_bytes == out_tile_rows ||
        out_required_out_bytes == out_tile_cols ||
        out_tile_rows == out_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (out_required_out_cells == out_cells_q32 ||
        out_required_out_bytes == out_cells_q32 ||
        out_tile_rows == out_cells_q32 ||
        out_tile_cols == out_cells_q32)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_tile_row_span = tile_row_span;
    snapshot_tile_col_span = tile_col_span;
    snapshot_out_capacity = out_cell_capacity;

    staged_required_out_cells = 0;
    staged_required_out_bytes = 0;
    staged_tile_rows = 0;
    staged_tile_cols = 0;

    status = MatMulQ4_0Q8_0Q32CheckedTiledNoPartialCommitOnlyPreflightOnlyParity(
        lhs_q4_blocks,
        lhs_q4_block_capacity,
        row_count,
        lhs_row_stride_blocks,
        rhs_q8_col_blocks,
        rhs_q8_block_capacity,
        col_count,
        rhs_col_stride_blocks,
        k_block_count,
        tile_row_span,
        tile_col_span,
        out_cells_q32,
        out_cell_capacity,
        out_row_stride_cells,
        &staged_required_out_cells,
        &staged_required_out_bytes,
        &staged_tile_rows,
        &staged_tile_cols);
    if (status != Q4_0_Q8_0_MATMUL_OK)
        return status;

    if (snapshot_row_count != row_count ||
        snapshot_col_count != col_count ||
        snapshot_k_block_count != k_block_count ||
        snapshot_tile_row_span != tile_row_span ||
        snapshot_tile_col_span != tile_col_span ||
        snapshot_out_capacity != out_cell_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(snapshot_row_count,
                                       out_row_stride_cells,
                                       &canonical_required_out_cells))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    if (canonical_required_out_cells > snapshot_out_capacity)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    if (!Q4_0Q8_0MatMulTryMulI64NonNeg(canonical_required_out_cells,
                                       sizeof(I64),
                                       &canonical_required_out_bytes))
        return Q4_0_Q8_0_MATMUL_ERR_OVERFLOW;

    canonical_tile_rows = snapshot_tile_row_span;
    canonical_tile_cols = snapshot_tile_col_span;

    if (staged_required_out_cells != canonical_required_out_cells ||
        staged_required_out_bytes != canonical_required_out_bytes ||
        staged_tile_rows != canonical_tile_rows ||
        staged_tile_cols != canonical_tile_cols)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_required_out_cells = staged_required_out_cells;
    *out_required_out_bytes = staged_required_out_bytes;
    *out_tile_rows = staged_tile_rows;
    *out_tile_cols = staged_tile_cols;
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
    I64 snapshot_lhs_q4_block_capacity;
    I64 snapshot_rhs_q8_block_capacity;
    I64 snapshot_out_cell_capacity;
    I64 snapshot_lhs_row_stride_blocks;
    I64 snapshot_rhs_col_stride_blocks;
    I64 snapshot_out_row_stride_cells;
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
    snapshot_lhs_q4_block_capacity = lhs_q4_block_capacity;
    snapshot_rhs_q8_block_capacity = rhs_q8_block_capacity;
    snapshot_out_cell_capacity = out_cell_capacity;
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks;
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks;
    snapshot_out_row_stride_cells = out_row_stride_cells;

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
        snapshot_iter_count != iter_count ||
        snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity ||
        snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity ||
        snapshot_out_cell_capacity != out_cell_capacity ||
        snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks ||
        snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks ||
        snapshot_out_row_stride_cells != out_row_stride_cells)
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

    I64 canonical_cells_per_iter;
    I64 canonical_block_dots_per_iter;
    I64 canonical_total_cells;
    I64 canonical_total_block_dots;

    I64 snapshot_row_count;
    I64 snapshot_col_count;
    I64 snapshot_k_block_count;
    I64 snapshot_iter_count;
    I64 snapshot_lhs_q4_block_capacity;
    I64 snapshot_rhs_q8_block_capacity;
    I64 snapshot_out_cell_capacity;
    I64 snapshot_lhs_row_stride_blocks;
    I64 snapshot_rhs_col_stride_blocks;
    I64 snapshot_out_row_stride_cells;
    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;
    snapshot_lhs_q4_block_capacity = lhs_q4_block_capacity;
    snapshot_rhs_q8_block_capacity = rhs_q8_block_capacity;
    snapshot_out_cell_capacity = out_cell_capacity;
    snapshot_lhs_row_stride_blocks = lhs_row_stride_blocks;
    snapshot_rhs_col_stride_blocks = rhs_col_stride_blocks;
    snapshot_out_row_stride_cells = out_row_stride_cells;

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
        snapshot_iter_count != iter_count ||
        snapshot_lhs_q4_block_capacity != lhs_q4_block_capacity ||
        snapshot_rhs_q8_block_capacity != rhs_q8_block_capacity ||
        snapshot_out_cell_capacity != out_cell_capacity ||
        snapshot_lhs_row_stride_blocks != lhs_row_stride_blocks ||
        snapshot_rhs_col_stride_blocks != rhs_col_stride_blocks ||
        snapshot_out_row_stride_cells != out_row_stride_cells)
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

    // IQ-858 strict parity gate:
    // 1) stage tuple from commit-only parity-preflight composition,
    // 2) stage canonical tuple from parity preflight-only helper,
    // 3) enforce immutable geometry snapshots and exact tuple equality,
    // 4) atomically publish diagnostics only on full success.
    commit_cells_per_iter = *out_cells_per_iter;
    commit_block_dots_per_iter = *out_block_dots_per_iter;
    commit_total_cells = *out_total_cells;
    commit_total_block_dots = *out_total_block_dots;

    preflight_cells_per_iter = 0;
    preflight_block_dots_per_iter = 0;
    preflight_total_cells = 0;
    preflight_total_block_dots = 0;

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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyParityCommitOnly(
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

    // IQ-864 commit-only diagnostics wrapper over IQ-858 parity gate:
    // stage parity tuple outputs, enforce immutable geometry snapshots,
    // and atomically publish only after successful full preflight pass.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyParity(
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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityPreflightOnlyCommitOnlyPreflightOnly(
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
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;

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

    staged_cells_per_iter = 0;
    staged_block_dots_per_iter = 0;
    staged_total_cells = 0;
    staged_total_block_dots = 0;

    canonical_cells_per_iter = 0;
    canonical_block_dots_per_iter = 0;
    canonical_total_cells = 0;
    canonical_total_block_dots = 0;

    // IQ-857 diagnostics-only no-write companion:
    // - stage tuple from parity commit-only wrapper,
    // - recompute canonical tuple under immutable geometry snapshot,
    // - reject on any mismatch with zero publish,
    // - atomically publish only after full agreement.
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

    if (staged_cells_per_iter != canonical_cells_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (staged_block_dots_per_iter != canonical_block_dots_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (staged_total_cells != canonical_total_cells)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (staged_total_block_dots != canonical_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;

    *out_cells_per_iter = staged_cells_per_iter;
    *out_block_dots_per_iter = staged_block_dots_per_iter;
    *out_total_cells = staged_total_cells;
    *out_total_block_dots = staged_total_block_dots;
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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParity(
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

    I64 preflight_cells_per_iter;
    I64 preflight_block_dots_per_iter;
    I64 preflight_total_cells;
    I64 preflight_total_block_dots;

    I64 commit_cells_per_iter;
    I64 commit_block_dots_per_iter;
    I64 commit_total_cells;
    I64 commit_total_block_dots;

    I32 status;

    if (!out_cells_per_iter || !out_block_dots_per_iter ||
        !out_total_cells || !out_total_block_dots)
        return Q4_0_Q8_0_MATMUL_ERR_NULL_PTR;

    snapshot_row_count = row_count;
    snapshot_col_count = col_count;
    snapshot_k_block_count = k_block_count;
    snapshot_iter_count = iter_count;

    preflight_cells_per_iter = *out_cells_per_iter;
    preflight_block_dots_per_iter = *out_block_dots_per_iter;
    preflight_total_cells = *out_total_cells;
    preflight_total_block_dots = *out_total_block_dots;

    commit_cells_per_iter = preflight_cells_per_iter;
    commit_block_dots_per_iter = preflight_block_dots_per_iter;
    commit_total_cells = preflight_total_cells;
    commit_total_block_dots = preflight_total_block_dots;

    // IQ-856 strict parity gate:
    // stage diagnostics via no-write preflight companion + canonical
    // commit-only parity wrapper, rederive canonical tuple from immutable
    // benchmark geometry, and publish atomically only on full agreement.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(
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

    if (preflight_cells_per_iter != commit_cells_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (preflight_block_dots_per_iter != commit_block_dots_per_iter)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (preflight_total_cells != commit_total_cells)
        return Q4_0_Q8_0_MATMUL_ERR_BAD_DST_LEN;
    if (preflight_total_block_dots != commit_total_block_dots)
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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnly(
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

    parity_cells_per_iter = 0;
    parity_block_dots_per_iter = 0;
    parity_total_cells = 0;
    parity_total_block_dots = 0;

    canonical_cells_per_iter = 0;
    canonical_block_dots_per_iter = 0;
    canonical_total_cells = 0;
    canonical_total_block_dots = 0;

    // IQ-862 diagnostics-only no-write companion over IQ-856 parity gate:
    // stage tuple outputs through the canonical parity wrapper, rederive
    // benchmark counts from immutable geometry, and publish only on exact
    // tuple agreement with strict zero-publish semantics on failure.
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParity(
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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnly(
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
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;

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

    staged_cells_per_iter = *out_cells_per_iter;
    staged_block_dots_per_iter = *out_block_dots_per_iter;
    staged_total_cells = *out_total_cells;
    staged_total_block_dots = *out_total_block_dots;

    // IQ-863 commit-only diagnostics wrapper over IQ-862:
    // 1) stage benchmark tuple outputs,
    // 2) enforce immutable geometry snapshots,
    // 3) atomically publish staged tuple (no partial writes on failure).
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnly(
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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnlyParity(
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

    I64 preflight_cells_per_iter;
    I64 preflight_block_dots_per_iter;
    I64 preflight_total_cells;
    I64 preflight_total_block_dots;

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

    // IQ-868 strict parity gate:
    // 1) stage tuple from IQ-863 commit-only wrapper;
    // 2) stage tuple from IQ-862 diagnostics-only wrapper;
    // 3) enforce immutable geometry snapshots + exact tuple parity;
    // 4) atomically publish only on full agreement.
    commit_cells_per_iter = *out_cells_per_iter;
    commit_block_dots_per_iter = *out_block_dots_per_iter;
    commit_total_cells = *out_total_cells;
    commit_total_block_dots = *out_total_block_dots;

    preflight_cells_per_iter = 0;
    preflight_block_dots_per_iter = 0;
    preflight_total_cells = 0;
    preflight_total_block_dots = 0;

    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnly(
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

    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnly(
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

I32 Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnlyPreflightOnly(
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
    I64 staged_cells_per_iter;
    I64 staged_block_dots_per_iter;
    I64 staged_total_cells;
    I64 staged_total_block_dots;

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

    staged_cells_per_iter = 0;
    staged_block_dots_per_iter = 0;
    staged_total_cells = 0;
    staged_total_block_dots = 0;

    // IQ-869 diagnostics-only no-write companion over IQ-863:
    // 1) stage benchmark tuple outputs through the commit-only helper,
    // 2) enforce immutable benchmark geometry snapshots,
    // 3) atomically publish only on full success (strict zero-write on fail).
    status = Q4_0Q8_0MatMulQ32TiledAVX2BenchRowsPreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityPreflightOnlyCommitOnly(
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
