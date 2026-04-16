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
