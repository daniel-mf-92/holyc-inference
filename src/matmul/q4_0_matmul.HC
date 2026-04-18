// q4_0_matmul.HC
// Checked tiled Q4_0 x Q4_0 matmul (Q32 accumulators) for TempleOS.
//
// Layout contract:
// - lhs_matrix_blocks: A matrix rows in row-major block layout.
// - rhs_col_blocks: B matrix columns pre-transposed into column-bank rows.
// - Each logical dot uses `k_block_count` active Q4_0 blocks.
// - Output is row-major Q32 (`I64`) with explicit output row stride.

#define Q4_0_MATMUL_OK            0
#define Q4_0_MATMUL_ERR_NULL_PTR  1
#define Q4_0_MATMUL_ERR_BAD_LEN   2
#define Q4_0_MATMUL_ERR_OVERFLOW  3
#define Q4_0_MATMUL_I64_MAX       0x7FFFFFFFFFFFFFFF

#define Q4_0_MATMUL_PACKED_BYTES  16

class Q4_0Block
{
    U16 d_fp16;
    U8 qs[Q4_0_MATMUL_PACKED_BYTES];
};

// Declared in src/quant/q4_0_avx2.HC.
I32 Q4_0DotRowsAVX2Q32Checked(
    Q4_0Block *matrix_blocks,
    I64 matrix_block_capacity,
    I64 row_count,
    I64 row_stride_blocks,
    Q4_0Block *vec_blocks,
    I64 vec_block_capacity,
    I64 vec_block_count,
    I64 *out_rows_q32);

I32 Q4_0DotBlocksAVX2Q32ToQ16Checked(
    Q4_0Block *lhs,
    Q4_0Block *rhs,
    I64 block_count,
    I64 *out_dot_q16);

Bool Q4_0MatMulTryMulI64NonNeg(I64 lhs, I64 rhs, I64 *out_prod)
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

    if (lhs > Q4_0_MATMUL_I64_MAX / rhs)
        return FALSE;

    *out_prod = lhs * rhs;
    return TRUE;
}

Bool Q4_0MatMulTryAddI64NonNeg(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return FALSE;
    if (lhs < 0 || rhs < 0)
        return FALSE;
    if (lhs > Q4_0_MATMUL_I64_MAX - rhs)
        return FALSE;

    *out_sum = lhs + rhs;
    return TRUE;
}

I64 Q4_0MatMulMinI64(I64 lhs, I64 rhs)
{
    if (lhs < rhs)
        return lhs;
    return rhs;
}

I32 Q4_0MatMulTiledAVX2Q32Checked(
    Q4_0Block *lhs_matrix_blocks,
    I64 lhs_block_capacity,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q4_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_mat_q32,
    I64 out_capacity,
    I64 out_row_stride_cols)
{
    I64 required_lhs_blocks;
    I64 required_rhs_blocks;
    I64 required_out_cells;
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
    Q4_0Block *rhs_col_ptr;
    I32 status;

    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q32)
        return Q4_0_MATMUL_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_capacity < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (lhs_rows < 0 || lhs_row_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (rhs_cols < 0 || rhs_col_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (k_block_count < 0 || out_row_stride_cols < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Active K span must fit in both row strides.
    if (lhs_rows > 0 && lhs_row_stride_blocks < k_block_count)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (rhs_cols > 0 && rhs_col_stride_blocks < k_block_count)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Output stride must be wide enough for the active output columns.
    if (lhs_rows > 0 && out_row_stride_cols < rhs_cols)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Capacity proofs for full strided views.
    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, lhs_row_stride_blocks, &required_lhs_blocks))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(rhs_cols, rhs_col_stride_blocks, &required_rhs_blocks))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, out_row_stride_cols, &required_out_cells))
        return Q4_0_MATMUL_ERR_OVERFLOW;

    if (required_lhs_blocks > lhs_block_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (required_rhs_blocks > rhs_block_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (required_out_cells > out_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Tile loops keep future AVX2 kernel boundaries explicit while routing
    // each row/column dot through the checked row-dot primitive.
    row_tile_start = 0;
    while (row_tile_start < lhs_rows)
    {
        if (!Q4_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q4_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q4_0MatMulMinI64(row_tile_end, lhs_rows);

        col_tile_start = 0;
        while (col_tile_start < rhs_cols)
        {
            if (!Q4_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q4_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q4_0MatMulMinI64(col_tile_end, rhs_cols);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q4_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q4_0_MATMUL_ERR_OVERFLOW;
                if (!Q4_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cols, &out_row_base))
                    return Q4_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_matrix_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q4_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q4_0_MATMUL_ERR_OVERFLOW;
                    if (!Q4_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q4_0_MATMUL_ERR_OVERFLOW;

                    rhs_col_ptr = &rhs_col_blocks[rhs_col_base];

                    // Use checked row-dot kernel with row_count=1.
                    // This preserves the exact Q4_0 AVX2-shape arithmetic contract.
                    status = Q4_0DotRowsAVX2Q32Checked(lhs_row_ptr,
                                                       lhs_block_capacity - lhs_row_base,
                                                       1,
                                                       lhs_row_stride_blocks,
                                                       rhs_col_ptr,
                                                       rhs_block_capacity - rhs_col_base,
                                                       k_block_count,
                                                       &cell_dot_q32);
                    if (status != Q4_0_MATMUL_OK)
                        return status;

                    out_mat_q32[out_index] = cell_dot_q32;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q4_0_MATMUL_OK;
}

I32 Q4_0MatMulTiledAVX2Q32CheckedDefaultTiles(
    Q4_0Block *lhs_matrix_blocks,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q4_0Block *rhs_col_blocks,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_mat_q32,
    I64 out_row_stride_cols)
{
    I64 lhs_block_capacity;
    I64 rhs_block_capacity;
    I64 out_capacity;

    // IQ-330 canonical default-tiles wrapper:
    // - Keep checked-core argument ordering/error precedence unchanged.
    // - Derive capacities only from visible matrix layout dimensions.
    // - Default tile traversal is 1x1, so behavior is exact checked-core
    //   behavior with explicit tile defaults.
    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q32)
        return Q4_0_MATMUL_ERR_NULL_PTR;

    if (lhs_rows < 0 || lhs_row_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (rhs_cols < 0 || rhs_col_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (k_block_count < 0 || out_row_stride_cols < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, lhs_row_stride_blocks, &lhs_block_capacity))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(rhs_cols, rhs_col_stride_blocks, &rhs_block_capacity))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, out_row_stride_cols, &out_capacity))
        return Q4_0_MATMUL_ERR_OVERFLOW;

    return Q4_0MatMulTiledAVX2Q32Checked(
        lhs_matrix_blocks,
        lhs_block_capacity,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        1,
        1,
        out_mat_q32,
        out_capacity,
        out_row_stride_cols);
}

I32 Q4_0MatMulTiledAVX2Q32ToQ16Checked(
    Q4_0Block *lhs_matrix_blocks,
    I64 lhs_block_capacity,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q4_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_mat_q16,
    I64 out_capacity,
    I64 out_row_stride_cols)
{
    I64 required_lhs_blocks;
    I64 required_rhs_blocks;
    I64 required_out_cells;
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
    Q4_0Block *rhs_col_ptr;
    I32 status;

    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q16)
        return Q4_0_MATMUL_ERR_NULL_PTR;

    if (lhs_block_capacity < 0 || rhs_block_capacity < 0 || out_capacity < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (lhs_rows < 0 || lhs_row_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (rhs_cols < 0 || rhs_col_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (k_block_count < 0 || out_row_stride_cols < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (tile_rows <= 0 || tile_cols <= 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Active K span must fit in both row strides.
    if (lhs_rows > 0 && lhs_row_stride_blocks < k_block_count)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (rhs_cols > 0 && rhs_col_stride_blocks < k_block_count)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Output stride must be wide enough for the active output columns.
    if (lhs_rows > 0 && out_row_stride_cols < rhs_cols)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, lhs_row_stride_blocks, &required_lhs_blocks))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(rhs_cols, rhs_col_stride_blocks, &required_rhs_blocks))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, out_row_stride_cols, &required_out_cells))
        return Q4_0_MATMUL_ERR_OVERFLOW;

    if (required_lhs_blocks > lhs_block_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (required_rhs_blocks > rhs_block_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (required_out_cells > out_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    // Tile loops keep explicit AVX2 execution boundaries while routing each
    // output cell through checked Q4_0 block-dot with one Q32->Q16 rounding.
    row_tile_start = 0;
    while (row_tile_start < lhs_rows)
    {
        if (!Q4_0MatMulTryAddI64NonNeg(row_tile_start, tile_rows, &row_tile_end))
            return Q4_0_MATMUL_ERR_OVERFLOW;
        row_tile_end = Q4_0MatMulMinI64(row_tile_end, lhs_rows);

        col_tile_start = 0;
        while (col_tile_start < rhs_cols)
        {
            if (!Q4_0MatMulTryAddI64NonNeg(col_tile_start, tile_cols, &col_tile_end))
                return Q4_0_MATMUL_ERR_OVERFLOW;
            col_tile_end = Q4_0MatMulMinI64(col_tile_end, rhs_cols);

            for (row_index = row_tile_start; row_index < row_tile_end; ++row_index)
            {
                if (!Q4_0MatMulTryMulI64NonNeg(row_index, lhs_row_stride_blocks, &lhs_row_base))
                    return Q4_0_MATMUL_ERR_OVERFLOW;
                if (!Q4_0MatMulTryMulI64NonNeg(row_index, out_row_stride_cols, &out_row_base))
                    return Q4_0_MATMUL_ERR_OVERFLOW;

                lhs_row_ptr = &lhs_matrix_blocks[lhs_row_base];

                for (col_index = col_tile_start; col_index < col_tile_end; ++col_index)
                {
                    if (!Q4_0MatMulTryMulI64NonNeg(col_index, rhs_col_stride_blocks, &rhs_col_base))
                        return Q4_0_MATMUL_ERR_OVERFLOW;
                    if (!Q4_0MatMulTryAddI64NonNeg(out_row_base, col_index, &out_index))
                        return Q4_0_MATMUL_ERR_OVERFLOW;

                    rhs_col_ptr = &rhs_col_blocks[rhs_col_base];
                    status = Q4_0DotBlocksAVX2Q32ToQ16Checked(lhs_row_ptr,
                                                              rhs_col_ptr,
                                                              k_block_count,
                                                              &cell_dot_q16);
                    if (status != Q4_0_MATMUL_OK)
                        return status;

                    out_mat_q16[out_index] = cell_dot_q16;
                }
            }

            col_tile_start = col_tile_end;
        }

        row_tile_start = row_tile_end;
    }

    return Q4_0_MATMUL_OK;
}

I32 Q4_0MatMulTiledAVX2Q32ToQ16CheckedNoPartial(
    Q4_0Block *lhs_matrix_blocks,
    I64 lhs_block_capacity,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q4_0Block *rhs_col_blocks,
    I64 rhs_block_capacity,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 tile_rows,
    I64 tile_cols,
    I64 *out_mat_q16,
    I64 out_capacity,
    I64 out_row_stride_cols)
{
    I64 required_out_cells;
    I64 stage_bytes;
    I64 out_index;
    I64 *staged_out_q16;
    I32 status;

    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q16)
        return Q4_0_MATMUL_ERR_NULL_PTR;

    if (!lhs_rows)
        return Q4_0MatMulTiledAVX2Q32ToQ16Checked(lhs_matrix_blocks,
                                                  lhs_block_capacity,
                                                  lhs_rows,
                                                  lhs_row_stride_blocks,
                                                  rhs_col_blocks,
                                                  rhs_block_capacity,
                                                  rhs_cols,
                                                  rhs_col_stride_blocks,
                                                  k_block_count,
                                                  tile_rows,
                                                  tile_cols,
                                                  out_mat_q16,
                                                  out_capacity,
                                                  out_row_stride_cols);

    if (lhs_rows < 0 || out_row_stride_cols < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, out_row_stride_cols, &required_out_cells))
        return Q4_0_MATMUL_ERR_OVERFLOW;

    if (required_out_cells > out_capacity)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    if (!required_out_cells)
        return Q4_0MatMulTiledAVX2Q32ToQ16Checked(lhs_matrix_blocks,
                                                  lhs_block_capacity,
                                                  lhs_rows,
                                                  lhs_row_stride_blocks,
                                                  rhs_col_blocks,
                                                  rhs_block_capacity,
                                                  rhs_cols,
                                                  rhs_col_stride_blocks,
                                                  k_block_count,
                                                  tile_rows,
                                                  tile_cols,
                                                  out_mat_q16,
                                                  out_capacity,
                                                  out_row_stride_cols);

    if (!Q4_0MatMulTryMulI64NonNeg(required_out_cells, sizeof(I64), &stage_bytes))
        return Q4_0_MATMUL_ERR_OVERFLOW;

    staged_out_q16 = MAlloc(stage_bytes);
    if (!staged_out_q16)
        return Q4_0_MATMUL_ERR_OVERFLOW;

    // Seed staging from caller output so logically inactive lanes
    // (e.g. row padding when out_row_stride_cols > rhs_cols) preserve
    // checked-core behavior on success.
    for (out_index = 0; out_index < required_out_cells; ++out_index)
        staged_out_q16[out_index] = out_mat_q16[out_index];

    status = Q4_0MatMulTiledAVX2Q32ToQ16Checked(lhs_matrix_blocks,
                                                lhs_block_capacity,
                                                lhs_rows,
                                                lhs_row_stride_blocks,
                                                rhs_col_blocks,
                                                rhs_block_capacity,
                                                rhs_cols,
                                                rhs_col_stride_blocks,
                                                k_block_count,
                                                tile_rows,
                                                tile_cols,
                                                staged_out_q16,
                                                required_out_cells,
                                                out_row_stride_cols);
    if (status != Q4_0_MATMUL_OK)
    {
        Free(staged_out_q16);
        return status;
    }

    // IQ-360 no-partial-write contract:
    // - Run full tiled conversion in a private stage buffer.
    // - Publish all visible output cells only after full success.
    // This preserves checked-core error codes while preventing early-cell
    // commits when later cells fail due to dot overflow/truncation.
    for (out_index = 0; out_index < required_out_cells; ++out_index)
        out_mat_q16[out_index] = staged_out_q16[out_index];

    Free(staged_out_q16);
    return Q4_0_MATMUL_OK;
}

I32 Q4_0MatMulTiledAVX2Q32ToQ16CheckedDefaultTiles(
    Q4_0Block *lhs_matrix_blocks,
    I64 lhs_rows,
    I64 lhs_row_stride_blocks,
    Q4_0Block *rhs_col_blocks,
    I64 rhs_cols,
    I64 rhs_col_stride_blocks,
    I64 k_block_count,
    I64 *out_mat_q16,
    I64 out_row_stride_cols)
{
    I64 lhs_block_capacity;
    I64 rhs_block_capacity;
    I64 out_capacity;

    if (!lhs_matrix_blocks || !rhs_col_blocks || !out_mat_q16)
        return Q4_0_MATMUL_ERR_NULL_PTR;

    if (lhs_rows < 0 || lhs_row_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (rhs_cols < 0 || rhs_col_stride_blocks < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;
    if (k_block_count < 0 || out_row_stride_cols < 0)
        return Q4_0_MATMUL_ERR_BAD_LEN;

    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, lhs_row_stride_blocks, &lhs_block_capacity))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(rhs_cols, rhs_col_stride_blocks, &rhs_block_capacity))
        return Q4_0_MATMUL_ERR_OVERFLOW;
    if (!Q4_0MatMulTryMulI64NonNeg(lhs_rows, out_row_stride_cols, &out_capacity))
        return Q4_0_MATMUL_ERR_OVERFLOW;

    return Q4_0MatMulTiledAVX2Q32ToQ16Checked(
        lhs_matrix_blocks,
        lhs_block_capacity,
        lhs_rows,
        lhs_row_stride_blocks,
        rhs_col_blocks,
        rhs_block_capacity,
        rhs_cols,
        rhs_col_stride_blocks,
        k_block_count,
        1,
        1,
        out_mat_q16,
        out_capacity,
        out_row_stride_cols);
}
