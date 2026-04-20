// q4_0_dot_bench.HC
// Deterministic TempleOS benchmark suite for integer-only Q4_0 dot kernels.
//
// This file intentionally keeps all math visible and auditable:
// - fixed suite shapes (no runtime randomness)
// - explicit LCG fixture generation for blocks
// - cycle accounting from TSC
// - integer-only throughput derivation

#define Q4_0_DOT_BENCH_OK                 0
#define Q4_0_DOT_BENCH_ERR_NULL_PTR       1
#define Q4_0_DOT_BENCH_ERR_BAD_PARAM      2
#define Q4_0_DOT_BENCH_ERR_OVERFLOW       3
#define Q4_0_DOT_BENCH_ERR_KERNEL         4

#define Q4_0_DOT_BENCH_MAX_BLOCKS         64
#define Q4_0_DOT_BENCH_MAX_ROWS           8
#define Q4_0_DOT_BENCH_FIXED_FP16_SCALE   0x3C00
#define Q4_0_DOT_BENCH_REPORT_FIELDS      "shape=%d kernel=%s block_count=%d row_count=%d row_stride_blocks=%d iters=%d elapsed_cycles=%d cycles_per_dot=%d dots_per_sec=%d"

class Q4_0DotBenchShape
{
    I64 block_count;
    I64 row_count;
    I64 row_stride_blocks;
    I64 iters;
};

U32 Q4_0DotBenchLCGNext(U32 *state)
{
    *state = (*state * 1664525) + 1013904223;
    return *state;
}

I32 Q4_0DotBenchTryMulI64(I64 lhs, I64 rhs, I64 *out)
{
    I64 result;

    if (!out)
        return Q4_0_DOT_BENCH_ERR_NULL_PTR;

    result = lhs * rhs;
    if (lhs && (result / lhs) != rhs)
        return Q4_0_DOT_BENCH_ERR_OVERFLOW;

    *out = result;
    return Q4_0_DOT_BENCH_OK;
}

I32 Q4_0DotBenchTryAddI64(I64 lhs, I64 rhs, I64 *out)
{
    I64 result;

    if (!out)
        return Q4_0_DOT_BENCH_ERR_NULL_PTR;

    result = lhs + rhs;
    if (((rhs > 0) && (result < lhs)) || ((rhs < 0) && (result > lhs)))
        return Q4_0_DOT_BENCH_ERR_OVERFLOW;

    *out = result;
    return Q4_0_DOT_BENCH_OK;
}

I32 Q4_0DotProductQ16(Q4_0Block *lhs, Q4_0Block *rhs, I64 block_count, I64 *out_dot_q16)
{
    return Q4_0DotProductBlocksQ32ToQ16(lhs, rhs, block_count, out_dot_q16);
}

I32 Q4_0DotProductQ16NoPartial(Q4_0Block *lhs, Q4_0Block *rhs, I64 block_count, I64 *out_dot_q16)
{
    I64 tmp_dot_q16;
    I32 status;

    if (!out_dot_q16)
        return Q4_0_DOT_BENCH_ERR_NULL_PTR;

    status = Q4_0DotProductBlocksQ32ToQ16(lhs, rhs, block_count, &tmp_dot_q16);
    if (status != Q4_0_OK)
        return Q4_0_DOT_BENCH_ERR_KERNEL;

    *out_dot_q16 = tmp_dot_q16;
    return Q4_0_DOT_BENCH_OK;
}

void Q4_0DotBenchInitBlocks(Q4_0Block *blocks, I64 block_count, U32 seed)
{
    I64 block_index;
    I64 byte_index;
    U32 state;

    state = seed;
    for (block_index = 0; block_index < block_count; ++block_index)
    {
        blocks[block_index].d_fp16 = Q4_0_DOT_BENCH_FIXED_FP16_SCALE;
        for (byte_index = 0; byte_index < Q4_0_PACKED_BYTES; ++byte_index)
            blocks[block_index].qs[byte_index] = Q4_0DotBenchLCGNext(&state) & 0xFF;
    }
}

I32 Q4_0DotBenchComputeDotsPerSec(I64 elapsed_cycles, I64 total_dots, I64 cpu_hz, I64 *out_dots_per_sec)
{
    I64 numerator;
    I32 status;

    if (!out_dots_per_sec)
        return Q4_0_DOT_BENCH_ERR_NULL_PTR;
    if (elapsed_cycles <= 0 || total_dots < 0 || cpu_hz <= 0)
        return Q4_0_DOT_BENCH_ERR_BAD_PARAM;

    status = Q4_0DotBenchTryMulI64(total_dots, cpu_hz, &numerator);
    if (status != Q4_0_DOT_BENCH_OK)
        return status;

    *out_dots_per_sec = numerator / elapsed_cycles;
    return Q4_0_DOT_BENCH_OK;
}

I32 Q4_0DotBenchRunShape(I64 shape_index,
                         Q4_0DotBenchShape *shape,
                         Q4_0Block *lhs,
                         Q4_0Block *rhs,
                         Q4_0Block *row_matrix,
                         I64 *row_out_q16,
                         I64 cpu_hz)
{
    I64 iter_index;
    I64 row_dot_q16;
    I64 elapsed_cycles;
    I64 cycles_per_dot;
    I64 dots_per_sec;
    I64 total_dots;
    I64 kernel_dots;
    I64 bench_sink;
    I64 start_tsc;
    I64 status;

    if (!shape || !lhs || !rhs || !row_matrix || !row_out_q16)
        return Q4_0_DOT_BENCH_ERR_NULL_PTR;

    if (shape->block_count <= 0 || shape->row_count <= 0 || shape->iters <= 0)
        return Q4_0_DOT_BENCH_ERR_BAD_PARAM;
    if (shape->row_stride_blocks < shape->block_count)
        return Q4_0_DOT_BENCH_ERR_BAD_PARAM;

    bench_sink = 0;

    start_tsc = TSC;
    for (iter_index = 0; iter_index < shape->iters; ++iter_index)
    {
        status = Q4_0DotProductQ16(lhs, rhs, shape->block_count, &row_dot_q16);
        if (status != Q4_0_DOT_BENCH_OK)
            return status;
        bench_sink ^= row_dot_q16;
    }
    elapsed_cycles = TSC - start_tsc;

    total_dots = shape->iters;
    cycles_per_dot = elapsed_cycles / total_dots;
    status = Q4_0DotBenchComputeDotsPerSec(elapsed_cycles, total_dots, cpu_hz, &dots_per_sec);
    if (status != Q4_0_DOT_BENCH_OK)
        return status;

    Printf("Q4_0_DOT_BENCH " Q4_0_DOT_BENCH_REPORT_FIELDS "\n",
           shape_index,
           "Q4_0DotProductQ16",
           shape->block_count,
           shape->row_count,
           shape->row_stride_blocks,
           shape->iters,
           elapsed_cycles,
           cycles_per_dot,
           dots_per_sec);

    start_tsc = TSC;
    for (iter_index = 0; iter_index < shape->iters; ++iter_index)
    {
        status = Q4_0DotProductQ16NoPartial(lhs, rhs, shape->block_count, &row_dot_q16);
        if (status != Q4_0_DOT_BENCH_OK)
            return status;
        bench_sink ^= row_dot_q16;
    }
    elapsed_cycles = TSC - start_tsc;

    total_dots = shape->iters;
    cycles_per_dot = elapsed_cycles / total_dots;
    status = Q4_0DotBenchComputeDotsPerSec(elapsed_cycles, total_dots, cpu_hz, &dots_per_sec);
    if (status != Q4_0_DOT_BENCH_OK)
        return status;

    Printf("Q4_0_DOT_BENCH " Q4_0_DOT_BENCH_REPORT_FIELDS "\n",
           shape_index,
           "Q4_0DotProductQ16NoPartial",
           shape->block_count,
           shape->row_count,
           shape->row_stride_blocks,
           shape->iters,
           elapsed_cycles,
           cycles_per_dot,
           dots_per_sec);

    start_tsc = TSC;
    for (iter_index = 0; iter_index < shape->iters; ++iter_index)
    {
        status = Q4_0DotRowsQ16MatrixVector(row_matrix,
                                            shape->row_count,
                                            shape->row_stride_blocks,
                                            rhs,
                                            shape->block_count,
                                            row_out_q16);
        if (status != Q4_0_OK)
            return Q4_0_DOT_BENCH_ERR_KERNEL;
        bench_sink ^= row_out_q16[0];
    }
    elapsed_cycles = TSC - start_tsc;

    status = Q4_0DotBenchTryMulI64(shape->iters, shape->row_count, &kernel_dots);
    if (status != Q4_0_DOT_BENCH_OK)
        return status;
    total_dots = kernel_dots;
    cycles_per_dot = elapsed_cycles / total_dots;
    status = Q4_0DotBenchComputeDotsPerSec(elapsed_cycles, total_dots, cpu_hz, &dots_per_sec);
    if (status != Q4_0_DOT_BENCH_OK)
        return status;

    Printf("Q4_0_DOT_BENCH " Q4_0_DOT_BENCH_REPORT_FIELDS "\n",
           shape_index,
           "Q4_0DotRowsQ16MatrixVector",
           shape->block_count,
           shape->row_count,
           shape->row_stride_blocks,
           shape->iters,
           elapsed_cycles,
           cycles_per_dot,
           dots_per_sec);

    // Keep result-dependent state observable so optimizer cannot erase the loop.
    Printf("Q4_0_DOT_BENCH_SINK shape=%d sink=%d\n", shape_index, bench_sink);
    return Q4_0_DOT_BENCH_OK;
}

I32 Q4_0DotBenchRunDefaultSuite(I64 cpu_hz)
{
    Q4_0DotBenchShape shapes[4];
    Q4_0Block lhs[Q4_0_DOT_BENCH_MAX_BLOCKS];
    Q4_0Block rhs[Q4_0_DOT_BENCH_MAX_BLOCKS];
    Q4_0Block row_matrix[Q4_0_DOT_BENCH_MAX_BLOCKS * Q4_0_DOT_BENCH_MAX_ROWS];
    I64 row_out_q16[Q4_0_DOT_BENCH_MAX_ROWS];
    I64 shape_index;
    I64 row_matrix_blocks;
    I32 status;

    if (cpu_hz <= 0)
        return Q4_0_DOT_BENCH_ERR_BAD_PARAM;

    // Fixed deterministic coverage:
    // - tiny scalar path
    // - medium scalar path
    // - wider rows path
    // - near-max configured row helper path
    shapes[0].block_count = 4;
    shapes[0].row_count = 1;
    shapes[0].row_stride_blocks = 4;
    shapes[0].iters = 8192;

    shapes[1].block_count = 8;
    shapes[1].row_count = 2;
    shapes[1].row_stride_blocks = 8;
    shapes[1].iters = 4096;

    shapes[2].block_count = 16;
    shapes[2].row_count = 4;
    shapes[2].row_stride_blocks = 16;
    shapes[2].iters = 2048;

    shapes[3].block_count = 32;
    shapes[3].row_count = 8;
    shapes[3].row_stride_blocks = 32;
    shapes[3].iters = 1024;

    for (shape_index = 0; shape_index < 4; ++shape_index)
    {
        status = Q4_0DotBenchTryMulI64(shapes[shape_index].row_count,
                                       shapes[shape_index].row_stride_blocks,
                                       &row_matrix_blocks);
        if (status != Q4_0_DOT_BENCH_OK)
            return status;

        if (shapes[shape_index].block_count > Q4_0_DOT_BENCH_MAX_BLOCKS)
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM;
        if (shapes[shape_index].row_count > Q4_0_DOT_BENCH_MAX_ROWS)
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM;
        if (row_matrix_blocks > (Q4_0_DOT_BENCH_MAX_BLOCKS * Q4_0_DOT_BENCH_MAX_ROWS))
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM;

        Q4_0DotBenchInitBlocks(lhs, shapes[shape_index].block_count, 0xA11CE000 + shape_index);
        Q4_0DotBenchInitBlocks(rhs, shapes[shape_index].block_count, 0xBEEFF000 + shape_index);
        Q4_0DotBenchInitBlocks(row_matrix, row_matrix_blocks, 0xC0FFEE00 + shape_index);

        status = Q4_0DotBenchRunShape(shape_index,
                                      &shapes[shape_index],
                                      lhs,
                                      rhs,
                                      row_matrix,
                                      row_out_q16,
                                      cpu_hz);
        if (status != Q4_0_DOT_BENCH_OK)
            return status;
    }

    return Q4_0_DOT_BENCH_OK;
}
