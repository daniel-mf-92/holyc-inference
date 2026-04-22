// q8_0_dot_bench.HC
// Deterministic integer-only helpers for Q8_0 benchmark reporting.

#define Q8_0_DOT_BENCH_OK                 0
#define Q8_0_DOT_BENCH_ERR_NULL_PTR       1
#define Q8_0_DOT_BENCH_ERR_BAD_PARAM      2
#define Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO    3
#define Q8_0_DOT_BENCH_ERR_OVERFLOW       4

#define Q8_0_DOT_BENCH_I64_MAX            0x7FFFFFFFFFFFFFFF
#define Q8_0_DOT_BENCH_MAX_BLOCKS         64
#define Q8_0_DOT_BENCH_MAX_ROWS           8

class Q8_0DotBenchShape
{
    I64 block_count;
    I64 row_count;
    I64 row_stride_blocks;
    I64 iters;
};

I32 Q8_0DotBenchTryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if ((rhs > 0) && (lhs > Q8_0_DOT_BENCH_I64_MAX - rhs))
        return Q8_0_DOT_BENCH_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchTryMulI64(I64 lhs, I64 rhs, I64 *out_prod)
{
    if (!out_prod)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (!lhs || !rhs)
    {
        *out_prod = 0;
        return Q8_0_DOT_BENCH_OK;
    }

    if (lhs < 0 || rhs < 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (lhs > (Q8_0_DOT_BENCH_I64_MAX / rhs))
        return Q8_0_DOT_BENCH_ERR_OVERFLOW;

    *out_prod = lhs * rhs;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchCyclesPerOpChecked(
    I64 total_cycles,
    I64 total_ops,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles)
{
    I64 half_ops;
    I64 rounded_numer;
    I64 staged_cycles_per_op;
    I64 staged_remainder_cycles;
    I32 status;

    if (!out_cycles_per_op || !out_remainder_cycles)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    // Bench counters are monotonic and non-negative.
    if (total_cycles < 0 || total_ops < 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (!total_ops)
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO;

    // Deterministic integer-only rounding:
    // cycles_per_op = Round(total_cycles / total_ops)
    //               = (total_cycles + floor(total_ops / 2)) / total_ops
    // Keep the bias add checked so hostile counters cannot wrap diagnostics.
    half_ops = total_ops >> 1;
    status = Q8_0DotBenchTryAddI64(total_cycles, half_ops, &rounded_numer);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    staged_cycles_per_op = rounded_numer / total_ops;
    staged_remainder_cycles = total_cycles % total_ops;

    // No-partial diagnostics publish: commit both outputs atomically at end.
    *out_cycles_per_op = staged_cycles_per_op;
    *out_remainder_cycles = staged_remainder_cycles;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchOpsPerCycleQ16Checked(
    I64 total_ops,
    I64 total_cycles,
    I64 *out_ops_per_cycle_q16,
    I64 *out_remainder_ops_q16)
{
    I64 total_ops_q16;
    I64 half_cycles;
    I64 rounded_numer;
    I64 staged_ops_per_cycle_q16;
    I64 staged_remainder_ops_q16;
    I32 status;

    if (!out_ops_per_cycle_q16 || !out_remainder_ops_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    // Counters are monotonic and non-negative.
    if (total_ops < 0 || total_cycles < 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (!total_cycles)
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO;

    // Convert integer ops to Q16 domain with checked shift.
    if (total_ops > (Q8_0_DOT_BENCH_I64_MAX >> 16))
        return Q8_0_DOT_BENCH_ERR_OVERFLOW;
    total_ops_q16 = total_ops << 16;

    // Deterministic integer-only rounding:
    // ops_per_cycle_q16 = Round((total_ops_q16 / total_cycles))
    //                   = (total_ops_q16 + floor(total_cycles / 2)) / total_cycles
    half_cycles = total_cycles >> 1;
    status = Q8_0DotBenchTryAddI64(total_ops_q16, half_cycles, &rounded_numer);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    staged_ops_per_cycle_q16 = rounded_numer / total_cycles;
    staged_remainder_ops_q16 = total_ops_q16 % total_cycles;

    // No-partial diagnostics publish: commit both outputs atomically at end.
    *out_ops_per_cycle_q16 = staged_ops_per_cycle_q16;
    *out_remainder_ops_q16 = staged_remainder_ops_q16;
    return Q8_0_DOT_BENCH_OK;
}

// Diagnostics-only planner for deterministic Q8_0 default bench shapes.
//
// This helper does not allocate buffers or run kernels. It only validates the
// fixed suite geometry and publishes required capacities for callers to stage
// benchmark working memory:
// - shape_count            : number of deterministic suite shapes
// - max_blocks             : max block_count among shapes
// - max_rows               : max row_count among shapes
// - max_row_matrix_blocks  : max (row_count * row_stride_blocks) among shapes
//
// Strict no-partial contract: outputs are committed only after all checks pass.
I32 Q8_0DotBenchRunDefaultSuitePreflightOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    Q8_0DotBenchShape shapes[4];
    I64 shape_count_snapshot;
    I64 max_blocks_snapshot;
    I64 max_rows_snapshot;
    I64 max_row_matrix_blocks_snapshot;
    I64 shape_index;
    I64 row_matrix_blocks;
    I64 cpu_hz_snapshot;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Immutable caller input snapshot for diagnostics determinism.
    cpu_hz_snapshot = cpu_hz;

    // Canonical deterministic suite geometry (small->large working sets).
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

    shape_count_snapshot = 4;
    max_blocks_snapshot = 0;
    max_rows_snapshot = 0;
    max_row_matrix_blocks_snapshot = 0;

    for (shape_index = 0; shape_index < shape_count_snapshot; ++shape_index)
    {
        if (shapes[shape_index].block_count <= 0 ||
            shapes[shape_index].row_count <= 0 ||
            shapes[shape_index].row_stride_blocks <= 0 ||
            shapes[shape_index].iters <= 0)
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

        if (shapes[shape_index].row_stride_blocks < shapes[shape_index].block_count)
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

        if (shapes[shape_index].block_count > Q8_0_DOT_BENCH_MAX_BLOCKS)
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
        if (shapes[shape_index].row_count > Q8_0_DOT_BENCH_MAX_ROWS)
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

        status = Q8_0DotBenchTryMulI64(shapes[shape_index].row_count,
                                       shapes[shape_index].row_stride_blocks,
                                       &row_matrix_blocks);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        if (row_matrix_blocks > (Q8_0_DOT_BENCH_MAX_BLOCKS * Q8_0_DOT_BENCH_MAX_ROWS))
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

        if (shapes[shape_index].block_count > max_blocks_snapshot)
            max_blocks_snapshot = shapes[shape_index].block_count;
        if (shapes[shape_index].row_count > max_rows_snapshot)
            max_rows_snapshot = shapes[shape_index].row_count;
        if (row_matrix_blocks > max_row_matrix_blocks_snapshot)
            max_row_matrix_blocks_snapshot = row_matrix_blocks;
    }

    if (cpu_hz != cpu_hz_snapshot)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (shape_count_snapshot <= 0 || max_blocks_snapshot <= 0 ||
        max_rows_snapshot <= 0 || max_row_matrix_blocks_snapshot <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = shape_count_snapshot;
    *out_max_blocks = max_blocks_snapshot;
    *out_max_rows = max_rows_snapshot;
    *out_max_row_matrix_blocks = max_row_matrix_blocks_snapshot;
    return Q8_0_DOT_BENCH_OK;
}

// Commit-only diagnostics wrapper over default-suite preflight planner.
//
// Contracts:
// - Revalidates immutable cpu_hz snapshot after preflight.
// - Enforces canonical deterministic suite tuple parity before publish.
// - Publishes {shape_count,max_blocks,max_rows,max_row_matrix_blocks}
//   atomically only when all checks pass.
I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 cpu_hz_snapshot;
    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;
    I64 canonical_shape_count;
    I64 canonical_max_blocks;
    I64 canonical_max_rows;
    I64 canonical_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    cpu_hz_snapshot = cpu_hz;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnly(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (cpu_hz != cpu_hz_snapshot)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Canonical deterministic suite tuple must remain stable.
    canonical_shape_count = 4;
    canonical_max_blocks = 32;
    canonical_max_rows = 8;

    status = Q8_0DotBenchTryMulI64(canonical_max_rows,
                                   canonical_max_blocks,
                                   &canonical_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (staged_shape_count != canonical_shape_count ||
        staged_max_blocks != canonical_max_blocks ||
        staged_max_rows != canonical_max_rows ||
        staged_max_row_matrix_blocks != canonical_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}
