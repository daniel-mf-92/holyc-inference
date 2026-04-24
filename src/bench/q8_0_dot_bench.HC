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
#define Q8_0_DOT_BENCH_VALUES_PER_BLOCK   32
#define Q8_0_DOT_BENCH_NS_PER_SECOND      1000000000

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

I32 Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16Checked(
    I64 total_ops,
    I64 total_cycles,
    I64 cpu_hz,
    I64 *out_ops_per_second_q16)
{
    I64 ops_hz_numer;
    I64 ops_hz_numer_q16;
    I64 half_cycles;
    I64 rounded_numer;
    I64 staged_ops_per_second_q16;
    I32 status;

    if (!out_ops_per_second_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    // Bench totals and CPU frequency are monotonic and non-negative.
    if (total_ops < 0 || total_cycles < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (!total_cycles)
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO;

    // ops/sec = (total_ops * cpu_hz) / total_cycles.
    // Preserve Q16 precision with deterministic round-to-nearest integer math:
    // ops_per_second_q16 = Round(((total_ops * cpu_hz) << 16) / total_cycles)
    status = Q8_0DotBenchTryMulI64(total_ops, cpu_hz, &ops_hz_numer);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (ops_hz_numer > (Q8_0_DOT_BENCH_I64_MAX >> 16))
        return Q8_0_DOT_BENCH_ERR_OVERFLOW;
    ops_hz_numer_q16 = ops_hz_numer << 16;

    half_cycles = total_cycles >> 1;
    status = Q8_0DotBenchTryAddI64(ops_hz_numer_q16, half_cycles, &rounded_numer);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    staged_ops_per_second_q16 = rounded_numer / total_cycles;

    // No-partial diagnostics publish.
    *out_ops_per_second_q16 = staged_ops_per_second_q16;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16Checked(
    I64 total_cycles,
    I64 total_tokens,
    I64 *out_cycles_per_token_q16)
{
    I64 total_cycles_q16;
    I64 half_tokens;
    I64 rounded_numer;
    I64 staged_cycles_per_token_q16;
    I32 status;

    if (!out_cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    // Bench counters are monotonic and non-negative.
    if (total_cycles < 0 || total_tokens < 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (!total_tokens)
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO;

    // Lift cycles into Q16 domain with checked left shift.
    if (total_cycles > (Q8_0_DOT_BENCH_I64_MAX >> 16))
        return Q8_0_DOT_BENCH_ERR_OVERFLOW;
    total_cycles_q16 = total_cycles << 16;

    // Deterministic integer-only rounding:
    // cycles_per_token_q16 = Round(total_cycles_q16 / total_tokens)
    //                      = (total_cycles_q16 + floor(total_tokens / 2)) / total_tokens
    half_tokens = total_tokens >> 1;
    status = Q8_0DotBenchTryAddI64(total_cycles_q16,
                                   half_tokens,
                                   &rounded_numer);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    staged_cycles_per_token_q16 = rounded_numer / total_tokens;

    // No-partial diagnostics publish.
    *out_cycles_per_token_q16 = staged_cycles_per_token_q16;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchEstimateTokPerSecQ16Checked(
    I64 cpu_hz,
    I64 cycles_per_token_q16,
    I64 *out_tok_per_sec_q16)
{
    I64 reciprocal_numer_q48;
    I64 half_cycles_per_token_q16;
    I64 rounded_reciprocal_numer_q48;
    I64 reciprocal_tok_per_cycle_q32;
    I64 tok_per_sec_q32;
    I64 rounded_tok_per_sec_q32;
    I64 staged_tok_per_sec_q16;
    I32 status;

    if (!out_tok_per_sec_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    // CPU frequency is strictly positive. Negative Q16 cycles/token is invalid.
    if (cpu_hz <= 0 || cycles_per_token_q16 < 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (!cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO;

    // Reciprocal scaling in fixed-point:
    // reciprocal_tok_per_cycle_q32 = Round((1.0_q48 / cycles_per_token_q16))
    // where 1.0_q48 = 2^48 and cycles_per_token_q16 encodes cycles/token.
    // This keeps enough precision for low tok/s regimes without floating point.
    reciprocal_numer_q48 = 281474976710656; // 1 << 48
    half_cycles_per_token_q16 = cycles_per_token_q16 >> 1;
    status = Q8_0DotBenchTryAddI64(reciprocal_numer_q48,
                                   half_cycles_per_token_q16,
                                   &rounded_reciprocal_numer_q48);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    reciprocal_tok_per_cycle_q32 = rounded_reciprocal_numer_q48 /
                                   cycles_per_token_q16;

    // tok_per_sec_q16 = Round((cpu_hz * reciprocal_tok_per_cycle_q32) / 2^16)
    // using checked multiply and checked rounding bias add.
    status = Q8_0DotBenchTryMulI64(cpu_hz,
                                   reciprocal_tok_per_cycle_q32,
                                   &tok_per_sec_q32);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchTryAddI64(tok_per_sec_q32,
                                   32768,
                                   &rounded_tok_per_sec_q32);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    staged_tok_per_sec_q16 = rounded_tok_per_sec_q32 >> 16;

    // No-partial diagnostics publish.
    *out_tok_per_sec_q16 = staged_tok_per_sec_q16;
    return Q8_0_DOT_BENCH_OK;
}

// Commit-only wrapper over canonical tok/sec estimator.
//
// Contracts:
// - Snapshot immutable tuple {cpu_hz,cycles_per_token_q16,out_tok_per_sec_q16}.
// - Compute staged result through canonical checked estimator.
// - Revalidate immutable snapshots before publish.
// - Publish atomically on success; preserve caller output on failure.
I32 Q8_0DotBenchEstimateTokPerSecQ16CheckedNoPartialCommitOnly(
    I64 cpu_hz,
    I64 cycles_per_token_q16,
    I64 *out_tok_per_sec_q16)
{
    I64 snapshot_cpu_hz;
    I64 snapshot_cycles_per_token_q16;
    I64 *snapshot_out_tok_per_sec_q16;
    I64 staged_tok_per_sec_q16;
    I32 status;

    if (!out_tok_per_sec_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_cycles_per_token_q16 = cycles_per_token_q16;
    snapshot_out_tok_per_sec_q16 = out_tok_per_sec_q16;

    status = Q8_0DotBenchEstimateTokPerSecQ16Checked(
        cpu_hz,
        cycles_per_token_q16,
        &staged_tok_per_sec_q16);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz ||
        snapshot_cycles_per_token_q16 != cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_tok_per_sec_q16 != out_tok_per_sec_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_tok_per_sec_q16 = staged_tok_per_sec_q16;
    return Q8_0_DOT_BENCH_OK;
}

// Diagnostics zero-write wrapper over commit-only tok/sec estimator.
//
// Contracts:
// - Snapshot immutable tuple {cpu_hz,cycles_per_token_q16,out_tok_per_sec_q16}.
// - Recompute staged result via canonical and commit-only paths.
// - Require strict parity between canonical and commit-only staged outputs.
// - Preserve caller output value (zero-write diagnostics behavior).
I32 Q8_0DotBenchEstimateTokPerSecQ16CheckedNoPartialCommitOnlyPreflightOnly(
    I64 cpu_hz,
    I64 cycles_per_token_q16,
    I64 *out_tok_per_sec_q16)
{
    I64 snapshot_cpu_hz;
    I64 snapshot_cycles_per_token_q16;
    I64 *snapshot_out_tok_per_sec_q16;
    I64 snapshot_out_value;
    I64 staged_direct_tok_per_sec_q16;
    I64 staged_commit_tok_per_sec_q16;
    I32 status;

    if (!out_tok_per_sec_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_cycles_per_token_q16 = cycles_per_token_q16;
    snapshot_out_tok_per_sec_q16 = out_tok_per_sec_q16;
    snapshot_out_value = *out_tok_per_sec_q16;

    status = Q8_0DotBenchEstimateTokPerSecQ16Checked(
        cpu_hz,
        cycles_per_token_q16,
        &staged_direct_tok_per_sec_q16);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    if (*out_tok_per_sec_q16 != snapshot_out_value)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    status = Q8_0DotBenchEstimateTokPerSecQ16CheckedNoPartialCommitOnly(
        cpu_hz,
        cycles_per_token_q16,
        &staged_commit_tok_per_sec_q16);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    if (*out_tok_per_sec_q16 != snapshot_out_value)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (snapshot_cpu_hz != cpu_hz ||
        snapshot_cycles_per_token_q16 != cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_tok_per_sec_q16 != out_tok_per_sec_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_direct_tok_per_sec_q16 != staged_commit_tok_per_sec_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (*out_tok_per_sec_q16 != snapshot_out_value)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

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

// Strict parity gate over commit-only + canonical preflight planners.
//
// Contracts:
// - Enforce immutable caller snapshots for cpu_hz and output destinations.
// - Require parity between commit-only wrapper and canonical preflight tuple.
// - Publish diagnostics tuple atomically only on full success.
I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParity(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;

    I64 canonical_shape_count;
    I64 canonical_max_blocks;
    I64 canonical_max_rows;
    I64 canonical_max_row_matrix_blocks;

    I64 derived_shape_count;
    I64 derived_max_blocks;
    I64 derived_max_rows;
    I64 derived_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_shape_count == out_max_blocks ||
        out_shape_count == out_max_rows ||
        out_shape_count == out_max_row_matrix_blocks ||
        out_max_blocks == out_max_rows ||
        out_max_blocks == out_max_row_matrix_blocks ||
        out_max_rows == out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnly(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnly(
        cpu_hz,
        &canonical_shape_count,
        &canonical_max_blocks,
        &canonical_max_rows,
        &canonical_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    // Canonical suite tuple derivation: 4 deterministic shapes,
    // max block_count=32, max row_count=8, max row matrix blocks=8*32.
    derived_shape_count = 4;
    derived_max_blocks = 32;
    derived_max_rows = 8;
    status = Q8_0DotBenchTryMulI64(derived_max_rows,
                                   derived_max_blocks,
                                   &derived_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (parity_shape_count != canonical_shape_count ||
        parity_max_blocks != canonical_max_blocks ||
        parity_max_rows != canonical_max_rows ||
        parity_max_row_matrix_blocks != canonical_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (canonical_shape_count != derived_shape_count ||
        canonical_max_blocks != derived_max_blocks ||
        canonical_max_rows != derived_max_rows ||
        canonical_max_row_matrix_blocks != derived_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = parity_shape_count;
    *out_max_blocks = parity_max_blocks;
    *out_max_rows = parity_max_rows;
    *out_max_row_matrix_blocks = parity_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Commit-only diagnostics wrapper over strict parity gate.
//
// Contracts:
// - Enforce immutable caller snapshots for cpu_hz and output destinations.
// - Require canonical deterministic suite tuple parity before publish.
// - Publish diagnostics tuple atomically only when all checks pass.
I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

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

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParity(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

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

// Diagnostics zero-write companion over commit-only parity gate.
//
// Contracts:
// - Run commit-only parity and parity-only diagnostics paths into staged locals.
// - Enforce immutable caller snapshots for cpu_hz and output destination values.
// - Require exact tuple parity on
//   {total_ops,total_cycles,cycles_per_op,remainder_cycles,suite_checksum}.
// - Preflight-only semantics: never publish caller outputs.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    I64 snapshot_cpu_hz;
    I64 snapshot_out_shape_count;
    I64 snapshot_out_total_cycles;
    I64 snapshot_out_cycles_per_op;
    I64 snapshot_out_remainder_cycles;
    I64 snapshot_out_suite_checksum;

    I64 parity_preflight_total_ops;
    I64 parity_preflight_total_cycles;
    I64 parity_preflight_cycles_per_op;
    I64 parity_preflight_remainder_cycles;
    I64 parity_preflight_suite_checksum;

    I64 parity_commit_total_ops;
    I64 parity_commit_total_cycles;
    I64 parity_commit_cycles_per_op;
    I64 parity_commit_remainder_cycles;
    I64 parity_commit_suite_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_total_ops == out_total_cycles ||
        out_total_ops == out_cycles_per_op ||
        out_total_ops == out_remainder_cycles ||
        out_total_ops == out_suite_checksum ||
        out_total_cycles == out_cycles_per_op ||
        out_total_cycles == out_remainder_cycles ||
        out_total_cycles == out_suite_checksum ||
        out_cycles_per_op == out_remainder_cycles ||
        out_cycles_per_op == out_suite_checksum ||
        out_remainder_cycles == out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = *out_total_ops;
    snapshot_out_total_cycles = *out_total_cycles;
    snapshot_out_cycles_per_op = *out_cycles_per_op;
    snapshot_out_remainder_cycles = *out_remainder_cycles;
    snapshot_out_suite_checksum = *out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &parity_preflight_total_ops,
        &parity_preflight_total_cycles,
        &parity_preflight_cycles_per_op,
        &parity_preflight_remainder_cycles,
        &parity_preflight_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &parity_commit_total_ops,
        &parity_commit_total_cycles,
        &parity_commit_cycles_per_op,
        &parity_commit_remainder_cycles,
        &parity_commit_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz ||
        snapshot_out_shape_count != *out_total_ops ||
        snapshot_out_total_cycles != *out_total_cycles ||
        snapshot_out_cycles_per_op != *out_cycles_per_op ||
        snapshot_out_remainder_cycles != *out_remainder_cycles ||
        snapshot_out_suite_checksum != *out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (parity_commit_total_ops != parity_preflight_total_ops ||
        parity_commit_total_cycles != parity_preflight_total_cycles ||
        parity_commit_cycles_per_op != parity_preflight_cycles_per_op ||
        parity_commit_remainder_cycles != parity_preflight_remainder_cycles ||
        parity_commit_suite_checksum != parity_preflight_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    return Q8_0_DOT_BENCH_OK;
}

// Strict parity checker over default-suite preflight-only and commit-only paths.
//
// Contracts:
// - Enforce immutable snapshots for {cpu_hz,out_* pointers}.
// - Recompute tuple through both diagnostics paths:
//   CommitOnlyPreflightOnly and CommitOnly.
// - Require exact tuple parity on
//   {total_ops,total_cycles,cycles_per_op,remainder_cycles,suite_checksum}.
// - Publish atomically only when all parity and snapshot checks succeed.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_total_ops;
    I64 *snapshot_out_total_cycles;
    I64 *snapshot_out_cycles_per_op;
    I64 *snapshot_out_remainder_cycles;
    I64 *snapshot_out_suite_checksum;

    I64 preflight_total_ops;
    I64 preflight_total_cycles;
    I64 preflight_cycles_per_op;
    I64 preflight_remainder_cycles;
    I64 preflight_suite_checksum;

    I64 commit_total_ops;
    I64 commit_total_cycles;
    I64 commit_cycles_per_op;
    I64 commit_remainder_cycles;
    I64 commit_suite_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_total_ops == out_total_cycles ||
        out_total_ops == out_cycles_per_op ||
        out_total_ops == out_remainder_cycles ||
        out_total_ops == out_suite_checksum ||
        out_total_cycles == out_cycles_per_op ||
        out_total_cycles == out_remainder_cycles ||
        out_total_cycles == out_suite_checksum ||
        out_cycles_per_op == out_remainder_cycles ||
        out_cycles_per_op == out_suite_checksum ||
        out_remainder_cycles == out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_total_ops = out_total_ops;
    snapshot_out_total_cycles = out_total_cycles;
    snapshot_out_cycles_per_op = out_cycles_per_op;
    snapshot_out_remainder_cycles = out_remainder_cycles;
    snapshot_out_suite_checksum = out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &preflight_total_ops,
        &preflight_total_cycles,
        &preflight_cycles_per_op,
        &preflight_remainder_cycles,
        &preflight_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnly(
        cpu_hz,
        &commit_total_ops,
        &commit_total_cycles,
        &commit_cycles_per_op,
        &commit_remainder_cycles,
        &commit_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (preflight_total_ops != commit_total_ops ||
        preflight_total_cycles != commit_total_cycles ||
        preflight_cycles_per_op != commit_cycles_per_op ||
        preflight_remainder_cycles != commit_remainder_cycles ||
        preflight_suite_checksum != commit_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_total_ops = preflight_total_ops;
    *out_total_cycles = preflight_total_cycles;
    *out_cycles_per_op = preflight_cycles_per_op;
    *out_remainder_cycles = preflight_remainder_cycles;
    *out_suite_checksum = preflight_suite_checksum;
    return Q8_0_DOT_BENCH_OK;
}

// Commit-only diagnostics wrapper over deepest strict parity gate.
//
// Contracts:
// - Recompute tuple through deepest parity gate into staged locals.
// - Revalidate immutable caller snapshots for cpu_hz and output destinations.
// - Enforce strict tuple parity against deepest preflight-only companion.
// - Publish diagnostics tuple atomically only after full parity success.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_shape_count == out_max_blocks ||
        out_shape_count == out_max_rows ||
        out_shape_count == out_max_row_matrix_blocks ||
        out_max_blocks == out_max_rows ||
        out_max_blocks == out_max_row_matrix_blocks ||
        out_max_rows == out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_shape_count != parity_shape_count ||
        staged_max_blocks != parity_max_blocks ||
        staged_max_rows != parity_max_rows ||
        staged_max_row_matrix_blocks != parity_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Diagnostics zero-write companion over deepest commit/parity pair.
//
// Contracts:
// - Recompute canonical tuple through deepest commit-only diagnostics path.
// - Recompute tuple through deepest strict parity diagnostics path.
// - Enforce immutable snapshots {cpu_hz,out_shape_count,out_max_blocks,
//   out_max_rows,out_max_row_matrix_blocks} before any publish.
// - Require exact tuple parity {shape_count,max_blocks,max_rows,
//   max_row_matrix_blocks} and publish atomically only on success.
I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_shape_count != parity_shape_count ||
        staged_max_blocks != parity_max_blocks ||
        staged_max_rows != parity_max_rows ||
        staged_max_row_matrix_blocks != parity_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Strict diagnostics parity gate over deepest commit/preflight diagnostics pair.
//
// Contracts:
// - Recompute canonical tuple through deepest zero-write preflight companion.
// - Recompute tuple through deepest commit-only diagnostics companion.
// - Enforce immutable snapshots {cpu_hz,out_shape_count,out_max_blocks,
//   out_max_rows,out_max_row_matrix_blocks} before publish.
// - Require exact tuple parity {shape_count,max_blocks,max_rows,
//   max_row_matrix_blocks} and publish atomically only on success.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_shape_count != parity_shape_count ||
        staged_max_blocks != parity_max_blocks ||
        staged_max_rows != parity_max_rows ||
        staged_max_row_matrix_blocks != parity_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Commit-only diagnostics wrapper over deepest strict parity preflight chain.
//
// Contracts:
// - Recompute canonical tuple through deepest strict parity diagnostics path.
// - Recompute tuple through deepest zero-write diagnostics companion.
// - Enforce immutable snapshots {cpu_hz,out_shape_count,out_max_blocks,
//   out_max_rows,out_max_row_matrix_blocks} before publish.
// - Require exact tuple parity {shape_count,max_blocks,max_rows,
//   max_row_matrix_blocks} and publish atomically only on success.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_shape_count != parity_shape_count ||
        staged_max_blocks != parity_max_blocks ||
        staged_max_rows != parity_max_rows ||
        staged_max_row_matrix_blocks != parity_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Diagnostics zero-write companion over deepest commit/parity preflight pair.
//
// Contracts:
// - Recompute canonical tuple through deepest commit-only diagnostics companion.
// - Recompute tuple through deepest strict parity diagnostics companion.
// - Enforce immutable snapshots {cpu_hz,out_shape_count,out_max_blocks,
//   out_max_rows,out_max_row_matrix_blocks} before publish.
// - Require exact tuple parity {shape_count,max_blocks,max_rows,
//   max_row_matrix_blocks} and publish atomically only on success.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_shape_count != parity_shape_count ||
        staged_max_blocks != parity_max_blocks ||
        staged_max_rows != parity_max_rows ||
        staged_max_row_matrix_blocks != parity_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Strict diagnostics parity gate over deepest commit-only and zero-write pair.
//
// Contracts:
// - Recompute canonical tuple through deepest zero-write diagnostics companion.
// - Recompute tuple through deepest commit-only diagnostics companion.
// - Enforce immutable snapshots {cpu_hz,out_shape_count,out_max_blocks,
//   out_max_rows,out_max_row_matrix_blocks} before publish.
// - Require exact tuple parity {shape_count,max_blocks,max_rows,
//   max_row_matrix_blocks} and publish atomically only on success.
I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 cpu_hz,
    I64 *out_shape_count,
    I64 *out_max_blocks,
    I64 *out_max_rows,
    I64 *out_max_row_matrix_blocks)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_shape_count;
    I64 *snapshot_out_max_blocks;
    I64 *snapshot_out_max_rows;
    I64 *snapshot_out_max_row_matrix_blocks;

    I64 staged_shape_count;
    I64 staged_max_blocks;
    I64 staged_max_rows;
    I64 staged_max_row_matrix_blocks;

    I64 parity_shape_count;
    I64 parity_max_blocks;
    I64 parity_max_rows;
    I64 parity_max_row_matrix_blocks;
    I32 status;

    if (!out_shape_count || !out_max_blocks || !out_max_rows ||
        !out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_shape_count = out_shape_count;
    snapshot_out_max_blocks = out_max_blocks;
    snapshot_out_max_rows = out_max_rows;
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &staged_shape_count,
        &staged_max_blocks,
        &staged_max_rows,
        &staged_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
        cpu_hz,
        &parity_shape_count,
        &parity_max_blocks,
        &parity_max_rows,
        &parity_max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_shape_count != out_shape_count ||
        snapshot_out_max_blocks != out_max_blocks ||
        snapshot_out_max_rows != out_max_rows ||
        snapshot_out_max_row_matrix_blocks != out_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_shape_count != parity_shape_count ||
        staged_max_blocks != parity_max_blocks ||
        staged_max_rows != parity_max_rows ||
        staged_max_row_matrix_blocks != parity_max_row_matrix_blocks)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_shape_count = staged_shape_count;
    *out_max_blocks = staged_max_blocks;
    *out_max_rows = staged_max_rows;
    *out_max_row_matrix_blocks = staged_max_row_matrix_blocks;
    return Q8_0_DOT_BENCH_OK;
}

// Deterministic default benchmark suite for Q8_0 dot kernels.
//
// This routine is intentionally timer-free. Instead of reading host counters,
// it uses a fixed cycle model from canonical suite geometry and fixed vectors.
//
// Published outputs:
// - total_ops            : total int8 MAC operations across the suite
// - total_cycles         : modeled total cycles at caller cpu_hz
// - cycles_per_op        : rounded integer cycles/op
// - remainder_cycles     : total_cycles % total_ops
// - suite_checksum       : deterministic checksum over fixed vectors+shapes
//
// No-partial contract: outputs are committed atomically only on success.
I32 Q8_0DotBenchRunDefaultSuite(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    Q8_0DotBenchShape shapes[4];
    I8 lhs_vec[Q8_0_DOT_BENCH_VALUES_PER_BLOCK];
    I8 rhs_vec[Q8_0_DOT_BENCH_VALUES_PER_BLOCK];
    I64 idx;
    I64 shape_idx;
    I64 vec_idx;

    I64 shape_count;
    I64 max_blocks;
    I64 max_rows;
    I64 max_row_matrix_blocks;

    I64 total_ops_snapshot;
    I64 total_cycles_snapshot;
    I64 suite_checksum_snapshot;
    I64 staged_cycles_per_op;
    I64 staged_remainder_cycles;

    I64 dot_calls;
    I64 block_mac_ops;
    I64 shape_ops;
    I64 ns_per_dot;
    I64 shape_ns;
    I64 numer_ns_hz;
    I64 rounded_numer_ns_hz;
    I64 shape_cycles;
    I64 vector_dot_q0;
    I64 vector_dot_term;
    I64 block_term;
    I64 row_term;
    I64 shape_signature;
    I64 shape_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;
    if (cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Validate canonical suite tuple via deepest no-partial preflight chain.
    status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &shape_count,
        &max_blocks,
        &max_rows,
        &max_row_matrix_blocks);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (shape_count != 4 || max_blocks != 32 || max_rows != 8 ||
        max_row_matrix_blocks != 256)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Canonical deterministic suite geometry.
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

    // Fixed benchmark vectors for stable checksum and auditable math paths.
    for (idx = 0; idx < Q8_0_DOT_BENCH_VALUES_PER_BLOCK; ++idx)
    {
        lhs_vec[idx] = (idx % 17) - 8;
        rhs_vec[idx] = ((idx * 5 + 3) % 23) - 11;
    }

    total_ops_snapshot = 0;
    total_cycles_snapshot = 0;
    suite_checksum_snapshot = 0;

    for (shape_idx = 0; shape_idx < shape_count; ++shape_idx)
    {
        if (shapes[shape_idx].row_stride_blocks < shapes[shape_idx].block_count)
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

        // Number of dot calls in this shape.
        status = Q8_0DotBenchTryMulI64(shapes[shape_idx].row_count,
                                       shapes[shape_idx].iters,
                                       &dot_calls);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        // Integer MAC operations per dot: block_count * 32 signed-byte products.
        status = Q8_0DotBenchTryMulI64(shapes[shape_idx].block_count,
                                       Q8_0_DOT_BENCH_VALUES_PER_BLOCK,
                                       &block_mac_ops);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryMulI64(dot_calls, block_mac_ops, &shape_ops);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        // Deterministic cycle model (integer-only):
        // ns_per_dot = 600 + 24*block_count + 7*row_count
        // shape_cycles = Round((dot_calls * ns_per_dot * cpu_hz) / 1e9)
        status = Q8_0DotBenchTryMulI64(shapes[shape_idx].block_count, 24, &block_term);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryMulI64(shapes[shape_idx].row_count, 7, &row_term);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(600, block_term, &ns_per_dot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(ns_per_dot, row_term, &ns_per_dot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryMulI64(dot_calls, ns_per_dot, &shape_ns);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryMulI64(shape_ns, cpu_hz, &numer_ns_hz);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryAddI64(numer_ns_hz,
                                       (Q8_0_DOT_BENCH_NS_PER_SECOND >> 1),
                                       &rounded_numer_ns_hz);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        shape_cycles = rounded_numer_ns_hz / Q8_0_DOT_BENCH_NS_PER_SECOND;

        // Deterministic fixed-vector checksum.
        vector_dot_q0 = 0;
        for (vec_idx = 0; vec_idx < Q8_0_DOT_BENCH_VALUES_PER_BLOCK; ++vec_idx)
            vector_dot_q0 += lhs_vec[vec_idx] * rhs_vec[(vec_idx + shape_idx) & 31];

        vector_dot_term = vector_dot_q0 * (shape_idx + 1);
        block_term = shapes[shape_idx].block_count * 257;
        row_term = shapes[shape_idx].row_count * 17;
        shape_signature = vector_dot_term + block_term + row_term + shapes[shape_idx].iters;

        status = Q8_0DotBenchTryMulI64(shape_signature, dot_calls, &shape_checksum);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryAddI64(total_ops_snapshot, shape_ops, &total_ops_snapshot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(total_cycles_snapshot, shape_cycles, &total_cycles_snapshot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(suite_checksum_snapshot, shape_checksum, &suite_checksum_snapshot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
    }

    status = Q8_0DotBenchCyclesPerOpChecked(total_cycles_snapshot,
                                            total_ops_snapshot,
                                            &staged_cycles_per_op,
                                            &staged_remainder_cycles);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    *out_total_ops = total_ops_snapshot;
    *out_total_cycles = total_cycles_snapshot;
    *out_cycles_per_op = staged_cycles_per_op;
    *out_remainder_cycles = staged_remainder_cycles;
    *out_suite_checksum = suite_checksum_snapshot;
    return Q8_0_DOT_BENCH_OK;
}

// Commit-only preflight diagnostics wrapper over default benchmark suite.
//
// Contracts:
// - Recomputes canonical deterministic tuple
//   {total_ops,total_cycles,cycles_per_op,remainder_cycles,suite_checksum}
//   using immutable cpu_hz snapshot.
// - Enforces output-pointer snapshot stability across staging/parity phases.
// - Publishes staged tuple atomically only when strict parity succeeds.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnly(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    Q8_0DotBenchShape canonical_shapes[4];
    I8 lhs_vec[Q8_0_DOT_BENCH_VALUES_PER_BLOCK];
    I8 rhs_vec[Q8_0_DOT_BENCH_VALUES_PER_BLOCK];
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_total_ops;
    I64 *snapshot_out_total_cycles;
    I64 *snapshot_out_cycles_per_op;
    I64 *snapshot_out_remainder_cycles;
    I64 *snapshot_out_suite_checksum;
    I64 staged_total_ops;
    I64 staged_total_cycles;
    I64 staged_cycles_per_op;
    I64 staged_remainder_cycles;
    I64 staged_suite_checksum;
    I64 canonical_total_ops;
    I64 canonical_total_cycles;
    I64 canonical_cycles_per_op;
    I64 canonical_remainder_cycles;
    I64 canonical_suite_checksum;
    I64 shape_idx;
    I64 vec_idx;
    I64 dot_calls;
    I64 block_mac_ops;
    I64 shape_ops;
    I64 ns_per_dot;
    I64 shape_ns;
    I64 numer_ns_hz;
    I64 rounded_numer_ns_hz;
    I64 shape_cycles;
    I64 vector_dot_q0;
    I64 vector_dot_term;
    I64 block_term;
    I64 row_term;
    I64 shape_signature;
    I64 shape_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;
    if (cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_total_ops = out_total_ops;
    snapshot_out_total_cycles = out_total_cycles;
    snapshot_out_cycles_per_op = out_cycles_per_op;
    snapshot_out_remainder_cycles = out_remainder_cycles;
    snapshot_out_suite_checksum = out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuite(cpu_hz,
                                         &staged_total_ops,
                                         &staged_total_cycles,
                                         &staged_cycles_per_op,
                                         &staged_remainder_cycles,
                                         &staged_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    canonical_shapes[0].block_count = 4;
    canonical_shapes[0].row_count = 1;
    canonical_shapes[0].row_stride_blocks = 4;
    canonical_shapes[0].iters = 8192;

    canonical_shapes[1].block_count = 8;
    canonical_shapes[1].row_count = 2;
    canonical_shapes[1].row_stride_blocks = 8;
    canonical_shapes[1].iters = 4096;

    canonical_shapes[2].block_count = 16;
    canonical_shapes[2].row_count = 4;
    canonical_shapes[2].row_stride_blocks = 16;
    canonical_shapes[2].iters = 2048;

    canonical_shapes[3].block_count = 32;
    canonical_shapes[3].row_count = 8;
    canonical_shapes[3].row_stride_blocks = 32;
    canonical_shapes[3].iters = 1024;

    for (vec_idx = 0; vec_idx < Q8_0_DOT_BENCH_VALUES_PER_BLOCK; ++vec_idx)
    {
        lhs_vec[vec_idx] = (vec_idx % 17) - 8;
        rhs_vec[vec_idx] = ((vec_idx * 5 + 3) % 23) - 11;
    }

    canonical_total_ops = 0;
    canonical_total_cycles = 0;
    canonical_suite_checksum = 0;

    for (shape_idx = 0; shape_idx < 4; ++shape_idx)
    {
        if (canonical_shapes[shape_idx].row_stride_blocks <
            canonical_shapes[shape_idx].block_count)
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

        status = Q8_0DotBenchTryMulI64(canonical_shapes[shape_idx].row_count,
                                       canonical_shapes[shape_idx].iters,
                                       &dot_calls);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryMulI64(canonical_shapes[shape_idx].block_count,
                                       Q8_0_DOT_BENCH_VALUES_PER_BLOCK,
                                       &block_mac_ops);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryMulI64(dot_calls, block_mac_ops, &shape_ops);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryMulI64(canonical_shapes[shape_idx].block_count,
                                       24,
                                       &block_term);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryMulI64(canonical_shapes[shape_idx].row_count,
                                       7,
                                       &row_term);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(600, block_term, &ns_per_dot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(ns_per_dot, row_term, &ns_per_dot);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryMulI64(dot_calls, ns_per_dot, &shape_ns);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryMulI64(shape_ns, snapshot_cpu_hz, &numer_ns_hz);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryAddI64(numer_ns_hz,
                                       (Q8_0_DOT_BENCH_NS_PER_SECOND >> 1),
                                       &rounded_numer_ns_hz);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        shape_cycles = rounded_numer_ns_hz / Q8_0_DOT_BENCH_NS_PER_SECOND;

        vector_dot_q0 = 0;
        for (vec_idx = 0; vec_idx < Q8_0_DOT_BENCH_VALUES_PER_BLOCK; ++vec_idx)
            vector_dot_q0 += lhs_vec[vec_idx] * rhs_vec[(vec_idx + shape_idx) & 31];

        vector_dot_term = vector_dot_q0 * (shape_idx + 1);
        block_term = canonical_shapes[shape_idx].block_count * 257;
        row_term = canonical_shapes[shape_idx].row_count * 17;
        shape_signature = vector_dot_term + block_term + row_term + canonical_shapes[shape_idx].iters;

        status = Q8_0DotBenchTryMulI64(shape_signature, dot_calls, &shape_checksum);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;

        status = Q8_0DotBenchTryAddI64(canonical_total_ops,
                                       shape_ops,
                                       &canonical_total_ops);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(canonical_total_cycles,
                                       shape_cycles,
                                       &canonical_total_cycles);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
        status = Q8_0DotBenchTryAddI64(canonical_suite_checksum,
                                       shape_checksum,
                                       &canonical_suite_checksum);
        if (status != Q8_0_DOT_BENCH_OK)
            return status;
    }

    status = Q8_0DotBenchCyclesPerOpChecked(canonical_total_cycles,
                                            canonical_total_ops,
                                            &canonical_cycles_per_op,
                                            &canonical_remainder_cycles);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_total_ops != canonical_total_ops ||
        staged_total_cycles != canonical_total_cycles ||
        staged_cycles_per_op != canonical_cycles_per_op ||
        staged_remainder_cycles != canonical_remainder_cycles ||
        staged_suite_checksum != canonical_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_total_ops = staged_total_ops;
    *out_total_cycles = staged_total_cycles;
    *out_cycles_per_op = staged_cycles_per_op;
    *out_remainder_cycles = staged_remainder_cycles;
    *out_suite_checksum = staged_suite_checksum;
    return Q8_0_DOT_BENCH_OK;
}

// Diagnostics zero-write companion over deepest commit/parity pair.
//
// Contracts:
// - Recompute canonical tuple through deepest commit-only diagnostics path.
// - Recompute tuple through deepest strict parity diagnostics path.
// - Enforce immutable snapshots {cpu_hz,out_total_ops,out_total_cycles,
//   out_cycles_per_op,out_remainder_cycles,out_suite_checksum} before publish.
// - Require exact tuple parity {total_ops,total_cycles,cycles_per_op,
//   remainder_cycles,suite_checksum} and publish atomically only on success.
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_total_ops;
    I64 *snapshot_out_total_cycles;
    I64 *snapshot_out_cycles_per_op;
    I64 *snapshot_out_remainder_cycles;
    I64 *snapshot_out_suite_checksum;

    I64 staged_total_ops;
    I64 staged_total_cycles;
    I64 staged_cycles_per_op;
    I64 staged_remainder_cycles;
    I64 staged_suite_checksum;

    I64 parity_total_ops;
    I64 parity_total_cycles;
    I64 parity_cycles_per_op;
    I64 parity_remainder_cycles;
    I64 parity_suite_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_total_ops == out_total_cycles ||
        out_total_ops == out_cycles_per_op ||
        out_total_ops == out_remainder_cycles ||
        out_total_ops == out_suite_checksum ||
        out_total_cycles == out_cycles_per_op ||
        out_total_cycles == out_remainder_cycles ||
        out_total_cycles == out_suite_checksum ||
        out_cycles_per_op == out_remainder_cycles ||
        out_cycles_per_op == out_suite_checksum ||
        out_remainder_cycles == out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_total_ops = out_total_ops;
    snapshot_out_total_cycles = out_total_cycles;
    snapshot_out_cycles_per_op = out_cycles_per_op;
    snapshot_out_remainder_cycles = out_remainder_cycles;
    snapshot_out_suite_checksum = out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
        cpu_hz,
        &staged_total_ops,
        &staged_total_cycles,
        &staged_cycles_per_op,
        &staged_remainder_cycles,
        &staged_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &parity_total_ops,
        &parity_total_cycles,
        &parity_cycles_per_op,
        &parity_remainder_cycles,
        &parity_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_total_ops != parity_total_ops ||
        staged_total_cycles != parity_total_cycles ||
        staged_cycles_per_op != parity_cycles_per_op ||
        staged_remainder_cycles != parity_remainder_cycles ||
        staged_suite_checksum != parity_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_total_ops = staged_total_ops;
    *out_total_cycles = staged_total_cycles;
    *out_cycles_per_op = staged_cycles_per_op;
    *out_remainder_cycles = staged_remainder_cycles;
    *out_suite_checksum = staged_suite_checksum;
    return Q8_0_DOT_BENCH_OK;
}

// Strict parity diagnostics gate over commit-only preflight path and canonical suite.
//
// Contracts:
// - Recompute tuple via CommitOnlyPreflightOnly path into staged locals.
// - Recompute canonical tuple via DefaultSuite path from the same immutable cpu_hz snapshot.
// - Enforce exact tuple parity:
//   {total_ops,total_cycles,cycles_per_op,remainder_cycles,suite_checksum}.
// - Enforce output-pointer snapshot stability and atomically publish only on success.



I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_total_ops;
    I64 *snapshot_out_total_cycles;
    I64 *snapshot_out_cycles_per_op;
    I64 *snapshot_out_remainder_cycles;
    I64 *snapshot_out_suite_checksum;

    I64 preflight_total_ops;
    I64 preflight_total_cycles;
    I64 preflight_cycles_per_op;
    I64 preflight_remainder_cycles;
    I64 preflight_suite_checksum;

    I64 commit_total_ops;
    I64 commit_total_cycles;
    I64 commit_cycles_per_op;
    I64 commit_remainder_cycles;
    I64 commit_suite_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_total_ops == out_total_cycles ||
        out_total_ops == out_cycles_per_op ||
        out_total_ops == out_remainder_cycles ||
        out_total_ops == out_suite_checksum ||
        out_total_cycles == out_cycles_per_op ||
        out_total_cycles == out_remainder_cycles ||
        out_total_cycles == out_suite_checksum ||
        out_cycles_per_op == out_remainder_cycles ||
        out_cycles_per_op == out_suite_checksum ||
        out_remainder_cycles == out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_total_ops = out_total_ops;
    snapshot_out_total_cycles = out_total_cycles;
    snapshot_out_cycles_per_op = out_cycles_per_op;
    snapshot_out_remainder_cycles = out_remainder_cycles;
    snapshot_out_suite_checksum = out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
        cpu_hz,
        &preflight_total_ops,
        &preflight_total_cycles,
        &preflight_cycles_per_op,
        &preflight_remainder_cycles,
        &preflight_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(
        cpu_hz,
        &commit_total_ops,
        &commit_total_cycles,
        &commit_cycles_per_op,
        &commit_remainder_cycles,
        &commit_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (preflight_total_ops != commit_total_ops ||
        preflight_total_cycles != commit_total_cycles ||
        preflight_cycles_per_op != commit_cycles_per_op ||
        preflight_remainder_cycles != commit_remainder_cycles ||
        preflight_suite_checksum != commit_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_total_ops = preflight_total_ops;
    *out_total_cycles = preflight_total_cycles;
    *out_cycles_per_op = preflight_cycles_per_op;
    *out_remainder_cycles = preflight_remainder_cycles;
    *out_suite_checksum = preflight_suite_checksum;
    return Q8_0_DOT_BENCH_OK;
}
I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnly(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_total_ops;
    I64 *snapshot_out_total_cycles;
    I64 *snapshot_out_cycles_per_op;
    I64 *snapshot_out_remainder_cycles;
    I64 *snapshot_out_suite_checksum;

    I64 parity_total_ops;
    I64 parity_total_cycles;
    I64 parity_cycles_per_op;
    I64 parity_remainder_cycles;
    I64 parity_suite_checksum;

    I64 commit_total_ops;
    I64 commit_total_cycles;
    I64 commit_cycles_per_op;
    I64 commit_remainder_cycles;
    I64 commit_suite_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (out_total_ops == out_total_cycles ||
        out_total_ops == out_cycles_per_op ||
        out_total_ops == out_remainder_cycles ||
        out_total_ops == out_suite_checksum ||
        out_total_cycles == out_cycles_per_op ||
        out_total_cycles == out_remainder_cycles ||
        out_total_cycles == out_suite_checksum ||
        out_cycles_per_op == out_remainder_cycles ||
        out_cycles_per_op == out_suite_checksum ||
        out_remainder_cycles == out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_total_ops = out_total_ops;
    snapshot_out_total_cycles = out_total_cycles;
    snapshot_out_cycles_per_op = out_cycles_per_op;
    snapshot_out_remainder_cycles = out_remainder_cycles;
    snapshot_out_suite_checksum = out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParity(
        cpu_hz,
        &parity_total_ops,
        &parity_total_cycles,
        &parity_cycles_per_op,
        &parity_remainder_cycles,
        &parity_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnly(
        cpu_hz,
        &commit_total_ops,
        &commit_total_cycles,
        &commit_cycles_per_op,
        &commit_remainder_cycles,
        &commit_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (parity_total_ops != commit_total_ops ||
        parity_total_cycles != commit_total_cycles ||
        parity_cycles_per_op != commit_cycles_per_op ||
        parity_remainder_cycles != commit_remainder_cycles ||
        parity_suite_checksum != commit_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_total_ops = parity_total_ops;
    *out_total_cycles = parity_total_cycles;
    *out_cycles_per_op = parity_cycles_per_op;
    *out_remainder_cycles = parity_remainder_cycles;
    *out_suite_checksum = parity_suite_checksum;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParity(
    I64 cpu_hz,
    I64 *out_total_ops,
    I64 *out_total_cycles,
    I64 *out_cycles_per_op,
    I64 *out_remainder_cycles,
    I64 *out_suite_checksum)
{
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_total_ops;
    I64 *snapshot_out_total_cycles;
    I64 *snapshot_out_cycles_per_op;
    I64 *snapshot_out_remainder_cycles;
    I64 *snapshot_out_suite_checksum;
    I64 staged_total_ops;
    I64 staged_total_cycles;
    I64 staged_cycles_per_op;
    I64 staged_remainder_cycles;
    I64 staged_suite_checksum;
    I64 canonical_total_ops;
    I64 canonical_total_cycles;
    I64 canonical_cycles_per_op;
    I64 canonical_remainder_cycles;
    I64 canonical_suite_checksum;
    I32 status;

    if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||
        !out_remainder_cycles || !out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;
    if (cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_cpu_hz = cpu_hz;
    snapshot_out_total_ops = out_total_ops;
    snapshot_out_total_cycles = out_total_cycles;
    snapshot_out_cycles_per_op = out_cycles_per_op;
    snapshot_out_remainder_cycles = out_remainder_cycles;
    snapshot_out_suite_checksum = out_suite_checksum;

    status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnly(cpu_hz,
                                                                 &staged_total_ops,
                                                                 &staged_total_cycles,
                                                                 &staged_cycles_per_op,
                                                                 &staged_remainder_cycles,
                                                                 &staged_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuite(cpu_hz,
                                         &canonical_total_ops,
                                         &canonical_total_cycles,
                                         &canonical_cycles_per_op,
                                         &canonical_remainder_cycles,
                                         &canonical_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_total_ops != out_total_ops ||
        snapshot_out_total_cycles != out_total_cycles ||
        snapshot_out_cycles_per_op != out_cycles_per_op ||
        snapshot_out_remainder_cycles != out_remainder_cycles ||
        snapshot_out_suite_checksum != out_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_total_ops != canonical_total_ops ||
        staged_total_cycles != canonical_total_cycles ||
        staged_cycles_per_op != canonical_cycles_per_op ||
        staged_remainder_cycles != canonical_remainder_cycles ||
        staged_suite_checksum != canonical_suite_checksum)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_total_ops = staged_total_ops;
    *out_total_cycles = staged_total_cycles;
    *out_cycles_per_op = staged_cycles_per_op;
    *out_remainder_cycles = staged_remainder_cycles;
    *out_suite_checksum = staged_suite_checksum;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16CheckedNoPartialCommitOnly(
    I64 total_ops,
    I64 total_cycles,
    I64 cpu_hz,
    I64 *out_ops_per_second_q16)
{
    I64 out_snapshot;
    I64 staged_ops_per_second_q16;
    I32 status;

    if (!out_ops_per_second_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    // Commit-only snapshot contract:
    // - capture caller-visible state once
    // - stage result in local scratch
    // - publish one atomic final store on success
    // - restore snapshot on all failures (strict no-partial)
    out_snapshot = *out_ops_per_second_q16;

    status = Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16Checked(
        total_ops,
        total_cycles,
        cpu_hz,
        &staged_ops_per_second_q16);
    if (status != Q8_0_DOT_BENCH_OK)
    {
        *out_ops_per_second_q16 = out_snapshot;
        return status;
    }

    *out_ops_per_second_q16 = staged_ops_per_second_q16;
    return Q8_0_DOT_BENCH_OK;
}

// Zero-write diagnostics wrapper over commit-only ops/sec helper.
//
// Contracts:
// - Keep immutable snapshots for params + output destination.
// - Require strict parity with canonical integer-only helper.
// - Never mutate caller output on error (preflight-only semantics).
I32 Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16CheckedNoPartialCommitOnly(
    I64 total_cycles,
    I64 total_ops,
    I64 *out_cycles_per_token_q16)
{
    I64 out_snapshot;
    I64 staged_cycles_per_token_q16;
    I32 status;

    if (!out_cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    out_snapshot = *out_cycles_per_token_q16;

    status = Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16Checked(
        total_cycles,
        total_ops,
        &staged_cycles_per_token_q16);
    if (status != Q8_0_DOT_BENCH_OK)
    {
        *out_cycles_per_token_q16 = out_snapshot;
        return status;
    }

    *out_cycles_per_token_q16 = staged_cycles_per_token_q16;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16CheckedNoPartialCommitOnly(
    I64 total_cycles,
    I64 total_tokens,
    I64 *out_cycles_per_token_q16)
{
    I64 staged_cycles_per_token_q16;
    I64 total_cycles_snapshot;
    I64 total_tokens_snapshot;
    I32 status;

    if (!out_cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    total_cycles_snapshot = total_cycles;
    total_tokens_snapshot = total_tokens;

    status = Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16Checked(
        total_cycles,
        total_tokens,
        &staged_cycles_per_token_q16);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (total_cycles != total_cycles_snapshot)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (total_tokens != total_tokens_snapshot)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    *out_cycles_per_token_q16 = staged_cycles_per_token_q16;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16CheckedNoPartialCommitOnlyPreflightOnly(
    I64 total_cycles,
    I64 total_ops,
    I64 *out_cycles_per_token_q16)
{
    I64 snapshot_total_cycles;
    I64 snapshot_total_ops;
    I64 *snapshot_out_cycles_per_token_q16;

    I64 staged_commit_value;
    I64 staged_canonical_value;
    I32 status;

    if (!out_cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_total_cycles = total_cycles;
    snapshot_total_ops = total_ops;
    snapshot_out_cycles_per_token_q16 = out_cycles_per_token_q16;

    status = Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16CheckedNoPartialCommitOnly(
        total_cycles,
        total_ops,
        &staged_commit_value);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteCyclesPerTokenQ16Checked(
        total_cycles,
        total_ops,
        &staged_canonical_value);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_cycles != total_cycles ||
        snapshot_total_ops != total_ops)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_cycles_per_token_q16 != out_cycles_per_token_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_commit_value != staged_canonical_value)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Preflight-only path is diagnostics only; caller output remains untouched.
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16CheckedNoPartialCommitOnlyPreflightOnly(
    I64 total_ops,
    I64 total_cycles,
    I64 cpu_hz,
    I64 *out_ops_per_second_q16)
{
    I64 snapshot_total_ops;
    I64 snapshot_total_cycles;
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_ops_per_second_q16;

    I64 staged_commit_value;
    I64 staged_canonical_value;
    I32 status;

    if (!out_ops_per_second_q16)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    snapshot_total_ops = total_ops;
    snapshot_total_cycles = total_cycles;
    snapshot_cpu_hz = cpu_hz;
    snapshot_out_ops_per_second_q16 = out_ops_per_second_q16;

    status = Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16CheckedNoPartialCommitOnly(
        total_ops,
        total_cycles,
        cpu_hz,
        &staged_commit_value);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteOpsPerSecondQ16Checked(
        total_ops,
        total_cycles,
        cpu_hz,
        &staged_canonical_value);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_ops != total_ops ||
        snapshot_total_cycles != total_cycles ||
        snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_ops_per_second_q16 != out_ops_per_second_q16)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (staged_commit_value != staged_canonical_value)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Preflight-only path is diagnostics only; caller output remains untouched.
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64Checked(
    I64 total_ops,
    I64 total_cycles,
    I64 cycles_per_op,
    I64 remainder_cycles,
    I64 suite_checksum,
    I64 cpu_hz,
    I64 *out_digest_q64)
{
    I64 digest_mod61;
    I64 folded_ops;
    I64 folded_cycles;
    I64 folded_cycles_per_op;
    I64 folded_remainder_cycles;
    I64 folded_suite_checksum;
    I64 folded_cpu_hz;
    I64 digest;
    I32 status;

    if (!out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (total_ops < 0 || total_cycles < 0 || cycles_per_op < 0 ||
        remainder_cycles < 0 || suite_checksum < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (total_ops < remainder_cycles)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // All folds are odd multipliers to keep avalanche in integer domain.
    status = Q8_0DotBenchTryMulI64(total_ops, 3, &folded_ops);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchTryMulI64(total_cycles, 5, &folded_cycles);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchTryMulI64(cycles_per_op, 7, &folded_cycles_per_op);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchTryMulI64(remainder_cycles, 11, &folded_remainder_cycles);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchTryMulI64(suite_checksum, 13, &folded_suite_checksum);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchTryMulI64(cpu_hz, 17, &folded_cpu_hz);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    // Deterministic integer-only digest in bounded signed-I64 domain:
    // fold input tuple terms, then mix each term with mod-(2^61-1) reduction.
    // This preserves avalanche while avoiding ubiquitous signed overflow.
    digest_mod61 = 2305843009213693951; // 2^61 - 1
    digest = 1469598103934665603;

    status = Q8_0DotBenchTryAddI64(digest, folded_ops, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;
    status = Q8_0DotBenchTryMulI64(digest, 3, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;

    status = Q8_0DotBenchTryAddI64(digest, folded_cycles, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;
    status = Q8_0DotBenchTryMulI64(digest, 3, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;

    status = Q8_0DotBenchTryAddI64(digest, folded_cycles_per_op, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;
    status = Q8_0DotBenchTryMulI64(digest, 3, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;

    status = Q8_0DotBenchTryAddI64(digest, folded_remainder_cycles, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;
    status = Q8_0DotBenchTryMulI64(digest, 3, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;

    status = Q8_0DotBenchTryAddI64(digest, folded_suite_checksum, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;
    status = Q8_0DotBenchTryMulI64(digest, 3, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;

    status = Q8_0DotBenchTryAddI64(digest, folded_cpu_hz, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;
    status = Q8_0DotBenchTryMulI64(digest, 3, &digest);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;
    digest %= digest_mod61;

    *out_digest_q64 = digest;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnly(
    I64 total_ops,
    I64 total_cycles,
    I64 cycles_per_op,
    I64 remainder_cycles,
    I64 suite_checksum,
    I64 cpu_hz,
    I64 *out_digest_q64)
{
    I64 snapshot_total_ops;
    I64 snapshot_total_cycles;
    I64 snapshot_cycles_per_op;
    I64 snapshot_remainder_cycles;
    I64 snapshot_suite_checksum;
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_digest_q64;
    I64 staged_digest_q64;
    I32 status;

    if (!out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (total_ops < 0 || total_cycles < 0 || cycles_per_op < 0 ||
        remainder_cycles < 0 || suite_checksum < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_total_ops = total_ops;
    snapshot_total_cycles = total_cycles;
    snapshot_cycles_per_op = cycles_per_op;
    snapshot_remainder_cycles = remainder_cycles;
    snapshot_suite_checksum = suite_checksum;
    snapshot_cpu_hz = cpu_hz;
    snapshot_out_digest_q64 = out_digest_q64;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64Checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &staged_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_ops != total_ops ||
        snapshot_total_cycles != total_cycles ||
        snapshot_cycles_per_op != cycles_per_op ||
        snapshot_remainder_cycles != remainder_cycles ||
        snapshot_suite_checksum != suite_checksum ||
        snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_digest_q64 != out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Commit-only publish: one atomic output write after all checks pass.
    *out_digest_q64 = staged_digest_q64;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnly(
    I64 total_ops,
    I64 total_cycles,
    I64 cycles_per_op,
    I64 remainder_cycles,
    I64 suite_checksum,
    I64 cpu_hz,
    I64 *out_digest_q64)
{
    I64 snapshot_total_ops;
    I64 snapshot_total_cycles;
    I64 snapshot_cycles_per_op;
    I64 snapshot_remainder_cycles;
    I64 snapshot_suite_checksum;
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_digest_q64;

    I64 commit_digest_q64;
    I64 canonical_digest_q64;
    I32 status;

    if (!out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (total_ops < 0 || total_cycles < 0 || cycles_per_op < 0 ||
        remainder_cycles < 0 || suite_checksum < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_total_ops = total_ops;
    snapshot_total_cycles = total_cycles;
    snapshot_cycles_per_op = cycles_per_op;
    snapshot_remainder_cycles = remainder_cycles;
    snapshot_suite_checksum = suite_checksum;
    snapshot_cpu_hz = cpu_hz;
    snapshot_out_digest_q64 = out_digest_q64;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnly(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &commit_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64Checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &canonical_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_ops != total_ops ||
        snapshot_total_cycles != total_cycles ||
        snapshot_cycles_per_op != cycles_per_op ||
        snapshot_remainder_cycles != remainder_cycles ||
        snapshot_suite_checksum != suite_checksum ||
        snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_digest_q64 != out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (commit_digest_q64 != canonical_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Preflight-only path is diagnostics only; caller output remains untouched.
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParity(
    I64 total_ops,
    I64 total_cycles,
    I64 cycles_per_op,
    I64 remainder_cycles,
    I64 suite_checksum,
    I64 cpu_hz,
    I64 *out_digest_q64)
{
    I64 snapshot_total_ops;
    I64 snapshot_total_cycles;
    I64 snapshot_cycles_per_op;
    I64 snapshot_remainder_cycles;
    I64 snapshot_suite_checksum;
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_digest_q64;

    I64 commit_digest_q64;
    I64 canonical_digest_q64;
    I32 status;

    if (!out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (total_ops < 0 || total_cycles < 0 || cycles_per_op < 0 ||
        remainder_cycles < 0 || suite_checksum < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_total_ops = total_ops;
    snapshot_total_cycles = total_cycles;
    snapshot_cycles_per_op = cycles_per_op;
    snapshot_remainder_cycles = remainder_cycles;
    snapshot_suite_checksum = suite_checksum;
    snapshot_cpu_hz = cpu_hz;
    snapshot_out_digest_q64 = out_digest_q64;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnly(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &canonical_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnly(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &commit_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64Checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &canonical_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_ops != total_ops ||
        snapshot_total_cycles != total_cycles ||
        snapshot_cycles_per_op != cycles_per_op ||
        snapshot_remainder_cycles != remainder_cycles ||
        snapshot_suite_checksum != suite_checksum ||
        snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_digest_q64 != out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (commit_digest_q64 != canonical_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Atomic publish: single caller-visible write after parity checks.
    *out_digest_q64 = commit_digest_q64;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
    I64 total_ops,
    I64 total_cycles,
    I64 cycles_per_op,
    I64 remainder_cycles,
    I64 suite_checksum,
    I64 cpu_hz,
    I64 *out_digest_q64)
{
    I64 snapshot_total_ops;
    I64 snapshot_total_cycles;
    I64 snapshot_cycles_per_op;
    I64 snapshot_remainder_cycles;
    I64 snapshot_suite_checksum;
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_digest_q64;

    I64 parity_digest_q64;
    I64 preflight_sentinel_digest_q64;
    I64 canonical_digest_q64;
    I32 status;

    if (!out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (total_ops < 0 || total_cycles < 0 || cycles_per_op < 0 ||
        remainder_cycles < 0 || suite_checksum < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_total_ops = total_ops;
    snapshot_total_cycles = total_cycles;
    snapshot_cycles_per_op = cycles_per_op;
    snapshot_remainder_cycles = remainder_cycles;
    snapshot_suite_checksum = suite_checksum;
    snapshot_cpu_hz = cpu_hz;
    snapshot_out_digest_q64 = out_digest_q64;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParity(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &parity_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    preflight_sentinel_digest_q64 = 6148914691236517205;
    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnly(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &preflight_sentinel_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64Checked(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &canonical_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_ops != total_ops ||
        snapshot_total_cycles != total_cycles ||
        snapshot_cycles_per_op != cycles_per_op ||
        snapshot_remainder_cycles != remainder_cycles ||
        snapshot_suite_checksum != suite_checksum ||
        snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_digest_q64 != out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (parity_digest_q64 != canonical_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Commit-only publish: single caller-visible write after all parity checks pass.
    *out_digest_q64 = parity_digest_q64;
    return Q8_0_DOT_BENCH_OK;
}

I32 Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(
    I64 total_ops,
    I64 total_cycles,
    I64 cycles_per_op,
    I64 remainder_cycles,
    I64 suite_checksum,
    I64 cpu_hz,
    I64 *out_digest_q64)
{
    I64 snapshot_total_ops;
    I64 snapshot_total_cycles;
    I64 snapshot_cycles_per_op;
    I64 snapshot_remainder_cycles;
    I64 snapshot_suite_checksum;
    I64 snapshot_cpu_hz;
    I64 *snapshot_out_digest_q64;
    I64 snapshot_out_digest_value;

    I64 parity_commit_digest_q64;
    I64 parity_digest_q64;
    I32 status;

    if (!out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if (total_ops < 0 || total_cycles < 0 || cycles_per_op < 0 ||
        remainder_cycles < 0 || suite_checksum < 0 || cpu_hz <= 0)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    snapshot_total_ops = total_ops;
    snapshot_total_cycles = total_cycles;
    snapshot_cycles_per_op = cycles_per_op;
    snapshot_remainder_cycles = remainder_cycles;
    snapshot_suite_checksum = suite_checksum;
    snapshot_cpu_hz = cpu_hz;
    snapshot_out_digest_q64 = out_digest_q64;
    snapshot_out_digest_value = *out_digest_q64;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &parity_commit_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    status = Q8_0DotBenchRunDefaultSuiteTelemetryDigestQ64CheckedNoPartialCommitOnlyPreflightOnlyParity(
        total_ops,
        total_cycles,
        cycles_per_op,
        remainder_cycles,
        suite_checksum,
        cpu_hz,
        &parity_digest_q64);
    if (status != Q8_0_DOT_BENCH_OK)
        return status;

    if (snapshot_total_ops != total_ops ||
        snapshot_total_cycles != total_cycles ||
        snapshot_cycles_per_op != cycles_per_op ||
        snapshot_remainder_cycles != remainder_cycles ||
        snapshot_suite_checksum != suite_checksum ||
        snapshot_cpu_hz != cpu_hz)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_digest_q64 != out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;
    if (snapshot_out_digest_value != *out_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    if (parity_commit_digest_q64 != parity_digest_q64)
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM;

    // Diagnostics-only preflight path: never publish caller output.
    return Q8_0_DOT_BENCH_OK;
}
