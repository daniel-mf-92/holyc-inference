// q8_0_dot_bench.HC
// Deterministic integer-only helpers for Q8_0 benchmark reporting.

#define Q8_0_DOT_BENCH_OK                 0
#define Q8_0_DOT_BENCH_ERR_NULL_PTR       1
#define Q8_0_DOT_BENCH_ERR_BAD_PARAM      2
#define Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO    3
#define Q8_0_DOT_BENCH_ERR_OVERFLOW       4

#define Q8_0_DOT_BENCH_I64_MAX            0x7FFFFFFFFFFFFFFF

I32 Q8_0DotBenchTryAddI64(I64 lhs, I64 rhs, I64 *out_sum)
{
    if (!out_sum)
        return Q8_0_DOT_BENCH_ERR_NULL_PTR;

    if ((rhs > 0) && (lhs > Q8_0_DOT_BENCH_I64_MAX - rhs))
        return Q8_0_DOT_BENCH_ERR_OVERFLOW;

    *out_sum = lhs + rhs;
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
