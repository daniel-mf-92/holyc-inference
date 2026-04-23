from pathlib import Path

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2


def canonical_tuple(cpu_hz: int) -> tuple[int, int, int, int, int] | None:
    if cpu_hz <= 0:
        return None

    shapes = [
        {"block_count": 4, "row_count": 1, "row_stride_blocks": 4, "iters": 8192},
        {"block_count": 8, "row_count": 2, "row_stride_blocks": 8, "iters": 4096},
        {"block_count": 16, "row_count": 4, "row_stride_blocks": 16, "iters": 2048},
        {"block_count": 32, "row_count": 8, "row_stride_blocks": 32, "iters": 1024},
    ]

    lhs = [((i * 3 + 7) & 0x7F) - 64 for i in range(32)]
    rhs = [((i * 5 + 11) & 0x7F) - 64 for i in range(32)]

    total_ops = 0
    total_cycles = 0
    suite_checksum = 0

    for shape_idx, shape in enumerate(shapes):
        dot_calls = shape["row_count"] * shape["iters"]
        shape_ops = dot_calls * shape["block_count"] * 32
        shape_cycles = shape_ops + (shape["block_count"] * shape["row_count"] * 37) + ((cpu_hz >> 20) & 1023)

        vector_dot_q0 = sum(lhs[i] * rhs[(i + shape_idx) & 31] for i in range(32))
        vector_dot_term = vector_dot_q0 * (shape_idx + 1)
        shape_signature = vector_dot_term + shape["block_count"] * 257 + shape["row_count"] * 17 + shape["iters"]
        shape_checksum = shape_signature * dot_calls

        total_ops += shape_ops
        total_cycles += shape_cycles
        suite_checksum += shape_checksum

    cycles_per_op = (total_cycles + (total_ops >> 1)) // total_ops
    remainder_cycles = total_cycles % total_ops

    return (total_ops, total_cycles, cycles_per_op, remainder_cycles, suite_checksum)


def parity_commit_only_preflight_only(
    cpu_hz: int,
    out_total_ops: list[int] | None,
    out_total_cycles: list[int] | None,
    out_cycles_per_op: list[int] | None,
    out_remainder_cycles: list[int] | None,
    out_suite_checksum: list[int] | None,
) -> int:
    if (
        out_total_ops is None
        or out_total_cycles is None
        or out_cycles_per_op is None
        or out_remainder_cycles is None
        or out_suite_checksum is None
    ):
        return Q8_0_DOT_BENCH_ERR_NULL_PTR

    if (
        out_total_ops is out_total_cycles
        or out_total_ops is out_cycles_per_op
        or out_total_ops is out_remainder_cycles
        or out_total_ops is out_suite_checksum
        or out_total_cycles is out_cycles_per_op
        or out_total_cycles is out_remainder_cycles
        or out_total_cycles is out_suite_checksum
        or out_cycles_per_op is out_remainder_cycles
        or out_cycles_per_op is out_suite_checksum
        or out_remainder_cycles is out_suite_checksum
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    snapshot_cpu_hz = cpu_hz
    snapshot_out_shape_count = out_total_ops[0]
    snapshot_out_total_cycles = out_total_cycles[0]
    snapshot_out_cycles_per_op = out_cycles_per_op[0]
    snapshot_out_remainder_cycles = out_remainder_cycles[0]
    snapshot_out_suite_checksum = out_suite_checksum[0]

    staged = canonical_tuple(cpu_hz)
    if staged is None:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    canonical = canonical_tuple(cpu_hz)
    if canonical is None:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    if (
        snapshot_cpu_hz != cpu_hz
        or snapshot_out_shape_count != out_total_ops[0]
        or snapshot_out_total_cycles != out_total_cycles[0]
        or snapshot_out_cycles_per_op != out_cycles_per_op[0]
        or snapshot_out_remainder_cycles != out_remainder_cycles[0]
        or snapshot_out_suite_checksum != out_suite_checksum[0]
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    if staged != canonical:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    # preflight-only => no publication
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1303_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||" in body
    assert "if (out_total_ops == out_total_cycles ||" in body
    assert "snapshot_cpu_hz = cpu_hz;" in body
    assert "snapshot_out_shape_count = *out_total_ops;" in body
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "if (parity_commit_total_ops != parity_preflight_total_ops ||" in body
    assert "return Q8_0_DOT_BENCH_OK;" in body


def test_null_and_alias_guards() -> None:
    assert parity_commit_only_preflight_only(3_200_000_000, None, [1], [1], [1], [1]) == Q8_0_DOT_BENCH_ERR_NULL_PTR

    same = [77]
    status = parity_commit_only_preflight_only(3_200_000_000, same, same, [3], [4], [5])
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM


def test_bad_frequency_no_partial_commit() -> None:
    out_total_ops = [111]
    out_total_cycles = [222]
    out_cycles_per_op = [333]
    out_remainder_cycles = [444]
    out_suite_checksum = [555]

    status = parity_commit_only_preflight_only(
        0,
        out_total_ops,
        out_total_cycles,
        out_cycles_per_op,
        out_remainder_cycles,
        out_suite_checksum,
    )
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert (out_total_ops[0], out_total_cycles[0], out_cycles_per_op[0], out_remainder_cycles[0], out_suite_checksum[0]) == (
        111,
        222,
        333,
        444,
        555,
    )


def test_preflight_only_preserves_outputs_on_success() -> None:
    out_total_ops = [19]
    out_total_cycles = [29]
    out_cycles_per_op = [39]
    out_remainder_cycles = [49]
    out_suite_checksum = [59]

    status = parity_commit_only_preflight_only(
        3_200_000_000,
        out_total_ops,
        out_total_cycles,
        out_cycles_per_op,
        out_remainder_cycles,
        out_suite_checksum,
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert (out_total_ops[0], out_total_cycles[0], out_cycles_per_op[0], out_remainder_cycles[0], out_suite_checksum[0]) == (
        19,
        29,
        39,
        49,
        59,
    )


if __name__ == "__main__":
    test_source_contains_iq1303_function_and_guards()
    test_null_and_alias_guards()
    test_bad_frequency_no_partial_commit()
    test_preflight_only_preserves_outputs_on_success()
    print("ok")
