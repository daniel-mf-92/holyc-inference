#!/usr/bin/env python3
from pathlib import Path

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2


def canonical_tuple(cpu_hz: int):
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


def parity_commit_only_preflight_only_parity_commit_only(
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

    staged = canonical_tuple(cpu_hz)
    if staged is None:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    parity = canonical_tuple(cpu_hz)
    if parity is None:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if staged != parity:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_total_ops[0] = staged[0]
    out_total_cycles[0] = staged[1]
    out_cycles_per_op[0] = staged[2]
    out_remainder_cycles[0] = staged[3]
    out_suite_checksum[0] = staged[4]
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1308_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "if (!out_total_ops || !out_total_cycles || !out_cycles_per_op ||" in body
    assert "if (out_total_ops == out_total_cycles ||" in body
    assert "snapshot_cpu_hz = cpu_hz;" in body
    assert "status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "if (staged_total_ops != preflight_total_ops ||" in body
    assert "*out_total_ops = staged_total_ops;" in body
    assert "*out_total_cycles = staged_total_cycles;" in body
    assert "*out_cycles_per_op = staged_cycles_per_op;" in body
    assert "*out_remainder_cycles = staged_remainder_cycles;" in body
    assert "*out_suite_checksum = staged_suite_checksum;" in body


def test_null_and_alias_guards() -> None:
    assert (
        parity_commit_only_preflight_only_parity_commit_only(3_200_000_000, None, [1], [1], [1], [1])
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )

    same = [77]
    status = parity_commit_only_preflight_only_parity_commit_only(3_200_000_000, same, same, [3], [4], [5])
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM


def test_bad_frequency_no_partial_commit() -> None:
    out_total_ops = [111]
    out_total_cycles = [222]
    out_cycles_per_op = [333]
    out_remainder_cycles = [444]
    out_suite_checksum = [555]

    status = parity_commit_only_preflight_only_parity_commit_only(
        0,
        out_total_ops,
        out_total_cycles,
        out_cycles_per_op,
        out_remainder_cycles,
        out_suite_checksum,
    )
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert (
        out_total_ops[0],
        out_total_cycles[0],
        out_cycles_per_op[0],
        out_remainder_cycles[0],
        out_suite_checksum[0],
    ) == (111, 222, 333, 444, 555)


def test_parity_publish_matches_canonical() -> None:
    out_total_ops = [0]
    out_total_cycles = [0]
    out_cycles_per_op = [0]
    out_remainder_cycles = [0]
    out_suite_checksum = [0]

    status = parity_commit_only_preflight_only_parity_commit_only(
        3_200_000_000,
        out_total_ops,
        out_total_cycles,
        out_cycles_per_op,
        out_remainder_cycles,
        out_suite_checksum,
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert (
        out_total_ops[0],
        out_total_cycles[0],
        out_cycles_per_op[0],
        out_remainder_cycles[0],
        out_suite_checksum[0],
    ) == canonical_tuple(3_200_000_000)


if __name__ == "__main__":
    test_source_contains_iq1308_function_and_guards()
    test_null_and_alias_guards()
    test_bad_frequency_no_partial_commit()
    test_parity_publish_matches_canonical()
    print("ok")
