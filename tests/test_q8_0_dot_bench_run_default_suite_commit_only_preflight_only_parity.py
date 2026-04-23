#!/usr/bin/env python3
"""Spec/parity harness for IQ-1200 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParity."""

from __future__ import annotations

from pathlib import Path

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2
Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO = 3
Q8_0_DOT_BENCH_ERR_OVERFLOW = 4
Q8_0_DOT_BENCH_I64_MAX = 0x7FFFFFFFFFFFFFFF
Q8_0_DOT_BENCH_VALUES_PER_BLOCK = 32
Q8_0_DOT_BENCH_NS_PER_SECOND = 1_000_000_000


def q8_0_dot_bench_try_add_i64(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > Q8_0_DOT_BENCH_I64_MAX - rhs:
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs + rhs


def q8_0_dot_bench_try_mul_i64(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return Q8_0_DOT_BENCH_OK, 0
    if lhs < 0 or rhs < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, 0
    if lhs > Q8_0_DOT_BENCH_I64_MAX // rhs:
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs * rhs


def q8_0_dot_bench_cycles_per_op_checked(total_cycles: int, total_ops: int) -> tuple[int, int, int]:
    if total_cycles < 0 or total_ops < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, 0, 0
    if total_ops == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO, 0, 0
    status, rounded_numer = q8_0_dot_bench_try_add_i64(total_cycles, total_ops >> 1)
    if status != Q8_0_DOT_BENCH_OK:
        return status, 0, 0
    return Q8_0_DOT_BENCH_OK, rounded_numer // total_ops, total_cycles % total_ops


def q8_0_dot_bench_run_default_suite(cpu_hz: int) -> tuple[int, tuple[int, int, int, int, int]]:
    if cpu_hz <= 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, (0, 0, 0, 0, 0)

    shapes = [
        {"block_count": 4, "row_count": 1, "row_stride_blocks": 4, "iters": 8192},
        {"block_count": 8, "row_count": 2, "row_stride_blocks": 8, "iters": 4096},
        {"block_count": 16, "row_count": 4, "row_stride_blocks": 16, "iters": 2048},
        {"block_count": 32, "row_count": 8, "row_stride_blocks": 32, "iters": 1024},
    ]

    lhs_vec = [(idx % 17) - 8 for idx in range(Q8_0_DOT_BENCH_VALUES_PER_BLOCK)]
    rhs_vec = [((idx * 5 + 3) % 23) - 11 for idx in range(Q8_0_DOT_BENCH_VALUES_PER_BLOCK)]

    total_ops = 0
    total_cycles = 0
    suite_checksum = 0

    for shape_idx, shape in enumerate(shapes):
        status, dot_calls = q8_0_dot_bench_try_mul_i64(shape["row_count"], shape["iters"])
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, block_mac_ops = q8_0_dot_bench_try_mul_i64(
            shape["block_count"], Q8_0_DOT_BENCH_VALUES_PER_BLOCK
        )
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, shape_ops = q8_0_dot_bench_try_mul_i64(dot_calls, block_mac_ops)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

        status, block_term = q8_0_dot_bench_try_mul_i64(shape["block_count"], 24)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, row_term = q8_0_dot_bench_try_mul_i64(shape["row_count"], 7)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

        status, ns_per_dot = q8_0_dot_bench_try_add_i64(600, block_term)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, ns_per_dot = q8_0_dot_bench_try_add_i64(ns_per_dot, row_term)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

        status, shape_ns = q8_0_dot_bench_try_mul_i64(dot_calls, ns_per_dot)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, numer_ns_hz = q8_0_dot_bench_try_mul_i64(shape_ns, cpu_hz)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

        status, rounded_numer_ns_hz = q8_0_dot_bench_try_add_i64(
            numer_ns_hz, Q8_0_DOT_BENCH_NS_PER_SECOND >> 1
        )
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        shape_cycles = rounded_numer_ns_hz // Q8_0_DOT_BENCH_NS_PER_SECOND

        vector_dot_q0 = 0
        for vec_idx in range(Q8_0_DOT_BENCH_VALUES_PER_BLOCK):
            vector_dot_q0 += lhs_vec[vec_idx] * rhs_vec[(vec_idx + shape_idx) & 31]

        vector_dot_term = vector_dot_q0 * (shape_idx + 1)
        block_term_sig = shape["block_count"] * 257
        row_term_sig = shape["row_count"] * 17
        shape_signature = vector_dot_term + block_term_sig + row_term_sig + shape["iters"]

        status, shape_checksum = q8_0_dot_bench_try_mul_i64(shape_signature, dot_calls)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

        status, total_ops = q8_0_dot_bench_try_add_i64(total_ops, shape_ops)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, total_cycles = q8_0_dot_bench_try_add_i64(total_cycles, shape_cycles)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        status, suite_checksum = q8_0_dot_bench_try_add_i64(suite_checksum, shape_checksum)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

    status, cycles_per_op, remainder_cycles = q8_0_dot_bench_cycles_per_op_checked(
        total_cycles, total_ops
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status, (0, 0, 0, 0, 0)

    return (
        Q8_0_DOT_BENCH_OK,
        (total_ops, total_cycles, cycles_per_op, remainder_cycles, suite_checksum),
    )


def q8_0_dot_bench_run_default_suite_commit_only_preflight_only_parity(
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
    if cpu_hz <= 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    snapshot_cpu_hz = cpu_hz

    status_a, tuple_a = q8_0_dot_bench_run_default_suite(cpu_hz)
    if status_a != Q8_0_DOT_BENCH_OK:
        return status_a

    status_b, tuple_b = q8_0_dot_bench_run_default_suite(cpu_hz)
    if status_b != Q8_0_DOT_BENCH_OK:
        return status_b

    if cpu_hz != snapshot_cpu_hz:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    if tuple_a != tuple_b:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_total_ops[0], out_total_cycles[0], out_cycles_per_op[0], out_remainder_cycles[0], out_suite_checksum[0] = tuple_a
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1200_function_and_parity_checks() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "snapshot_cpu_hz = cpu_hz;" in body
    assert "Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnly(" in body
    assert "Q8_0DotBenchRunDefaultSuite(" in body
    assert "if (snapshot_cpu_hz != cpu_hz)" in body
    assert "if (staged_total_ops != canonical_total_ops ||" in body
    assert "*out_total_ops = staged_total_ops;" in body
    assert "*out_total_cycles = staged_total_cycles;" in body
    assert "*out_cycles_per_op = staged_cycles_per_op;" in body
    assert "*out_remainder_cycles = staged_remainder_cycles;" in body
    assert "*out_suite_checksum = staged_suite_checksum;" in body


def test_null_and_bad_frequency_keep_outputs_unchanged() -> None:
    out_ops = [11]
    out_cycles = [22]
    out_cpo = [33]
    out_rem = [44]
    out_checksum = [55]

    status = q8_0_dot_bench_run_default_suite_commit_only_preflight_only_parity(
        cpu_hz=0,
        out_total_ops=out_ops,
        out_total_cycles=out_cycles,
        out_cycles_per_op=out_cpo,
        out_remainder_cycles=out_rem,
        out_suite_checksum=out_checksum,
    )
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert (out_ops[0], out_cycles[0], out_cpo[0], out_rem[0], out_checksum[0]) == (
        11,
        22,
        33,
        44,
        55,
    )


def test_parity_happy_path_tuple_matches_canonical() -> None:
    out_ops = [-1]
    out_cycles = [-1]
    out_cpo = [-1]
    out_rem = [-1]
    out_checksum = [-1]

    status = q8_0_dot_bench_run_default_suite_commit_only_preflight_only_parity(
        cpu_hz=3_200_000_000,
        out_total_ops=out_ops,
        out_total_cycles=out_cycles,
        out_cycles_per_op=out_cpo,
        out_remainder_cycles=out_rem,
        out_suite_checksum=out_checksum,
    )
    assert status == Q8_0_DOT_BENCH_OK

    status_ref, ref_tuple = q8_0_dot_bench_run_default_suite(3_200_000_000)
    assert status_ref == Q8_0_DOT_BENCH_OK
    assert (out_ops[0], out_cycles[0], out_cpo[0], out_rem[0], out_checksum[0]) == ref_tuple


if __name__ == "__main__":
    test_source_contains_iq1200_function_and_parity_checks()
    test_null_and_bad_frequency_keep_outputs_unchanged()
    test_parity_happy_path_tuple_matches_canonical()
    print("ok")
