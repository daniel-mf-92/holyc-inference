#!/usr/bin/env python3
"""Parity harness for IQ-1185 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnly."""

from __future__ import annotations

from pathlib import Path
import random

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2
Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO = 3
Q8_0_DOT_BENCH_ERR_OVERFLOW = 4
Q8_0_DOT_BENCH_I64_MAX = 0x7FFFFFFFFFFFFFFF
Q8_0_DOT_BENCH_NS_PER_SECOND = 1_000_000_000
Q8_0_DOT_BENCH_VALUES_PER_BLOCK = 32


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
    status, rounded = q8_0_dot_bench_try_add_i64(total_cycles, total_ops >> 1)
    if status != Q8_0_DOT_BENCH_OK:
        return status, 0, 0
    return Q8_0_DOT_BENCH_OK, rounded // total_ops, total_cycles % total_ops


def q8_0_dot_bench_run_default_suite_tuple(cpu_hz: int) -> tuple[int, tuple[int, int, int, int, int]]:
    if cpu_hz <= 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, (0, 0, 0, 0, 0)

    shapes = [
        {"block_count": 4, "row_count": 1, "row_stride_blocks": 4, "iters": 8192},
        {"block_count": 8, "row_count": 2, "row_stride_blocks": 8, "iters": 4096},
        {"block_count": 16, "row_count": 4, "row_stride_blocks": 16, "iters": 2048},
        {"block_count": 32, "row_count": 8, "row_stride_blocks": 32, "iters": 1024},
    ]

    lhs = [(i % 17) - 8 for i in range(Q8_0_DOT_BENCH_VALUES_PER_BLOCK)]
    rhs = [((i * 5 + 3) % 23) - 11 for i in range(Q8_0_DOT_BENCH_VALUES_PER_BLOCK)]

    total_ops = 0
    total_cycles = 0
    suite_checksum = 0

    for shape_idx, shape in enumerate(shapes):
        if shape["row_stride_blocks"] < shape["block_count"]:
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM, (0, 0, 0, 0, 0)

        status, dot_calls = q8_0_dot_bench_try_mul_i64(shape["row_count"], shape["iters"])
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)

        status, block_mac_ops = q8_0_dot_bench_try_mul_i64(shape["block_count"], Q8_0_DOT_BENCH_VALUES_PER_BLOCK)
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
        status, rounded_numer = q8_0_dot_bench_try_add_i64(numer_ns_hz, Q8_0_DOT_BENCH_NS_PER_SECOND >> 1)
        if status != Q8_0_DOT_BENCH_OK:
            return status, (0, 0, 0, 0, 0)
        shape_cycles = rounded_numer // Q8_0_DOT_BENCH_NS_PER_SECOND

        vector_dot_q0 = sum(lhs[i] * rhs[(i + shape_idx) & 31] for i in range(Q8_0_DOT_BENCH_VALUES_PER_BLOCK))
        vector_dot_term = vector_dot_q0 * (shape_idx + 1)
        block_sig = shape["block_count"] * 257
        row_sig = shape["row_count"] * 17
        shape_signature = vector_dot_term + block_sig + row_sig + shape["iters"]

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

    status, cycles_per_op, remainder_cycles = q8_0_dot_bench_cycles_per_op_checked(total_cycles, total_ops)
    if status != Q8_0_DOT_BENCH_OK:
        return status, (0, 0, 0, 0, 0)

    return Q8_0_DOT_BENCH_OK, (total_ops, total_cycles, cycles_per_op, remainder_cycles, suite_checksum)


def q8_0_dot_bench_run_default_suite_commit_only_preflight_only(
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
    snapshot_out_total_ops = out_total_ops
    snapshot_out_total_cycles = out_total_cycles
    snapshot_out_cycles_per_op = out_cycles_per_op
    snapshot_out_remainder_cycles = out_remainder_cycles
    snapshot_out_suite_checksum = out_suite_checksum

    staged_status, staged_tuple = q8_0_dot_bench_run_default_suite_tuple(cpu_hz)
    if staged_status != Q8_0_DOT_BENCH_OK:
        return staged_status

    if snapshot_cpu_hz != cpu_hz:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if (
        snapshot_out_total_ops is not out_total_ops
        or snapshot_out_total_cycles is not out_total_cycles
        or snapshot_out_cycles_per_op is not out_cycles_per_op
        or snapshot_out_remainder_cycles is not out_remainder_cycles
        or snapshot_out_suite_checksum is not out_suite_checksum
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    canonical_status, canonical_tuple = q8_0_dot_bench_run_default_suite_tuple(snapshot_cpu_hz)
    if canonical_status != Q8_0_DOT_BENCH_OK:
        return canonical_status

    if staged_tuple != canonical_tuple:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_total_ops[0] = staged_tuple[0]
    out_total_cycles[0] = staged_tuple[1]
    out_cycles_per_op[0] = staged_tuple[2]
    out_remainder_cycles[0] = staged_tuple[3]
    out_suite_checksum[0] = staged_tuple[4]
    return Q8_0_DOT_BENCH_OK


def explicit_checked_composition(
    cpu_hz: int,
    out_total_ops: list[int] | None,
    out_total_cycles: list[int] | None,
    out_cycles_per_op: list[int] | None,
    out_remainder_cycles: list[int] | None,
    out_suite_checksum: list[int] | None,
) -> int:
    return q8_0_dot_bench_run_default_suite_commit_only_preflight_only(
        cpu_hz,
        out_total_ops,
        out_total_cycles,
        out_cycles_per_op,
        out_remainder_cycles,
        out_suite_checksum,
    )


def test_source_contains_iq1185_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnly("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "status = Q8_0DotBenchRunDefaultSuite(cpu_hz," in body
    assert "snapshot_cpu_hz = cpu_hz;" in body
    assert "snapshot_out_total_ops = out_total_ops;" in body
    assert "snapshot_out_total_cycles = out_total_cycles;" in body
    assert "snapshot_out_cycles_per_op = out_cycles_per_op;" in body
    assert "snapshot_out_remainder_cycles = out_remainder_cycles;" in body
    assert "snapshot_out_suite_checksum = out_suite_checksum;" in body
    assert "if (snapshot_cpu_hz != cpu_hz)" in body
    assert "canonical_shapes[3].block_count = 32;" in body
    assert "if (staged_total_ops != canonical_total_ops ||" in body
    assert "*out_total_ops = staged_total_ops;" in body
    assert "*out_total_cycles = staged_total_cycles;" in body
    assert "*out_cycles_per_op = staged_cycles_per_op;" in body
    assert "*out_remainder_cycles = staged_remainder_cycles;" in body
    assert "*out_suite_checksum = staged_suite_checksum;" in body


def test_null_bad_frequency_and_no_partial_on_error() -> None:
    assert (
        q8_0_dot_bench_run_default_suite_commit_only_preflight_only(
            3_200_000_000,
            None,
            [1],
            [2],
            [3],
            [4],
        )
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )

    out_total_ops = [111]
    out_total_cycles = [222]
    out_cycles_per_op = [333]
    out_remainder_cycles = [444]
    out_suite_checksum = [555]

    status = q8_0_dot_bench_run_default_suite_commit_only_preflight_only(
        cpu_hz=0,
        out_total_ops=out_total_ops,
        out_total_cycles=out_total_cycles,
        out_cycles_per_op=out_cycles_per_op,
        out_remainder_cycles=out_remainder_cycles,
        out_suite_checksum=out_suite_checksum,
    )
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert (out_total_ops[0], out_total_cycles[0], out_cycles_per_op[0], out_remainder_cycles[0], out_suite_checksum[0]) == (
        111,
        222,
        333,
        444,
        555,
    )


def test_output_alias_vector_is_deterministic() -> None:
    aliased = [-1]

    status = q8_0_dot_bench_run_default_suite_commit_only_preflight_only(
        cpu_hz=3_200_000_000,
        out_total_ops=aliased,
        out_total_cycles=aliased,
        out_cycles_per_op=aliased,
        out_remainder_cycles=aliased,
        out_suite_checksum=aliased,
    )
    assert status == Q8_0_DOT_BENCH_OK

    _, canonical_tuple = q8_0_dot_bench_run_default_suite_tuple(3_200_000_000)
    assert aliased[0] == canonical_tuple[4]


def test_randomized_parity_vectors() -> None:
    rng = random.Random(20260423_1185)

    for _ in range(4000):
        cpu_hz = rng.randint(0, 6_000_000_000)
        if rng.random() < 0.1:
            cpu_hz = -rng.randint(1, 1_000_000)

        got_total_ops = [0x11]
        got_total_cycles = [0x22]
        got_cycles_per_op = [0x33]
        got_remainder_cycles = [0x44]
        got_suite_checksum = [0x55]

        want_total_ops = [0x66]
        want_total_cycles = [0x77]
        want_cycles_per_op = [0x88]
        want_remainder_cycles = [0x99]
        want_suite_checksum = [0xAA]

        got_status = explicit_checked_composition(
            cpu_hz,
            got_total_ops,
            got_total_cycles,
            got_cycles_per_op,
            got_remainder_cycles,
            got_suite_checksum,
        )

        want_status, want_tuple = q8_0_dot_bench_run_default_suite_tuple(cpu_hz)

        if want_status != Q8_0_DOT_BENCH_OK:
            want_status = Q8_0_DOT_BENCH_ERR_BAD_PARAM if cpu_hz <= 0 else want_status

        assert got_status == want_status

        if got_status == Q8_0_DOT_BENCH_OK:
            want_total_ops[0] = want_tuple[0]
            want_total_cycles[0] = want_tuple[1]
            want_cycles_per_op[0] = want_tuple[2]
            want_remainder_cycles[0] = want_tuple[3]
            want_suite_checksum[0] = want_tuple[4]
            assert (
                got_total_ops[0],
                got_total_cycles[0],
                got_cycles_per_op[0],
                got_remainder_cycles[0],
                got_suite_checksum[0],
            ) == (
                want_total_ops[0],
                want_total_cycles[0],
                want_cycles_per_op[0],
                want_remainder_cycles[0],
                want_suite_checksum[0],
            )
        else:
            assert (
                got_total_ops[0],
                got_total_cycles[0],
                got_cycles_per_op[0],
                got_remainder_cycles[0],
                got_suite_checksum[0],
            ) == (0x11, 0x22, 0x33, 0x44, 0x55)


if __name__ == "__main__":
    test_source_contains_iq1185_function_and_guards()
    test_null_bad_frequency_and_no_partial_on_error()
    test_output_alias_vector_is_deterministic()
    test_randomized_parity_vectors()
    print("ok")
