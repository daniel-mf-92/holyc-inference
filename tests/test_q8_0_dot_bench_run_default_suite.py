#!/usr/bin/env python3
"""Parity harness for IQ-1142 Q8_0DotBenchRunDefaultSuite."""

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


def q8_0_dot_bench_cycles_per_op_checked(
    total_cycles: int,
    total_ops: int,
    out_cycles_per_op: list[int] | None,
    out_remainder_cycles: list[int] | None,
) -> int:
    if out_cycles_per_op is None or out_remainder_cycles is None:
        return Q8_0_DOT_BENCH_ERR_NULL_PTR
    if total_cycles < 0 or total_ops < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if total_ops == 0:
        return Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO

    half_ops = total_ops >> 1
    status, rounded = q8_0_dot_bench_try_add_i64(total_cycles, half_ops)
    if status != Q8_0_DOT_BENCH_OK:
        return status

    out_cycles_per_op[0] = rounded // total_ops
    out_remainder_cycles[0] = total_cycles % total_ops
    return Q8_0_DOT_BENCH_OK


def q8_0_dot_bench_run_default_suite_preflight_only_companion(
    cpu_hz: int,
    out_shape_count: list[int] | None,
    out_max_blocks: list[int] | None,
    out_max_rows: list[int] | None,
    out_max_row_matrix_blocks: list[int] | None,
) -> int:
    if (
        out_shape_count is None
        or out_max_blocks is None
        or out_max_rows is None
        or out_max_row_matrix_blocks is None
    ):
        return Q8_0_DOT_BENCH_ERR_NULL_PTR
    if cpu_hz <= 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_shape_count[0] = 4
    out_max_blocks[0] = 32
    out_max_rows[0] = 8
    out_max_row_matrix_blocks[0] = 256
    return Q8_0_DOT_BENCH_OK


def q8_0_dot_bench_run_default_suite(
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

    shape_count = [0]
    max_blocks = [0]
    max_rows = [0]
    max_row_matrix_blocks = [0]
    status = q8_0_dot_bench_run_default_suite_preflight_only_companion(
        cpu_hz,
        shape_count,
        max_blocks,
        max_rows,
        max_row_matrix_blocks,
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status
    if (
        shape_count[0] != 4
        or max_blocks[0] != 32
        or max_rows[0] != 8
        or max_row_matrix_blocks[0] != 256
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    shapes = [
        {"block_count": 4, "row_count": 1, "row_stride_blocks": 4, "iters": 8192},
        {"block_count": 8, "row_count": 2, "row_stride_blocks": 8, "iters": 4096},
        {"block_count": 16, "row_count": 4, "row_stride_blocks": 16, "iters": 2048},
        {"block_count": 32, "row_count": 8, "row_stride_blocks": 32, "iters": 1024},
    ]

    lhs = [(i % 17) - 8 for i in range(32)]
    rhs = [((i * 5 + 3) % 23) - 11 for i in range(32)]

    total_ops = 0
    total_cycles = 0
    checksum = 0

    for shape_idx, shape in enumerate(shapes):
        if shape["row_stride_blocks"] < shape["block_count"]:
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM

        status, dot_calls = q8_0_dot_bench_try_mul_i64(shape["row_count"], shape["iters"])
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, block_mac = q8_0_dot_bench_try_mul_i64(shape["block_count"], 32)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, shape_ops = q8_0_dot_bench_try_mul_i64(dot_calls, block_mac)
        if status != Q8_0_DOT_BENCH_OK:
            return status

        status, block_term = q8_0_dot_bench_try_mul_i64(shape["block_count"], 24)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, row_term = q8_0_dot_bench_try_mul_i64(shape["row_count"], 7)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, ns_per_dot = q8_0_dot_bench_try_add_i64(600, block_term)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, ns_per_dot = q8_0_dot_bench_try_add_i64(ns_per_dot, row_term)
        if status != Q8_0_DOT_BENCH_OK:
            return status

        status, shape_ns = q8_0_dot_bench_try_mul_i64(dot_calls, ns_per_dot)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, numer_ns_hz = q8_0_dot_bench_try_mul_i64(shape_ns, cpu_hz)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, rounded = q8_0_dot_bench_try_add_i64(
            numer_ns_hz, Q8_0_DOT_BENCH_NS_PER_SECOND >> 1
        )
        if status != Q8_0_DOT_BENCH_OK:
            return status
        shape_cycles = rounded // Q8_0_DOT_BENCH_NS_PER_SECOND

        vector_dot_q0 = sum(lhs[i] * rhs[(i + shape_idx) & 31] for i in range(32))
        vector_dot_term = vector_dot_q0 * (shape_idx + 1)
        block_sig = shape["block_count"] * 257
        row_sig = shape["row_count"] * 17
        shape_signature = vector_dot_term + block_sig + row_sig + shape["iters"]
        status, shape_checksum = q8_0_dot_bench_try_mul_i64(shape_signature, dot_calls)
        if status != Q8_0_DOT_BENCH_OK:
            return status

        status, total_ops = q8_0_dot_bench_try_add_i64(total_ops, shape_ops)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, total_cycles = q8_0_dot_bench_try_add_i64(total_cycles, shape_cycles)
        if status != Q8_0_DOT_BENCH_OK:
            return status
        status, checksum = q8_0_dot_bench_try_add_i64(checksum, shape_checksum)
        if status != Q8_0_DOT_BENCH_OK:
            return status

    status = q8_0_dot_bench_cycles_per_op_checked(
        total_cycles,
        total_ops,
        out_cycles_per_op,
        out_remainder_cycles,
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    out_total_ops[0] = total_ops
    out_total_cycles[0] = total_cycles
    out_suite_checksum[0] = checksum
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1142_function_and_core_math() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuite("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "if (shape_count != 4 || max_blocks != 32 || max_rows != 8 ||" in body
    assert "ns_per_dot = 600 + 24*block_count + 7*row_count" in body
    assert "shape_cycles = rounded_numer_ns_hz / Q8_0_DOT_BENCH_NS_PER_SECOND;" in body
    assert "vector_dot_q0 += lhs_vec[vec_idx] * rhs_vec[(vec_idx + shape_idx) & 31];" in body
    assert "status = Q8_0DotBenchCyclesPerOpChecked(total_cycles_snapshot," in body
    assert "*out_total_ops = total_ops_snapshot;" in body
    assert "*out_total_cycles = total_cycles_snapshot;" in body
    assert "*out_cycles_per_op = staged_cycles_per_op;" in body
    assert "*out_remainder_cycles = staged_remainder_cycles;" in body
    assert "*out_suite_checksum = suite_checksum_snapshot;" in body


def test_null_and_domain_guards_no_partial_publish() -> None:
    out_ops = [111]
    out_cycles = [222]
    out_cpo = [333]
    out_rem = [444]
    out_ck = [555]

    assert (
        q8_0_dot_bench_run_default_suite(
            3_200_000_000, None, out_cycles, out_cpo, out_rem, out_ck
        )
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )
    assert (
        q8_0_dot_bench_run_default_suite(
            0, out_ops, out_cycles, out_cpo, out_rem, out_ck
        )
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )

    assert out_ops[0] == 111
    assert out_cycles[0] == 222
    assert out_cpo[0] == 333
    assert out_rem[0] == 444
    assert out_ck[0] == 555


def test_known_tuple_for_3p2ghz() -> None:
    out_ops = [-1]
    out_cycles = [-1]
    out_cpo = [-1]
    out_rem = [-1]
    out_ck = [-1]

    status = q8_0_dot_bench_run_default_suite(
        3_200_000_000,
        out_ops,
        out_cycles,
        out_cpo,
        out_rem,
        out_ck,
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert (out_ops[0], out_cycles[0], out_cpo[0], out_rem[0], out_ck[0]) == (
        15_728_640,
        103_415_808,
        7,
        9_043_968,
        256_606_208,
    )


def test_randomized_reference_parity() -> None:
    rng = random.Random(20260422_1142)

    for _ in range(2500):
        cpu_hz = rng.randint(1, 7_000_000_000)

        got_ops = [0x11]
        got_cycles = [0x22]
        got_cpo = [0x33]
        got_rem = [0x44]
        got_ck = [0x55]

        status = q8_0_dot_bench_run_default_suite(
            cpu_hz,
            got_ops,
            got_cycles,
            got_cpo,
            got_rem,
            got_ck,
        )

        assert status == Q8_0_DOT_BENCH_OK
        assert got_ops[0] == 15_728_640
        assert got_ck[0] == 256_606_208
        assert got_cycles[0] > 0
        assert got_cpo[0] >= 0
        assert 0 <= got_rem[0] < got_ops[0]


if __name__ == "__main__":
    test_source_contains_iq1142_function_and_core_math()
    test_null_and_domain_guards_no_partial_publish()
    test_known_tuple_for_3p2ghz()
    test_randomized_reference_parity()
    print("ok")
