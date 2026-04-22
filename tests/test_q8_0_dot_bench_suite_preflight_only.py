#!/usr/bin/env python3
"""Spec/parity harness for IQ-1176 Q8_0DotBenchRunDefaultSuitePreflightOnly."""

from __future__ import annotations

from pathlib import Path

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2
Q8_0_DOT_BENCH_ERR_DIV_BY_ZERO = 3
Q8_0_DOT_BENCH_ERR_OVERFLOW = 4
Q8_0_DOT_BENCH_I64_MAX = 0x7FFFFFFFFFFFFFFF
Q8_0_DOT_BENCH_MAX_BLOCKS = 64
Q8_0_DOT_BENCH_MAX_ROWS = 8


def q8_0_dot_bench_try_mul_i64(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return Q8_0_DOT_BENCH_OK, 0
    if lhs < 0 or rhs < 0:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM, 0
    if lhs > Q8_0_DOT_BENCH_I64_MAX // rhs:
        return Q8_0_DOT_BENCH_ERR_OVERFLOW, 0
    return Q8_0_DOT_BENCH_OK, lhs * rhs


def q8_0_dot_bench_run_default_suite_preflight_only(
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

    cpu_hz_snapshot = cpu_hz

    shapes = [
        {"block_count": 4, "row_count": 1, "row_stride_blocks": 4, "iters": 8192},
        {"block_count": 8, "row_count": 2, "row_stride_blocks": 8, "iters": 4096},
        {"block_count": 16, "row_count": 4, "row_stride_blocks": 16, "iters": 2048},
        {"block_count": 32, "row_count": 8, "row_stride_blocks": 32, "iters": 1024},
    ]

    shape_count_snapshot = 4
    max_blocks_snapshot = 0
    max_rows_snapshot = 0
    max_row_matrix_blocks_snapshot = 0

    for shape in shapes:
        if (
            shape["block_count"] <= 0
            or shape["row_count"] <= 0
            or shape["row_stride_blocks"] <= 0
            or shape["iters"] <= 0
        ):
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM

        if shape["row_stride_blocks"] < shape["block_count"]:
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM

        if shape["block_count"] > Q8_0_DOT_BENCH_MAX_BLOCKS:
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM
        if shape["row_count"] > Q8_0_DOT_BENCH_MAX_ROWS:
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM

        status, row_matrix_blocks = q8_0_dot_bench_try_mul_i64(
            shape["row_count"], shape["row_stride_blocks"]
        )
        if status != Q8_0_DOT_BENCH_OK:
            return status

        if row_matrix_blocks > (Q8_0_DOT_BENCH_MAX_BLOCKS * Q8_0_DOT_BENCH_MAX_ROWS):
            return Q8_0_DOT_BENCH_ERR_BAD_PARAM

        max_blocks_snapshot = max(max_blocks_snapshot, shape["block_count"])
        max_rows_snapshot = max(max_rows_snapshot, shape["row_count"])
        max_row_matrix_blocks_snapshot = max(
            max_row_matrix_blocks_snapshot, row_matrix_blocks
        )

    if cpu_hz != cpu_hz_snapshot:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if (
        shape_count_snapshot <= 0
        or max_blocks_snapshot <= 0
        or max_rows_snapshot <= 0
        or max_row_matrix_blocks_snapshot <= 0
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_shape_count[0] = shape_count_snapshot
    out_max_blocks[0] = max_blocks_snapshot
    out_max_rows[0] = max_rows_snapshot
    out_max_row_matrix_blocks[0] = max_row_matrix_blocks_snapshot
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1176_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuitePreflightOnly("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "if (cpu_hz <= 0)" in body
    assert "cpu_hz_snapshot = cpu_hz;" in body
    assert "shapes[3].row_stride_blocks = 32;" in body
    assert "status = Q8_0DotBenchTryMulI64(shapes[shape_index].row_count," in body
    assert "if (cpu_hz != cpu_hz_snapshot)" in body
    assert "*out_shape_count = shape_count_snapshot;" in body
    assert "*out_max_blocks = max_blocks_snapshot;" in body
    assert "*out_max_rows = max_rows_snapshot;" in body
    assert "*out_max_row_matrix_blocks = max_row_matrix_blocks_snapshot;" in body


def test_preflight_only_tuple_and_no_partial_on_error() -> None:
    out_shape = [111]
    out_blocks = [222]
    out_rows = [333]
    out_row_matrix = [444]

    status = q8_0_dot_bench_run_default_suite_preflight_only(
        cpu_hz=0,
        out_shape_count=out_shape,
        out_max_blocks=out_blocks,
        out_max_rows=out_rows,
        out_max_row_matrix_blocks=out_row_matrix,
    )
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert out_shape[0] == 111
    assert out_blocks[0] == 222
    assert out_rows[0] == 333
    assert out_row_matrix[0] == 444


def test_preflight_only_happy_path_outputs() -> None:
    out_shape = [-1]
    out_blocks = [-1]
    out_rows = [-1]
    out_row_matrix = [-1]

    status = q8_0_dot_bench_run_default_suite_preflight_only(
        cpu_hz=3_200_000_000,
        out_shape_count=out_shape,
        out_max_blocks=out_blocks,
        out_max_rows=out_rows,
        out_max_row_matrix_blocks=out_row_matrix,
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert out_shape[0] == 4
    assert out_blocks[0] == 32
    assert out_rows[0] == 8
    assert out_row_matrix[0] == 256


if __name__ == "__main__":
    test_source_contains_iq1176_function_and_guards()
    test_preflight_only_tuple_and_no_partial_on_error()
    test_preflight_only_happy_path_outputs()
    print("ok")
