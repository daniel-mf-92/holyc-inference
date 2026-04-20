#!/usr/bin/env python3
"""Parity harness for Q4_0DotBenchRunDefaultSuiteNoPartialPreflightOnly (IQ-692)."""

from __future__ import annotations

from pathlib import Path
import random

Q4_0_DOT_BENCH_OK = 0
Q4_0_DOT_BENCH_ERR_NULL_PTR = 1
Q4_0_DOT_BENCH_ERR_BAD_PARAM = 2
Q4_0_DOT_BENCH_ERR_OVERFLOW = 3

Q4_0_DOT_BENCH_MAX_BLOCKS = 64
Q4_0_DOT_BENCH_MAX_ROWS = 8
I64_MAX = (1 << 63) - 1


def _try_mul_i64(lhs: int, rhs: int) -> tuple[int, int | None]:
    value = lhs * rhs
    if lhs != 0 and value // lhs != rhs:
        return Q4_0_DOT_BENCH_ERR_OVERFLOW, None
    if value < -I64_MAX - 1 or value > I64_MAX:
        return Q4_0_DOT_BENCH_ERR_OVERFLOW, None
    return Q4_0_DOT_BENCH_OK, value


def q4_0_dot_bench_preflight_only_reference(
    *,
    cpu_hz: int,
    out_shape_count: object | None,
    out_required_max_blocks: object | None,
    out_required_max_rows: object | None,
    out_required_row_matrix_blocks: object | None,
) -> tuple[int, dict[str, int] | None]:
    if (
        out_shape_count is None
        or out_required_max_blocks is None
        or out_required_max_rows is None
        or out_required_row_matrix_blocks is None
    ):
        return Q4_0_DOT_BENCH_ERR_NULL_PTR, None

    if cpu_hz <= 0:
        return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None

    shapes = [
        {"block_count": 4, "row_count": 1, "row_stride_blocks": 4, "iters": 8192},
        {"block_count": 8, "row_count": 2, "row_stride_blocks": 8, "iters": 4096},
        {"block_count": 16, "row_count": 4, "row_stride_blocks": 16, "iters": 2048},
        {"block_count": 32, "row_count": 8, "row_stride_blocks": 32, "iters": 1024},
    ]

    required_max_blocks = 0
    required_max_rows = 0
    required_row_matrix_blocks = 0

    for shape in shapes:
        if (
            shape["block_count"] <= 0
            or shape["row_count"] <= 0
            or shape["row_stride_blocks"] <= 0
            or shape["iters"] <= 0
        ):
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None

        if shape["row_stride_blocks"] < shape["block_count"]:
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None

        if shape["block_count"] > Q4_0_DOT_BENCH_MAX_BLOCKS:
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None
        if shape["row_count"] > Q4_0_DOT_BENCH_MAX_ROWS:
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None

        status, row_matrix_blocks = _try_mul_i64(
            shape["row_count"], shape["row_stride_blocks"]
        )
        if status != Q4_0_DOT_BENCH_OK:
            return status, None
        assert row_matrix_blocks is not None

        if row_matrix_blocks > (Q4_0_DOT_BENCH_MAX_BLOCKS * Q4_0_DOT_BENCH_MAX_ROWS):
            return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None

        required_max_blocks = max(required_max_blocks, shape["block_count"])
        required_max_rows = max(required_max_rows, shape["row_count"])
        required_row_matrix_blocks = max(required_row_matrix_blocks, row_matrix_blocks)

    if (
        required_max_blocks <= 0
        or required_max_rows <= 0
        or required_row_matrix_blocks <= 0
    ):
        return Q4_0_DOT_BENCH_ERR_BAD_PARAM, None

    return Q4_0_DOT_BENCH_OK, {
        "shape_count": len(shapes),
        "required_max_blocks": required_max_blocks,
        "required_max_rows": required_max_rows,
        "required_row_matrix_blocks": required_row_matrix_blocks,
    }


def test_source_contains_q4_0_dot_bench_preflight_only() -> None:
    source = Path("src/bench/q4_0_dot_bench.HC").read_text(encoding="utf-8")
    assert "I32 Q4_0DotBenchRunDefaultSuiteNoPartialPreflightOnly(" in source
    assert "Q4_0DotBenchTryMulI64(shapes[shape_index].row_count," in source
    assert "*out_required_row_matrix_blocks = required_row_matrix_blocks;" in source


def test_preflight_only_expected_diagnostics() -> None:
    status, diagnostics = q4_0_dot_bench_preflight_only_reference(
        cpu_hz=3_200_000_000,
        out_shape_count=object(),
        out_required_max_blocks=object(),
        out_required_max_rows=object(),
        out_required_row_matrix_blocks=object(),
    )

    assert status == Q4_0_DOT_BENCH_OK
    assert diagnostics == {
        "shape_count": 4,
        "required_max_blocks": 32,
        "required_max_rows": 8,
        "required_row_matrix_blocks": 256,
    }


def test_preflight_only_null_and_bad_param_errors() -> None:
    status, diagnostics = q4_0_dot_bench_preflight_only_reference(
        cpu_hz=3_200_000_000,
        out_shape_count=None,
        out_required_max_blocks=object(),
        out_required_max_rows=object(),
        out_required_row_matrix_blocks=object(),
    )
    assert status == Q4_0_DOT_BENCH_ERR_NULL_PTR
    assert diagnostics is None

    status, diagnostics = q4_0_dot_bench_preflight_only_reference(
        cpu_hz=0,
        out_shape_count=object(),
        out_required_max_blocks=object(),
        out_required_max_rows=object(),
        out_required_row_matrix_blocks=object(),
    )
    assert status == Q4_0_DOT_BENCH_ERR_BAD_PARAM
    assert diagnostics is None


def test_preflight_only_randomized_cpu_hz_success() -> None:
    rng = random.Random(20260420_692)
    for _ in range(300):
        cpu_hz = rng.randint(1, 5_000_000_000)
        status, diagnostics = q4_0_dot_bench_preflight_only_reference(
            cpu_hz=cpu_hz,
            out_shape_count=object(),
            out_required_max_blocks=object(),
            out_required_max_rows=object(),
            out_required_row_matrix_blocks=object(),
        )
        assert status == Q4_0_DOT_BENCH_OK
        assert diagnostics is not None
        assert diagnostics["shape_count"] == 4
        assert diagnostics["required_max_blocks"] == 32
        assert diagnostics["required_max_rows"] == 8
        assert diagnostics["required_row_matrix_blocks"] == 256


if __name__ == "__main__":
    test_source_contains_q4_0_dot_bench_preflight_only()
    test_preflight_only_expected_diagnostics()
    test_preflight_only_null_and_bad_param_errors()
    test_preflight_only_randomized_cpu_hz_success()
    print("q4_0_dot_bench_suite_preflight_only_reference_checks=ok")
