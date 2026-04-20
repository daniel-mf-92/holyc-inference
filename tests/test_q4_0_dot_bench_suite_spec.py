#!/usr/bin/env python3
"""Spec/parity harness for Q4_0DotBenchRunDefaultSuite benchmark contract."""

from __future__ import annotations

from pathlib import Path


def _function_body(source: str, signature: str) -> str:
    start = source.find(signature)
    assert start != -1, f"missing signature: {signature}"

    brace_start = source.find("{", start)
    assert brace_start != -1, f"missing opening brace: {signature}"

    depth = 0
    index = brace_start
    while index < len(source):
        ch = source[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start + 1 : index]
        index += 1

    raise AssertionError(f"missing closing brace: {signature}")


def test_q4_0_dot_bench_file_exists_and_contracts_present() -> None:
    source_path = Path("src/bench/q4_0_dot_bench.HC")
    assert source_path.exists(), "expected HolyC bench source at src/bench/q4_0_dot_bench.HC"
    source = source_path.read_text(encoding="utf-8")

    assert "I32 Q4_0DotBenchRunDefaultSuite(I64 cpu_hz)" in source
    assert "#define Q4_0_DOT_BENCH_REPORT_FIELDS" in source
    assert "cycles_per_dot=%d" in source
    assert "dots_per_sec=%d" in source


def test_q4_0_dot_bench_default_suite_shapes_are_deterministic() -> None:
    source = Path("src/bench/q4_0_dot_bench.HC").read_text(encoding="utf-8")
    body = _function_body(source, "I32 Q4_0DotBenchRunDefaultSuite(I64 cpu_hz)")

    expected_shape_assignments = [
        "shapes[0].block_count = 4;",
        "shapes[0].row_count = 1;",
        "shapes[0].row_stride_blocks = 4;",
        "shapes[0].iters = 8192;",
        "shapes[1].block_count = 8;",
        "shapes[1].row_count = 2;",
        "shapes[1].row_stride_blocks = 8;",
        "shapes[1].iters = 4096;",
        "shapes[2].block_count = 16;",
        "shapes[2].row_count = 4;",
        "shapes[2].row_stride_blocks = 16;",
        "shapes[2].iters = 2048;",
        "shapes[3].block_count = 32;",
        "shapes[3].row_count = 8;",
        "shapes[3].row_stride_blocks = 32;",
        "shapes[3].iters = 1024;",
    ]

    for assignment in expected_shape_assignments:
        assert assignment in body, f"missing deterministic default-suite shape assignment: {assignment}"


def test_q4_0_dot_bench_runs_all_required_kernels_and_tsc_metrics() -> None:
    source = Path("src/bench/q4_0_dot_bench.HC").read_text(encoding="utf-8")
    body = _function_body(
        source,
        "I32 Q4_0DotBenchRunShape(I64 shape_index,",
    )

    assert "Q4_0DotProductQ16(lhs, rhs, shape->block_count, &row_dot_q16);" in body
    assert "Q4_0DotProductQ16NoPartial(lhs, rhs, shape->block_count, &row_dot_q16);" in body
    assert "Q4_0DotRowsQ16MatrixVector(row_matrix," in body

    assert "start_tsc = TSC;" in body
    assert "elapsed_cycles = TSC - start_tsc;" in body
    assert "Q4_0DotBenchComputeDotsPerSec" in body

    assert '"Q4_0DotProductQ16"' in body
    assert '"Q4_0DotProductQ16NoPartial"' in body
    assert '"Q4_0DotRowsQ16MatrixVector"' in body
    assert 'Printf("Q4_0_DOT_BENCH " Q4_0_DOT_BENCH_REPORT_FIELDS "\\n",' in body


def test_q4_0_dot_bench_capacity_bounds_are_explicit() -> None:
    source = Path("src/bench/q4_0_dot_bench.HC").read_text(encoding="utf-8")

    assert "#define Q4_0_DOT_BENCH_MAX_BLOCKS         64" in source
    assert "#define Q4_0_DOT_BENCH_MAX_ROWS           8" in source
    assert "if (shapes[shape_index].block_count > Q4_0_DOT_BENCH_MAX_BLOCKS)" in source
    assert "if (shapes[shape_index].row_count > Q4_0_DOT_BENCH_MAX_ROWS)" in source
    assert "if (row_matrix_blocks > (Q4_0_DOT_BENCH_MAX_BLOCKS * Q4_0_DOT_BENCH_MAX_ROWS))" in source


if __name__ == "__main__":
    test_q4_0_dot_bench_file_exists_and_contracts_present()
    test_q4_0_dot_bench_default_suite_shapes_are_deterministic()
    test_q4_0_dot_bench_runs_all_required_kernels_and_tsc_metrics()
    test_q4_0_dot_bench_capacity_bounds_are_explicit()
    print("ok")
