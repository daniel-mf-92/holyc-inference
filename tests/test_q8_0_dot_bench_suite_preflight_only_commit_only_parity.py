#!/usr/bin/env python3
"""Parity harness for IQ-1178 ...PreflightOnlyCommitOnlyParity."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from test_q8_0_dot_bench_suite_preflight_only_commit_only import (
    Q8_0_DOT_BENCH_ERR_BAD_PARAM,
    Q8_0_DOT_BENCH_ERR_NULL_PTR,
    Q8_0_DOT_BENCH_OK,
    q8_0_dot_bench_run_default_suite_preflight_only,
    q8_0_dot_bench_run_default_suite_preflight_only_commit_only,
    q8_0_dot_bench_try_mul_i64,
)


def q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity(
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

    if (
        out_shape_count is out_max_blocks
        or out_shape_count is out_max_rows
        or out_shape_count is out_max_row_matrix_blocks
        or out_max_blocks is out_max_rows
        or out_max_blocks is out_max_row_matrix_blocks
        or out_max_rows is out_max_row_matrix_blocks
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    snapshot_cpu_hz = cpu_hz
    snapshot_out_shape_count = out_shape_count
    snapshot_out_max_blocks = out_max_blocks
    snapshot_out_max_rows = out_max_rows
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks

    parity_shape_count = [-1]
    parity_max_blocks = [-1]
    parity_max_rows = [-1]
    parity_max_row_matrix_blocks = [-1]

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only(
        cpu_hz,
        parity_shape_count,
        parity_max_blocks,
        parity_max_rows,
        parity_max_row_matrix_blocks,
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    canonical_shape_count = [-1]
    canonical_max_blocks = [-1]
    canonical_max_rows = [-1]
    canonical_max_row_matrix_blocks = [-1]

    status = q8_0_dot_bench_run_default_suite_preflight_only(
        cpu_hz,
        canonical_shape_count,
        canonical_max_blocks,
        canonical_max_rows,
        canonical_max_row_matrix_blocks,
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    derived_shape_count = 4
    derived_max_blocks = 32
    derived_max_rows = 8
    status, derived_max_row_matrix_blocks = q8_0_dot_bench_try_mul_i64(
        derived_max_rows, derived_max_blocks
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    if snapshot_cpu_hz != cpu_hz:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM
    if (
        snapshot_out_shape_count is not out_shape_count
        or snapshot_out_max_blocks is not out_max_blocks
        or snapshot_out_max_rows is not out_max_rows
        or snapshot_out_max_row_matrix_blocks is not out_max_row_matrix_blocks
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    if (
        parity_shape_count[0] != canonical_shape_count[0]
        or parity_max_blocks[0] != canonical_max_blocks[0]
        or parity_max_rows[0] != canonical_max_rows[0]
        or parity_max_row_matrix_blocks[0] != canonical_max_row_matrix_blocks[0]
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    if (
        canonical_shape_count[0] != derived_shape_count
        or canonical_max_blocks[0] != derived_max_blocks
        or canonical_max_rows[0] != derived_max_rows
        or canonical_max_row_matrix_blocks[0] != derived_max_row_matrix_blocks
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_shape_count[0] = parity_shape_count[0]
    out_max_blocks[0] = parity_max_blocks[0]
    out_max_rows[0] = parity_max_rows[0]
    out_max_row_matrix_blocks[0] = parity_max_row_matrix_blocks[0]
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1178_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParity("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnly(" in body
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnly(" in body
    assert "derived_shape_count = 4;" in body
    assert "derived_max_blocks = 32;" in body
    assert "derived_max_rows = 8;" in body
    assert "status = Q8_0DotBenchTryMulI64(derived_max_rows," in body
    assert "if (snapshot_cpu_hz != cpu_hz)" in body
    assert "if (snapshot_out_shape_count != out_shape_count ||" in body
    assert "if (parity_shape_count != canonical_shape_count ||" in body
    assert "if (canonical_shape_count != derived_shape_count ||" in body
    assert "*out_shape_count = parity_shape_count;" in body
    assert "*out_max_blocks = parity_max_blocks;" in body
    assert "*out_max_rows = parity_max_rows;" in body
    assert "*out_max_row_matrix_blocks = parity_max_row_matrix_blocks;" in body


def test_parity_null_alias_and_error_no_publish() -> None:
    assert (
        q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity(
            3_200_000_000,
            None,
            [1],
            [2],
            [3],
        )
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )

    shared = [77]
    assert (
        q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity(
            3_200_000_000,
            shared,
            shared,
            [2],
            [3],
        )
        == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    )

    out_shape = [111]
    out_blocks = [222]
    out_rows = [333]
    out_row_matrix = [444]

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity(
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


def test_parity_happy_path_and_composition_match() -> None:
    out_shape = [-1]
    out_blocks = [-1]
    out_rows = [-1]
    out_row_matrix = [-1]

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity(
        cpu_hz=3_200_000_000,
        out_shape_count=out_shape,
        out_max_blocks=out_blocks,
        out_max_rows=out_rows,
        out_max_row_matrix_blocks=out_row_matrix,
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert (out_shape[0], out_blocks[0], out_rows[0], out_row_matrix[0]) == (
        4,
        32,
        8,
        256,
    )

    for cpu_hz in [1, 1_800_000_000, 2_400_000_000, 3_200_000_000, 5_000_000_000]:
        got_shape = [-9]
        got_blocks = [-9]
        got_rows = [-9]
        got_row_matrix = [-9]
        want_shape = [-8]
        want_blocks = [-8]
        want_rows = [-8]
        want_row_matrix = [-8]

        got_status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity(
            cpu_hz,
            got_shape,
            got_blocks,
            got_rows,
            got_row_matrix,
        )

        parity_status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only(
            cpu_hz,
            want_shape,
            want_blocks,
            want_rows,
            want_row_matrix,
        )
        assert got_status == parity_status == Q8_0_DOT_BENCH_OK
        assert (got_shape[0], got_blocks[0], got_rows[0], got_row_matrix[0]) == (
            want_shape[0],
            want_blocks[0],
            want_rows[0],
            want_row_matrix[0],
        )


if __name__ == "__main__":
    test_source_contains_iq1178_function_and_guards()
    test_parity_null_alias_and_error_no_publish()
    test_parity_happy_path_and_composition_match()
    print("q8_0_dot_bench_suite_preflight_only_commit_only_parity_reference_checks=ok")
