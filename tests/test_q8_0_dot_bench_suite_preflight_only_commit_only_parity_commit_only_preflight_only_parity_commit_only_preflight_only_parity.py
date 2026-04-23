#!/usr/bin/env python3
"""Parity harness for IQ-1207 ...PreflightOnly...PreflightOnlyParity."""

from __future__ import annotations

from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from test_q8_0_dot_bench_suite_preflight_only_commit_only import (
    Q8_0_DOT_BENCH_ERR_BAD_PARAM,
    Q8_0_DOT_BENCH_ERR_NULL_PTR,
    Q8_0_DOT_BENCH_OK,
)
from test_q8_0_dot_bench_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only import (
    q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_q8_0_dot_bench_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    snapshot_cpu_hz = cpu_hz
    snapshot_out_shape_count = out_shape_count
    snapshot_out_max_blocks = out_max_blocks
    snapshot_out_max_rows = out_max_rows
    snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks

    staged_shape_count = [-1]
    staged_max_blocks = [-1]
    staged_max_rows = [-1]
    staged_max_row_matrix_blocks = [-1]

    parity_shape_count = [-2]
    parity_max_blocks = [-2]
    parity_max_rows = [-2]
    parity_max_row_matrix_blocks = [-2]

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        cpu_hz,
        staged_shape_count,
        staged_max_blocks,
        staged_max_rows,
        staged_max_row_matrix_blocks,
    )
    if status != Q8_0_DOT_BENCH_OK:
        return status

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only(
        cpu_hz,
        parity_shape_count,
        parity_max_blocks,
        parity_max_rows,
        parity_max_row_matrix_blocks,
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
        staged_shape_count[0] != parity_shape_count[0]
        or staged_max_blocks[0] != parity_max_blocks[0]
        or staged_max_rows[0] != parity_max_rows[0]
        or staged_max_row_matrix_blocks[0] != parity_max_row_matrix_blocks[0]
    ):
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_shape_count[0] = staged_shape_count[0]
    out_max_blocks[0] = staged_max_blocks[0]
    out_max_rows[0] = staged_max_rows[0]
    out_max_row_matrix_blocks[0] = staged_max_row_matrix_blocks[0]
    return Q8_0_DOT_BENCH_OK


def explicit_checked_composition(
    cpu_hz: int,
    out_shape_count: list[int] | None,
    out_max_blocks: list[int] | None,
    out_max_rows: list[int] | None,
    out_max_row_matrix_blocks: list[int] | None,
) -> int:
    return q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        cpu_hz,
        out_shape_count,
        out_max_blocks,
        out_max_rows,
        out_max_row_matrix_blocks,
    )


def test_source_contains_iq1207_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "snapshot_cpu_hz = cpu_hz;" in body
    assert "snapshot_out_shape_count = out_shape_count;" in body
    assert "snapshot_out_max_blocks = out_max_blocks;" in body
    assert "snapshot_out_max_rows = out_max_rows;" in body
    assert "snapshot_out_max_row_matrix_blocks = out_max_row_matrix_blocks;" in body
    assert "if (snapshot_cpu_hz != cpu_hz)" in body
    assert "if (snapshot_out_shape_count != out_shape_count ||" in body
    assert "if (staged_shape_count != parity_shape_count ||" in body
    assert "*out_shape_count = staged_shape_count;" in body
    assert "*out_max_blocks = staged_max_blocks;" in body
    assert "*out_max_rows = staged_max_rows;" in body
    assert "*out_max_row_matrix_blocks = staged_max_row_matrix_blocks;" in body


def test_preflight_only_parity_null_and_error_no_publish() -> None:
    assert (
        q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            3_200_000_000,
            None,
            [1],
            [2],
            [3],
        )
        == Q8_0_DOT_BENCH_ERR_NULL_PTR
    )

    out_shape = [111]
    out_blocks = [222]
    out_rows = [333]
    out_row_matrix = [444]

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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


def test_preflight_only_parity_happy_path() -> None:
    out_shape = [-1]
    out_blocks = [-1]
    out_rows = [-1]
    out_row_matrix = [-1]

    status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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


def test_preflight_only_parity_randomized_parity() -> None:
    rng = random.Random(20260423_1207)

    for _ in range(3000):
        cpu_hz = rng.randint(0, 6_000_000_000)
        if rng.random() < 0.1:
            cpu_hz = -rng.randint(1, 1_000_000)

        got_shape = [0x11]
        got_blocks = [0x22]
        got_rows = [0x33]
        got_row_matrix = [0x44]

        want_shape = [0x55]
        want_blocks = [0x66]
        want_rows = [0x77]
        want_row_matrix = [0x88]

        got_status = explicit_checked_composition(
            cpu_hz,
            got_shape,
            got_blocks,
            got_rows,
            got_row_matrix,
        )
        want_status = q8_0_dot_bench_run_default_suite_preflight_only_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            cpu_hz,
            want_shape,
            want_blocks,
            want_rows,
            want_row_matrix,
        )

        assert got_status == want_status
        if got_status == Q8_0_DOT_BENCH_OK:
            assert (got_shape[0], got_blocks[0], got_rows[0], got_row_matrix[0]) == (
                want_shape[0],
                want_blocks[0],
                want_rows[0],
                want_row_matrix[0],
            )
        else:
            assert (got_shape[0], got_blocks[0], got_rows[0], got_row_matrix[0]) == (
                0x11,
                0x22,
                0x33,
                0x44,
            )


if __name__ == "__main__":
    test_source_contains_iq1207_function_and_guards()
    test_preflight_only_parity_null_and_error_no_publish()
    test_preflight_only_parity_happy_path()
    test_preflight_only_parity_randomized_parity()
    print("ok")
