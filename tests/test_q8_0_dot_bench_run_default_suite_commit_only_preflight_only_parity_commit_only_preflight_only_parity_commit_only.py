#!/usr/bin/env python3
from pathlib import Path

Q8_0_DOT_BENCH_OK = 0
Q8_0_DOT_BENCH_ERR_NULL_PTR = 1
Q8_0_DOT_BENCH_ERR_BAD_PARAM = 2


def canonical_shape_tuple(cpu_hz: int) -> tuple[int, int, int, int] | None:
    if cpu_hz <= 0:
        return None
    return (4, 32, 8, 256)


def parity_commit_only_preflight_only_parity_commit_only(
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

    staged = canonical_shape_tuple(cpu_hz)
    if staged is None:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    parity = canonical_shape_tuple(cpu_hz)
    if parity is None:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    if staged != parity:
        return Q8_0_DOT_BENCH_ERR_BAD_PARAM

    out_shape_count[0] = staged[0]
    out_max_blocks[0] = staged[1]
    out_max_rows[0] = staged[2]
    out_max_row_matrix_blocks[0] = staged[3]
    return Q8_0_DOT_BENCH_OK


def test_source_contains_iq1305_function_and_guards() -> None:
    source = Path("src/bench/q8_0_dot_bench.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "if (!out_shape_count || !out_max_blocks || !out_max_rows ||" in body
    assert "if (out_shape_count == out_max_blocks ||" in body
    assert "snapshot_cpu_hz = cpu_hz;" in body
    assert "status = Q8_0DotBenchRunDefaultSuiteCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "status = Q8_0DotBenchRunDefaultSuitePreflightOnlyCommitOnlyParityCommitOnlyPreflightOnly(" in body
    assert "if (staged_shape_count != parity_shape_count ||" in body
    assert "*out_shape_count = staged_shape_count;" in body
    assert "*out_max_blocks = staged_max_blocks;" in body
    assert "*out_max_rows = staged_max_rows;" in body
    assert "*out_max_row_matrix_blocks = staged_max_row_matrix_blocks;" in body


def test_null_and_alias_guards() -> None:
    assert parity_commit_only_preflight_only_parity_commit_only(3_200_000_000, None, [1], [1], [1]) == Q8_0_DOT_BENCH_ERR_NULL_PTR

    same = [9]
    status = parity_commit_only_preflight_only_parity_commit_only(3_200_000_000, same, same, [3], [4])
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM


def test_bad_frequency_no_partial_commit() -> None:
    out_shape_count = [111]
    out_max_blocks = [222]
    out_max_rows = [333]
    out_max_row_matrix_blocks = [444]

    status = parity_commit_only_preflight_only_parity_commit_only(
        0,
        out_shape_count,
        out_max_blocks,
        out_max_rows,
        out_max_row_matrix_blocks,
    )
    assert status == Q8_0_DOT_BENCH_ERR_BAD_PARAM
    assert (
        out_shape_count[0],
        out_max_blocks[0],
        out_max_rows[0],
        out_max_row_matrix_blocks[0],
    ) == (111, 222, 333, 444)


def test_parity_publish_matches_canonical() -> None:
    out_shape_count = [0]
    out_max_blocks = [0]
    out_max_rows = [0]
    out_max_row_matrix_blocks = [0]

    status = parity_commit_only_preflight_only_parity_commit_only(
        3_200_000_000,
        out_shape_count,
        out_max_blocks,
        out_max_rows,
        out_max_row_matrix_blocks,
    )
    assert status == Q8_0_DOT_BENCH_OK
    assert (
        out_shape_count[0],
        out_max_blocks[0],
        out_max_rows[0],
        out_max_row_matrix_blocks[0],
    ) == canonical_shape_tuple(3_200_000_000)


if __name__ == "__main__":
    test_source_contains_iq1305_function_and_guards()
    test_null_and_alias_guards()
    test_bad_frequency_no_partial_commit()
    test_parity_publish_matches_canonical()
    print("ok")
