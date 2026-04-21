#!/usr/bin/env python3
"""Parity harness for FPQ16...ParityCommitOnlyPreflightOnly (IQ-923)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
)
from test_fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial_commit_only_preflight_only_parity_commit_only import (
    fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only,
)


def fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_count: list[int] | None,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
) -> int:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if out_count is None or out_required_cells is None or out_required_bytes is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_k or x_q16 is out_r_q16 or out_k is out_r_q16:
        return FP_Q16_ERR_BAD_PARAM

    if out_count is out_required_cells or out_count is out_required_bytes or out_required_cells is out_required_bytes:
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_count is x_q16
        or out_count is out_k
        or out_count is out_r_q16
        or out_required_cells is x_q16
        or out_required_cells is out_k
        or out_required_cells is out_r_q16
        or out_required_bytes is x_q16
        or out_required_bytes is out_k
        or out_required_bytes is out_r_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    snapshot = (x_q16, out_k, out_r_q16, count)

    staged_count = [0]
    staged_cells = [0]
    staged_bytes = [0]
    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
        x_q16,
        out_k,
        out_r_q16,
        count,
        staged_count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if snapshot != (x_q16, out_k, out_r_q16, count):
        return FP_Q16_ERR_BAD_PARAM

    canonical_count = count
    if canonical_count > (I64_MAX_VALUE >> 1):
        return FP_Q16_ERR_OVERFLOW

    canonical_required_cells = canonical_count << 1
    if canonical_required_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    canonical_required_bytes = canonical_required_cells << 3

    if staged_count[0] != canonical_count:
        return FP_Q16_ERR_BAD_PARAM
    if staged_cells[0] != canonical_required_cells:
        return FP_Q16_ERR_BAD_PARAM
    if staged_bytes[0] != canonical_required_bytes:
        return FP_Q16_ERR_BAD_PARAM

    out_count[0] = staged_count[0]
    out_required_cells[0] = staged_cells[0]
    out_required_bytes[0] = staged_bytes[0]
    return FP_Q16_OK


def explicit_composition(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
) -> tuple[int, tuple[int, int, int]]:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR, (0, 0, 0)
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, (0, 0, 0)

    staged_count = [0]
    staged_cells = [0]
    staged_bytes = [0]
    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
        x_q16,
        out_k,
        out_r_q16,
        count,
        staged_count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0, 0)

    canonical_count = count
    if canonical_count > (I64_MAX_VALUE >> 1):
        return FP_Q16_ERR_OVERFLOW, (0, 0, 0)

    canonical_required_cells = canonical_count << 1
    if canonical_required_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, (0, 0, 0)

    canonical_required_bytes = canonical_required_cells << 3

    if staged_count[0] != canonical_count:
        return FP_Q16_ERR_BAD_PARAM, (0, 0, 0)
    if staged_cells[0] != canonical_required_cells:
        return FP_Q16_ERR_BAD_PARAM, (0, 0, 0)
    if staged_bytes[0] != canonical_required_bytes:
        return FP_Q16_ERR_BAD_PARAM, (0, 0, 0)

    return FP_Q16_OK, (staged_count[0], staged_cells[0], staged_bytes[0])


def test_source_contains_iq923_helper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes", 1)[0]
    assert "status = FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "canonical_count = count;" in body
    assert "canonical_required_cells = canonical_count << 1;" in body
    assert "canonical_required_bytes = canonical_required_cells << 3;" in body
    assert "if (snapshot_count != count)" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_null_alias_and_bad_param_paths() -> None:
    x = [0, 1, 2]
    out_k = [0, 0, 0]
    out_r = [0, 0, 0]
    out_count = [7]
    out_cells = [11]
    out_bytes = [13]

    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
            None, out_k, out_r, 3, out_count, out_cells, out_bytes
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
            x, out_k, out_r, 3, None, out_cells, out_bytes
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
            x, out_k, out_r, -1, out_count, out_cells, out_bytes
        )
        == FP_Q16_ERR_BAD_PARAM
    )
    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
            x, out_k, out_r, 3, out_count, out_count, out_bytes
        )
        == FP_Q16_ERR_BAD_PARAM
    )


def test_known_vectors() -> None:
    x = [-(1 << 20), -65_536, -1, 0, 1, 65_536, 1 << 20]
    out_k = [0] * len(x)
    out_r = [0] * len(x)
    out_count = [999]
    out_cells = [123]
    out_bytes = [456]

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        x,
        out_k,
        out_r,
        len(x),
        out_count,
        out_cells,
        out_bytes,
    )
    assert status == FP_Q16_OK
    assert out_count[0] == len(x)
    assert out_cells[0] == len(x) * 2
    assert out_bytes[0] == len(x) * 16


def test_failure_paths_preserve_outputs() -> None:
    x = [0, 1 << 62, -(1 << 62)]
    out_k = [123, 456, 789]
    out_r = [111, 222, 333]
    out_count = [444]
    out_cells = [555]
    out_bytes = [666]

    before_k = out_k.copy()
    before_r = out_r.copy()
    before_tuple = (out_count[0], out_cells[0], out_bytes[0])

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        x,
        out_k,
        out_r,
        len(x),
        out_count,
        out_cells,
        out_bytes,
    )

    if status != FP_Q16_OK:
        assert out_k == before_k
        assert out_r == before_r
        assert (out_count[0], out_cells[0], out_bytes[0]) == before_tuple


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260421_923)

    for _ in range(1200):
        count = rng.randint(0, 96)
        x = [rng.randint(-(1 << 22), (1 << 22)) for _ in range(count)]
        out_k = [rng.randint(-10, 10) for _ in range(count)]
        out_r = [rng.randint(-10, 10) for _ in range(count)]

        if rng.random() < 0.12 and count > 0:
            x = [rng.choice([1 << 60, -(1 << 60)]) for _ in range(count)]

        out_count = [0xAA]
        out_cells = [0xBB]
        out_bytes = [0xCC]

        status_impl = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
            x,
            out_k,
            out_r,
            count,
            out_count,
            out_cells,
            out_bytes,
        )
        status_ref, expected_tuple = explicit_composition(x, out_k, out_r, count)

        assert status_impl == status_ref
        if status_impl == FP_Q16_OK:
            assert (out_count[0], out_cells[0], out_bytes[0]) == expected_tuple
        else:
            assert out_count == [0xAA]
            assert out_cells == [0xBB]
            assert out_bytes == [0xCC]


def test_large_magnitude_lane_vector_preserves_contract() -> None:
    x = [1 << 62]
    out_k = [0]
    out_r = [0]
    out_count = [111]
    out_cells = [222]
    out_bytes = [333]

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        x,
        out_k,
        out_r,
        len(x),
        out_count,
        out_cells,
        out_bytes,
    )
    assert status == FP_Q16_OK
    assert out_count == [1]
    assert out_cells == [2]
    assert out_bytes == [16]


if __name__ == "__main__":
    test_source_contains_iq923_helper()
    test_null_alias_and_bad_param_paths()
    test_known_vectors()
    test_failure_paths_preserve_outputs()
    test_randomized_parity_vs_explicit_composition()
    test_large_magnitude_lane_vector_preserves_contract()
    print("ok")
