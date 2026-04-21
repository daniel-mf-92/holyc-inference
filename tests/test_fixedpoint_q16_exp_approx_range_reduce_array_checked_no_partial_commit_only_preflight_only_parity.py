#!/usr/bin/env python3
"""Parity harness for FPQ16...CommitOnlyPreflightOnlyParity (IQ-907)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_OK,
    fpq16_exp_approx_range_reduce_checked_no_partial_array,
)
from test_fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial_commit_only_preflight_only import (
    FP_Q16_ERR_OVERFLOW,
    fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only,
)


def fp_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    i64_min = -(1 << 63)
    i64_max = (1 << 63) - 1
    prod = lhs * rhs
    if prod < i64_min or prod > i64_max:
        return FP_Q16_ERR_OVERFLOW, 0
    return FP_Q16_OK, prod


def fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
) -> int:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if out_required_cells is None or out_required_bytes is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_k or x_q16 is out_r_q16 or out_k is out_r_q16:
        return FP_Q16_ERR_BAD_PARAM

    if out_required_cells is out_required_bytes:
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_cells is x_q16
        or out_required_cells is out_k
        or out_required_cells is out_r_q16
        or out_required_bytes is x_q16
        or out_required_bytes is out_k
        or out_required_bytes is out_r_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    snapshot = (x_q16, out_k, out_r_q16, count)

    staged_count = [0]
    staged_required_cells = [0]
    staged_required_bytes = [0]
    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
        x_q16,
        out_k,
        out_r_q16,
        count,
        staged_count,
        staged_required_cells,
        staged_required_bytes,
    )
    if status != FP_Q16_OK:
        return status

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array(x_q16, out_k, out_r_q16, count)
    if status != FP_Q16_OK:
        return status

    if snapshot != (x_q16, out_k, out_r_q16, count):
        return FP_Q16_ERR_BAD_PARAM

    status, canonical_required_cells = fp_try_mul_i64_checked(count, 2)
    if status != FP_Q16_OK:
        return status

    status, canonical_required_bytes = fp_try_mul_i64_checked(canonical_required_cells, 8)
    if status != FP_Q16_OK:
        return status

    if staged_count[0] != count:
        return FP_Q16_ERR_BAD_PARAM
    if staged_required_cells[0] != canonical_required_cells:
        return FP_Q16_ERR_BAD_PARAM
    if staged_required_bytes[0] != canonical_required_bytes:
        return FP_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_required_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    return FP_Q16_OK


def explicit_composition(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
) -> tuple[int, tuple[int, int]]:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR, (0, 0)

    staged_count = [0]
    staged_cells = [0]
    staged_bytes = [0]
    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
        x_q16,
        out_k,
        out_r_q16,
        count,
        staged_count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array(x_q16, out_k, out_r_q16, count)
    if status != FP_Q16_OK:
        return status, (0, 0)

    status, cells = fp_try_mul_i64_checked(count, 2)
    if status != FP_Q16_OK:
        return status, (0, 0)
    status, bytes_ = fp_try_mul_i64_checked(cells, 8)
    if status != FP_Q16_OK:
        return status, (0, 0)

    if staged_count[0] != count or staged_cells[0] != cells or staged_bytes[0] != bytes_:
        return FP_Q16_ERR_BAD_PARAM, (0, 0)

    return FP_Q16_OK, (cells, bytes_)


def test_source_contains_iq907_helper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnlyParity(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes", 1)[0]
    assert "FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16ExpApproxRangeReduceCheckedNoPartialArray(x_q16," in body
    assert "status = FPTryMulI64Checked(count, 2, &canonical_required_cells);" in body
    assert "status = FPTryMulI64Checked(canonical_required_cells," in body
    assert "*out_required_cells = preflight_required_cells;" in body
    assert "*out_required_bytes = preflight_required_bytes;" in body


def test_known_vector_success() -> None:
    x = [-(1 << 20), -65_536, -1, 0, 1, 65_536, 1 << 20]
    out_k = [0] * len(x)
    out_r = [0] * len(x)
    out_cells = [123]
    out_bytes = [456]

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity(
        x,
        out_k,
        out_r,
        len(x),
        out_cells,
        out_bytes,
    )
    assert status == FP_Q16_OK
    assert out_cells[0] == len(x) * 2
    assert out_bytes[0] == len(x) * 16


def test_null_and_alias_fail_no_publish() -> None:
    x = [0, 1]
    out_k = [0, 0]
    out_r = [0, 0]
    out_cells = [111]
    out_bytes = [222]

    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity(
            None, out_k, out_r, 2, out_cells, out_bytes
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert out_cells == [111]
    assert out_bytes == [222]

    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity(
            x, out_k, out_r, 2, out_cells, out_cells
        )
        == FP_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(907)

    for _ in range(1000):
        count = rng.randint(0, 128)
        x = [rng.randint(-(1 << 22), (1 << 22)) for _ in range(count)]
        out_k = [rng.randint(-10, 10) for _ in range(count)]
        out_r = [rng.randint(-10, 10) for _ in range(count)]

        if rng.random() < 0.15 and count > 0:
            x = [rng.choice([1 << 60, -(1 << 60)]) for _ in range(count)]

        out_cells = [0xAA]
        out_bytes = [0xBB]

        status_a = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only_parity(
            x,
            out_k,
            out_r,
            count,
            out_cells,
            out_bytes,
        )

        expected_status, expected_tuple = explicit_composition(x, out_k, out_r, count)
        assert status_a == expected_status

        if status_a == FP_Q16_OK:
            assert (out_cells[0], out_bytes[0]) == expected_tuple
        else:
            assert out_cells == [0xAA]
            assert out_bytes == [0xBB]


if __name__ == "__main__":
    test_source_contains_iq907_helper()
    test_known_vector_success()
    test_null_and_alias_fail_no_publish()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
