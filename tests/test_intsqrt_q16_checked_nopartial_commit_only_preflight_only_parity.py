#!/usr/bin/env python3
"""Parity harness for FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-1236)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_checked import (  # noqa: E402
    FP_Q16_ERR_DOMAIN,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    q16_from_float,
)

FP_Q16_SHIFT = 16
I64_MAX_VALUE = (1 << 63) - 1
FP_Q16_ERR_BAD_PARAM = 2


def fpq16_sqrt(x_q16: int) -> int:
    if x_q16 <= 0:
        return 0
    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        shifted = I64_MAX_VALUE
    else:
        shifted = x_q16 << FP_Q16_SHIFT
    return int(shifted**0.5)


def fpq16_sqrt_checked_nopartial_commit_only_reference(
    x_q16: int,
    out_q16: list[int] | None,
) -> int:
    if out_q16 is None:
        return FP_Q16_ERR_NULL_PTR

    if x_q16 < 0:
        return FP_Q16_ERR_DOMAIN

    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW

    snapshot_x_q16 = x_q16
    staged_out_q16 = fpq16_sqrt(x_q16)

    if snapshot_x_q16 != x_q16:
        return FP_Q16_ERR_BAD_PARAM

    out_q16[0] = staged_out_q16
    return FP_Q16_OK


def fpq16_sqrt_checked_nopartial_commit_only_preflight_only_reference(
    x_q16: int,
    required_out_cells_out: list[int] | None,
) -> int:
    if required_out_cells_out is None:
        return FP_Q16_ERR_NULL_PTR

    required_out_cells_out[0] = 0
    snapshot_x_q16 = x_q16

    staged_out_q16 = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_reference(x_q16, staged_out_q16)
    if status != FP_Q16_OK:
        return status

    if snapshot_x_q16 != x_q16:
        return FP_Q16_ERR_BAD_PARAM

    canonical_out_q16 = fpq16_sqrt(x_q16)
    if canonical_out_q16 != staged_out_q16[0]:
        return FP_Q16_ERR_BAD_PARAM

    staged_required_out_cells = 1
    if staged_required_out_cells != 1:
        return FP_Q16_ERR_BAD_PARAM

    required_out_cells_out[0] = staged_required_out_cells
    return FP_Q16_OK


def fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(
    x_q16: int,
    required_out_cells_out: list[int] | None,
    *,
    mutate_snapshot: bool = False,
    force_tuple_parity_break: bool = False,
    force_required_mismatch: bool = False,
) -> int:
    if required_out_cells_out is None:
        return FP_Q16_ERR_NULL_PTR

    required_out_cells_out[0] = 0
    snapshot_x_q16 = x_q16

    staged_required_out_cells = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_reference(
        x_q16,
        staged_required_out_cells,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        x_q16 += 1

    if snapshot_x_q16 != x_q16:
        return FP_Q16_ERR_BAD_PARAM

    canonical_out_q16 = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_reference(x_q16, canonical_out_q16)
    if status != FP_Q16_OK:
        return status

    canonical_required_out_cells = 1

    if force_required_mismatch:
        staged_required_out_cells[0] += 1

    if staged_required_out_cells[0] != canonical_required_out_cells:
        return FP_Q16_ERR_BAD_PARAM

    if force_tuple_parity_break:
        canonical_required_out_cells = 2

    if canonical_required_out_cells != 1:
        return FP_Q16_ERR_BAD_PARAM

    required_out_cells_out[0] = staged_required_out_cells[0]
    return FP_Q16_OK


def explicit_parity_composition(x_q16: int, required_out_cells_out: list[int] | None) -> int:
    if required_out_cells_out is None:
        return FP_Q16_ERR_NULL_PTR

    required_out_cells_out[0] = 0

    staged_required = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_reference(x_q16, staged_required)
    if status != FP_Q16_OK:
        return status

    canonical_out = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_reference(x_q16, canonical_out)
    if status != FP_Q16_OK:
        return status

    if staged_required[0] != 1:
        return FP_Q16_ERR_BAD_PARAM

    required_out_cells_out[0] = staged_required[0]
    return FP_Q16_OK


def test_source_contains_iq1236_function() -> None:
    source = Path("src/math/intsqrt.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParity(I64 x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (!required_out_cells_out)" in body
    assert "*required_out_cells_out = 0;" in body
    assert "snapshot_x_q16 = x_q16;" in body
    assert "status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnly(x_q16," in body
    assert "status = FPQ16SqrtCheckedNoPartialCommitOnly(x_q16," in body
    assert "if (snapshot_x_q16 != x_q16)" in body
    assert "canonical_required_out_cells = 1;" in body
    assert "if (staged_required_out_cells != canonical_required_out_cells)" in body
    assert "if (canonical_required_out_cells != 1)" in body
    assert "*required_out_cells_out = staged_required_out_cells;" in body


def test_null_domain_overflow_keep_required_zero() -> None:
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(123, None)
        == FP_Q16_ERR_NULL_PTR
    )

    required = [77]
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(-1, required)
        == FP_Q16_ERR_DOMAIN
    )
    assert required == [0]

    required = [88]
    too_large = (I64_MAX_VALUE >> FP_Q16_SHIFT) + 1
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(too_large, required)
        == FP_Q16_ERR_OVERFLOW
    )
    assert required == [0]


def test_success_and_explicit_parity_composition() -> None:
    vectors = [
        0,
        1,
        q16_from_float(0.25),
        q16_from_float(1.0),
        q16_from_float(2.0),
        q16_from_float(123.5),
        (I64_MAX_VALUE >> FP_Q16_SHIFT),
    ]

    for x_q16 in vectors:
        req_a = [0]
        req_b = [0]
        err_a = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(x_q16, req_a)
        err_b = explicit_parity_composition(x_q16, req_b)
        assert err_a == FP_Q16_OK
        assert err_b == FP_Q16_OK
        assert req_a == [1]
        assert req_a == req_b


def test_snapshot_and_tuple_parity_guards() -> None:
    required = [5]
    err = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(
        q16_from_float(16.0), required, mutate_snapshot=True
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert required == [0]

    required = [6]
    err = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(
        q16_from_float(25.0), required, force_required_mismatch=True
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert required == [0]

    required = [7]
    err = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(
        q16_from_float(36.0), required, force_tuple_parity_break=True
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert required == [0]


def test_randomized_requirement_stability() -> None:
    rng = random.Random(20260423_1236)

    for _ in range(5000):
        x_q16 = q16_from_float(rng.uniform(0.0, 50000.0))
        required = [0]
        err = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_reference(x_q16, required)
        assert err == FP_Q16_OK
        assert required == [1]


def run() -> None:
    test_source_contains_iq1236_function()
    test_null_domain_overflow_keep_required_zero()
    test_success_and_explicit_parity_composition()
    test_snapshot_and_tuple_parity_guards()
    test_randomized_requirement_stability()
    print("intsqrt_q16_checked_nopartial_commit_only_preflight_only_parity_reference_checks=ok")


if __name__ == "__main__":
    run()
