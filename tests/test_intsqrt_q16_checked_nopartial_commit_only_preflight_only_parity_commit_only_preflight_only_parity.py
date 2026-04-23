#!/usr/bin/env python3
"""Harness for FPQ16Sqrt...ParityCommitOnlyPreflightOnlyParity (IQ-1239)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_checked import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_DOMAIN,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    q16_from_float,
)
from test_intsqrt_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only import (  # noqa: E402
    fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference,
)
from test_intsqrt_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only import (  # noqa: E402
    fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference,
)

FP_Q16_SHIFT = 16
I64_MAX_VALUE = (1 << 63) - 1


def fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
    x_q16: int,
    required_out_cells_out: list[int] | None,
    *,
    mutate_snapshot: bool = False,
    mutate_tuple: bool = False,
) -> int:
    if required_out_cells_out is None:
        return FP_Q16_ERR_NULL_PTR

    required_out_cells_out[0] = 0
    snapshot_x_q16 = x_q16

    staged_required = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        x_q16,
        staged_required,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        x_q16 += 1

    if snapshot_x_q16 != x_q16:
        return FP_Q16_ERR_BAD_PARAM

    canonical_required = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        x_q16,
        canonical_required,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_tuple:
        canonical_required[0] += 1

    if staged_required[0] != canonical_required[0]:
        return FP_Q16_ERR_BAD_PARAM

    if canonical_required[0] != 1:
        return FP_Q16_ERR_BAD_PARAM

    required_out_cells_out[0] = staged_required[0]
    return FP_Q16_OK


def explicit_composition(x_q16: int, required_out_cells_out: list[int] | None) -> int:
    if required_out_cells_out is None:
        return FP_Q16_ERR_NULL_PTR

    required_out_cells_out[0] = 0

    staged_required = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        x_q16,
        staged_required,
    )
    if status != FP_Q16_OK:
        return status

    canonical_required = [0]
    status = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        x_q16,
        canonical_required,
    )
    if status != FP_Q16_OK:
        return status

    if staged_required[0] != canonical_required[0]:
        return FP_Q16_ERR_BAD_PARAM

    if canonical_required[0] != 1:
        return FP_Q16_ERR_BAD_PARAM

    required_out_cells_out[0] = staged_required[0]
    return FP_Q16_OK


def test_source_contains_iq1239_function() -> None:
    source = Path("src/math/intsqrt.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "if (!required_out_cells_out)" in body
    assert "*required_out_cells_out = 0;" in body
    assert "snapshot_x_q16 = x_q16;" in body
    assert "status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16SqrtCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (snapshot_x_q16 != x_q16)" in body
    assert "if (staged_required_out_cells != canonical_required_out_cells)" in body
    assert "if (canonical_required_out_cells != 1)" in body
    assert "*required_out_cells_out = staged_required_out_cells;" in body


def test_null_domain_overflow_keep_required_zero() -> None:
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
            123,
            None,
        )
        == FP_Q16_ERR_NULL_PTR
    )

    out = [77]
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
            -1,
            out,
        )
        == FP_Q16_ERR_DOMAIN
    )
    assert out == [0]

    too_large = (I64_MAX_VALUE >> FP_Q16_SHIFT) + 1
    out = [88]
    assert (
        fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
            too_large,
            out,
        )
        == FP_Q16_ERR_OVERFLOW
    )
    assert out == [0]


def test_snapshot_and_tuple_parity_guards() -> None:
    out = [11]
    err = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
        q16_from_float(49.0),
        out,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [0]

    out = [12]
    err = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
        q16_from_float(64.0),
        out,
        mutate_tuple=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [0]


def test_success_and_explicit_composition_parity() -> None:
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
        out_a = [0]
        out_b = [0]

        err_a = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
            x_q16,
            out_a,
        )
        err_b = explicit_composition(x_q16, out_b)

        assert err_a == FP_Q16_OK
        assert err_b == FP_Q16_OK
        assert out_a == out_b == [1]


def test_randomized_parity_against_components() -> None:
    rng = random.Random(20260423_1239)

    for _ in range(12000):
        x_q16 = q16_from_float(rng.uniform(0.0, 50000.0))

        out_new = [9]
        err_new = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
            x_q16,
            out_new,
        )

        out_preflight = [0]
        err_preflight = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
            x_q16,
            out_preflight,
        )

        out_commit_only = [0]
        err_commit_only = fpq16_sqrt_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
            x_q16,
            out_commit_only,
        )

        assert err_new == err_preflight == err_commit_only
        if err_new == FP_Q16_OK:
            assert out_new == [1]
            assert out_preflight == [1]
            assert out_commit_only == [1]


if __name__ == "__main__":
    test_source_contains_iq1239_function()
    test_null_domain_overflow_keep_required_zero()
    test_snapshot_and_tuple_parity_guards()
    test_success_and_explicit_composition_parity()
    test_randomized_parity_against_components()
    print("intsqrt_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity=ok")
