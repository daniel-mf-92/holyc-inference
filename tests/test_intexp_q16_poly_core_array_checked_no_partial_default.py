#!/usr/bin/env python3
"""Parity checks for FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartialDefault."""

from __future__ import annotations

import pathlib
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_intexp_q16_poly_core_array_checked import (
    EXP_Q16_LN2,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
)
from test_intexp_q16_poly_core_array_checked_no_partial import (
    fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial,
)


def fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
    residual_q16: list[int] | None,
    out_poly_q16: list[int] | None,
    count: int,
    *,
    residual_addr: int = 0x1100,
    out_addr: int = 0x2200,
) -> int:
    return fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(
        residual_q16,
        out_poly_q16,
        count,
        residual_addr=residual_addr,
        out_addr=out_addr,
    )


def test_source_contains_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartialDefault" in source
    assert "status = FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartial(residual_q16," in source
    assert "return status;" in source


def test_null_bad_count_and_zero_count_surfaces() -> None:
    lanes = [0, 1, 2]
    out = [101, 202, 303]

    assert (
        fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(None, out, 1)
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(lanes, None, 1)
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(lanes, out, -1)
        == FP_Q16_ERR_BAD_PARAM
    )

    sentinel = out.copy()
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(lanes, out, 0)
    assert err == FP_Q16_OK
    assert out == sentinel


def test_overflow_passthrough_and_no_partial_write() -> None:
    lanes = [0, 1]
    out = [77, 88]

    huge_count = (I64_MAX_VALUE >> 3) + 2
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
        lanes,
        out,
        huge_count,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]

    near_end = U64_MAX_VALUE - ((len(lanes) - 1) << 3) + 1
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
        lanes,
        out,
        len(lanes),
        residual_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]

    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
        lanes,
        out,
        len(lanes),
        out_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]


def test_boundary_residuals_and_randomized_parity() -> None:
    boundary = [0, 1, EXP_Q16_LN2 - 2, EXP_Q16_LN2 - 1]

    out_default = [0x66] * len(boundary)
    out_ref = [0x66] * len(boundary)
    err_default = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
        boundary,
        out_default,
        len(boundary),
    )
    err_ref = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(
        boundary,
        out_ref,
        len(boundary),
    )
    assert err_default == err_ref == FP_Q16_OK
    assert out_default == out_ref

    invalid = [0, 2, EXP_Q16_LN2, 3]
    sentinel = [9, 8, 7, 6]
    out_invalid = sentinel.copy()
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
        invalid,
        out_invalid,
        len(invalid),
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_invalid == sentinel

    rng = random.Random(20260419_502)
    for _ in range(1200):
        count = rng.randint(1, 192)
        lanes = [rng.randint(0, EXP_Q16_LN2 - 1) for _ in range(count)]
        out_default = [0x55AA] * count
        out_ref = [0x55AA] * count

        err_default = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial_default(
            lanes,
            out_default,
            count,
        )
        err_ref = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(
            lanes,
            out_ref,
            count,
        )
        assert err_default == err_ref
        if err_ref == FP_Q16_OK:
            assert out_default == out_ref
        else:
            assert out_default == [0x55AA] * count


def run() -> None:
    test_source_contains_default_wrapper_shape()
    test_null_bad_count_and_zero_count_surfaces()
    test_overflow_passthrough_and_no_partial_write()
    test_boundary_residuals_and_randomized_parity()
    print("intexp_q16_poly_core_array_checked_no_partial_default=ok")


if __name__ == "__main__":
    run()
