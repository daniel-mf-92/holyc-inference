#!/usr/bin/env python3
"""Parity checks for FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartial."""

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
    fpq16_exp_poly_core_ln2_residual_q16_array_checked,
)


def fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(
    residual_q16: list[int] | None,
    out_poly_q16: list[int] | None,
    count: int,
    *,
    residual_addr: int = 0x1100,
    out_addr: int = 0x2200,
) -> int:
    if residual_q16 is None or out_poly_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if residual_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if out_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    staging = [0] * count
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(
        residual_q16,
        staging,
        count,
        residual_addr=residual_addr,
        out_addr=out_addr,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        out_poly_q16[i] = staging[i]

    return FP_Q16_OK


def test_source_contains_wrapper_and_delegate_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartial" in source
    body = source.split("I32 FPQ16ExpPolyCoreLn2ResidualQ16ArrayCheckedNoPartial", 1)[1].split(
        "// Compatibility alias retained for existing call sites/tests.", 1
    )[0]
    assert "FPQ16ExpPolyCoreLn2ResidualQ16ArrayChecked(residual_q16," in body
    assert "MAlloc(" in body


def test_null_bad_count_and_overflow_surfaces() -> None:
    lanes = [0, 1]
    out = [11, 22]

    assert fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(None, out, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(lanes, None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(lanes, out, -1) == FP_Q16_ERR_BAD_PARAM

    huge_count = (I64_MAX_VALUE >> 3) + 2
    assert (
        fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial([0], [0], huge_count)
        == FP_Q16_ERR_OVERFLOW
    )


def test_zero_count_keeps_output_unchanged() -> None:
    lanes = [10, 20, 30]
    out = [700, 800, 900]
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(lanes, out, 0)
    assert err == FP_Q16_OK
    assert out == [700, 800, 900]


def test_no_partial_write_on_invalid_lane() -> None:
    residuals = [10, 20, EXP_Q16_LN2, 30]
    out = [101, 202, 303, 404]

    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(
        residuals,
        out,
        len(residuals),
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 202, 303, 404]


def test_randomized_parity_vs_checked_array() -> None:
    rng = random.Random(20260419_488)

    for _ in range(500):
        count = rng.randint(1, 192)
        residuals = [rng.randint(0, EXP_Q16_LN2 - 1) for _ in range(count)]

        out_nopartial = [0x7F7F7F] * count
        out_ref = [0x7F7F7F] * count

        err_nopartial = fpq16_exp_poly_core_ln2_residual_q16_array_checked_no_partial(
            residuals,
            out_nopartial,
            count,
        )
        err_ref = fpq16_exp_poly_core_ln2_residual_q16_array_checked(
            residuals,
            out_ref,
            count,
        )

        assert err_nopartial == err_ref == FP_Q16_OK
        assert out_nopartial == out_ref


def run() -> None:
    test_source_contains_wrapper_and_delegate_shape()
    test_null_bad_count_and_overflow_surfaces()
    test_zero_count_keeps_output_unchanged()
    test_no_partial_write_on_invalid_lane()
    test_randomized_parity_vs_checked_array()
    print("intexp_q16_poly_core_array_checked_no_partial_parity=ok")


if __name__ == "__main__":
    run()
