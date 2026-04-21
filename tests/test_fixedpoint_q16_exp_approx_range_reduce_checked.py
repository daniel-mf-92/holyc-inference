#!/usr/bin/env python3
"""Parity checks for FPQ16ExpApproxRangeReduceChecked (IQ-900)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2

EXP_LN2_Q16 = 45_426
EXP_HALF_LN2_Q16 = 22_713


def fpq16_exp_approx_range_reduce_checked(
    x_q16: int,
    out_k_present: bool = True,
    out_r_present: bool = True,
    alias_outputs: bool = False,
) -> tuple[int, int, int, bool]:
    if not out_k_present or not out_r_present:
        return FP_Q16_ERR_NULL_PTR, 0, 0, False
    if alias_outputs:
        return FP_Q16_ERR_BAD_PARAM, 0, 0, False

    if x_q16 >= 0:
        k = (x_q16 + EXP_HALF_LN2_Q16) // EXP_LN2_Q16
    else:
        k = -(((-x_q16) + EXP_HALF_LN2_Q16) // EXP_LN2_Q16)

    r_q16 = x_q16 - (k * EXP_LN2_Q16)
    if r_q16 < -EXP_HALF_LN2_Q16 or r_q16 > EXP_HALF_LN2_Q16:
        return FP_Q16_ERR_BAD_PARAM, 0, 0, False

    return FP_Q16_OK, k, r_q16, True


def test_source_contains_iq900_helper_and_exp_callsite() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    assert "I32 FPQ16ExpApproxRangeReduceChecked(I64 x_q16," in source
    assert "status = FPQ16ExpApproxRangeReduceChecked(x_q16," in source
    assert "x_q16 = k*ln(2) + r_q16" in source


def test_null_pointer_and_alias_guards() -> None:
    err, _, _, wrote = fpq16_exp_approx_range_reduce_checked(0, out_k_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert wrote is False

    err, _, _, wrote = fpq16_exp_approx_range_reduce_checked(0, out_r_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert wrote is False

    err, _, _, wrote = fpq16_exp_approx_range_reduce_checked(0, alias_outputs=True)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert wrote is False


def test_reconstruction_identity_on_adversarial_grid() -> None:
    vectors = [
        -(1 << 63) + 1,
        -4_194_304,
        -1_048_576,
        -65_536,
        -1,
        0,
        1,
        65_536,
        1_048_576,
        4_194_304,
        (1 << 63) - 1,
    ]

    for x_q16 in vectors:
        err, k, r_q16, wrote = fpq16_exp_approx_range_reduce_checked(x_q16)
        assert err == FP_Q16_OK
        assert wrote is True
        assert -EXP_HALF_LN2_Q16 <= r_q16 <= EXP_HALF_LN2_Q16
        assert x_q16 == (k * EXP_LN2_Q16) + r_q16


def test_residual_minimizes_distance_to_nearest_ln2_multiple() -> None:
    random.seed(900)

    for _ in range(2000):
        x_q16 = random.randint(-50_000_000, 50_000_000)
        err, k, r_q16, wrote = fpq16_exp_approx_range_reduce_checked(x_q16)
        assert err == FP_Q16_OK
        assert wrote is True

        this_dist = abs(r_q16)
        prev_dist = abs(x_q16 - ((k - 1) * EXP_LN2_Q16))
        next_dist = abs(x_q16 - ((k + 1) * EXP_LN2_Q16))

        assert this_dist <= prev_dist
        assert this_dist <= next_dist


def test_shifted_ln2_roundtrip_stability() -> None:
    base_values = [-200_000, -100_000, -1, 0, 1, 100_000, 200_000]

    for base in base_values:
        err0, k0, r0, wrote0 = fpq16_exp_approx_range_reduce_checked(base)
        assert err0 == FP_Q16_OK
        assert wrote0 is True

        for delta_k in (-7, -3, -1, 1, 3, 7):
            x_shift = base + (delta_k * EXP_LN2_Q16)
            err1, k1, r1, wrote1 = fpq16_exp_approx_range_reduce_checked(x_shift)
            assert err1 == FP_Q16_OK
            assert wrote1 is True
            assert r1 == r0
            assert k1 == (k0 + delta_k)
