#!/usr/bin/env python3
"""Reference checks for FPQ16ExpSplitBase2FromClampedInputChecked semantics."""

from __future__ import annotations

import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360


def fpq16_exp_split_base2_from_clamped_input_checked(
    clamped_input_q16: int,
    out_k_present: bool = True,
    out_r_present: bool = True,
) -> tuple[int, int, int]:
    if not out_k_present or not out_r_present:
        return FP_Q16_ERR_NULL_PTR, 0, 0

    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    if EXP_Q16_LN2 <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    k = int(clamped_input_q16 / EXP_Q16_LN2)
    rem = clamped_input_q16 - (k * EXP_Q16_LN2)
    if clamped_input_q16 < 0 and rem:
        k -= 1

    r_q16 = clamped_input_q16 - (k * EXP_Q16_LN2)
    if r_q16 < 0 or r_q16 >= EXP_Q16_LN2:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    return FP_Q16_OK, k, r_q16


def test_null_pointer_surfaces() -> None:
    err, _, _ = fpq16_exp_split_base2_from_clamped_input_checked(0, out_k_present=False)
    assert err == FP_Q16_ERR_NULL_PTR

    err, _, _ = fpq16_exp_split_base2_from_clamped_input_checked(0, out_r_present=False)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_out_of_domain() -> None:
    err, _, _ = fpq16_exp_split_base2_from_clamped_input_checked(EXP_Q16_MIN_INPUT - 1)
    assert err == FP_Q16_ERR_BAD_PARAM

    err, _, _ = fpq16_exp_split_base2_from_clamped_input_checked(EXP_Q16_MAX_INPUT + 1)
    assert err == FP_Q16_ERR_BAD_PARAM


def test_remainder_range_and_reconstruction_edges() -> None:
    samples = [
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -500000,
        -1,
        0,
        1,
        500000,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
    ]

    for x_q16 in samples:
        err, k, r_q16 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
        assert err == FP_Q16_OK
        assert 0 <= r_q16 < EXP_Q16_LN2
        assert x_q16 == (k * EXP_Q16_LN2) + r_q16


def test_floor_division_behavior_for_negative_inputs() -> None:
    # For negative non-multiples of ln2, k is floor(x/ln2), not trunc toward zero.
    x_q16 = -1
    err, k, r_q16 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
    assert err == FP_Q16_OK

    trunc_k = x_q16 // EXP_Q16_LN2
    # Python // is floor already; emulate trunc-toward-zero for contrast.
    trunc_toward_zero_k = int(x_q16 / EXP_Q16_LN2)

    assert k == trunc_k
    assert k <= trunc_toward_zero_k
    assert r_q16 == x_q16 - (k * EXP_Q16_LN2)
    assert 0 <= r_q16 < EXP_Q16_LN2


def test_randomized_invariants() -> None:
    rng = random.Random(20260417_181)

    for _ in range(10000):
        x_q16 = rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT)
        err, k, r_q16 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
        assert err == FP_Q16_OK
        assert 0 <= r_q16 < EXP_Q16_LN2
        assert x_q16 == (k * EXP_Q16_LN2) + r_q16


def run() -> None:
    test_null_pointer_surfaces()
    test_bad_param_out_of_domain()
    test_remainder_range_and_reconstruction_edges()
    test_floor_division_behavior_for_negative_inputs()
    test_randomized_invariants()
    print("intexp_split_base2_from_clamped_input_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
