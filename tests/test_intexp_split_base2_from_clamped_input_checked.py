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


def _holyc_floor_div_k(x_q16: int) -> int:
    """Mirror HolyC logic: trunc div + negative remainder correction."""
    k = int(x_q16 / EXP_Q16_LN2)
    if x_q16 < 0 and (x_q16 % EXP_Q16_LN2):
        k -= 1
    return k


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

    k = _holyc_floor_div_k(clamped_input_q16)

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

    floor_k = x_q16 // EXP_Q16_LN2
    # Python // is floor; contrast against trunc-toward-zero.
    trunc_toward_zero_k = int(x_q16 / EXP_Q16_LN2)

    assert k == floor_k
    assert k <= trunc_toward_zero_k
    assert r_q16 == x_q16 - (k * EXP_Q16_LN2)
    assert 0 <= r_q16 < EXP_Q16_LN2


def test_exact_multiples_have_zero_remainder() -> None:
    for k in range(-14, 15):
        x_q16 = k * EXP_Q16_LN2
        if x_q16 < EXP_Q16_MIN_INPUT or x_q16 > EXP_Q16_MAX_INPUT:
            continue

        err, got_k, r_q16 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
        assert err == FP_Q16_OK
        assert got_k == k
        assert r_q16 == 0


def test_adjacent_to_negative_multiples_uses_floor() -> None:
    # For x = -(m*ln2)-1, floor-division k is one less than trunc-toward-zero.
    for m in range(1, 15):
        x_q16 = -(m * EXP_Q16_LN2) - 1
        if x_q16 < EXP_Q16_MIN_INPUT or x_q16 > EXP_Q16_MAX_INPUT:
            continue

        err, k, r_q16 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
        assert err == FP_Q16_OK
        assert k == _holyc_floor_div_k(x_q16)
        assert k == (int(x_q16 / EXP_Q16_LN2) - 1)
        assert 0 <= r_q16 < EXP_Q16_LN2
        assert x_q16 == (k * EXP_Q16_LN2) + r_q16


def test_exhaustive_domain_reconstruction_and_remainder_range() -> None:
    # Full domain sweep to lock in splitter behavior for every legal lane.
    for x_q16 in range(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT + 1):
        err, k, r_q16 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
        assert err == FP_Q16_OK
        assert k == _holyc_floor_div_k(x_q16)
        assert 0 <= r_q16 < EXP_Q16_LN2
        assert x_q16 == (k * EXP_Q16_LN2) + r_q16


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
    test_exact_multiples_have_zero_remainder()
    test_adjacent_to_negative_multiples_uses_floor()
    test_exhaustive_domain_reconstruction_and_remainder_range()
    test_randomized_invariants()
    print("intexp_split_base2_from_clamped_input_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
