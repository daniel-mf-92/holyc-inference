#!/usr/bin/env python3
"""Reference checks for FPQ16ExpRangeReduceLn2Checked semantics."""

from __future__ import annotations

import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360


def fpq16_exp_clamp_to_input_domain_checked(input_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0
    if EXP_Q16_MIN_INPUT > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0
    if input_q16 < EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, EXP_Q16_MIN_INPUT
    if input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, EXP_Q16_MAX_INPUT
    return FP_Q16_OK, input_q16


def _holyc_floor_div_k(x_q16: int) -> int:
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


def fpq16_exp_range_reduce_ln2_checked(
    input_q16: int,
    out_k_present: bool = True,
    out_r_present: bool = True,
) -> tuple[int, int, int]:
    if not out_k_present or not out_r_present:
        return FP_Q16_ERR_NULL_PTR, 0, 0

    err, clamped_q16 = fpq16_exp_clamp_to_input_domain_checked(input_q16)
    if err != FP_Q16_OK:
        return err, 0, 0

    return fpq16_exp_split_base2_from_clamped_input_checked(clamped_q16)


def test_null_pointer_surfaces() -> None:
    err, _, _ = fpq16_exp_range_reduce_ln2_checked(0, out_k_present=False)
    assert err == FP_Q16_ERR_NULL_PTR

    err, _, _ = fpq16_exp_range_reduce_ln2_checked(0, out_r_present=False)
    assert err == FP_Q16_ERR_NULL_PTR


def test_clamp_then_split_out_of_domain_inputs() -> None:
    # Below-domain values clamp to MIN before split.
    err, k, r = fpq16_exp_range_reduce_ln2_checked(EXP_Q16_MIN_INPUT - 12345)
    assert err == FP_Q16_OK
    assert (k * EXP_Q16_LN2 + r) == EXP_Q16_MIN_INPUT
    assert 0 <= r < EXP_Q16_LN2

    # Above-domain values clamp to MAX before split.
    err, k, r = fpq16_exp_range_reduce_ln2_checked(EXP_Q16_MAX_INPUT + 67890)
    assert err == FP_Q16_OK
    assert (k * EXP_Q16_LN2 + r) == EXP_Q16_MAX_INPUT
    assert 0 <= r < EXP_Q16_LN2


def test_boundary_vectors_and_signed_behavior() -> None:
    samples = [
        EXP_Q16_MIN_INPUT - 1,
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -500000,
        -1,
        0,
        1,
        500000,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
        EXP_Q16_MAX_INPUT + 1,
    ]

    for x_q16 in samples:
        err, k, r_q16 = fpq16_exp_range_reduce_ln2_checked(x_q16)
        assert err == FP_Q16_OK

        _, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
        assert k == _holyc_floor_div_k(clamped)
        assert 0 <= r_q16 < EXP_Q16_LN2
        assert clamped == (k * EXP_Q16_LN2) + r_q16


def test_randomized_clamp_and_reconstruction() -> None:
    rng = random.Random(20260419_441)

    for _ in range(20000):
        x_q16 = rng.randint(EXP_Q16_MIN_INPUT - 200000, EXP_Q16_MAX_INPUT + 200000)
        err, k, r_q16 = fpq16_exp_range_reduce_ln2_checked(x_q16)
        assert err == FP_Q16_OK

        _, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
        assert 0 <= r_q16 < EXP_Q16_LN2
        assert clamped == (k * EXP_Q16_LN2) + r_q16


def test_matches_splitter_on_in_domain_inputs() -> None:
    for x_q16 in [
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 123,
        -45426,
        -1,
        0,
        1,
        45426,
        EXP_Q16_MAX_INPUT - 123,
        EXP_Q16_MAX_INPUT,
    ]:
        err0, k0, r0 = fpq16_exp_range_reduce_ln2_checked(x_q16)
        err1, k1, r1 = fpq16_exp_split_base2_from_clamped_input_checked(x_q16)
        assert err0 == err1 == FP_Q16_OK
        assert (k0, r0) == (k1, r1)


def run() -> None:
    test_null_pointer_surfaces()
    test_clamp_then_split_out_of_domain_inputs()
    test_boundary_vectors_and_signed_behavior()
    test_randomized_clamp_and_reconstruction()
    test_matches_splitter_on_in_domain_inputs()
    print("intexp_q16_range_reduce_reference_checks=ok")


if __name__ == "__main__":
    run()
