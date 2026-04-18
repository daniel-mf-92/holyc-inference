#!/usr/bin/env python3
"""Parity checks for FPQ16ExpRangeReduceLn2CheckedNoPartial semantics."""

from __future__ import annotations

import pathlib
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


def fpq16_exp_range_reduce_ln2_checked_no_partial(
    input_q16: int,
    out_k: list[int] | None,
    out_r: list[int] | None,
) -> int:
    if out_k is None or out_r is None:
        return FP_Q16_ERR_NULL_PTR

    staged_k = out_k[0]
    staged_r = out_r[0]

    err, got_k, got_r = fpq16_exp_range_reduce_ln2_checked(input_q16)
    if err != FP_Q16_OK:
        return err

    staged_k = got_k
    staged_r = got_r
    out_k[0] = staged_k
    out_r[0] = staged_r
    return FP_Q16_OK


def test_source_contains_no_partial_wrapper() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpRangeReduceLn2CheckedNoPartial" in source
    assert "FPQ16ExpRangeReduceLn2Checked(input_q16," in source
    assert "staged_k = 0;" in source
    assert "staged_r_q16 = 0;" in source


def test_null_pointer_surfaces() -> None:
    out_k = [111]
    out_r = [222]

    err = fpq16_exp_range_reduce_ln2_checked_no_partial(0, None, out_r)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_r == [222]

    err = fpq16_exp_range_reduce_ln2_checked_no_partial(0, out_k, None)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_k == [111]


def test_success_matches_core_outputs() -> None:
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
        out_k = [777]
        out_r = [888]

        err = fpq16_exp_range_reduce_ln2_checked_no_partial(x_q16, out_k, out_r)
        core_err, core_k, core_r = fpq16_exp_range_reduce_ln2_checked(x_q16)

        assert err == core_err == FP_Q16_OK
        assert (out_k[0], out_r[0]) == (core_k, core_r)
        assert 0 <= out_r[0] < EXP_Q16_LN2


def test_no_partial_commit_on_error() -> None:
    # Emulate inner helper failure by temporarily invalidating ln2 constant.
    global EXP_Q16_LN2
    saved_ln2 = EXP_Q16_LN2

    out_k = [12345]
    out_r = [67890]

    try:
        EXP_Q16_LN2 = 0
        err = fpq16_exp_range_reduce_ln2_checked_no_partial(0, out_k, out_r)
        assert err == FP_Q16_ERR_BAD_PARAM
        assert out_k == [12345]
        assert out_r == [67890]
    finally:
        EXP_Q16_LN2 = saved_ln2


def test_success_is_independent_of_prior_output_values() -> None:
    sample_inputs = [
        EXP_Q16_MIN_INPUT,
        -123456,
        -1,
        0,
        1,
        123456,
        EXP_Q16_MAX_INPUT,
    ]

    for x_q16 in sample_inputs:
        err_a, core_k, core_r = fpq16_exp_range_reduce_ln2_checked(x_q16)
        assert err_a == FP_Q16_OK

        out_k_a = [111111111]
        out_r_a = [222222222]
        err = fpq16_exp_range_reduce_ln2_checked_no_partial(x_q16, out_k_a, out_r_a)
        assert err == FP_Q16_OK

        out_k_b = [-333333333]
        out_r_b = [-444444444]
        err = fpq16_exp_range_reduce_ln2_checked_no_partial(x_q16, out_k_b, out_r_b)
        assert err == FP_Q16_OK

        assert out_k_a == [core_k]
        assert out_r_a == [core_r]
        assert out_k_b == [core_k]
        assert out_r_b == [core_r]
        assert out_k_a == out_k_b
        assert out_r_a == out_r_b


def test_randomized_parity_and_reconstruction() -> None:
    rng = random.Random(20260419_451)

    for _ in range(30000):
        x_q16 = rng.randint(EXP_Q16_MIN_INPUT - 300000, EXP_Q16_MAX_INPUT + 300000)

        out_k = [-1]
        out_r = [-1]
        err = fpq16_exp_range_reduce_ln2_checked_no_partial(x_q16, out_k, out_r)

        core_err, core_k, core_r = fpq16_exp_range_reduce_ln2_checked(x_q16)
        assert err == core_err
        if err == FP_Q16_OK:
            assert (out_k[0], out_r[0]) == (core_k, core_r)
            _, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
            assert clamped == (out_k[0] * EXP_Q16_LN2) + out_r[0]
            assert 0 <= out_r[0] < EXP_Q16_LN2


def run() -> None:
    test_source_contains_no_partial_wrapper()
    test_null_pointer_surfaces()
    test_success_matches_core_outputs()
    test_no_partial_commit_on_error()
    test_success_is_independent_of_prior_output_values()
    test_randomized_parity_and_reconstruction()
    print("intexp_q16_range_reduce_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
