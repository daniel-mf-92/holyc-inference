#!/usr/bin/env python3
"""Reference checks for FPQ16ExpFromClampedInputCheckedNoPartial semantics."""

from __future__ import annotations

import pathlib
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX_VALUE = (1 << 63) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2


def fpq16_mul(a: int, b: int) -> int:
    prod = a * b
    if prod >= 0:
        return (prod + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT
    return -(((-prod) + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT)


def fpq16_exp_from_clamped_input_checked(
    clamped_input_q16: int,
    out_present: bool = True,
) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0

    if clamped_input_q16 >= EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, I64_MAX_VALUE
    if clamped_input_q16 <= EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, 0

    k = clamped_input_q16 // EXP_Q16_LN2
    if clamped_input_q16 < 0 and (clamped_input_q16 % EXP_Q16_LN2):
        k -= 1

    r = clamped_input_q16 - (k * EXP_Q16_LN2)

    r2 = fpq16_mul(r, r)
    r3 = fpq16_mul(r2, r)
    r4 = fpq16_mul(r3, r)

    poly = FP_Q16_ONE + r + (r2 // 2) + (r3 // 6) + (r4 // 24)

    if k >= 0:
        if k >= 30:
            return FP_Q16_OK, I64_MAX_VALUE

        if poly > (I64_MAX_VALUE >> k):
            return FP_Q16_OK, I64_MAX_VALUE

        return FP_Q16_OK, poly << k

    if k <= -63:
        return FP_Q16_OK, 0

    return FP_Q16_OK, poly >> (-k)


def fpq16_exp_from_clamped_input_checked_no_partial(
    clamped_input_q16: int,
    out_exp: list[int] | None,
) -> int:
    if out_exp is None:
        return FP_Q16_ERR_NULL_PTR

    staged = 0
    err, got = fpq16_exp_from_clamped_input_checked(clamped_input_q16)
    if err != FP_Q16_OK:
        return err

    staged = got
    out_exp[0] = staged
    return FP_Q16_OK


def test_source_contains_no_partial_wrapper() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpFromClampedInputCheckedNoPartial" in source
    assert "FPQ16ExpFromClampedInputChecked(clamped_input_q16," in source
    assert "staged_exp_q16 = 0;" in source


def test_null_pointer_surface() -> None:
    err = fpq16_exp_from_clamped_input_checked_no_partial(0, None)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_no_partial_commit() -> None:
    sentinel = [123456789]

    err = fpq16_exp_from_clamped_input_checked_no_partial(EXP_Q16_MIN_INPUT - 1, sentinel)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert sentinel == [123456789]

    err = fpq16_exp_from_clamped_input_checked_no_partial(EXP_Q16_MAX_INPUT + 1, sentinel)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert sentinel == [123456789]


def test_success_matches_core() -> None:
    samples = [
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -250000,
        -1,
        0,
        1,
        250000,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
    ]

    for x_q16 in samples:
        out = [777]
        err = fpq16_exp_from_clamped_input_checked_no_partial(x_q16, out)
        core_err, core_out = fpq16_exp_from_clamped_input_checked(x_q16)

        assert err == core_err == FP_Q16_OK
        assert out == [core_out]


def test_success_independent_of_prior_output_value() -> None:
    samples = [-123456, -1, 0, 1, 123456]
    for x_q16 in samples:
        err_core, core_out = fpq16_exp_from_clamped_input_checked(x_q16)
        assert err_core == FP_Q16_OK

        a = [111111]
        b = [-999999]
        err_a = fpq16_exp_from_clamped_input_checked_no_partial(x_q16, a)
        err_b = fpq16_exp_from_clamped_input_checked_no_partial(x_q16, b)

        assert err_a == err_b == FP_Q16_OK
        assert a == [core_out]
        assert b == [core_out]


def test_randomized_parity() -> None:
    rng = random.Random(20260419_454)
    for _ in range(20000):
        x_q16 = rng.randint(EXP_Q16_MIN_INPUT - 200000, EXP_Q16_MAX_INPUT + 200000)
        out = [42424242]

        err = fpq16_exp_from_clamped_input_checked_no_partial(x_q16, out)
        core_err, core_out = fpq16_exp_from_clamped_input_checked(x_q16)

        assert err == core_err
        if err == FP_Q16_OK:
            assert out == [core_out]
        else:
            assert out == [42424242]


def run() -> None:
    test_source_contains_no_partial_wrapper()
    test_null_pointer_surface()
    test_bad_param_no_partial_commit()
    test_success_matches_core()
    test_success_independent_of_prior_output_value()
    test_randomized_parity()
    print("intexp_exp_from_clamped_input_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
