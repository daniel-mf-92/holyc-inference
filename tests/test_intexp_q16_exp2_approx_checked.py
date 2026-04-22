#!/usr/bin/env python3
"""Reference checks for FPQ16Exp2ApproxChecked (IQ-1133)."""

from __future__ import annotations

import math
from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360
EXP_Q16_SAT_MIN_OUTPUT = 0
EXP_Q16_SAT_MAX_OUTPUT = I64_MAX_VALUE


# Same half-away-from-zero behavior used by fixedpoint checked multiply.
def _mul_q16_checked(a_q16: int, b_q16: int) -> tuple[int, int | None]:
    abs_a = abs(a_q16)
    abs_b = abs(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a == 0 or abs_b == 0:
        return FP_Q16_OK, 0

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, None

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    limit = (1 << 63) if is_negative else I64_MAX_VALUE
    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, None

    out = -rounded_mag if is_negative else rounded_mag
    if out < I64_MIN_VALUE or out > I64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, None
    return FP_Q16_OK, out


def fpq16_exp2_approx_checked_reference(input_q16: int, *, out_present: bool = True) -> tuple[int, int | None]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, None

    status, exp_input_q16 = _mul_q16_checked(input_q16, EXP_Q16_LN2)
    if status == FP_Q16_ERR_OVERFLOW:
        if input_q16 >= 0:
            return FP_Q16_OK, EXP_Q16_SAT_MAX_OUTPUT
        return FP_Q16_OK, EXP_Q16_SAT_MIN_OUTPUT
    if status != FP_Q16_OK:
        return status, None

    if exp_input_q16 >= EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, EXP_Q16_SAT_MAX_OUTPUT
    if exp_input_q16 <= EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, EXP_Q16_SAT_MIN_OUTPUT

    x = exp_input_q16 / FP_Q16_ONE
    out = int(round(math.exp(x) * FP_Q16_ONE))
    if out < 0:
        out = 0
    if out > I64_MAX_VALUE:
        out = I64_MAX_VALUE
    return FP_Q16_OK, out


def _q16(v: float) -> int:
    return int(round(v * FP_Q16_ONE))


def test_source_contains_iq1133_signature_and_contract() -> None:
    source = Path("src/math/intexp.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16Exp2ApproxChecked(I64 input_q16,"
    assert source.count(sig) >= 1

    def_start = -1
    search_from = 0
    while True:
        idx = source.find(sig, search_from)
        if idx < 0:
            break
        brace_idx = source.find("{", idx)
        semi_idx = source.find(";", idx)
        if brace_idx >= 0 and (semi_idx < 0 or brace_idx < semi_idx):
            def_start = idx
            break
        search_from = idx + len(sig)

    assert def_start >= 0
    body = source[def_start:]
    assert "FPQ16MulChecked(input_q16," in body
    assert "EXP_Q16_LN2" in body
    assert "FPQ16ExpFromClampedInputChecked(exp_input_q16," in body
    assert "if (status == FP_Q16_ERR_OVERFLOW)" in body


def test_nullptr_contract() -> None:
    status, out = fpq16_exp2_approx_checked_reference(_q16(1.0), out_present=False)
    assert status == FP_Q16_ERR_NULL_PTR
    assert out is None


def test_known_points() -> None:
    for x, expected in [
        (0.0, 1.0),
        (1.0, 2.0),
        (-1.0, 0.5),
        (0.5, math.sqrt(2.0)),
        (2.0, 4.0),
    ]:
        status, out_q16 = fpq16_exp2_approx_checked_reference(_q16(x))
        assert status == FP_Q16_OK
        got = out_q16 / FP_Q16_ONE
        assert abs(got - expected) < 0.06


def test_saturation_surfaces() -> None:
    status, out_q16 = fpq16_exp2_approx_checked_reference(_q16(20.0))
    assert status == FP_Q16_OK
    assert out_q16 == EXP_Q16_SAT_MAX_OUTPUT

    status, out_q16 = fpq16_exp2_approx_checked_reference(_q16(-20.0))
    assert status == FP_Q16_OK
    assert out_q16 == EXP_Q16_SAT_MIN_OUTPUT


def test_monotonicity_over_core_domain() -> None:
    prev = -1
    for i in range(-32, 33):
        x_q16 = _q16(i / 8.0)
        status, out_q16 = fpq16_exp2_approx_checked_reference(x_q16)
        assert status == FP_Q16_OK
        assert out_q16 >= prev
        prev = out_q16


def run() -> None:
    test_source_contains_iq1133_signature_and_contract()
    test_nullptr_contract()
    test_known_points()
    test_saturation_surfaces()
    test_monotonicity_over_core_domain()
    print("intexp_q16_exp2_approx_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
