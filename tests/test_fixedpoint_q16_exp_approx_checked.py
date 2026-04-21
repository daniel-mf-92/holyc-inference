#!/usr/bin/env python3
"""Parity and accuracy checks for FPQ16ExpApproxChecked (IQ-897)."""

from __future__ import annotations

import math
import random
from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_DOMAIN = 3
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1

EXP_LN2_Q16 = 45_426
EXP_HALF_LN2_Q16 = 22_713


def _abs_to_u64(value: int) -> int:
    if value >= 0:
        return value
    return (-(value + 1)) + 1


def _try_apply_sign_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if is_negative:
        if mag == (1 << 63):
            return FP_Q16_OK, I64_MIN_VALUE
        return FP_Q16_OK, -mag

    return FP_Q16_OK, mag


def _fpq16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    if not a_q16 or not b_q16:
        return FP_Q16_OK, 0

    abs_a = _abs_to_u64(a_q16)
    abs_b = _abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > (U64_MAX_VALUE - round_bias):
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63
    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    return _try_apply_sign_checked(rounded_mag, is_negative)


def _round_shift_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if shift >= 63:
        return 0

    is_negative = value < 0
    abs_value = _abs_to_u64(value)
    round_bias = 1 << (shift - 1)

    if abs_value > (U64_MAX_VALUE - round_bias):
        rounded = U64_MAX_VALUE >> shift
    else:
        rounded = (abs_value + round_bias) >> shift

    if is_negative:
        if rounded >= (1 << 63):
            return I64_MIN_VALUE
        return -rounded

    if rounded > I64_MAX_VALUE:
        return I64_MAX_VALUE
    return rounded


def fpq16_exp_approx_checked(x_q16: int, out_present: bool = True) -> tuple[int, int, bool]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0, False

    if x_q16 >= 0:
        k = (x_q16 + EXP_HALF_LN2_Q16) // EXP_LN2_Q16
    else:
        k = -(((-x_q16) + EXP_HALF_LN2_Q16) // EXP_LN2_Q16)

    abs_k_u64 = _abs_to_u64(k)
    if abs_k_u64 > (U64_MAX_VALUE // EXP_LN2_Q16):
        return FP_Q16_ERR_OVERFLOW, 0, False

    if k >= 0:
        k_ln2_q16 = abs_k_u64 * EXP_LN2_Q16
    else:
        k_ln2_q16 = -(abs_k_u64 * EXP_LN2_Q16)

    if (k_ln2_q16 > 0 and x_q16 < (I64_MIN_VALUE + k_ln2_q16)) or (
        k_ln2_q16 < 0 and x_q16 > (I64_MAX_VALUE + k_ln2_q16)
    ):
        return FP_Q16_ERR_OVERFLOW, 0, False

    residual_q16 = x_q16 - k_ln2_q16

    err, residual2_q16 = _fpq16_mul_checked(residual_q16, residual_q16)
    if err != FP_Q16_OK:
        return err, 0, False

    err, residual3_q16 = _fpq16_mul_checked(residual2_q16, residual_q16)
    if err != FP_Q16_OK:
        return err, 0, False

    err, residual4_q16 = _fpq16_mul_checked(residual3_q16, residual_q16)
    if err != FP_Q16_OK:
        return err, 0, False

    poly_q16 = FP_Q16_ONE

    for term_q16 in (
        residual_q16,
        residual2_q16 // 2,
        residual3_q16 // 6,
        residual4_q16 // 24,
    ):
        if term_q16 > 0 and poly_q16 > (I64_MAX_VALUE - term_q16):
            return FP_Q16_ERR_OVERFLOW, 0, False
        if term_q16 < 0 and poly_q16 < (I64_MIN_VALUE - term_q16):
            return FP_Q16_ERR_OVERFLOW, 0, False
        poly_q16 += term_q16

    if poly_q16 <= 0:
        staged = 0
    elif k >= 0:
        if k >= 63 or poly_q16 > (I64_MAX_VALUE >> k):
            return FP_Q16_ERR_OVERFLOW, I64_MAX_VALUE, True
        staged = poly_q16 << k
    else:
        shift_mag = -k
        if shift_mag >= 63:
            staged = 0
        else:
            staged = _round_shift_signed(poly_q16, shift_mag)

    if staged < 0:
        staged = 0

    return FP_Q16_OK, staged, True


def q16_to_float(value_q16: int) -> float:
    return value_q16 / float(FP_Q16_ONE)


def test_source_contains_iq897_function_and_core_math() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    assert "I32 FPQ16ExpApproxChecked(I64 x_q16," in source
    assert "x = k*ln(2) + r" in source
    assert "residual4_q16 / 24" in source
    assert "*out_q16 = I64_MAX_VALUE;" in source


def test_null_output_pointer() -> None:
    err, out, wrote = fpq16_exp_approx_checked(0, out_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == 0
    assert wrote is False


def test_overflow_saturates_positive_side() -> None:
    err, out, wrote = fpq16_exp_approx_checked(I64_MAX_VALUE)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == I64_MAX_VALUE
    assert wrote is True


def test_large_negative_underflows_toward_zero() -> None:
    err, out, wrote = fpq16_exp_approx_checked(I64_MIN_VALUE + 1)
    assert err == FP_Q16_OK
    assert out == 0
    assert wrote is True


def test_monotonicity_over_adversarial_grid() -> None:
    grid = [
        -4_194_304,
        -2_097_152,
        -1_048_576,
        -524_288,
        -262_144,
        -131_072,
        -65_536,
        -1,
        0,
        1,
        65_536,
        131_072,
        262_144,
        524_288,
        1_048_576,
        2_097_152,
        4_194_304,
    ]

    prev = None
    for x_q16 in grid:
        err, out, wrote = fpq16_exp_approx_checked(x_q16)
        assert wrote is True
        if err == FP_Q16_ERR_OVERFLOW:
            assert out == I64_MAX_VALUE
            break
        assert err == FP_Q16_OK
        if prev is not None:
            assert out >= prev
        prev = out


def test_accuracy_vs_high_precision_reference_in_safe_window() -> None:
    # Safe window avoids saturation and extreme underflow.
    rng = random.Random(20260421_897)

    # Deterministic boundary probes plus random adversarial values.
    probes = [
        -8 * FP_Q16_ONE,
        -6 * FP_Q16_ONE,
        -4 * FP_Q16_ONE,
        -2 * FP_Q16_ONE,
        -FP_Q16_ONE,
        -(FP_Q16_ONE // 2),
        -1,
        0,
        1,
        FP_Q16_ONE // 2,
        FP_Q16_ONE,
        2 * FP_Q16_ONE,
        4 * FP_Q16_ONE,
        6 * FP_Q16_ONE,
        8 * FP_Q16_ONE,
    ]
    probes.extend(rng.randint(-8 * FP_Q16_ONE, 8 * FP_Q16_ONE) for _ in range(3000))

    worst_rel = 0.0
    for x_q16 in probes:
        err, out_q16, wrote = fpq16_exp_approx_checked(x_q16)
        assert wrote is True
        assert err == FP_Q16_OK

        x = q16_to_float(x_q16)
        ref = math.exp(x)
        got = q16_to_float(out_q16)

        rel = abs(got - ref) / max(ref, 1e-12)
        worst_rel = max(worst_rel, rel)

        # Tight enough to catch broken range-reduction/poly wiring.
        assert rel < 0.035

    # Keep this as a coarse regression signal.
    assert worst_rel < 0.035


def run() -> None:
    test_source_contains_iq897_function_and_core_math()
    test_null_output_pointer()
    test_overflow_saturates_positive_side()
    test_large_negative_underflows_toward_zero()
    test_monotonicity_over_adversarial_grid()
    test_accuracy_vs_high_precision_reference_in_safe_window()
    print("fixedpoint_q16_exp_approx_checked=ok")


if __name__ == "__main__":
    run()
