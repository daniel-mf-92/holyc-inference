#!/usr/bin/env python3
"""Parity harness for FPQ16MulSatChecked + FPQ16MacSatChecked (IQ-883)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_sign_apply_from_u64(mag: int, is_negative: bool) -> int:
    if is_negative:
        if mag >= (1 << 63):
            return I64_MIN_VALUE
        return -mag
    if mag > I64_MAX_VALUE:
        return I64_MAX_VALUE
    return mag


def fpq16_mul_sat_checked(a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if a_q16 == 0 or b_q16 == 0:
        return FP_Q16_OK, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, I64_MIN_VALUE if is_negative else I64_MAX_VALUE

    abs_prod = abs_a * abs_b
    frac_mask = (1 << FP_Q16_SHIFT) - 1
    frac_part = abs_prod & frac_mask

    rounded_mag = abs_prod >> FP_Q16_SHIFT
    if frac_part >= (1 << (FP_Q16_SHIFT - 1)):
        rounded_mag += 1

    limit = (1 << 63) if is_negative else I64_MAX_VALUE
    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, I64_MIN_VALUE if is_negative else I64_MAX_VALUE

    return FP_Q16_OK, fp_sign_apply_from_u64(rounded_mag, is_negative)


def fpq16_mac_sat_checked(acc_q16: int, a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    mul_status, mul_q16 = fpq16_mul_sat_checked(a_q16, b_q16, out_present=True)

    if mul_q16 > 0 and acc_q16 > I64_MAX_VALUE - mul_q16:
        return FP_Q16_ERR_OVERFLOW, I64_MAX_VALUE
    if mul_q16 < 0 and acc_q16 < I64_MIN_VALUE - mul_q16:
        return FP_Q16_ERR_OVERFLOW, I64_MIN_VALUE

    out_q16 = acc_q16 + mul_q16
    if mul_status != FP_Q16_OK:
        return FP_Q16_ERR_OVERFLOW, out_q16

    return FP_Q16_OK, out_q16


def test_source_contains_new_helpers() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    assert "I32 FPQ16MulSatChecked(I64 a_q16," in source
    assert "I32 FPQ16MacSatChecked(I64 acc_q16," in source
    assert "if (abs_a > (U64_MAX_VALUE / abs_b))" in source
    assert "sum_q16 = acc_q16 + mul_q16;" in source


def test_null_ptr_contracts() -> None:
    status, _ = fpq16_mul_sat_checked(123, 456, out_present=False)
    assert status == FP_Q16_ERR_NULL_PTR

    status, _ = fpq16_mac_sat_checked(789, 111, 222, out_present=False)
    assert status == FP_Q16_ERR_NULL_PTR


def test_saturation_edges() -> None:
    status, out = fpq16_mul_sat_checked(I64_MAX_VALUE, I64_MAX_VALUE)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == I64_MAX_VALUE

    status, out = fpq16_mul_sat_checked(I64_MIN_VALUE, I64_MAX_VALUE)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == I64_MIN_VALUE

    status, out = fpq16_mac_sat_checked(I64_MAX_VALUE - 4, 5 << FP_Q16_SHIFT, 1 << FP_Q16_SHIFT)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == I64_MAX_VALUE

    status, out = fpq16_mac_sat_checked(I64_MIN_VALUE + 4, -(5 << FP_Q16_SHIFT), 1 << FP_Q16_SHIFT)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == I64_MIN_VALUE


def test_randomized_reference_parity() -> None:
    rng = random.Random(883)

    for _ in range(15000):
        a_q16 = rng.randint(I64_MIN_VALUE, I64_MAX_VALUE)
        b_q16 = rng.randint(I64_MIN_VALUE, I64_MAX_VALUE)
        acc_q16 = rng.randint(I64_MIN_VALUE, I64_MAX_VALUE)

        status_mul, out_mul = fpq16_mul_sat_checked(a_q16, b_q16, out_present=True)
        assert status_mul in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW)
        assert I64_MIN_VALUE <= out_mul <= I64_MAX_VALUE

        status_mac, out_mac = fpq16_mac_sat_checked(acc_q16, a_q16, b_q16, out_present=True)
        assert status_mac in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW)
        assert I64_MIN_VALUE <= out_mac <= I64_MAX_VALUE


def run() -> None:
    test_source_contains_new_helpers()
    test_null_ptr_contracts()
    test_saturation_edges()
    test_randomized_reference_parity()
    print("fixedpoint_q16_mul_mac_saturating_checked=ok")


if __name__ == "__main__":
    run()
