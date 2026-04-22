#!/usr/bin/env python3
"""Reference checks for FPQ16MulShiftRoundChecked (IQ-1132)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_try_apply_sign_from_u64_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if is_negative:
        if mag >= (1 << 63):
            return FP_Q16_OK, I64_MIN_VALUE
        return FP_Q16_OK, -mag

    return FP_Q16_OK, mag


def fpq16_mul_shift_round_checked_reference(
    lhs: int,
    rhs: int,
    shift: int,
    *,
    out_present: bool = True,
) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0
    if shift < 0 or shift > 62:
        return FP_Q16_ERR_BAD_PARAM, 0

    if lhs == 0 or rhs == 0:
        return FP_Q16_OK, 0

    abs_lhs = fp_abs_to_u64(lhs)
    abs_rhs = fp_abs_to_u64(rhs)
    is_negative = (lhs < 0) ^ (rhs < 0)

    if abs_lhs > (U64_MAX_VALUE // abs_rhs):
        return FP_Q16_ERR_OVERFLOW, 0

    product_mag = abs_lhs * abs_rhs

    if shift == 0:
        return fp_try_apply_sign_from_u64_checked(product_mag, is_negative)

    quotient = product_mag >> shift
    remainder = product_mag & (((1 << shift) - 1))
    half = 1 << (shift - 1)

    if remainder >= half:
        if quotient == U64_MAX_VALUE:
            return FP_Q16_ERR_OVERFLOW, 0
        quotient += 1

    return fp_try_apply_sign_from_u64_checked(quotient, is_negative)


def round_half_away_signed_div_pow2(value: int, shift: int) -> int:
    if shift == 0:
        return value

    denom = 1 << shift
    abs_value = abs(value)
    quotient = abs_value >> shift
    remainder = abs_value & (denom - 1)

    if remainder >= (denom >> 1):
        quotient += 1

    return -quotient if value < 0 else quotient


def test_source_contains_iq1132_function_and_rounding_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulShiftRoundChecked(I64 lhs,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatChecked", 1)[0]
    assert "if (shift < 0 || shift > 62)" in body
    assert "signed_limit = (U64)I64_MAX_VALUE;" in body
    assert "if (remainder >= half)" in body
    assert "if (quotient >= signed_limit)" in body
    assert "return FPTryApplySignFromU64Checked(quotient," in body


def test_nullptr_and_shift_domain_errors() -> None:
    status, _ = fpq16_mul_shift_round_checked_reference(7, 9, 4, out_present=False)
    assert status == FP_Q16_ERR_NULL_PTR

    status, _ = fpq16_mul_shift_round_checked_reference(7, 9, -1)
    assert status == FP_Q16_ERR_BAD_PARAM

    status, _ = fpq16_mul_shift_round_checked_reference(7, 9, 63)
    assert status == FP_Q16_ERR_BAD_PARAM


def test_half_away_from_zero_ties() -> None:
    status, out = fpq16_mul_shift_round_checked_reference(3, 1, 1)
    assert status == FP_Q16_OK
    assert out == 2

    status, out = fpq16_mul_shift_round_checked_reference(-3, 1, 1)
    assert status == FP_Q16_OK
    assert out == -2

    status, out = fpq16_mul_shift_round_checked_reference(5, 1, 2)
    assert status == FP_Q16_OK
    assert out == 1

    status, out = fpq16_mul_shift_round_checked_reference(6, 1, 2)
    assert status == FP_Q16_OK
    assert out == 2


def test_overflow_surface_and_sign_restore() -> None:
    status, _ = fpq16_mul_shift_round_checked_reference(I64_MAX_VALUE, I64_MAX_VALUE, 16)
    assert status == FP_Q16_ERR_OVERFLOW

    # 0xFFFFFFFF * 0x100000001 == U64_MAX; shift=1 gives quotient=I64_MAX
    # and remainder=1, so tie-away increment must report overflow.
    status, out = fpq16_mul_shift_round_checked_reference(0xFFFFFFFF, 0x100000001, 1)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == 0

    status, out = fpq16_mul_shift_round_checked_reference(I64_MIN_VALUE, 1, 0)
    assert status == FP_Q16_OK
    assert out == I64_MIN_VALUE

    status, out = fpq16_mul_shift_round_checked_reference(I64_MIN_VALUE, -1, 1)
    assert status == FP_Q16_OK
    assert out == (1 << 62)


def test_randomized_reference_sanity() -> None:
    rng = random.Random(20260422_1132)

    for _ in range(10000):
        lhs = rng.randint(I64_MIN_VALUE, I64_MAX_VALUE)
        rhs = rng.randint(I64_MIN_VALUE, I64_MAX_VALUE)
        shift = rng.randint(0, 62)

        status, out = fpq16_mul_shift_round_checked_reference(lhs, rhs, shift)

        assert status in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW)
        if status == FP_Q16_OK:
            assert I64_MIN_VALUE <= out <= I64_MAX_VALUE


def test_randomized_exact_rounding_parity_bounded_domain() -> None:
    rng = random.Random(20260422_1132_2)

    for _ in range(20000):
        lhs = rng.randint(-(1 << 31), (1 << 31) - 1)
        rhs = rng.randint(-(1 << 31), (1 << 31) - 1)
        shift = rng.randint(0, 31)

        status, out = fpq16_mul_shift_round_checked_reference(lhs, rhs, shift)
        exact = round_half_away_signed_div_pow2(lhs * rhs, shift)

        assert status == FP_Q16_OK
        assert out == exact


def test_i64_min_negative_tie_rounds_away_from_zero() -> None:
    status, out = fpq16_mul_shift_round_checked_reference(I64_MIN_VALUE, 1, 1)
    assert status == FP_Q16_OK
    assert out == -(1 << 62)


def run() -> None:
    test_source_contains_iq1132_function_and_rounding_contract()
    test_nullptr_and_shift_domain_errors()
    test_half_away_from_zero_ties()
    test_overflow_surface_and_sign_restore()
    test_randomized_reference_sanity()
    test_randomized_exact_rounding_parity_bounded_domain()
    test_i64_min_negative_tie_rounds_away_from_zero()
    print("fixedpoint_q16_mul_shift_round_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
