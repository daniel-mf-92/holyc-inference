#!/usr/bin/env python3
"""Parity harness for FPQ16MulSatCheckedNoPartialArray (IQ-885)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_SHIFT = 16
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


def fpq16_mul_sat_checked_no_partial_array(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
) -> int:
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    if count > len(lhs_q16) or count > len(rhs_q16) or count > len(out_q16):
        return FP_Q16_ERR_BAD_PARAM

    # Preflight all lanes first; no output writes on hard failure.
    for i in range(count):
        err, _ = fpq16_mul_sat_checked(lhs_q16[i], rhs_q16[i], out_present=True)
        if err not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
            return err

    overflow_seen = False
    for i in range(count):
        err, lane = fpq16_mul_sat_checked(lhs_q16[i], rhs_q16[i], out_present=True)
        out_q16[i] = lane
        if err == FP_Q16_ERR_OVERFLOW:
            overflow_seen = True
        elif err != FP_Q16_OK:
            return err

    if overflow_seen:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def explicit_reference_array(
    lhs_q16: list[int],
    rhs_q16: list[int],
    count: int,
) -> tuple[int, list[int]]:
    out = [0] * count
    status = FP_Q16_OK
    for i in range(count):
        err, lane = fpq16_mul_sat_checked(lhs_q16[i], rhs_q16[i], out_present=True)
        out[i] = lane
        if err == FP_Q16_ERR_OVERFLOW:
            status = FP_Q16_ERR_OVERFLOW
    return status, out


def test_source_contains_mul_sat_no_partial_array_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulSatCheckedNoPartialArray(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16DivArrayChecked", 1)[0]
    assert "FPQ16MulSatChecked(lhs_q16[i]," in body
    assert "if (err != FP_Q16_OK && err != FP_Q16_ERR_OVERFLOW)" in body
    assert "if (overflow_seen)" in body


def test_null_ptr_and_bad_count_no_writes() -> None:
    out = [11, 22, 33]

    status = fpq16_mul_sat_checked_no_partial_array(None, [1, 2, 3], out, 3)
    assert status == FP_Q16_ERR_NULL_PTR
    assert out == [11, 22, 33]

    status = fpq16_mul_sat_checked_no_partial_array([1, 2, 3], [1, 2, 3], out, -1)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [11, 22, 33]

    status = fpq16_mul_sat_checked_no_partial_array([1, 2], [3, 4], out, 0)
    assert status == FP_Q16_OK
    assert out == [11, 22, 33]


def test_known_vectors_saturation_and_commit() -> None:
    lhs = [1 << FP_Q16_SHIFT, -(2 << FP_Q16_SHIFT), I64_MAX_VALUE]
    rhs = [3 << FP_Q16_SHIFT, 4 << FP_Q16_SHIFT, I64_MAX_VALUE]
    out = [0x55, 0x66, 0x77]

    status = fpq16_mul_sat_checked_no_partial_array(lhs, rhs, out, 3)
    assert status == FP_Q16_ERR_OVERFLOW

    assert out[0] == 3 << FP_Q16_SHIFT
    assert out[1] == -(8 << FP_Q16_SHIFT)
    assert out[2] == I64_MAX_VALUE


def test_randomized_reference_parity_and_alias_adversarial() -> None:
    rng = random.Random(20260421_885)

    for _ in range(5000):
        count = rng.randint(1, 48)
        lhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        rhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]

        expected_status, expected_out = explicit_reference_array(lhs, rhs, count)

        out = [0x1234] * count
        status = fpq16_mul_sat_checked_no_partial_array(lhs.copy(), rhs.copy(), out, count)
        assert status == expected_status
        assert out == expected_out

        # Alias stress: out aliases lhs; lane-local semantics must still match.
        lhs_alias = lhs.copy()
        status_alias = fpq16_mul_sat_checked_no_partial_array(lhs_alias, rhs.copy(), lhs_alias, count)
        assert status_alias == expected_status
        assert lhs_alias == expected_out


def run() -> None:
    test_source_contains_mul_sat_no_partial_array_contract()
    test_null_ptr_and_bad_count_no_writes()
    test_known_vectors_saturation_and_commit()
    test_randomized_reference_parity_and_alias_adversarial()
    print("fixedpoint_q16_mul_sat_checked_no_partial_array=ok")


if __name__ == "__main__":
    run()
