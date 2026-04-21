#!/usr/bin/env python3
"""Parity harness for FPQ16MulCheckedNoPartialArray (IQ-964)."""

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


def fp_try_apply_sign_from_u64_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = 1 << 63 if is_negative else I64_MAX_VALUE
    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0
    if is_negative:
        if mag >= (1 << 63):
            return FP_Q16_OK, I64_MIN_VALUE
        return FP_Q16_OK, -mag
    return FP_Q16_OK, mag


def fpq16_mul_checked(a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if a_q16 == 0 or b_q16 == 0:
        return FP_Q16_OK, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    limit = 1 << 63 if is_negative else I64_MAX_VALUE
    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    return fp_try_apply_sign_from_u64_checked(rounded_mag, is_negative)


def fpq16_mul_checked_no_partial_array(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    *,
    alias_lhs_rhs: bool = False,
    alias_lhs_out: bool = False,
    alias_rhs_out: bool = False,
) -> int:
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if alias_lhs_rhs or alias_lhs_out or alias_rhs_out:
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK

    if count > len(lhs_q16) or count > len(rhs_q16) or count > len(out_q16):
        return FP_Q16_ERR_BAD_PARAM

    for i in range(count):
        err, _ = fpq16_mul_checked(lhs_q16[i], rhs_q16[i], out_present=True)
        if err != FP_Q16_OK:
            return err

    for i in range(count):
        err, lane = fpq16_mul_checked(lhs_q16[i], rhs_q16[i], out_present=True)
        if err != FP_Q16_OK:
            return err
        out_q16[i] = lane

    return FP_Q16_OK


def explicit_reference_array(
    lhs_q16: list[int],
    rhs_q16: list[int],
    count: int,
) -> tuple[int, list[int]]:
    out = [0] * count
    for i in range(count):
        err, lane = fpq16_mul_checked(lhs_q16[i], rhs_q16[i], out_present=True)
        if err != FP_Q16_OK:
            return err, [0] * count
        out[i] = lane
    return FP_Q16_OK, out


def test_source_contains_mul_checked_no_partial_array_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArray(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatCheckedNoPartialArray", 1)[0]
    assert "if (lhs_q16 == rhs_q16 || lhs_q16 == out_q16 || rhs_q16 == out_q16)" in body
    assert "if ((lhs_end > rhs_base && rhs_end > lhs_base) ||" in body
    assert "FPQ16MulChecked(lhs_q16[i]," in body


def test_null_ptr_bad_count_and_alias_rejection_no_writes() -> None:
    out = [111, 222, 333]

    status = fpq16_mul_checked_no_partial_array(None, [1, 2, 3], out, 3)
    assert status == FP_Q16_ERR_NULL_PTR
    assert out == [111, 222, 333]

    status = fpq16_mul_checked_no_partial_array([1, 2, 3], [1, 2, 3], out, -1)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [111, 222, 333]

    status = fpq16_mul_checked_no_partial_array([1, 2, 3], [1, 2, 3], out, 3, alias_lhs_out=True)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [111, 222, 333]


def test_known_vectors_rounding_and_overflow_no_partial_commit() -> None:
    lhs = [1 << FP_Q16_SHIFT, -(2 << FP_Q16_SHIFT), I64_MAX_VALUE]
    rhs = [3 << FP_Q16_SHIFT, 4 << FP_Q16_SHIFT, I64_MAX_VALUE]
    out = [0xAA, 0xBB, 0xCC]

    status = fpq16_mul_checked_no_partial_array(lhs, rhs, out, 3)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == [0xAA, 0xBB, 0xCC]

    lhs_ok = [1 << FP_Q16_SHIFT, -(2 << FP_Q16_SHIFT), 3 << FP_Q16_SHIFT]
    rhs_ok = [3 << FP_Q16_SHIFT, 4 << FP_Q16_SHIFT, -(5 << FP_Q16_SHIFT)]
    out_ok = [0, 0, 0]
    status_ok = fpq16_mul_checked_no_partial_array(lhs_ok, rhs_ok, out_ok, 3)
    assert status_ok == FP_Q16_OK
    assert out_ok == [3 << FP_Q16_SHIFT, -(8 << FP_Q16_SHIFT), -(15 << FP_Q16_SHIFT)]


def test_randomized_reference_parity_adversarial_sign_saturation_count() -> None:
    rng = random.Random(20260421_964)

    for _ in range(6000):
        count = rng.randint(0, 64)
        lhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        rhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]

        expected_status, expected_out = explicit_reference_array(lhs, rhs, count)

        out = [0x12345678] * count
        status = fpq16_mul_checked_no_partial_array(lhs.copy(), rhs.copy(), out, count)
        assert status == expected_status
        if expected_status == FP_Q16_OK:
            assert out == expected_out
        else:
            assert out == [0x12345678] * count


def run() -> None:
    test_source_contains_mul_checked_no_partial_array_contract()
    test_null_ptr_bad_count_and_alias_rejection_no_writes()
    test_known_vectors_rounding_and_overflow_no_partial_commit()
    test_randomized_reference_parity_adversarial_sign_saturation_count()
    print("fixedpoint_q16_mul_checked_no_partial_array=ok")


if __name__ == "__main__":
    run()
