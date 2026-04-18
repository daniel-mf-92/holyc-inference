#!/usr/bin/env python3
"""Reference checks for FPQ16ExpMulPow2Checked semantics."""

from __future__ import annotations

import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
EXP_Q16_RIGHT_SHIFT_ZERO_K = -63


def fpq16_exp_mul_pow2_checked(
    base_q16: int,
    k: int,
    out_present: bool = True,
) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if base_q16 < 0:
        return FP_Q16_ERR_BAD_PARAM, 0

    if base_q16 == 0:
        return FP_Q16_OK, 0

    if k >= 0:
        if k >= 63:
            return FP_Q16_ERR_OVERFLOW, 0
        if base_q16 > (I64_MAX_VALUE >> k):
            return FP_Q16_ERR_OVERFLOW, 0
        return FP_Q16_OK, base_q16 << k

    if k <= EXP_Q16_RIGHT_SHIFT_ZERO_K:
        return FP_Q16_OK, 0

    right_shift = -k
    if right_shift >= 63:
        return FP_Q16_OK, 0

    return FP_Q16_OK, base_q16 >> right_shift


def test_null_pointer_surface() -> None:
    err, out = fpq16_exp_mul_pow2_checked(1 << 16, 0, out_present=False)
    assert err == FP_Q16_ERR_NULL_PTR
    assert out == 0


def test_negative_base_rejected() -> None:
    err, out = fpq16_exp_mul_pow2_checked(-1, 3)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == 0


def test_zero_preserving_semantics() -> None:
    for k in (-100, -63, -10, -1, 0, 1, 20, 62, 100):
        err, out = fpq16_exp_mul_pow2_checked(0, k)
        assert err == FP_Q16_OK
        assert out == 0


def test_non_negative_scaling_examples() -> None:
    vectors = [
        (1, 0),
        (1, 5),
        (12345, 7),
        ((1 << 16), 1),
        (I64_MAX_VALUE >> 20, 20),
    ]

    for base_q16, k in vectors:
        err, out = fpq16_exp_mul_pow2_checked(base_q16, k)
        assert err == FP_Q16_OK
        assert out == (base_q16 << k)


def test_overflow_guards_for_positive_shift() -> None:
    err, _ = fpq16_exp_mul_pow2_checked(1, 63)
    assert err == FP_Q16_ERR_OVERFLOW

    err, _ = fpq16_exp_mul_pow2_checked(I64_MAX_VALUE, 1)
    assert err == FP_Q16_ERR_OVERFLOW


def test_negative_shift_domain_and_underflow_to_zero() -> None:
    base_q16 = 123456789

    for k in (-63, -64, -128):
        err, out = fpq16_exp_mul_pow2_checked(base_q16, k)
        assert err == FP_Q16_OK
        assert out == 0

    err, out = fpq16_exp_mul_pow2_checked(base_q16, -1)
    assert err == FP_Q16_OK
    assert out == (base_q16 >> 1)

    err, out = fpq16_exp_mul_pow2_checked(base_q16, -7)
    assert err == FP_Q16_OK
    assert out == (base_q16 >> 7)


def test_roundtrip_consistency_when_no_overflow() -> None:
    rng = random.Random(20260419_444)

    for _ in range(10000):
        base_q16 = rng.randint(0, I64_MAX_VALUE)
        k = rng.randint(-70, 62)

        err, out = fpq16_exp_mul_pow2_checked(base_q16, k)

        if base_q16 == 0:
            assert err == FP_Q16_OK
            assert out == 0
            continue

        if k >= 63:
            assert err == FP_Q16_ERR_OVERFLOW
            continue

        if k >= 0 and base_q16 > (I64_MAX_VALUE >> k):
            assert err == FP_Q16_ERR_OVERFLOW
            continue

        assert err == FP_Q16_OK
        if k >= 0:
            assert out == (base_q16 << k)
        elif k <= EXP_Q16_RIGHT_SHIFT_ZERO_K:
            assert out == 0
        else:
            assert out == (base_q16 >> (-k))


def run() -> None:
    test_null_pointer_surface()
    test_negative_base_rejected()
    test_zero_preserving_semantics()
    test_non_negative_scaling_examples()
    test_overflow_guards_for_positive_shift()
    test_negative_shift_domain_and_underflow_to_zero()
    test_roundtrip_consistency_when_no_overflow()
    print("intexp_q16_mul_pow2_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
