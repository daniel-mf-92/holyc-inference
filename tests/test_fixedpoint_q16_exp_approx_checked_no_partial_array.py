#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxCheckedNoPartialArray (IQ-905)."""

from __future__ import annotations

import random
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


def _span_checked(base_addr: int | None, count: int) -> tuple[int, int, int]:
    if base_addr is None:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if count > (U64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    byte_count = (count << 3) & U64_MAX_VALUE
    base = base_addr & U64_MAX_VALUE

    if base > (U64_MAX_VALUE - byte_count):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    return FP_Q16_OK, base, (base + byte_count) & U64_MAX_VALUE


def _ranges_overlap(a_base: int, a_end: int, b_base: int, b_end: int) -> int:
    if a_end <= a_base:
        return 0
    if b_end <= b_base:
        return 0
    if a_end <= b_base:
        return 0
    if b_end <= a_base:
        return 0
    return 1


def fpq16_exp_approx_checked_no_partial_array(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    x_addr: int = 0x1000,
    out_addr: int = 0x4000,
) -> int:
    if x_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if x_q16 is out_q16:
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK
    if count > len(x_q16) or count > len(out_q16):
        return FP_Q16_ERR_BAD_PARAM

    status, x_base, x_end = _span_checked(x_addr, count)
    if status != FP_Q16_OK:
        return status
    status, out_base, out_end = _span_checked(out_addr, count)
    if status != FP_Q16_OK:
        return status

    if _ranges_overlap(x_base, x_end, out_base, out_end):
        return FP_Q16_ERR_BAD_PARAM

    for i in range(count):
        status, _, _ = fpq16_exp_approx_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status

    for i in range(count):
        status, lane_out, _ = fpq16_exp_approx_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status
        out_q16[i] = lane_out

    return FP_Q16_OK


def test_source_contains_iq905_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxCheckedNoPartialArray(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "if (x_q16 == out_q16)" in body
    assert "status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);" in body
    assert "if (FPAddressRangesOverlap(x_base, x_end, out_base, out_end))" in body
    assert "status = FPQ16ExpApproxChecked(x_q16[i], &staged_lane_q16);" in body


def test_null_bad_count_alias_and_overlap_guards() -> None:
    x = [0, FP_Q16_ONE]
    out = [0x1111, 0x2222]

    assert fpq16_exp_approx_checked_no_partial_array(None, out, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array(x, None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array(x, out, -1) == FP_Q16_ERR_BAD_PARAM
    assert fpq16_exp_approx_checked_no_partial_array(x, x, 1) == FP_Q16_ERR_BAD_PARAM

    before = out.copy()
    err = fpq16_exp_approx_checked_no_partial_array(
        x,
        out,
        2,
        x_addr=0x1000,
        out_addr=0x1008,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == before


def test_no_partial_on_overflow_lane() -> None:
    x = [0, I64_MAX_VALUE, 1]
    out = [0xAAAA, 0xBBBB, 0xCCCC]
    before = out.copy()

    err = fpq16_exp_approx_checked_no_partial_array(x, out, len(x))
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == before


def test_known_vectors_match_scalar() -> None:
    x = [
        -(8 * FP_Q16_ONE),
        -(4 * FP_Q16_ONE),
        -FP_Q16_ONE,
        -1,
        0,
        1,
        FP_Q16_ONE,
        4 * FP_Q16_ONE,
        8 * FP_Q16_ONE,
    ]
    out = [0x5555] * len(x)

    err = fpq16_exp_approx_checked_no_partial_array(x, out, len(x))
    assert err == FP_Q16_OK

    for i, lane_x in enumerate(x):
        lane_err, lane_out, lane_wrote = fpq16_exp_approx_checked(lane_x)
        assert lane_err == FP_Q16_OK
        assert lane_wrote is True
        assert out[i] == lane_out


def test_randomized_parity() -> None:
    rng = random.Random(20260421_905)

    for _ in range(3000):
        count = rng.randint(1, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(count)]

        out_expected = [0x7878] * count
        out_got = [0x7878] * count

        expected_status = fpq16_exp_approx_checked_no_partial_array(
            x,
            out_expected,
            count,
            x_addr=0x1000,
            out_addr=0x6000,
        )
        got_status = fpq16_exp_approx_checked_no_partial_array(
            x,
            out_got,
            count,
            x_addr=0x1000,
            out_addr=0x6000,
        )

        assert got_status == expected_status
        assert out_got == out_expected


def run() -> None:
    test_source_contains_iq905_function()
    test_null_bad_count_alias_and_overlap_guards()
    test_no_partial_on_overflow_lane()
    test_known_vectors_match_scalar()
    test_randomized_parity()
    print("fixedpoint_q16_exp_approx_checked_no_partial_array=ok")


if __name__ == "__main__":
    run()
