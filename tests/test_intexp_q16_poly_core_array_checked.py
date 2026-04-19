#!/usr/bin/env python3
"""Reference checks for FPQ16ExpPolyCoreLn2ResidualQ16ArrayChecked semantics."""

from __future__ import annotations

import pathlib
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

EXP_Q16_LN2 = 45426
EXP_Q16_POLY_C2_DIV = 2
EXP_Q16_POLY_C3_DIV = 6
EXP_Q16_POLY_C4_DIV = 24

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1


def _abs_i64_to_u64(value: int) -> int:
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
            return FP_Q16_OK, -(1 << 63)
        return FP_Q16_OK, -mag
    return FP_Q16_OK, mag


def fpq16_mul_checked(a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if not a_q16 or not b_q16:
        return FP_Q16_OK, 0

    abs_a = _abs_i64_to_u64(a_q16)
    abs_b = _abs_i64_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63
    if rounded_mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    return _try_apply_sign_checked(rounded_mag, is_negative)


def fpq16_exp_poly_core_ln2_residual_q16_checked(
    residual_q16: int,
    out_present: bool = True,
) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if residual_q16 < 0 or residual_q16 >= EXP_Q16_LN2:
        return FP_Q16_ERR_BAD_PARAM, 0

    err, r2 = fpq16_mul_checked(residual_q16, residual_q16)
    if err != FP_Q16_OK:
        return err, 0
    err, r3 = fpq16_mul_checked(r2, residual_q16)
    if err != FP_Q16_OK:
        return err, 0
    err, r4 = fpq16_mul_checked(r3, residual_q16)
    if err != FP_Q16_OK:
        return err, 0

    t1 = residual_q16
    t2 = r2 // EXP_Q16_POLY_C2_DIV
    t3 = r3 // EXP_Q16_POLY_C3_DIV
    t4 = r4 // EXP_Q16_POLY_C4_DIV

    poly = FP_Q16_ONE
    for term in (t1, t2, t3, t4):
        if poly > I64_MAX_VALUE - term:
            return FP_Q16_ERR_OVERFLOW, 0
        poly += term

    return FP_Q16_OK, poly


def fpq16_exp_poly_core_ln2_residual_q16_array_checked(
    residual_q16: list[int] | None,
    out_poly_q16: list[int] | None,
    count: int,
    *,
    residual_addr: int = 0x1100,
    out_addr: int = 0x2200,
) -> int:
    if residual_q16 is None or out_poly_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if residual_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if out_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(residual_q16) < count or len(out_poly_q16) < count:
        raise ValueError("test harness requires count <= provided buffer lengths")

    # Pass 1: validate without writes.
    for i in range(count):
        err, _ = fpq16_exp_poly_core_ln2_residual_q16_checked(residual_q16[i])
        if err != FP_Q16_OK:
            return err

    # Pass 2: commit after preflight success.
    for i in range(count):
        err, poly = fpq16_exp_poly_core_ln2_residual_q16_checked(residual_q16[i])
        if err != FP_Q16_OK:
            return err
        out_poly_q16[i] = poly

    return FP_Q16_OK


def test_source_contains_helper_and_two_pass_pattern() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpPolyCoreLn2ResidualQ16ArrayChecked" in source
    assert "// Pass 1: validate every lane without caller-buffer writes." in source
    assert "// Pass 2: deterministic commit after successful preflight." in source


def test_null_and_bad_count_surfaces() -> None:
    lanes = [0, 1]
    out = [111, 222]

    assert fpq16_exp_poly_core_ln2_residual_q16_array_checked(None, out, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_poly_core_ln2_residual_q16_array_checked(lanes, None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_poly_core_ln2_residual_q16_array_checked(lanes, out, -1) == FP_Q16_ERR_BAD_PARAM


def test_zero_count_no_write() -> None:
    lanes = [10, 20, 30]
    out = [777, 888, 999]
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(lanes, out, 0)
    assert err == FP_Q16_OK
    assert out == [777, 888, 999]


def test_pointer_span_overflow_surfaces() -> None:
    huge_count = (I64_MAX_VALUE >> 3) + 2
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked([0], [0], huge_count)
    assert err == FP_Q16_ERR_OVERFLOW

    count = 4
    near_end = U64_MAX_VALUE - ((count - 1) << 3) + 1

    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(
        [0] * count,
        [0] * count,
        count,
        residual_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW

    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(
        [0] * count,
        [0] * count,
        count,
        out_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW


def test_boundary_and_scalar_composition_parity() -> None:
    residuals = [
        0,
        1,
        2,
        EXP_Q16_LN2 // 2,
        EXP_Q16_LN2 - 2,
        EXP_Q16_LN2 - 1,
    ]

    out = [0] * len(residuals)
    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(residuals, out, len(residuals))
    assert err == FP_Q16_OK

    for i, residual in enumerate(residuals):
        serr, scalar = fpq16_exp_poly_core_ln2_residual_q16_checked(residual)
        assert serr == FP_Q16_OK
        assert out[i] == scalar


def test_no_partial_write_on_invalid_lane() -> None:
    residuals = [10, 20, EXP_Q16_LN2, 30]
    out = [101, 202, 303, 404]

    err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(residuals, out, len(residuals))
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 202, 303, 404]


def test_randomized_parity() -> None:
    rng = random.Random(20260419_473)

    for _ in range(500):
        count = rng.randint(1, 192)
        residuals = [rng.randint(0, EXP_Q16_LN2 - 1) for _ in range(count)]
        out = [-1] * count

        err = fpq16_exp_poly_core_ln2_residual_q16_array_checked(residuals, out, count)
        assert err == FP_Q16_OK

        for i, residual in enumerate(residuals):
            serr, scalar = fpq16_exp_poly_core_ln2_residual_q16_checked(residual)
            assert serr == FP_Q16_OK
            assert out[i] == scalar


def run() -> None:
    test_source_contains_helper_and_two_pass_pattern()
    test_null_and_bad_count_surfaces()
    test_zero_count_no_write()
    test_pointer_span_overflow_surfaces()
    test_boundary_and_scalar_composition_parity()
    test_no_partial_write_on_invalid_lane()
    test_randomized_parity()
    print("intexp_q16_poly_core_array_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
