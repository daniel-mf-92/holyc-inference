#!/usr/bin/env python3
"""Reference checks for FPQ16MulShiftRoundCheckedNoPartialArray (IQ-1148)."""

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


def fpq16_mul_shift_round_checked_reference(lhs: int, rhs: int, shift: int) -> tuple[int, int]:
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
        signed_limit = I64_MAX_VALUE if not is_negative else (1 << 63)
        if quotient >= signed_limit:
            return FP_Q16_ERR_OVERFLOW, 0
        quotient += 1

    return fp_try_apply_sign_from_u64_checked(quotient, is_negative)


def fpq16_mul_shift_round_checked_nopartial_array_reference(
    lhs_values: list[int] | None,
    lhs_capacity: int,
    rhs_values: list[int] | None,
    rhs_capacity: int,
    out_values: list[int] | None,
    out_capacity: int,
    count: int,
    shift: int,
    *,
    alias_lhs_rhs: bool = False,
    alias_lhs_out: bool = False,
    alias_rhs_out: bool = False,
) -> int:
    if lhs_values is None or rhs_values is None or out_values is None:
        return FP_Q16_ERR_NULL_PTR

    if lhs_capacity < 0 or rhs_capacity < 0 or out_capacity < 0 or count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if alias_lhs_rhs or alias_lhs_out or alias_rhs_out:
        return FP_Q16_ERR_BAD_PARAM

    if count > lhs_capacity or count > rhs_capacity or count > out_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if count > len(lhs_values) or count > len(rhs_values) or count > len(out_values):
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK

    for i in range(count):
        status, _ = fpq16_mul_shift_round_checked_reference(lhs_values[i], rhs_values[i], shift)
        if status != FP_Q16_OK:
            return status

    for i in range(count):
        status, lane = fpq16_mul_shift_round_checked_reference(lhs_values[i], rhs_values[i], shift)
        if status != FP_Q16_OK:
            return status
        out_values[i] = lane

    return FP_Q16_OK


def explicit_expected(lhs_values: list[int], rhs_values: list[int], count: int, shift: int) -> tuple[int, list[int]]:
    out = [0] * count
    for i in range(count):
        status, lane = fpq16_mul_shift_round_checked_reference(lhs_values[i], rhs_values[i], shift)
        if status != FP_Q16_OK:
            return status, [0] * count
        out[i] = lane
    return FP_Q16_OK, out


def test_source_contains_iq1148_function_and_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulShiftRoundCheckedNoPartialArray(I64 *lhs_values,"
    assert sig in source
    body = source.split(sig, 1)[1].split("// Geometry diagnostics helper for FPQ16MulCheckedNoPartialArray.", 1)[0]
    assert "if (lhs_values == rhs_values || lhs_values == out_values || rhs_values == out_values)" in body
    assert "required_lhs = snapshot_count;" in body
    assert "if (required_lhs > snapshot_lhs_capacity ||" in body
    assert "status = FPQ16MulShiftRoundChecked(lhs_values[i]," in body
    assert "if (snapshot_count != count ||" in body


def test_null_alias_and_capacity_guards_no_writes() -> None:
    out = [111, 222, 333]

    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        None,
        3,
        [1, 2, 3],
        3,
        out,
        3,
        3,
        2,
    )
    assert status == FP_Q16_ERR_NULL_PTR
    assert out == [111, 222, 333]

    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        [1, 2, 3],
        2,
        [1, 2, 3],
        3,
        out,
        3,
        3,
        2,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [111, 222, 333]

    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        [1, 2, 3],
        3,
        [1, 2, 3],
        3,
        out,
        3,
        3,
        2,
        alias_lhs_out=True,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [111, 222, 333]


def test_overflow_in_preflight_preserves_output() -> None:
    lhs = [1, I64_MAX_VALUE]
    rhs = [1, I64_MAX_VALUE]
    out = [0x55, 0x66]

    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        lhs,
        2,
        rhs,
        2,
        out,
        2,
        2,
        0,
    )
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == [0x55, 0x66]


def test_known_vectors_and_randomized_parity() -> None:
    lhs = [3, -3, 6, I64_MIN_VALUE]
    rhs = [1, 1, 1, 1]
    out = [0, 0, 0, 0]
    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        lhs,
        4,
        rhs,
        4,
        out,
        4,
        4,
        1,
    )
    assert status == FP_Q16_OK
    assert out == [2, -2, 3, -(1 << 62)]

    rng = random.Random(20260422_1148)
    for _ in range(12000):
        count = rng.randint(0, 48)
        shift = rng.randint(0, 62)
        lhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        rhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        out = [0xA5A5A5A5] * count

        expected_status, expected_out = explicit_expected(lhs, rhs, count, shift)
        status = fpq16_mul_shift_round_checked_nopartial_array_reference(
            lhs,
            count,
            rhs,
            count,
            out,
            count,
            count,
            shift,
        )

        assert status == expected_status
        if status == FP_Q16_OK:
            assert out == expected_out
        else:
            assert out == [0xA5A5A5A5] * count


def run() -> None:
    test_source_contains_iq1148_function_and_contract()
    test_null_alias_and_capacity_guards_no_writes()
    test_overflow_in_preflight_preserves_output()
    test_known_vectors_and_randomized_parity()
    print("fixedpoint_q16_mul_shift_round_checked_nopartial_array=ok")


if __name__ == "__main__":
    run()
