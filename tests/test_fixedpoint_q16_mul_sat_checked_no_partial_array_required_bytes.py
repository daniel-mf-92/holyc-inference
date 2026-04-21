#!/usr/bin/env python3
"""Parity harness for FPQ16MulSatCheckedNoPartialArrayRequiredBytes (IQ-892)."""

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


def fpq16_mul_sat_checked_no_partial_array_required_bytes(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_present: bool,
) -> tuple[int, int]:
    if not out_required_output_bytes_present:
        return FP_Q16_ERR_NULL_PTR, -1

    status = fpq16_mul_sat_checked_no_partial_array(lhs_q16, rhs_q16, out_q16, count)
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status, -1

    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, -1
    if count > (U64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, -1

    required_output_bytes = count << 3
    return status, required_output_bytes


def test_source_contains_required_bytes_helper_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulSatCheckedNoPartialArrayRequiredBytes(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPArrayI64SpanChecked", 1)[0]
    assert "FPQ16MulSatCheckedNoPartialArray(lhs_q16," in body
    assert "if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)" in body
    assert "if (count > (I64_MAX_VALUE >> 3))" in body
    assert "if (out_required_output_bytes == lhs_q16 ||" in body
    assert "if ((required_end > lhs_base && lhs_end > required_base) ||" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_null_required_bytes_ptr() -> None:
    lhs = [1 << FP_Q16_SHIFT]
    rhs = [2 << FP_Q16_SHIFT]
    out = [0x55]

    status, required = fpq16_mul_sat_checked_no_partial_array_required_bytes(
        lhs, rhs, out, 1, out_required_output_bytes_present=False
    )
    assert status == FP_Q16_ERR_NULL_PTR
    assert required == -1
    assert out == [0x55]


def test_known_vectors_and_required_bytes_publication() -> None:
    lhs = [1 << FP_Q16_SHIFT, -(2 << FP_Q16_SHIFT), I64_MAX_VALUE]
    rhs = [3 << FP_Q16_SHIFT, 4 << FP_Q16_SHIFT, I64_MAX_VALUE]
    out = [0x11, 0x22, 0x33]

    status, required = fpq16_mul_sat_checked_no_partial_array_required_bytes(
        lhs, rhs, out, 3, out_required_output_bytes_present=True
    )

    assert status == FP_Q16_ERR_OVERFLOW
    assert required == 24
    assert out[0] == 3 << FP_Q16_SHIFT
    assert out[1] == -(8 << FP_Q16_SHIFT)
    assert out[2] == I64_MAX_VALUE


def test_randomized_reference_parity_and_no_write_on_hard_failures() -> None:
    rng = random.Random(20260421_892)

    for _ in range(5000):
        count = rng.randint(1, 64)
        lhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        rhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        out = [0x7A7A] * count

        exp_status, exp_required = fpq16_mul_sat_checked_no_partial_array_required_bytes(
            lhs.copy(), rhs.copy(), [0x7A7A] * count, count, out_required_output_bytes_present=True
        )

        got_status, got_required = fpq16_mul_sat_checked_no_partial_array_required_bytes(
            lhs.copy(), rhs.copy(), out, count, out_required_output_bytes_present=True
        )

        assert got_status == exp_status
        assert got_required == exp_required

        # Hard failure case: malformed count should not mutate output.
        before = out.copy()
        bad_status = fpq16_mul_sat_checked_no_partial_array(lhs, rhs, out, count + 1)
        assert bad_status == FP_Q16_ERR_BAD_PARAM
        assert out == before


def run() -> None:
    test_source_contains_required_bytes_helper_contract()
    test_null_required_bytes_ptr()
    test_known_vectors_and_required_bytes_publication()
    test_randomized_reference_parity_and_no_write_on_hard_failures()
    print("fixedpoint_q16_mul_sat_checked_no_partial_array_required_bytes=ok")


if __name__ == "__main__":
    run()
