#!/usr/bin/env python3
"""Parity harness for FPQ16MacSatCheckedNoPartialArrayRequiredBytes (IQ-895)."""

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


def fpq16_mac_sat_checked(acc_q16: int, a_q16: int, b_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    mul_status, mul_q16 = fpq16_mul_sat_checked(a_q16, b_q16, out_present=True)

    if mul_q16 > 0 and acc_q16 > I64_MAX_VALUE - mul_q16:
        return FP_Q16_ERR_OVERFLOW, I64_MAX_VALUE
    if mul_q16 < 0 and acc_q16 < I64_MIN_VALUE - mul_q16:
        return FP_Q16_ERR_OVERFLOW, I64_MIN_VALUE

    sum_q16 = acc_q16 + mul_q16
    if mul_status != FP_Q16_OK:
        return FP_Q16_ERR_OVERFLOW, sum_q16
    return FP_Q16_OK, sum_q16


def fpq16_mac_sat_checked_no_partial_array(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
) -> int:
    if acc_q16 is None or a_q16 is None or b_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if (
        acc_q16 is a_q16
        or acc_q16 is b_q16
        or acc_q16 is out_q16
        or a_q16 is b_q16
        or a_q16 is out_q16
        or b_q16 is out_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK

    if count > len(acc_q16) or count > len(a_q16) or count > len(b_q16) or count > len(out_q16):
        return FP_Q16_ERR_BAD_PARAM

    for i in range(count):
        err, _ = fpq16_mac_sat_checked(acc_q16[i], a_q16[i], b_q16[i], out_present=True)
        if err not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
            return err

    overflow_seen = False
    for i in range(count):
        err, lane = fpq16_mac_sat_checked(acc_q16[i], a_q16[i], b_q16[i], out_present=True)
        out_q16[i] = lane
        if err == FP_Q16_ERR_OVERFLOW:
            overflow_seen = True
        elif err != FP_Q16_OK:
            return err

    if overflow_seen:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def fpq16_mac_sat_checked_no_partial_array_required_bytes(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_present: bool,
) -> tuple[int, int]:
    if not out_required_output_bytes_present:
        return FP_Q16_ERR_NULL_PTR, -1

    status = fpq16_mac_sat_checked_no_partial_array(acc_q16, a_q16, b_q16, out_q16, count)
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status, -1

    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, -1

    required_output_bytes = count << 3
    return status, required_output_bytes


def test_source_contains_required_bytes_helper_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MacSatCheckedNoPartialArrayRequiredBytes(I64 *acc_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16DivArrayChecked", 1)[0]
    assert "FPQ16MacSatCheckedNoPartialArray(acc_q16," in body
    assert "if (!acc_q16 || !a_q16 || !b_q16 || !out_q16 || !out_required_output_bytes)" in body
    assert "if (out_required_output_bytes == acc_q16 ||" in body
    assert "status = FPArrayI64SpanChecked(acc_q16, count, &acc_base, &acc_end);" in body
    assert "if (FPAddressRangesOverlap(required_base, required_end, acc_base, acc_end) ||" in body
    assert "if (status != FP_Q16_OK && status != FP_Q16_ERR_OVERFLOW)" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_null_required_bytes_ptr_and_hard_fail_no_write() -> None:
    acc = [1 << FP_Q16_SHIFT]
    a = [2 << FP_Q16_SHIFT]
    b = [3 << FP_Q16_SHIFT]
    out = [0x55]

    status, required = fpq16_mac_sat_checked_no_partial_array_required_bytes(
        acc, a, b, out, 1, out_required_output_bytes_present=False
    )
    assert status == FP_Q16_ERR_NULL_PTR
    assert required == -1
    assert out == [0x55]

    before = out.copy()
    status, required = fpq16_mac_sat_checked_no_partial_array_required_bytes(
        acc, a, b, out, 2, out_required_output_bytes_present=True
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert required == -1
    assert out == before


def test_known_vectors_and_required_bytes_publication() -> None:
    acc = [1 << FP_Q16_SHIFT, -(3 << FP_Q16_SHIFT), I64_MAX_VALUE]
    a = [2 << FP_Q16_SHIFT, 5 << FP_Q16_SHIFT, I64_MAX_VALUE]
    b = [4 << FP_Q16_SHIFT, 2 << FP_Q16_SHIFT, I64_MAX_VALUE]
    out = [0x11, 0x22, 0x33]

    status, required = fpq16_mac_sat_checked_no_partial_array_required_bytes(
        acc, a, b, out, 3, out_required_output_bytes_present=True
    )

    assert status == FP_Q16_ERR_OVERFLOW
    assert required == 24
    assert out[0] == 9 << FP_Q16_SHIFT
    assert out[1] == 7 << FP_Q16_SHIFT
    assert out[2] == I64_MAX_VALUE


def test_randomized_reference_parity() -> None:
    rng = random.Random(20260421_895)

    for _ in range(5000):
        count = rng.randint(1, 64)
        acc = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        a = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        b = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]

        exp_status, exp_required = fpq16_mac_sat_checked_no_partial_array_required_bytes(
            acc.copy(), a.copy(), b.copy(), [0x7A7A] * count, count, out_required_output_bytes_present=True
        )

        out = [0x7A7A] * count
        got_status, got_required = fpq16_mac_sat_checked_no_partial_array_required_bytes(
            acc.copy(), a.copy(), b.copy(), out, count, out_required_output_bytes_present=True
        )

        assert got_status == exp_status
        assert got_required == exp_required


def run() -> None:
    test_source_contains_required_bytes_helper_contract()
    test_null_required_bytes_ptr_and_hard_fail_no_write()
    test_known_vectors_and_required_bytes_publication()
    test_randomized_reference_parity()
    print("fixedpoint_q16_mac_sat_checked_no_partial_array_required_bytes=ok")


if __name__ == "__main__":
    run()
