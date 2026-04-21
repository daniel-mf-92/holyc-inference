#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly (IQ-910)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1

FP_Q16_ONE = 1 << 16
EXP_LN2_Q16 = 45_426
EXP_HALF_LN2_Q16 = 22_713


def _i64_wrap(value: int) -> int:
    value &= U64_MAX_VALUE
    if value >= (1 << 63):
        value -= (1 << 64)
    return value


def _u64(value: int) -> int:
    return value & U64_MAX_VALUE


def fp_address_ranges_overlap(a_base: int, a_end_exclusive: int, b_base: int, b_end_exclusive: int) -> int:
    if a_end_exclusive <= a_base:
        return 0
    if b_end_exclusive <= b_base:
        return 0
    if a_end_exclusive <= b_base:
        return 0
    if b_end_exclusive <= a_base:
        return 0
    return 1


def fp_array_i64_span_checked(base_addr: int | None, count: int) -> tuple[int, int, int]:
    if base_addr is None:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if count > (U64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    byte_count = _u64(count << 3)
    if _u64(base_addr) > (U64_MAX_VALUE - byte_count):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    base = _u64(base_addr)
    return FP_Q16_OK, base, _u64(base + byte_count)


def fpq16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    product = a_q16 * b_q16
    rounded = product + (1 << 15) if product >= 0 else product - (1 << 15)
    shifted = rounded >> 16
    if shifted < -(1 << 63) or shifted > I64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, 0
    return FP_Q16_OK, _i64_wrap(shifted)


def fpq16_exp_approx_range_reduce_checked(x_q16: int) -> tuple[int, int, int]:
    x_q16 = _i64_wrap(x_q16)

    if x_q16 >= 0:
        k = (x_q16 + EXP_HALF_LN2_Q16) // EXP_LN2_Q16
    else:
        k = -(((-x_q16) + EXP_HALF_LN2_Q16) // EXP_LN2_Q16)

    abs_k = abs(k)
    if abs_k > (U64_MAX_VALUE // EXP_LN2_Q16):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    k_ln2_q16 = _i64_wrap(k * EXP_LN2_Q16)
    residual_q16 = _i64_wrap(x_q16 - k_ln2_q16)

    if residual_q16 < -EXP_HALF_LN2_Q16 or residual_q16 > EXP_HALF_LN2_Q16:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    return FP_Q16_OK, _i64_wrap(k), residual_q16


def fpq16_exp_approx_checked(x_q16: int) -> tuple[int, int]:
    status, k, residual = fpq16_exp_approx_range_reduce_checked(x_q16)
    if status != FP_Q16_OK:
        return status, 0

    terms = [FP_Q16_ONE, residual]

    status, r2 = fpq16_mul_checked(residual, residual)
    if status != FP_Q16_OK:
        return status, 0
    terms.append(r2 // 2)

    status, r3 = fpq16_mul_checked(r2, residual)
    if status != FP_Q16_OK:
        return status, 0
    terms.append(r3 // 6)

    status, r4 = fpq16_mul_checked(r2, r2)
    if status != FP_Q16_OK:
        return status, 0
    terms.append(r4 // 24)

    poly_q16 = _i64_wrap(sum(terms))

    if k >= 0:
        if k >= 63:
            return FP_Q16_ERR_OVERFLOW, I64_MAX_VALUE
        out = poly_q16 << k
        if out > I64_MAX_VALUE:
            return FP_Q16_ERR_OVERFLOW, I64_MAX_VALUE
        if out < 0:
            return FP_Q16_OK, 0
        return FP_Q16_OK, _i64_wrap(out)

    shift_mag = -k
    if shift_mag >= 63:
        return FP_Q16_OK, 0
    out = poly_q16 >> shift_mag
    if out < 0:
        return FP_Q16_OK, 0
    return FP_Q16_OK, _i64_wrap(out)


def fpq16_exp_approx_checked_no_partial_array(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    x_addr: int = 0x1000,
    out_addr: int = 0x3000,
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

    status, x_base, x_end = fp_array_i64_span_checked(x_addr, count)
    if status != FP_Q16_OK:
        return status
    status, out_base, out_end = fp_array_i64_span_checked(out_addr, count)
    if status != FP_Q16_OK:
        return status

    if fp_address_ranges_overlap(x_base, x_end, out_base, out_end):
        return FP_Q16_ERR_BAD_PARAM

    staged: list[int] = []
    for i in range(count):
        status, lane = fpq16_exp_approx_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status
        staged.append(lane)

    for i in range(count):
        out_q16[i] = staged[i]

    return FP_Q16_OK


def fpq16_exp_approx_checked_no_partial_array_required_bytes(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_slot: list[int] | None,
    x_addr: int = 0x1000,
    out_addr: int = 0x3000,
    out_required_addr: int = 0x5000,
) -> int:
    if x_q16 is None or out_q16 is None or out_required_output_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if out_required_output_bytes_slot is x_q16 or out_required_output_bytes_slot is out_q16:
        return FP_Q16_ERR_BAD_PARAM

    if count:
        status, x_base, x_end = fp_array_i64_span_checked(x_addr, count)
        if status != FP_Q16_OK:
            return status
        status, out_base, out_end = fp_array_i64_span_checked(out_addr, count)
        if status != FP_Q16_OK:
            return status

        if out_required_addr > (U64_MAX_VALUE - 7):
            return FP_Q16_ERR_OVERFLOW
        required_end = out_required_addr + 8

        if (
            fp_address_ranges_overlap(out_required_addr, required_end, x_base, x_end)
            or fp_address_ranges_overlap(out_required_addr, required_end, out_base, out_end)
        ):
            return FP_Q16_ERR_BAD_PARAM

    status = fpq16_exp_approx_checked_no_partial_array(x_q16, out_q16, count, x_addr=x_addr, out_addr=out_addr)
    if status != FP_Q16_OK:
        return status

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    out_required_output_bytes_slot[0] = count << 3
    return FP_Q16_OK


def fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_slot: list[int] | None,
) -> int:
    if out_required_output_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR

    snapshot_x_q16 = x_q16
    snapshot_out_q16 = out_q16
    snapshot_count = count

    staged_required_output_bytes = [0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
        x_q16,
        out_q16,
        count,
        staged_required_output_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if snapshot_x_q16 is not x_q16:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_out_q16 is not out_q16:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_count != count:
        return FP_Q16_ERR_BAD_PARAM

    out_required_output_bytes_slot[0] = staged_required_output_bytes[0]
    return FP_Q16_OK


def explicit_commit_only_composition(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_slot: list[int] | None,
) -> int:
    if out_required_output_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR

    staged = [0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
        x_q16,
        out_q16,
        count,
        staged,
    )
    if status != FP_Q16_OK:
        return status

    out_required_output_bytes_slot[0] = staged[0]
    return FP_Q16_OK


def test_source_contains_iq910_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes(" in body
    assert "if (snapshot_x_q16 != x_q16)" in body
    assert "if (snapshot_out_q16 != out_q16)" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_null_bad_param_paths() -> None:
    x = [0, FP_Q16_ONE]
    out = [0x1111, 0x2222]
    required = [0x9999]

    assert fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(x, out, 2, None) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(None, out, 1, required) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(x, None, 1, required) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(x, out, -1, required) == FP_Q16_ERR_BAD_PARAM


def test_no_publish_on_hard_failure() -> None:
    x = [0, I64_MAX_VALUE, 1]
    out = [0xAAAA, 0xBBBB, 0xCCCC]
    required = [0xDDDD]

    out_before = out.copy()
    req_before = required[0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(x, out, len(x), required)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == out_before
    assert required[0] == req_before


def test_zero_count_commit() -> None:
    x: list[int] = []
    out: list[int] = []
    required = [0x7777]

    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(x, out, 0, required)
    assert status == FP_Q16_OK
    assert required[0] == 0


def test_known_vectors_parity() -> None:
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

    out_impl = [0x1111] * len(x)
    out_ref = [0x1111] * len(x)
    req_impl = [0x2222]
    req_ref = [0x2222]

    got = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(
        x,
        out_impl,
        len(x),
        req_impl,
    )
    exp = explicit_commit_only_composition(
        x,
        out_ref,
        len(x),
        req_ref,
    )

    assert got == exp
    assert out_impl == out_ref
    assert req_impl == req_ref


def test_randomized_parity() -> None:
    rng = random.Random(20260421_910)

    for _ in range(3000):
        count = rng.randint(0, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(count)]

        out_expected = [0x4141] * count
        out_got = [0x4141] * count
        required_expected = [0x5151]
        required_got = [0x5151]

        expected_status = explicit_commit_only_composition(
            x,
            out_expected,
            count,
            required_expected,
        )
        got_status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(
            x,
            out_got,
            count,
            required_got,
        )

        assert got_status == expected_status
        assert out_got == out_expected
        assert required_got == required_expected


def run() -> None:
    test_source_contains_iq910_function()
    test_null_bad_param_paths()
    test_no_publish_on_hard_failure()
    test_zero_count_commit()
    test_known_vectors_parity()
    test_randomized_parity()
    print("fixedpoint_q16_exp_approx_checked_no_partial_array_required_bytes_commit_only=ok")


if __name__ == "__main__":
    run()
