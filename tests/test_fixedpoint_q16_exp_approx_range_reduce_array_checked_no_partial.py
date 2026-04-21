#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxRangeReduceCheckedNoPartialArray (IQ-902)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1

EXP_LN2_Q16 = 45_426
EXP_HALF_LN2_Q16 = 22_713


def _i64_wrap(value: int) -> int:
    value &= U64_MAX_VALUE
    if value >= (1 << 63):
        value -= (1 << 64)
    return value


def _u64(value: int) -> int:
    return value & U64_MAX_VALUE


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


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


def fpq16_exp_approx_range_reduce_checked(
    x_q16: int,
    out_k_present: bool = True,
    out_r_present: bool = True,
    alias_outputs: bool = False,
) -> tuple[int, int, int]:
    if not out_k_present or not out_r_present:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if alias_outputs:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    x_q16 = _i64_wrap(x_q16)

    if x_q16 >= 0:
        k = (x_q16 + EXP_HALF_LN2_Q16) // EXP_LN2_Q16
    else:
        k = -(((-x_q16) + EXP_HALF_LN2_Q16) // EXP_LN2_Q16)

    abs_k_u64 = fp_abs_to_u64(k)
    if abs_k_u64 > (U64_MAX_VALUE // EXP_LN2_Q16):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    if k >= 0:
        k_ln2_q16 = _i64_wrap(abs_k_u64 * EXP_LN2_Q16)
    else:
        k_ln2_q16 = _i64_wrap(-(abs_k_u64 * EXP_LN2_Q16))

    if (k_ln2_q16 > 0 and x_q16 < _i64_wrap(-(1 << 63) + k_ln2_q16)) or (
        k_ln2_q16 < 0 and x_q16 > _i64_wrap(I64_MAX_VALUE + k_ln2_q16)
    ):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    residual_q16 = _i64_wrap(x_q16 - k_ln2_q16)

    if residual_q16 < -EXP_HALF_LN2_Q16 or residual_q16 > EXP_HALF_LN2_Q16:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    return FP_Q16_OK, _i64_wrap(k), residual_q16


def fpq16_exp_approx_range_reduce_checked_no_partial_array(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    x_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
) -> int:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_k or x_q16 is out_r_q16 or out_k is out_r_q16:
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK
    if count > len(x_q16) or count > len(out_k) or count > len(out_r_q16):
        return FP_Q16_ERR_BAD_PARAM

    status, x_base, x_end = fp_array_i64_span_checked(x_addr, count)
    if status != FP_Q16_OK:
        return status
    status, k_base, k_end = fp_array_i64_span_checked(out_k_addr, count)
    if status != FP_Q16_OK:
        return status
    status, r_base, r_end = fp_array_i64_span_checked(out_r_addr, count)
    if status != FP_Q16_OK:
        return status

    if (
        fp_address_ranges_overlap(x_base, x_end, k_base, k_end)
        or fp_address_ranges_overlap(x_base, x_end, r_base, r_end)
        or fp_address_ranges_overlap(k_base, k_end, r_base, r_end)
    ):
        return FP_Q16_ERR_BAD_PARAM

    for i in range(count):
        status, _, _ = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status

    for i in range(count):
        status, lane_k, lane_r = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status
        out_k[i] = lane_k
        out_r_q16[i] = lane_r

    return FP_Q16_OK


def test_source_contains_iq902_no_partial_array_helper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArray(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16ExpApproxChecked", 1)[0]
    assert "if (x_q16 == out_k || x_q16 == out_r_q16 || out_k == out_r_q16)" in body
    assert "status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);" in body
    assert "FPAddressRangesOverlap(x_base, x_end, k_base, k_end)" in body
    assert "status = FPQ16ExpApproxRangeReduceChecked(x_q16[i]," in body


def test_null_bad_count_alias_and_overlap_guards() -> None:
    x = [0, 1]
    out_k = [0x11, 0x22]
    out_r = [0x33, 0x44]

    assert fpq16_exp_approx_range_reduce_checked_no_partial_array(None, out_k, out_r, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_range_reduce_checked_no_partial_array(x, None, out_r, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_range_reduce_checked_no_partial_array(x, out_k, None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_range_reduce_checked_no_partial_array(x, out_k, out_r, -1) == FP_Q16_ERR_BAD_PARAM
    assert fpq16_exp_approx_range_reduce_checked_no_partial_array(x, x, out_r, 1) == FP_Q16_ERR_BAD_PARAM

    before_k = out_k.copy()
    before_r = out_r.copy()
    err = fpq16_exp_approx_range_reduce_checked_no_partial_array(
        x,
        out_k,
        out_r,
        2,
        x_addr=0x1000,
        out_k_addr=0x1008,
        out_r_addr=0x3000,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_k == before_k
    assert out_r == before_r


def test_known_vectors_and_no_partial_commit() -> None:
    x = [
        -(1 << 63) + 1,
        -4_194_304,
        -65_536,
        -1,
        0,
        1,
        65_536,
        4_194_304,
        (1 << 63) - 1,
    ]
    out_k = [0xAAAA] * len(x)
    out_r = [0xBBBB] * len(x)

    err = fpq16_exp_approx_range_reduce_checked_no_partial_array(x, out_k, out_r, len(x))
    assert err == FP_Q16_OK

    for i, x_q16 in enumerate(x):
        lane_status, lane_k, lane_r = fpq16_exp_approx_range_reduce_checked(x_q16)
        assert lane_status == FP_Q16_OK
        assert out_k[i] == lane_k
        assert out_r[i] == lane_r
        assert -EXP_HALF_LN2_Q16 <= out_r[i] <= EXP_HALF_LN2_Q16
        assert _i64_wrap((out_k[i] * EXP_LN2_Q16) + out_r[i]) == _i64_wrap(x_q16)


def test_randomized_parity() -> None:
    rng = random.Random(20260421_902)

    for _ in range(3000):
        count = rng.randint(1, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(count)]
        out_k = [0x7A7A] * count
        out_r = [0x5B5B] * count

        expected_k = [0x7A7A] * count
        expected_r = [0x5B5B] * count
        expected_status = fpq16_exp_approx_range_reduce_checked_no_partial_array(
            x,
            expected_k,
            expected_r,
            count,
            x_addr=0x1000,
            out_k_addr=0x4000,
            out_r_addr=0x8000,
        )

        got_status = fpq16_exp_approx_range_reduce_checked_no_partial_array(
            x,
            out_k,
            out_r,
            count,
            x_addr=0x1000,
            out_k_addr=0x4000,
            out_r_addr=0x8000,
        )

        assert got_status == expected_status
        assert out_k == expected_k
        assert out_r == expected_r


def run() -> None:
    test_source_contains_iq902_no_partial_array_helper()
    test_null_bad_count_alias_and_overlap_guards()
    test_known_vectors_and_no_partial_commit()
    test_randomized_parity()
    print("fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial=ok")


if __name__ == "__main__":
    run()
