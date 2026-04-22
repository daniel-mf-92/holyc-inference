#!/usr/bin/env python3
"""Reference checks for FPQ16LnRatioArrayCheckedNoPartial HolyC semantics."""

from __future__ import annotations

from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT
LOG_Q16_LN2 = 45426
LOG_Q16_SPLIT = (FP_Q16_ONE * 3) // 2

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)

INTLOG_STATUS_OK = 0
INTLOG_STATUS_BADPTR = -1
INTLOG_STATUS_BADCOUNT = -2
INTLOG_STATUS_OVERFLOW = -3
INTLOG_STATUS_DOMAIN = -4


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    half = 1 << (shift - 1)
    if value >= 0:
        return (value + half) >> shift
    return -(((-value) + half) >> shift)


def fpq16_mul(a_q16: int, b_q16: int) -> int:
    return round_shift_right_signed(a_q16 * b_q16, FP_Q16_SHIFT)


def fpq16_ln1p_poly(y_q16: int) -> int:
    y2 = fpq16_mul(y_q16, y_q16)
    y3 = fpq16_mul(y2, y_q16)
    y4 = fpq16_mul(y3, y_q16)
    y5 = fpq16_mul(y4, y_q16)
    return y_q16 - (y2 // 2) + (y3 // 3) - (y4 // 4) + (y5 // 5)


def fpq16_ln_reduce(x_q16: int) -> tuple[int, int]:
    if x_q16 <= 0:
        return 0, 0

    m_q16 = x_q16
    k = 0
    while m_q16 >= (FP_Q16_ONE << 1):
        m_q16 >>= 1
        k += 1
    while m_q16 < FP_Q16_ONE:
        m_q16 <<= 1
        k -= 1
    return m_q16, k


def fpq16_ln_ratio(num_q16: int, den_q16: int) -> int:
    if num_q16 <= 0 or den_q16 <= 0:
        return I64_MIN_VALUE

    num_m_q16, num_k = fpq16_ln_reduce(num_q16)
    den_m_q16, den_k = fpq16_ln_reduce(den_q16)

    if num_m_q16 >= LOG_Q16_SPLIT:
        num_m_q16 = (num_m_q16 + 1) >> 1
        num_k += 1
    if den_m_q16 >= LOG_Q16_SPLIT:
        den_m_q16 = (den_m_q16 + 1) >> 1
        den_k += 1

    num_poly_q16 = fpq16_ln1p_poly(num_m_q16 - FP_Q16_ONE)
    den_poly_q16 = fpq16_ln1p_poly(den_m_q16 - FP_Q16_ONE)

    if den_poly_q16 < 0 and num_poly_q16 > I64_MAX_VALUE + den_poly_q16:
        return I64_MAX_VALUE
    if den_poly_q16 > 0 and num_poly_q16 < I64_MIN_VALUE + den_poly_q16:
        return I64_MIN_VALUE
    poly_delta_q16 = num_poly_q16 - den_poly_q16

    if den_k < 0 and num_k > I64_MAX_VALUE + den_k:
        return I64_MAX_VALUE
    if den_k > 0 and num_k < I64_MIN_VALUE + den_k:
        return I64_MIN_VALUE
    k_delta = num_k - den_k

    abs_k_delta = -k_delta if k_delta < 0 else k_delta
    max_k_delta = I64_MAX_VALUE // LOG_Q16_LN2
    if abs_k_delta > max_k_delta:
        return I64_MIN_VALUE if k_delta < 0 else I64_MAX_VALUE
    base_q16 = k_delta * LOG_Q16_LN2

    if base_q16 > 0 and poly_delta_q16 > I64_MAX_VALUE - base_q16:
        return I64_MAX_VALUE
    if base_q16 < 0 and poly_delta_q16 < I64_MIN_VALUE - base_q16:
        return I64_MIN_VALUE

    return base_q16 + poly_delta_q16


def fpq16_ln_ratio_array_checked_nopartial_preflight(
    num_q16: list[int] | None,
    den_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
) -> tuple[int, int]:
    if count < 0:
        return INTLOG_STATUS_BADCOUNT, 0
    if num_q16 is None or den_q16 is None or out_q16 is None:
        return INTLOG_STATUS_BADPTR, 0

    if count == 0:
        return INTLOG_STATUS_OK, 0
    if count > (I64_MAX_VALUE // 8):
        return INTLOG_STATUS_OVERFLOW, 0

    for i in range(count):
        if num_q16[i] <= 0 or den_q16[i] <= 0:
            return INTLOG_STATUS_DOMAIN, 0

    return INTLOG_STATUS_OK, count * 8


def fpq16_ln_ratio_array_checked_nopartial(
    num_q16: list[int] | None,
    den_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
    out_capacity_bytes: int,
) -> int:
    status, needed = fpq16_ln_ratio_array_checked_nopartial_preflight(
        num_q16, den_q16, count, out_q16
    )
    if status != INTLOG_STATUS_OK:
        return status
    if out_capacity_bytes < needed:
        return INTLOG_STATUS_BADCOUNT

    assert num_q16 is not None
    assert den_q16 is not None
    assert out_q16 is not None
    for i in range(count):
        out_q16[i] = fpq16_ln_ratio(num_q16[i], den_q16[i])
    return INTLOG_STATUS_OK


def fpq16_ln_ratio_array_checked_nopartial_default(
    num_q16: list[int] | None,
    den_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
) -> int:
    status, needed = fpq16_ln_ratio_array_checked_nopartial_preflight(
        num_q16, den_q16, count, out_q16
    )
    if status != INTLOG_STATUS_OK:
        return status
    return fpq16_ln_ratio_array_checked_nopartial(num_q16, den_q16, count, out_q16, needed)


def test_signatures_present_in_holyc() -> None:
    src = Path("src/math/intlog.HC").read_text(encoding="utf-8")
    assert "I32 FPQ16LnRatioArrayCheckedNoPartialPreflight(" in src
    assert "I32 FPQ16LnRatioArrayCheckedNoPartial(" in src
    assert "I32 FPQ16LnRatioArrayCheckedNoPartialDefault(" in src
    assert "#define INTLOG_STATUS_DOMAIN -4" in src


def test_preflight_rejects_bad_geometry() -> None:
    st, need = fpq16_ln_ratio_array_checked_nopartial_preflight(None, [1], 1, [0])
    assert st == INTLOG_STATUS_BADPTR
    assert need == 0

    st, need = fpq16_ln_ratio_array_checked_nopartial_preflight([1], [1], -1, [0])
    assert st == INTLOG_STATUS_BADCOUNT
    assert need == 0


def test_preflight_detects_domain_violations_without_writes() -> None:
    out = [777, 777, 777]
    st, need = fpq16_ln_ratio_array_checked_nopartial_preflight(
        [FP_Q16_ONE, 0, FP_Q16_ONE],
        [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE],
        3,
        out,
    )
    assert st == INTLOG_STATUS_DOMAIN
    assert need == 0
    assert out == [777, 777, 777]


def test_preflight_detects_required_bytes_overflow() -> None:
    huge = (I64_MAX_VALUE // 8) + 1
    st, need = fpq16_ln_ratio_array_checked_nopartial_preflight([1], [1], huge, [0])
    assert st == INTLOG_STATUS_OVERFLOW
    assert need == 0


def test_commit_rejects_small_capacity_without_writing() -> None:
    num = [FP_Q16_ONE, FP_Q16_ONE * 2, FP_Q16_ONE * 3]
    den = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]
    out = [123456, 123456, 123456]
    st = fpq16_ln_ratio_array_checked_nopartial(num, den, len(num), out, 8)
    assert st == INTLOG_STATUS_BADCOUNT
    assert out == [123456, 123456, 123456]


def test_commit_rejects_domain_without_writing() -> None:
    num = [FP_Q16_ONE, 0, FP_Q16_ONE * 3]
    den = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]
    out = [99, 99, 99]
    st = fpq16_ln_ratio_array_checked_nopartial(num, den, len(num), out, len(num) * 8)
    assert st == INTLOG_STATUS_DOMAIN
    assert out == [99, 99, 99]


def test_commit_writes_expected_ln_ratio_lanes() -> None:
    num = [
        FP_Q16_ONE,
        FP_Q16_ONE * 2,
        FP_Q16_ONE * 3,
        FP_Q16_ONE * 8,
    ]
    den = [
        FP_Q16_ONE,
        FP_Q16_ONE,
        FP_Q16_ONE * 2,
        FP_Q16_ONE // 2,
    ]
    out = [0] * len(num)

    st = fpq16_ln_ratio_array_checked_nopartial(num, den, len(num), out, len(num) * 8)
    assert st == INTLOG_STATUS_OK

    want = [fpq16_ln_ratio(a, b) for a, b in zip(num, den)]
    assert out == want


def test_default_wrapper_matches_explicit_capacity_path() -> None:
    num = [FP_Q16_ONE // 8, FP_Q16_ONE // 2, FP_Q16_ONE, FP_Q16_ONE * 8]
    den = [FP_Q16_ONE // 16, FP_Q16_ONE // 4, FP_Q16_ONE, FP_Q16_ONE * 2]
    out_default = [0] * len(num)
    out_explicit = [0] * len(num)

    st0 = fpq16_ln_ratio_array_checked_nopartial_default(num, den, len(num), out_default)
    st1 = fpq16_ln_ratio_array_checked_nopartial(num, den, len(num), out_explicit, len(num) * 8)

    assert st0 == INTLOG_STATUS_OK
    assert st1 == INTLOG_STATUS_OK
    assert out_default == out_explicit


def run() -> None:
    test_signatures_present_in_holyc()
    test_preflight_rejects_bad_geometry()
    test_preflight_detects_domain_violations_without_writes()
    test_preflight_detects_required_bytes_overflow()
    test_commit_rejects_small_capacity_without_writing()
    test_commit_rejects_domain_without_writing()
    test_commit_writes_expected_ln_ratio_lanes()
    test_default_wrapper_matches_explicit_capacity_path()
    print("intlog_ln_ratio_array_checked_nopartial_reference_checks=ok")


if __name__ == "__main__":
    run()
