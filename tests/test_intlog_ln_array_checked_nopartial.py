#!/usr/bin/env python3
"""Reference checks for FPQ16LnArrayCheckedNoPartial HolyC semantics."""

from __future__ import annotations

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


def fpq16_ln(x_q16: int) -> int:
    if x_q16 <= 0:
        return I64_MIN_VALUE

    m_q16, k = fpq16_ln_reduce(x_q16)
    if m_q16 >= LOG_Q16_SPLIT:
        m_q16 = (m_q16 + 1) >> 1
        k += 1

    y_q16 = m_q16 - FP_Q16_ONE
    poly_q16 = fpq16_ln1p_poly(y_q16)
    base_q16 = k * LOG_Q16_LN2

    if base_q16 > 0 and poly_q16 > I64_MAX_VALUE - base_q16:
        return I64_MAX_VALUE
    if base_q16 < 0 and poly_q16 < I64_MIN_VALUE - base_q16:
        return I64_MIN_VALUE

    return base_q16 + poly_q16


def fpq16_ln_array_checked_no_partial_preflight(
    input_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
) -> tuple[int, int]:
    if count < 0:
        return INTLOG_STATUS_BADCOUNT, 0
    if input_q16 is None or out_q16 is None:
        return INTLOG_STATUS_BADPTR, 0
    if count == 0:
        return INTLOG_STATUS_OK, 0
    if count > (I64_MAX_VALUE // 8):
        return INTLOG_STATUS_OVERFLOW, 0
    return INTLOG_STATUS_OK, count * 8


def fpq16_ln_array_checked_no_partial(
    input_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
    out_capacity_bytes: int,
) -> int:
    status, needed = fpq16_ln_array_checked_no_partial_preflight(input_q16, count, out_q16)
    if status != INTLOG_STATUS_OK:
        return status
    if out_capacity_bytes < needed:
        return INTLOG_STATUS_BADCOUNT

    assert input_q16 is not None
    assert out_q16 is not None
    for i in range(count):
        out_q16[i] = fpq16_ln(input_q16[i])
    return INTLOG_STATUS_OK


def fpq16_ln_array_checked_no_partial_default(
    input_q16: list[int] | None,
    count: int,
    out_q16: list[int] | None,
) -> int:
    status, needed = fpq16_ln_array_checked_no_partial_preflight(input_q16, count, out_q16)
    if status != INTLOG_STATUS_OK:
        return status
    return fpq16_ln_array_checked_no_partial(input_q16, count, out_q16, needed)


def test_preflight_rejects_bad_geometry() -> None:
    st, need = fpq16_ln_array_checked_no_partial_preflight(None, 4, [0, 0, 0, 0])
    assert st == INTLOG_STATUS_BADPTR
    assert need == 0

    st, need = fpq16_ln_array_checked_no_partial_preflight([FP_Q16_ONE], -1, [0])
    assert st == INTLOG_STATUS_BADCOUNT
    assert need == 0


def test_preflight_detects_required_bytes_overflow() -> None:
    huge = (I64_MAX_VALUE // 8) + 1
    st, need = fpq16_ln_array_checked_no_partial_preflight([1], huge, [0])
    assert st == INTLOG_STATUS_OVERFLOW
    assert need == 0


def test_commit_rejects_small_capacity_without_writing() -> None:
    inp = [FP_Q16_ONE, FP_Q16_ONE * 2, FP_Q16_ONE // 2]
    out = [123456, 123456, 123456]
    st = fpq16_ln_array_checked_no_partial(inp, len(inp), out, 8)
    assert st == INTLOG_STATUS_BADCOUNT
    assert out == [123456, 123456, 123456]


def test_commit_writes_expected_ln_lanes() -> None:
    inp = [
        0,
        -1,
        FP_Q16_ONE // 4,
        FP_Q16_ONE,
        FP_Q16_ONE * 2,
        FP_Q16_ONE * 3,
        FP_Q16_ONE * 16,
    ]
    out = [0] * len(inp)

    st = fpq16_ln_array_checked_no_partial(inp, len(inp), out, len(inp) * 8)
    assert st == INTLOG_STATUS_OK

    want = [fpq16_ln(x) for x in inp]
    assert out == want
    assert out[0] == I64_MIN_VALUE
    assert out[1] == I64_MIN_VALUE


def test_default_wrapper_matches_explicit_capacity_path() -> None:
    inp = [FP_Q16_ONE // 8, FP_Q16_ONE // 2, FP_Q16_ONE, FP_Q16_ONE * 8]
    out_default = [0] * len(inp)
    out_explicit = [0] * len(inp)

    st0 = fpq16_ln_array_checked_no_partial_default(inp, len(inp), out_default)
    st1 = fpq16_ln_array_checked_no_partial(inp, len(inp), out_explicit, len(inp) * 8)

    assert st0 == INTLOG_STATUS_OK
    assert st1 == INTLOG_STATUS_OK
    assert out_default == out_explicit


def run() -> None:
    test_preflight_rejects_bad_geometry()
    test_preflight_detects_required_bytes_overflow()
    test_commit_rejects_small_capacity_without_writing()
    test_commit_writes_expected_ln_lanes()
    test_default_wrapper_matches_explicit_capacity_path()
    print("intlog_ln_array_checked_nopartial_reference_checks=ok")


if __name__ == "__main__":
    run()
