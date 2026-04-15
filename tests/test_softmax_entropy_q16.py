#!/usr/bin/env python3
"""Host-side parity checks for FPQ16EntropyFromProbs behavior."""

from __future__ import annotations

import math
import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT
LOG_Q16_LN2 = 45426
LOG_Q16_SPLIT = (FP_Q16_ONE * 3) // 2
I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
I64_MIN_VALUE = -(1 << 63)


def q16_from_float(value: float) -> int:
    return int(round(value * FP_Q16_ONE))


def q16_to_float(value: int) -> float:
    return value / FP_Q16_ONE


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    half = 1 << (shift - 1)
    return (value + half) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


def fpq16_mul(a: int, b: int) -> int:
    return round_shift_right_signed(a * b, FP_Q16_SHIFT)


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


def fpq16_entropy_from_probs(probs: list[int]) -> int:
    if not probs:
        return 0

    sum_plnp_q16 = 0
    for p_q16 in probs:
        if p_q16 <= 0:
            continue

        if p_q16 > FP_Q16_ONE:
            p_q16 = FP_Q16_ONE

        ln_p_q16 = fpq16_ln(p_q16)
        if ln_p_q16 == I64_MIN_VALUE:
            continue

        plnp_q16 = fpq16_mul(p_q16, ln_p_q16)
        if plnp_q16 > 0:
            plnp_q16 = 0

        if sum_plnp_q16 < I64_MIN_VALUE - plnp_q16:
            sum_plnp_q16 = I64_MIN_VALUE
        else:
            sum_plnp_q16 += plnp_q16

    if sum_plnp_q16 == I64_MIN_VALUE:
        return I64_MAX_VALUE

    return -sum_plnp_q16


def test_entropy_clamps_and_degenerate_inputs() -> None:
    assert fpq16_entropy_from_probs([]) == 0
    assert fpq16_entropy_from_probs([0, 0, 0]) == 0
    assert fpq16_entropy_from_probs([FP_Q16_ONE, 0, 0, 0]) == 0
    assert fpq16_entropy_from_probs([FP_Q16_ONE + 12345]) == 0
    assert fpq16_entropy_from_probs([-7, 0, FP_Q16_ONE]) == 0


def test_entropy_known_vectors_against_float_reference() -> None:
    vectors = [
        [0.5, 0.5],
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.2, 0.1],
        [0.9, 0.1],
        [1.0, 0.0, 0.0],
    ]

    for vector in vectors:
        probs_q16 = [q16_from_float(v) for v in vector]
        got = q16_to_float(fpq16_entropy_from_probs(probs_q16))

        clamped = [min(1.0, max(0.0, v)) for v in vector]
        total = sum(clamped)
        if total > 0:
            clamped = [v / total for v in clamped]

        want = 0.0
        for p in clamped:
            if p > 0:
                want -= p * math.log(p)

        assert abs(got - want) <= 0.015


def test_entropy_random_simplex_vectors_against_float_reference() -> None:
    rng = random.Random(20260415)
    max_abs_err = 0.0

    for _ in range(2000):
        count = rng.randint(2, 16)
        raw = [rng.random() for _ in range(count)]
        total = sum(raw)
        probs = [v / total for v in raw]

        probs_q16 = [q16_from_float(v) for v in probs]
        q16_sum = sum(probs_q16)
        probs_q16[0] += FP_Q16_ONE - q16_sum

        got = q16_to_float(fpq16_entropy_from_probs(probs_q16))

        want = 0.0
        for p_q16 in probs_q16:
            p = q16_to_float(max(0, min(FP_Q16_ONE, p_q16)))
            if p > 0:
                want -= p * math.log(p)

        abs_err = abs(got - want)
        max_abs_err = max(max_abs_err, abs_err)
        assert abs_err <= 0.02

    assert max_abs_err > 0.0


def run() -> None:
    test_entropy_clamps_and_degenerate_inputs()
    test_entropy_known_vectors_against_float_reference()
    test_entropy_random_simplex_vectors_against_float_reference()
    print("softmax_entropy_q16_reference_checks=ok")


if __name__ == "__main__":
    run()
