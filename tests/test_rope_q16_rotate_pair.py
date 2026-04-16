#!/usr/bin/env python3
"""Reference checks for RoPE Q16 rotate-pair helper semantics."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref

ROPE_Q16_PI = 205887
ROPE_Q16_TWO_PI = 411775
ROPE_Q16_HALF_PI = 102944


def c_div_trunc(a: int, b: int) -> int:
    assert b != 0
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def c_mod(a: int, b: int) -> int:
    return a - (c_div_trunc(a, b) * b)


def rope_try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs + rhs
    if out < ref.I64_MIN or out > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, out


def rope_try_sub_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    return rope_try_add_i64_checked(lhs, -rhs)


def rope_q16_div_by_positive_int_rounded_checked(num_q16: int, den: int) -> tuple[int, int]:
    if den <= 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    is_negative = num_q16 < 0
    abs_num = abs(num_q16)
    q = abs_num // den
    r = abs_num % den

    limit = (1 << 63) if is_negative else ref.I64_MAX
    if q > limit:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0

    if r >= ((den + 1) >> 1):
        if q == limit:
            return ref.ROPE_Q16_ERR_OVERFLOW, 0
        q += 1

    out = -q if is_negative else q
    if out < ref.I64_MIN or out > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, out


def rope_q16_normalize_angle_pi_checked(angle_q16: int) -> tuple[int, int]:
    reduced = c_mod(angle_q16, ROPE_Q16_TWO_PI)

    if reduced > ROPE_Q16_PI:
        err, reduced = rope_try_sub_i64_checked(reduced, ROPE_Q16_TWO_PI)
        if err != ref.ROPE_Q16_OK:
            return err, 0
    elif reduced < -ROPE_Q16_PI:
        err, reduced = rope_try_add_i64_checked(reduced, ROPE_Q16_TWO_PI)
        if err != ref.ROPE_Q16_OK:
            return err, 0

    return ref.ROPE_Q16_OK, reduced


def rope_q16_sin_approx_checked(angle_q16: int) -> tuple[int, int]:
    err, x = rope_q16_normalize_angle_pi_checked(angle_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    if x > ROPE_Q16_HALF_PI:
        err, x = rope_try_sub_i64_checked(ROPE_Q16_PI, x)
        if err != ref.ROPE_Q16_OK:
            return err, 0
    elif x < -ROPE_Q16_HALF_PI:
        err, x = rope_try_sub_i64_checked(-ROPE_Q16_PI, x)
        if err != ref.ROPE_Q16_OK:
            return err, 0

    err, x2 = ref.rope_q16_mul_checked(x, x)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    err, x3 = ref.rope_q16_mul_checked(x2, x)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    err, x5 = ref.rope_q16_mul_checked(x3, x2)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    err, x7 = ref.rope_q16_mul_checked(x5, x2)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, t1 = rope_q16_div_by_positive_int_rounded_checked(x3, 6)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    err, t2 = rope_q16_div_by_positive_int_rounded_checked(x5, 120)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    err, t3 = rope_q16_div_by_positive_int_rounded_checked(x7, 5040)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, sum1 = rope_try_sub_i64_checked(x, t1)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    err, sum2 = rope_try_add_i64_checked(sum1, t2)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    return rope_try_sub_i64_checked(sum2, t3)


def rope_q16_cos_approx_checked(angle_q16: int) -> tuple[int, int]:
    err, shifted = rope_try_add_i64_checked(angle_q16, ROPE_Q16_HALF_PI)
    if err != ref.ROPE_Q16_OK:
        return err, 0
    return rope_q16_sin_approx_checked(shifted)


def rope_q16_rotate_pair_checked(x_q16: int, y_q16: int, angle_q16: int) -> tuple[int, int, int]:
    err, sin_q16 = rope_q16_sin_approx_checked(angle_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, cos_q16 = rope_q16_cos_approx_checked(angle_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, x_cos = ref.rope_q16_mul_checked(x_q16, cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, y_sin = ref.rope_q16_mul_checked(y_q16, sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, x_sin = ref.rope_q16_mul_checked(x_q16, sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, y_cos = ref.rope_q16_mul_checked(y_q16, cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, out_x = rope_try_sub_i64_checked(x_cos, y_sin)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    err, out_y = rope_try_add_i64_checked(x_sin, y_cos)
    if err != ref.ROPE_Q16_OK:
        return err, 0, 0

    return ref.ROPE_Q16_OK, out_x, out_y


def test_known_quadrant_rotations() -> None:
    one = ref.q16_from_float(1.0)
    zero = 0

    for angle_f, want_x, want_y in [
        (0.0, 1.0, 0.0),
        (math.pi / 2.0, 0.0, 1.0),
        (math.pi, -1.0, 0.0),
        (-math.pi / 2.0, 0.0, -1.0),
    ]:
        angle_q16 = ref.q16_from_float(angle_f)
        err, got_x_q16, got_y_q16 = rope_q16_rotate_pair_checked(one, zero, angle_q16)
        assert err == ref.ROPE_Q16_OK

        got_x = ref.q16_to_float(got_x_q16)
        got_y = ref.q16_to_float(got_y_q16)

        assert abs(got_x - want_x) <= 0.09
        assert abs(got_y - want_y) <= 0.09


def test_checked_overflow_surfaces() -> None:
    angle_q16 = ref.q16_from_float(math.pi / 4.0)
    err, _, _ = rope_q16_rotate_pair_checked(ref.I64_MAX, ref.I64_MAX, angle_q16)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_randomized_against_float_rotation() -> None:
    rng = random.Random(20260416135)

    for _ in range(3000):
        x = rng.uniform(-4.0, 4.0)
        y = rng.uniform(-4.0, 4.0)
        angle = rng.uniform(-12.0 * math.pi, 12.0 * math.pi)

        x_q16 = ref.q16_from_float(x)
        y_q16 = ref.q16_from_float(y)
        angle_q16 = ref.q16_from_float(angle)

        err, got_x_q16, got_y_q16 = rope_q16_rotate_pair_checked(x_q16, y_q16, angle_q16)
        assert err == ref.ROPE_Q16_OK

        got_x = ref.q16_to_float(got_x_q16)
        got_y = ref.q16_to_float(got_y_q16)

        want_x = x * math.cos(angle) - y * math.sin(angle)
        want_y = x * math.sin(angle) + y * math.cos(angle)

        assert abs(got_x - want_x) <= 0.16
        assert abs(got_y - want_y) <= 0.16


def run() -> None:
    test_known_quadrant_rotations()
    test_checked_overflow_surfaces()
    test_randomized_against_float_rotation()
    print("rope_q16_rotate_pair_reference_checks=ok")


if __name__ == "__main__":
    run()
