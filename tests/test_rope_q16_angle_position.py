#!/usr/bin/env python3
"""Reference checks for RoPE Q16 angle-for-position helper semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref


def rope_q16_angle_for_position_checked(angle_step_q16: int, position: int) -> tuple[int, int]:
    if angle_step_q16 <= 0:
        return ref.ROPE_Q16_ERR_DOMAIN, 0
    if position < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    err, position_q16 = ref.rope_q16_from_int_checked(position)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    return ref.rope_q16_mul_checked(position_q16, angle_step_q16)


def test_domain_and_bad_param_contracts() -> None:
    assert rope_q16_angle_for_position_checked(0, 0)[0] == ref.ROPE_Q16_ERR_DOMAIN
    assert rope_q16_angle_for_position_checked(-1, 0)[0] == ref.ROPE_Q16_ERR_DOMAIN

    step_q16 = ref.q16_from_float(0.125)
    assert rope_q16_angle_for_position_checked(step_q16, -1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_position_yields_zero_angle() -> None:
    step_q16 = ref.q16_from_float(0.75)
    err, angle_q16 = rope_q16_angle_for_position_checked(step_q16, 0)
    assert err == ref.ROPE_Q16_OK
    assert angle_q16 == 0


def test_checked_overflow_surfaces() -> None:
    too_large_position = (ref.I64_MAX >> ref.FP_Q16_SHIFT) + 1
    step_q16 = ref.q16_from_float(0.125)

    err, _ = rope_q16_angle_for_position_checked(step_q16, too_large_position)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW

    err, _ = rope_q16_angle_for_position_checked(ref.I64_MAX, 2)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_composes_with_angle_step_helper() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    head_dim = 128

    for pair_index in [0, 1, 7, 31, 63]:
        err, step_q16 = ref.rope_q16_angle_step_checked(base_q16, head_dim, pair_index)
        assert err == ref.ROPE_Q16_OK

        for position in [0, 1, 2, 7, 31, 127, 511, 2048]:
            err, angle_q16 = rope_q16_angle_for_position_checked(step_q16, position)
            assert err == ref.ROPE_Q16_OK

            got = ref.q16_to_float(angle_q16)
            want = float(position) * (10000.0 ** (-(2.0 * pair_index) / head_dim))
            abs_err = abs(got - want)
            assert abs_err <= 16.0
            if abs(want) >= 1.0:
                rel_err = abs_err / abs(want)
                assert rel_err <= 0.05


def test_randomized_model_like_ranges() -> None:
    rng = random.Random(20260416134)
    dims = [32, 64, 80, 128, 160, 256]
    bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1600):
        head_dim = rng.choice(dims)
        pair_index = rng.randint(0, (head_dim // 2) - 1)
        base = rng.choice(bases)
        position = rng.randint(0, 4096)

        base_q16 = ref.q16_from_float(base)
        err, step_q16 = ref.rope_q16_angle_step_checked(base_q16, head_dim, pair_index)
        assert err == ref.ROPE_Q16_OK

        err, angle_q16 = rope_q16_angle_for_position_checked(step_q16, position)
        if step_q16 <= 0:
            assert err == ref.ROPE_Q16_ERR_DOMAIN
            continue
        assert err == ref.ROPE_Q16_OK

        got = ref.q16_to_float(angle_q16)
        want = float(position) * (base ** (-(2.0 * pair_index) / head_dim))
        abs_err = abs(got - want)
        assert abs_err <= 32.0
        if abs(want) >= 1.0:
            rel_err = abs_err / abs(want)
            assert rel_err <= 0.05
        else:
            assert abs_err <= 0.2


def run() -> None:
    test_domain_and_bad_param_contracts()
    test_zero_position_yields_zero_angle()
    test_checked_overflow_surfaces()
    test_composes_with_angle_step_helper()
    test_randomized_model_like_ranges()
    print("rope_q16_angle_position_reference_checks=ok")


if __name__ == "__main__":
    run()
