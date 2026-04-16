#!/usr/bin/env python3
"""Reference checks for composed RoPE Q16 rotate-by-position helper semantics."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as step_ref
import test_rope_q16_angle_position as pos_ref
import test_rope_q16_rotate_pair as rot_ref


def rope_q16_rotate_pair_by_position_checked(
    x_q16: int,
    y_q16: int,
    freq_base_q16: int,
    head_dim: int,
    pair_index: int,
    position: int,
) -> tuple[int, int, int]:
    err, angle_step_q16 = step_ref.rope_q16_angle_step_checked(freq_base_q16, head_dim, pair_index)
    if err != step_ref.ROPE_Q16_OK:
        return err, 0, 0

    err, angle_q16 = pos_ref.rope_q16_angle_for_position_checked(angle_step_q16, position)
    if err != step_ref.ROPE_Q16_OK:
        return err, 0, 0

    return rot_ref.rope_q16_rotate_pair_checked(x_q16, y_q16, angle_q16)


def test_null_contract_surfaces_via_composed_helpers() -> None:
    base_q16 = step_ref.q16_from_float(10000.0)

    err, _, _ = rope_q16_rotate_pair_by_position_checked(0, 0, 0, 64, 0, 0)
    assert err == step_ref.ROPE_Q16_ERR_DOMAIN

    err, _, _ = rope_q16_rotate_pair_by_position_checked(0, 0, base_q16, 63, 0, 0)
    assert err == step_ref.ROPE_Q16_ERR_BAD_PARAM

    err, _, _ = rope_q16_rotate_pair_by_position_checked(0, 0, base_q16, 64, 32, 0)
    assert err == step_ref.ROPE_Q16_ERR_BAD_PARAM

    err, _, _ = rope_q16_rotate_pair_by_position_checked(0, 0, base_q16, 64, 0, -1)
    assert err == step_ref.ROPE_Q16_ERR_BAD_PARAM


def test_known_quadrant_case_pair0_position_maps_to_direct_rotation() -> None:
    base_q16 = step_ref.q16_from_float(10000.0)
    one = step_ref.q16_from_float(1.0)

    # For pair_index=0, angle_step=1 rad/token. position=pi/2 gives quarter turn.
    position = int(round(math.pi / 2.0))
    err, x_q16, y_q16 = rope_q16_rotate_pair_by_position_checked(one, 0, base_q16, 64, 0, position)
    assert err == step_ref.ROPE_Q16_OK

    got_x = step_ref.q16_to_float(x_q16)
    got_y = step_ref.q16_to_float(y_q16)
    assert abs(got_x - math.cos(position)) <= 0.12
    assert abs(got_y - math.sin(position)) <= 0.12


def test_matches_manual_composition_exactly() -> None:
    rng = random.Random(20260416136)
    dims = [32, 64, 80, 96, 128, 160]
    bases = [5000.0, 10000.0, 50000.0]

    for _ in range(2500):
        head_dim = rng.choice(dims)
        pair_index = rng.randint(0, (head_dim // 2) - 1)
        position = rng.randint(0, 4096)
        base_q16 = step_ref.q16_from_float(rng.choice(bases))

        x_q16 = step_ref.q16_from_float(rng.uniform(-4.0, 4.0))
        y_q16 = step_ref.q16_from_float(rng.uniform(-4.0, 4.0))

        err, got_x_q16, got_y_q16 = rope_q16_rotate_pair_by_position_checked(
            x_q16, y_q16, base_q16, head_dim, pair_index, position
        )

        err_step, angle_step_q16 = step_ref.rope_q16_angle_step_checked(base_q16, head_dim, pair_index)
        if err_step != step_ref.ROPE_Q16_OK:
            assert err == err_step
            continue

        err_angle, angle_q16 = pos_ref.rope_q16_angle_for_position_checked(angle_step_q16, position)
        if err_angle != step_ref.ROPE_Q16_OK:
            assert err == err_angle
            continue

        err_rot, want_x_q16, want_y_q16 = rot_ref.rope_q16_rotate_pair_checked(x_q16, y_q16, angle_q16)
        assert err_rot == step_ref.ROPE_Q16_OK
        assert err == step_ref.ROPE_Q16_OK
        assert got_x_q16 == want_x_q16
        assert got_y_q16 == want_y_q16


def test_randomized_against_float_rope_formula() -> None:
    rng = random.Random(20260416137)
    dims = [32, 64, 128, 256]
    bases = [10000.0, 50000.0]

    checked = 0
    for _ in range(8000):
        head_dim = rng.choice(dims)
        pair_index = rng.randint(0, (head_dim // 2) - 1)
        position = rng.randint(0, 4096)
        freq_base = rng.choice(bases)

        x = rng.uniform(-3.0, 3.0)
        y = rng.uniform(-3.0, 3.0)

        theta_step = freq_base ** (-(2.0 * pair_index) / float(head_dim))
        angle = float(position) * theta_step

        # The polynomial sin/cos used by the HolyC kernel is validated in
        # `test_rope_q16_rotate_pair.py` over roughly +/-12pi. Keep this
        # composition parity check in the same regime so failures isolate
        # composition errors, not known high-angle approximation drift.
        if abs(angle) > (12.0 * math.pi):
            continue

        x_q16 = step_ref.q16_from_float(x)
        y_q16 = step_ref.q16_from_float(y)
        base_q16 = step_ref.q16_from_float(freq_base)

        err, got_x_q16, got_y_q16 = rope_q16_rotate_pair_by_position_checked(
            x_q16, y_q16, base_q16, head_dim, pair_index, position
        )

        err_step, angle_step_q16 = step_ref.rope_q16_angle_step_checked(base_q16, head_dim, pair_index)
        if err_step != step_ref.ROPE_Q16_OK:
            assert err == err_step
            continue

        err_angle, angle_q16 = pos_ref.rope_q16_angle_for_position_checked(angle_step_q16, position)
        if err_angle != step_ref.ROPE_Q16_OK:
            assert err == err_angle
            continue

        err_rot, want_x_q16, want_y_q16 = rot_ref.rope_q16_rotate_pair_checked(x_q16, y_q16, angle_q16)
        assert err_rot == step_ref.ROPE_Q16_OK
        assert err == step_ref.ROPE_Q16_OK
        assert got_x_q16 == want_x_q16
        assert got_y_q16 == want_y_q16

        got_x = step_ref.q16_to_float(got_x_q16)
        got_y = step_ref.q16_to_float(got_y_q16)

        want_x = x * math.cos(angle) - y * math.sin(angle)
        want_y = x * math.sin(angle) + y * math.cos(angle)

        assert abs(got_x - want_x) <= 1.5
        assert abs(got_y - want_y) <= 1.5
        checked += 1

    assert checked >= 800


def run() -> None:
    test_null_contract_surfaces_via_composed_helpers()
    test_known_quadrant_case_pair0_position_maps_to_direct_rotation()
    test_matches_manual_composition_exactly()
    test_randomized_against_float_rope_formula()
    print("rope_q16_rotate_pair_position_reference_checks=ok")


if __name__ == "__main__":
    run()
