#!/usr/bin/env python3
"""Reference checks for RoPEQ16ApplyPairwiseChecked semantics (IQ-1362)."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_apply_pair_checked as apply_pair_ref


def rope_q16_apply_pairwise_checked(
    head_cells_q16: list[int],
    head_cell_capacity: int,
    lane_x_index: int,
    lane_y_index: int,
    angle_q16: int,
) -> tuple[int, list[int]]:
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if lane_x_index < 0 or lane_y_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if lane_x_index >= head_cell_capacity or lane_y_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if lane_x_index >= lane_y_index:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    out = list(head_cells_q16)
    x_q16 = out[lane_x_index]
    y_q16 = out[lane_y_index]

    err, sin_q16, cos_q16 = apply_pair_ref.rope_q16_lookup_sincos_checked(angle_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, x_cos_q16 = ref.rope_q16_mul_checked(x_q16, cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []
    err, y_sin_q16 = ref.rope_q16_mul_checked(y_q16, sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []
    err, x_sin_q16 = ref.rope_q16_mul_checked(x_q16, sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []
    err, y_cos_q16 = ref.rope_q16_mul_checked(y_q16, cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, x_rot_q16 = apply_pair_ref.rope_try_add_i64_checked(x_cos_q16, -y_sin_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []
    err, y_rot_q16 = apply_pair_ref.rope_try_add_i64_checked(x_sin_q16, y_cos_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    out[lane_x_index] = x_rot_q16
    out[lane_y_index] = y_rot_q16
    return ref.ROPE_Q16_OK, out


def q16_to_float(x_q16: int) -> float:
    return x_q16 / float(1 << 16)


def float_to_q16(x: float) -> int:
    return int(round(x * (1 << 16)))


def rope_pair_rotate_float_reference(x_q16: int, y_q16: int, angle_q16: int) -> tuple[int, int]:
    x = q16_to_float(x_q16)
    y = q16_to_float(y_q16)
    angle = q16_to_float(angle_q16)

    c = math.cos(angle)
    s = math.sin(angle)
    x_rot = float_to_q16((x * c) - (y * s))
    y_rot = float_to_q16((x * s) + (y * c))
    return x_rot, y_rot


def test_bad_param_contracts() -> None:
    buf = [ref.q16_from_float(0.25)] * 12
    angle_q16 = ref.q16_from_float(1.0)

    assert rope_q16_apply_pairwise_checked(buf, -1, 1, 2, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pairwise_checked(buf, 12, -1, 2, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pairwise_checked(buf, 12, 1, -2, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pairwise_checked(buf, 12, 5, 5, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pairwise_checked(buf, 12, 9, 4, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_apply_pairwise_checked(buf, 12, 1, 12, angle_q16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_float_reference_error_bound() -> None:
    rng = random.Random(2026042402)
    max_lane_error = 0

    for _ in range(3000):
        capacity = rng.randint(8, 96)
        lane_x = rng.randint(0, capacity - 2)
        lane_y = rng.randint(lane_x + 1, capacity - 1)
        angle_q16 = rng.randint(-6 * apply_pair_ref.ROPE_Q16_TWO_PI, 6 * apply_pair_ref.ROPE_Q16_TWO_PI)
        buf = [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]

        err, got = rope_q16_apply_pairwise_checked(buf, capacity, lane_x, lane_y, angle_q16)
        assert err == ref.ROPE_Q16_OK

        want_x, want_y = rope_pair_rotate_float_reference(buf[lane_x], buf[lane_y], angle_q16)
        lane_error_x = abs(got[lane_x] - want_x)
        lane_error_y = abs(got[lane_y] - want_y)
        max_lane_error = max(max_lane_error, lane_error_x, lane_error_y)

        for idx in range(capacity):
            if idx != lane_x and idx != lane_y:
                assert got[idx] == buf[idx]

    # Fixed-point bound against float sin/cos reference. LUT approximation
    # plus Q16 rounding should stay comfortably under this threshold.
    assert max_lane_error <= 1300


def test_overflow_propagates_from_checked_multiply() -> None:
    capacity = 4
    lane_x = 0
    lane_y = 1
    buf = [ref.I64_MAX, ref.q16_from_float(0.125), 0, 0]
    angle_q16 = 0

    err, _ = rope_q16_apply_pairwise_checked(buf, capacity, lane_x, lane_y, angle_q16)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_source_contract_and_checked_paths_exist() -> None:
    source = Path("src/transformer/rope.HC").read_text(encoding="utf-8")
    assert "I32 RoPEQ16ApplyPairwiseChecked(" in source
    assert "if (lane_x_index >= lane_y_index)" in source
    assert "status = RoPEQ16LookupSinCosChecked(angle_q16," in source
    assert source.count("status = RoPEQ16MulChecked(") >= 4
    assert "status = RoPETrySubI64Checked(x_cos_q16," in source
    assert "status = RoPETryAddI64Checked(x_sin_q16," in source


def run() -> None:
    test_bad_param_contracts()
    test_float_reference_error_bound()
    test_overflow_propagates_from_checked_multiply()
    test_source_contract_and_checked_paths_exist()
    print("rope_q16_apply_pairwise_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
