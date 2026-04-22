#!/usr/bin/env python3
"""Reference checks for RoPEQ16RotateHeadByPositionApplyPairChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_apply_pair_checked as apply_pair_ref


def rope_try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs + rhs
    if out < ref.I64_MIN or out > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, out


def rope_q16_validate_head_span_for_dim_checked(
    head_cell_capacity: int,
    head_base_index: int,
    head_dim: int,
    pair_stride_cells: int,
) -> tuple[int, int]:
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_dim <= 0 or (head_dim & 1):
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if pair_stride_cells < 2:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_base_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    pair_count = head_dim >> 1
    last_pair_y = head_base_index + ((pair_count - 1) * pair_stride_cells) + 1
    if last_pair_y < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if last_pair_y > ref.I64_MAX:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    if last_pair_y >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    return ref.ROPE_Q16_OK, last_pair_y


def rope_q16_rotate_head_by_position_apply_pair_checked(
    head_cells_q16: list[int],
    head_cell_capacity: int,
    head_base_index: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position: int,
) -> tuple[int, list[int]]:
    err, last_pair_y = rope_q16_validate_head_span_for_dim_checked(
        head_cell_capacity,
        head_base_index,
        head_dim,
        pair_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    out = list(head_cells_q16)
    pair_count = head_dim >> 1
    lane_x_index = head_base_index

    for pair_index in range(pair_count):
        err, lane_y_index = rope_try_add_i64_checked(lane_x_index, 1)
        if err != ref.ROPE_Q16_OK:
            return err, []

        if lane_x_index < 0 or lane_y_index < 0:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []
        if lane_y_index > last_pair_y:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []

        err, angle_step_q16 = ref.rope_q16_angle_step_checked(freq_base_q16, head_dim, pair_index)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, angle_q16 = apply_pair_ref.rope_try_mul_i64_checked(angle_step_q16, position)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, out = apply_pair_ref.rope_q16_apply_pair_checked(
            out,
            head_cell_capacity,
            head_base_index,
            pair_index,
            pair_stride_cells,
            angle_q16,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        if pair_index + 1 < pair_count:
            err, lane_x_index = rope_try_add_i64_checked(lane_x_index, pair_stride_cells)
            if err != ref.ROPE_Q16_OK:
                return err, []

    return ref.ROPE_Q16_OK, out


def rope_q16_rotate_head_by_position_pair_kernel_reference(
    head_cells_q16: list[int],
    head_cell_capacity: int,
    head_base_index: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position: int,
) -> tuple[int, list[int]]:
    out = list(head_cells_q16)
    pair_count = head_dim >> 1

    for pair_index in range(pair_count):
        err, angle_step_q16 = ref.rope_q16_angle_step_checked(freq_base_q16, head_dim, pair_index)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, angle_q16 = apply_pair_ref.rope_try_mul_i64_checked(angle_step_q16, position)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, out = apply_pair_ref.rope_q16_apply_pair_checked(
            out,
            head_cell_capacity,
            head_base_index,
            pair_index,
            pair_stride_cells,
            angle_q16,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

    return ref.ROPE_Q16_OK, out


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 1)

    assert (
        rope_q16_rotate_head_by_position_apply_pair_checked(buf, -1, 0, 32, 2, base_q16, 7)[0]
        == ref.ROPE_Q16_ERR_BAD_PARAM
    )
    assert (
        rope_q16_rotate_head_by_position_apply_pair_checked(buf, 64, -1, 32, 2, base_q16, 7)[0]
        == ref.ROPE_Q16_ERR_BAD_PARAM
    )
    assert (
        rope_q16_rotate_head_by_position_apply_pair_checked(buf, 64, 0, 31, 2, base_q16, 7)[0]
        == ref.ROPE_Q16_ERR_BAD_PARAM
    )
    assert (
        rope_q16_rotate_head_by_position_apply_pair_checked(buf, 64, 0, 32, 1, base_q16, 7)[0]
        == ref.ROPE_Q16_ERR_BAD_PARAM
    )


def test_matches_pair_kernel_reference_for_interleaved_layout() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    head_dim = 16
    pair_stride = 2
    cap = 96
    base = 10
    pos = 37
    inp = make_head_buffer(cap, 3)

    err, got = rope_q16_rotate_head_by_position_apply_pair_checked(
        inp,
        cap,
        base,
        head_dim,
        pair_stride,
        base_q16,
        pos,
    )
    assert err == ref.ROPE_Q16_OK

    err_ref, want = rope_q16_rotate_head_by_position_pair_kernel_reference(
        inp,
        cap,
        base,
        head_dim,
        pair_stride,
        base_q16,
        pos,
    )
    assert err_ref == ref.ROPE_Q16_OK
    assert got == want


def test_matches_pair_kernel_reference_for_wide_stride_layout() -> None:
    base_q16 = ref.q16_from_float(50000.0)
    head_dim = 12
    pair_stride = 5
    cap = 256
    base = 17
    pos = 91
    inp = make_head_buffer(cap, 4)

    err, got = rope_q16_rotate_head_by_position_apply_pair_checked(
        inp,
        cap,
        base,
        head_dim,
        pair_stride,
        base_q16,
        pos,
    )
    assert err == ref.ROPE_Q16_OK

    err_ref, want = rope_q16_rotate_head_by_position_pair_kernel_reference(
        inp,
        cap,
        base,
        head_dim,
        pair_stride,
        base_q16,
        pos,
    )
    assert err_ref == ref.ROPE_Q16_OK
    assert got == want


def test_randomized_reference_parity() -> None:
    rng = random.Random(2026042301)
    dims = [8, 16, 24, 32]
    bases = [5000.0, 10000.0, 50000.0]

    for _ in range(2000):
        head_dim = rng.choice(dims)
        pair_stride = rng.randint(2, 7)
        pair_count = head_dim // 2
        needed = pair_count * pair_stride + 1

        cap = rng.randint(needed, needed + 96)
        base = rng.randint(0, cap - needed)
        pos = rng.randint(0, 4096)
        freq_base_q16 = ref.q16_from_float(rng.choice(bases))

        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err, got = rope_q16_rotate_head_by_position_apply_pair_checked(
            inp,
            cap,
            base,
            head_dim,
            pair_stride,
            freq_base_q16,
            pos,
        )
        err_ref, want = rope_q16_rotate_head_by_position_pair_kernel_reference(
            inp,
            cap,
            base,
            head_dim,
            pair_stride,
            freq_base_q16,
            pos,
        )

        assert err == err_ref
        if err == ref.ROPE_Q16_OK:
            assert got == want


def test_source_calls_apply_pair_and_angle_derivation_helpers() -> None:
    source = Path("src/model/rope.HC").read_text(encoding="utf-8")
    marker = "I32 RoPEQ16RotateHeadByPositionApplyPairChecked("
    assert marker in source

    start = source.rindex(marker)
    tail = source[start:]
    next_fn = tail.find("\nI32 ", 1)
    body = tail if next_fn == -1 else tail[:next_fn]

    assert "status = RoPEQ16AngleStepChecked(freq_base_q16," in body
    assert "status = RoPEQ16AngleForPositionChecked(angle_step_q16," in body
    assert "status = RoPEQ16ApplyPairChecked(head_cells_q16," in body


def run() -> None:
    test_bad_param_contracts()
    test_matches_pair_kernel_reference_for_interleaved_layout()
    test_matches_pair_kernel_reference_for_wide_stride_layout()
    test_randomized_reference_parity()
    test_source_calls_apply_pair_and_angle_derivation_helpers()
    print("rope_q16_rotate_head_position_apply_pair_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
