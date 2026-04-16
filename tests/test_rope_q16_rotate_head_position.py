#!/usr/bin/env python3
"""Reference checks for RoPE Q16 rotate-head-by-position helper semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_pair_position as pair_ref


def rope_try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > ref.I64_MAX - rhs:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < ref.I64_MIN - rhs:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, lhs + rhs


def rope_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    prod = lhs * rhs
    if prod > ref.I64_MAX or prod < ref.I64_MIN:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, prod


def rope_q16_rotate_head_by_position_checked(
    head_cells_q16: list[int],
    head_cell_capacity: int,
    head_base_index: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position: int,
) -> tuple[int, list[int]]:
    if head_cells_q16 is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []

    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if head_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if head_dim <= 0 or (head_dim & 1):
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if pair_stride_cells < 2:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if head_base_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    work = list(head_cells_q16)
    pair_count = head_dim // 2

    for pair_index in range(pair_count):
        err, pair_offset = rope_try_mul_i64_checked(pair_index, pair_stride_cells)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, x_index = rope_try_add_i64_checked(head_base_index, pair_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, y_index = rope_try_add_i64_checked(x_index, 1)
        if err != ref.ROPE_Q16_OK:
            return err, []

        if x_index < 0 or y_index < 0:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []
        if x_index >= head_cell_capacity or y_index >= head_cell_capacity:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []

        err, x_rot_q16, y_rot_q16 = pair_ref.rope_q16_rotate_pair_by_position_checked(
            work[x_index],
            work[y_index],
            freq_base_q16,
            head_dim,
            pair_index,
            position,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        work[x_index] = x_rot_q16
        work[y_index] = y_rot_q16

    return ref.ROPE_Q16_OK, work


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 1)

    assert rope_q16_rotate_head_by_position_checked(buf, -1, 0, 32, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_by_position_checked(buf, 64, -1, 32, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_by_position_checked(buf, 64, 0, 31, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_by_position_checked(buf, 64, 0, 32, 1, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_capacity_bounds_guard() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(20, 2)

    err, _ = rope_q16_rotate_head_by_position_checked(
        buf,
        20,
        15,
        8,
        2,
        base_q16,
        2,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_interleaved_layout_rotates_in_place_like_pair_composition() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    head_dim = 16
    pair_stride = 2
    cap = 96
    base = 10
    pos = 37

    inp = make_head_buffer(cap, 3)

    err, got = rope_q16_rotate_head_by_position_checked(
        inp,
        cap,
        base,
        head_dim,
        pair_stride,
        base_q16,
        pos,
    )
    assert err == ref.ROPE_Q16_OK

    want = list(inp)
    for pair_index in range(head_dim // 2):
        x_index = base + (pair_index * pair_stride)
        y_index = x_index + 1
        err_pair, x_rot, y_rot = pair_ref.rope_q16_rotate_pair_by_position_checked(
            want[x_index],
            want[y_index],
            base_q16,
            head_dim,
            pair_index,
            pos,
        )
        assert err_pair == ref.ROPE_Q16_OK
        want[x_index] = x_rot
        want[y_index] = y_rot

    assert got == want


def test_wide_stride_layout_rotates_expected_pairs_only() -> None:
    base_q16 = ref.q16_from_float(50000.0)
    head_dim = 12
    pair_stride = 5
    cap = 256
    base = 17
    pos = 91

    inp = make_head_buffer(cap, 4)
    original = list(inp)

    err, got = rope_q16_rotate_head_by_position_checked(
        inp,
        cap,
        base,
        head_dim,
        pair_stride,
        base_q16,
        pos,
    )
    assert err == ref.ROPE_Q16_OK

    touched = set()
    for pair_index in range(head_dim // 2):
        x_index = base + (pair_index * pair_stride)
        y_index = x_index + 1
        touched.add(x_index)
        touched.add(y_index)

        err_pair, x_rot, y_rot = pair_ref.rope_q16_rotate_pair_by_position_checked(
            original[x_index],
            original[y_index],
            base_q16,
            head_dim,
            pair_index,
            pos,
        )
        assert err_pair == ref.ROPE_Q16_OK
        assert got[x_index] == x_rot
        assert got[y_index] == y_rot

    for idx in range(cap):
        if idx not in touched:
            assert got[idx] == original[idx]


def test_randomized_contract_and_determinism() -> None:
    rng = random.Random(20260416137)
    dims = [8, 16, 24, 32]
    bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1500):
        head_dim = rng.choice(dims)
        pair_stride = rng.randint(2, 7)
        pair_count = head_dim // 2
        needed = pair_count * pair_stride + 1

        cap = rng.randint(needed, needed + 64)
        base = rng.randint(0, cap - needed)
        pos = rng.randint(0, 4096)
        freq_base_q16 = ref.q16_from_float(rng.choice(bases))

        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err1, out1 = rope_q16_rotate_head_by_position_checked(
            inp,
            cap,
            base,
            head_dim,
            pair_stride,
            freq_base_q16,
            pos,
        )
        err2, out2 = rope_q16_rotate_head_by_position_checked(
            inp,
            cap,
            base,
            head_dim,
            pair_stride,
            freq_base_q16,
            pos,
        )

        assert err1 == err2
        if err1 == ref.ROPE_Q16_OK:
            assert out1 == out2


def run() -> None:
    test_bad_param_contracts()
    test_capacity_bounds_guard()
    test_interleaved_layout_rotates_in_place_like_pair_composition()
    test_wide_stride_layout_rotates_expected_pairs_only()
    test_randomized_contract_and_determinism()
    print("rope_q16_rotate_head_position_reference_checks=ok")


if __name__ == "__main__":
    run()
