#!/usr/bin/env python3
"""Reference checks for RoPEQ16RotateHeadRangeByPositionCheckedStrided semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_preflighted_position as preflight_ref


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


def rope_q16_rotate_head_range_by_position_checked_strided(
    head_cells_q16: list[int] | None,
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    token_index: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position: int,
) -> tuple[int, list[int]]:
    if head_cells_q16 is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []
    if token_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if token_stride_q16 < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    err, token_offset = rope_try_mul_i64_checked(token_index, token_stride_q16)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, token_base = rope_try_add_i64_checked(token_base_index, token_offset)
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, abs_range_base = rope_try_add_i64_checked(token_base, range_base_index)
    if err != ref.ROPE_Q16_OK:
        return err, []

    return preflight_ref.rope_q16_rotate_head_range_by_position_preflighted_checked(
        head_cells_q16,
        head_cell_capacity,
        abs_range_base,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        freq_base_q16,
        position,
    )


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(512, 81)

    assert rope_q16_rotate_head_range_by_position_checked_strided(
        None,
        512,
        0,
        128,
        1,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        3,
    )[0] == ref.ROPE_Q16_ERR_NULL_PTR

    assert rope_q16_rotate_head_range_by_position_checked_strided(
        buf,
        512,
        0,
        128,
        -1,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        3,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM

    assert rope_q16_rotate_head_range_by_position_checked_strided(
        buf,
        512,
        0,
        -1,
        1,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        3,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_token_offset_overflow_surfaces_err_overflow() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(512, 82)

    err, _ = rope_q16_rotate_head_range_by_position_checked_strided(
        buf,
        512,
        0,
        ref.I64_MAX,
        2,
        0,
        1,
        32,
        16,
        2,
        base_q16,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_absolute_range_base_overflow_surfaces_err_overflow() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(512, 83)

    err, _ = rope_q16_rotate_head_range_by_position_checked_strided(
        buf,
        512,
        ref.I64_MAX - 3,
        2,
        1,
        5,
        1,
        32,
        16,
        2,
        base_q16,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_strided_helper_matches_explicit_absolute_base_composition() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 4096
    token_base_index = 9
    token_stride = 256
    token_index = 4
    range_base = 13
    head_count = 3
    head_stride = 40
    head_dim = 16
    pair_stride = 2
    position = 91

    inp = make_head_buffer(cap, 84)

    err, got = rope_q16_rotate_head_range_by_position_checked_strided(
        inp,
        cap,
        token_base_index,
        token_stride,
        token_index,
        range_base,
        head_count,
        head_stride,
        head_dim,
        pair_stride,
        base_q16,
        position,
    )
    assert err == ref.ROPE_Q16_OK

    abs_base = token_base_index + token_index * token_stride + range_base
    err_want, want = preflight_ref.rope_q16_rotate_head_range_by_position_preflighted_checked(
        inp,
        cap,
        abs_base,
        head_count,
        head_stride,
        head_dim,
        pair_stride,
        base_q16,
        position,
    )
    assert err_want == ref.ROPE_Q16_OK
    assert got == want


def test_strided_helper_rejects_out_of_capacity_span() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 256
    buf = make_head_buffer(cap, 85)

    err, _ = rope_q16_rotate_head_range_by_position_checked_strided(
        buf,
        cap,
        40,
        96,
        2,
        20,
        2,
        40,
        16,
        2,
        base_q16,
        17,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


if __name__ == "__main__":
    test_bad_param_contracts()
    test_token_offset_overflow_surfaces_err_overflow()
    test_absolute_range_base_overflow_surfaces_err_overflow()
    test_strided_helper_matches_explicit_absolute_base_composition()
    test_strided_helper_rejects_out_of_capacity_span()
    print("ok")
