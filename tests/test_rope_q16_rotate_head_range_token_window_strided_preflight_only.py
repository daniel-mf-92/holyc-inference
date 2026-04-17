#!/usr/bin/env python3
"""Reference checks for RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedPreflightOnly."""

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


def rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    position_start: int,
    token_count: int,
    position_step: int,
) -> tuple[int, tuple[int, int, int, int]]:
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)
    if token_stride_q16 < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)
    if token_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

    if token_count == 0 or head_count == 0:
        return (
            ref.ROPE_Q16_OK,
            (range_base_index, range_base_index, range_base_index, position_start),
        )

    last_abs_range_base = 0
    last_head_base = 0
    last_last_pair_y = 0
    last_token_position = 0

    for token_index in range(token_count):
        err, token_base_offset = rope_try_mul_i64_checked(token_index, token_stride_q16)
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)

        err, token_base = rope_try_add_i64_checked(token_base_index, token_base_offset)
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)

        err, abs_range_base = rope_try_add_i64_checked(token_base, range_base_index)
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)

        err, token_position_offset = rope_try_mul_i64_checked(token_index, position_step)
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)

        err, token_position = rope_try_add_i64_checked(position_start, token_position_offset)
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)
        if token_position < 0:
            return ref.ROPE_Q16_ERR_BAD_PARAM, (0, 0, 0, 0)

        err, last_head_base = preflight_ref.cap_ref.rope_q16_validate_head_range_capacity_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)

        err, last_last_pair_y = preflight_ref.rope_q16_validate_head_range_span_for_dim_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, (0, 0, 0, 0)

        last_abs_range_base = abs_range_base
        last_token_position = token_position

    return ref.ROPE_Q16_OK, (
        last_abs_range_base,
        last_head_base,
        last_last_pair_y,
        last_token_position,
    )


def expected_by_direct_composition(
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    position_start: int,
    token_count: int,
    position_step: int,
) -> tuple[int, tuple[int, int, int, int]]:
    return rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        head_cell_capacity,
        token_base_index,
        token_stride_q16,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        position_start,
        token_count,
        position_step,
    )


def test_bad_param_contracts() -> None:
    assert rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        -1,
        0,
        32,
        4,
        2,
        24,
        16,
        2,
        0,
        3,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM

    assert rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        1024,
        0,
        -1,
        4,
        2,
        24,
        16,
        2,
        0,
        3,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM

    assert rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        1024,
        0,
        32,
        4,
        2,
        24,
        16,
        2,
        0,
        -1,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_noop_diagnostics_contract() -> None:
    err, diag = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        2048,
        77,
        128,
        13,
        3,
        24,
        16,
        2,
        91,
        0,
        5,
    )
    assert err == ref.ROPE_Q16_OK
    assert diag == (13, 13, 13, 91)

    err, diag = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        2048,
        77,
        128,
        13,
        0,
        24,
        16,
        2,
        91,
        6,
        5,
    )
    assert err == ref.ROPE_Q16_OK
    assert diag == (13, 13, 13, 91)


def test_known_value_diagnostics() -> None:
    err, diag = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        8192,
        10,
        128,
        7,
        3,
        40,
        16,
        2,
        100,
        4,
        3,
    )
    assert err == ref.ROPE_Q16_OK

    token_index = 3
    abs_range = 10 + token_index * 128 + 7
    last_head_base = abs_range + (3 - 1) * 40
    head_span = ((16 // 2) - 1) * 2 + 1
    last_pair_y = last_head_base + head_span
    token_pos = 100 + token_index * 3

    assert diag == (abs_range, last_head_base, last_pair_y, token_pos)


def test_negative_position_rejected() -> None:
    err, _ = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        4096,
        40,
        96,
        11,
        2,
        32,
        16,
        2,
        4,
        5,
        -2,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_randomized_parity() -> None:
    rng = random.Random(2026041718)

    for _ in range(6000):
        head_dim = rng.choice([8, 16, 24, 32])
        pair_stride = rng.randint(2, 6)
        head_span = ((head_dim // 2) - 1) * pair_stride + 1

        head_count = rng.randint(0, 4)
        token_count = rng.randint(0, 7)
        token_stride = rng.randint(0, 256)
        head_stride = rng.randint(1, 80)
        position_start = rng.randint(0, 8000)
        position_step = rng.randint(0, 9)

        if head_count == 0 or token_count == 0:
            cap = rng.randint(1, 10_000)
            token_base = rng.randint(0, 1000)
            range_base = rng.randint(0, 1000)
        else:
            token_span = (token_count - 1) * token_stride
            needed = token_span + ((head_count - 1) * head_stride) + head_span
            cap = rng.randint(max(needed + 1, 1), max(needed + 512, 2))
            max_token_base = max(0, cap - needed - 1)
            token_base = rng.randint(0, max_token_base)
            max_range_base = max(0, cap - token_base - token_span - ((head_count - 1) * head_stride) - head_span - 1)
            range_base = rng.randint(0, max_range_base)

        if rng.random() < 0.16:
            pick = rng.randint(0, 4)
            if pick == 0:
                token_stride = -1
            elif pick == 1:
                token_count = -1
            elif pick == 2:
                head_stride = 0
            elif pick == 3:
                position_step = -rng.randint(1, 4)
                position_start = rng.randint(0, 5)
                token_count = rng.randint(2, 7)
            else:
                token_base = ref.I64_MAX
                token_stride = ref.I64_MAX

        got_err, got_diag = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
            cap,
            token_base,
            token_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            position_start,
            token_count,
            position_step,
        )

        want_err, want_diag = expected_by_direct_composition(
            cap,
            token_base,
            token_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            position_start,
            token_count,
            position_step,
        )

        assert got_err == want_err
        if got_err == ref.ROPE_Q16_OK:
            assert got_diag == want_diag


def run() -> None:
    test_bad_param_contracts()
    test_noop_diagnostics_contract()
    test_known_value_diagnostics()
    test_negative_position_rejected()
    test_randomized_parity()
    print("rope_q16_rotate_head_range_token_window_strided_preflight_only_reference_checks=ok")


if __name__ == "__main__":
    run()
