#!/usr/bin/env python3
"""Parity checks for RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedPreflightOnlyDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_token_window_strided_preflight_only as base_ref


def rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only_default_stride(
    head_cell_capacity: int,
    token_base_index: int,
    head_cell_stride: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    position_start: int,
    token_count: int,
    position_step: int,
) -> tuple[int, tuple[int, int, int, int]]:
    return base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
        head_cell_capacity,
        token_base_index,
        head_cell_stride,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        position_start,
        token_count,
        position_step,
    )


def test_matches_core_preflight_for_same_stride() -> None:
    args = (8192, 12, 144, 9, 3, 48, 16, 2, 77, 5, 4)
    got = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only_default_stride(*args)
    expected = base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(*args)
    assert got == expected


def test_known_value_diagnostics() -> None:
    err, diag = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only_default_stride(
        16_384,
        21,
        160,
        8,
        4,
        36,
        16,
        2,
        200,
        6,
        3,
    )
    assert err == ref.ROPE_Q16_OK

    token_index = 5
    abs_range_base = 21 + token_index * 160 + 8
    last_head_base = abs_range_base + (4 - 1) * 36
    head_span = ((16 // 2) - 1) * 2 + 1
    last_pair_y = last_head_base + head_span
    token_position = 200 + token_index * 3
    assert diag == (abs_range_base, last_head_base, last_pair_y, token_position)


def test_bad_param_and_overflow_surface_passthrough() -> None:
    err, _ = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only_default_stride(
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
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM

    err, _ = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only_default_stride(
        1024,
        ref.I64_MAX,
        ref.I64_MAX,
        1,
        1,
        24,
        16,
        2,
        0,
        2,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_randomized_parity_with_core_preflight() -> None:
    rng = random.Random(2026041719)

    for _ in range(6000):
        head_dim = rng.choice([8, 16, 24, 32])
        pair_stride = rng.randint(2, 6)
        head_span = ((head_dim // 2) - 1) * pair_stride + 1

        head_count = rng.randint(0, 4)
        token_count = rng.randint(0, 7)
        head_cell_stride = rng.randint(0, 300)
        head_stride = rng.randint(1, 80)
        position_start = rng.randint(0, 9000)
        position_step = rng.randint(0, 11)

        if head_count == 0 or token_count == 0:
            cap = rng.randint(1, 10_000)
            token_base = rng.randint(0, 2000)
            range_base = rng.randint(0, 2000)
        else:
            token_span = (token_count - 1) * head_cell_stride
            needed = token_span + ((head_count - 1) * head_stride) + head_span
            cap = rng.randint(max(needed + 1, 1), max(needed + 1024, 2))
            max_token_base = max(0, cap - needed - 1)
            token_base = rng.randint(0, max_token_base)
            max_range_base = max(0, cap - token_base - token_span - ((head_count - 1) * head_stride) - head_span - 1)
            range_base = rng.randint(0, max_range_base)

        if rng.random() < 0.16:
            pick = rng.randint(0, 4)
            if pick == 0:
                head_cell_stride = -1
            elif pick == 1:
                token_count = -1
            elif pick == 2:
                head_stride = 0
            elif pick == 3:
                position_step = -rng.randint(1, 5)
                position_start = rng.randint(0, 8)
                token_count = rng.randint(2, 7)
            else:
                token_base = ref.I64_MAX
                head_cell_stride = ref.I64_MAX

        got = rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only_default_stride(
            cap,
            token_base,
            head_cell_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            position_start,
            token_count,
            position_step,
        )
        expected = base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided_preflight_only(
            cap,
            token_base,
            head_cell_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            position_start,
            token_count,
            position_step,
        )
        assert got == expected


if __name__ == "__main__":
    test_matches_core_preflight_for_same_stride()
    test_known_value_diagnostics()
    test_bad_param_and_overflow_surface_passthrough()
    test_randomized_parity_with_core_preflight()
    print("ok")
