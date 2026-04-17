#!/usr/bin/env python3
"""Reference checks for RoPEQ16RotateHeadRangeByTokenWindowCheckedStrided semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_position_strided as pos_strided_ref
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


def rope_q16_rotate_head_range_by_token_window_checked_strided(
    head_cells_q16: list[int] | None,
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position_start: int,
    token_count: int,
    position_step: int,
) -> tuple[int, list[int]]:
    if head_cells_q16 is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if token_stride_q16 < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if token_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if token_count == 0:
        err, abs_range_base = rope_try_add_i64_checked(token_base_index, range_base_index)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, _ = preflight_ref.cap_ref.rope_q16_validate_head_range_capacity_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, _ = preflight_ref.rope_q16_validate_head_range_span_for_dim_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        return ref.ROPE_Q16_OK, list(head_cells_q16)

    for token_index in range(token_count):
        err, token_base_offset = rope_try_mul_i64_checked(token_index, token_stride_q16)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_base = rope_try_add_i64_checked(token_base_index, token_base_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, abs_range_base = rope_try_add_i64_checked(token_base, range_base_index)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position_offset = rope_try_mul_i64_checked(token_index, position_step)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position = rope_try_add_i64_checked(position_start, token_position_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []
        if token_position < 0:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []

        err, _ = preflight_ref.cap_ref.rope_q16_validate_head_range_capacity_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, _ = preflight_ref.rope_q16_validate_head_range_span_for_dim_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

    if head_count == 0:
        return ref.ROPE_Q16_OK, list(head_cells_q16)

    work = list(head_cells_q16)
    for token_index in range(token_count):
        err, token_base_offset = rope_try_mul_i64_checked(token_index, token_stride_q16)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_base = rope_try_add_i64_checked(token_base_index, token_base_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, abs_range_base = rope_try_add_i64_checked(token_base, range_base_index)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position_offset = rope_try_mul_i64_checked(token_index, position_step)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position = rope_try_add_i64_checked(position_start, token_position_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, work = preflight_ref.rope_q16_rotate_head_range_by_position_preflighted_checked(
            work,
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            freq_base_q16,
            token_position,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

    return ref.ROPE_Q16_OK, work


def rope_q16_rotate_head_range_by_token_window_checked_strided_composed_reference(
    head_cells_q16: list[int] | None,
    head_cell_capacity: int,
    token_base_index: int,
    token_stride_q16: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position_start: int,
    token_count: int,
    position_step: int,
) -> tuple[int, list[int]]:
    if head_cells_q16 is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []
    if token_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if token_count == 0:
        err, abs_range_base = rope_try_add_i64_checked(token_base_index, range_base_index)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, _ = preflight_ref.cap_ref.rope_q16_validate_head_range_capacity_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, _ = preflight_ref.rope_q16_validate_head_range_span_for_dim_checked(
            head_cell_capacity,
            abs_range_base,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        return ref.ROPE_Q16_OK, list(head_cells_q16)

    work = list(head_cells_q16)
    for token_index in range(token_count):
        err, work = pos_strided_ref.rope_q16_rotate_head_range_by_position_checked_strided(
            work,
            head_cell_capacity,
            token_base_index,
            token_stride_q16,
            token_index,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            freq_base_q16,
            position_start + token_index * position_step,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

    return ref.ROPE_Q16_OK, work


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(512, 91)

    assert rope_q16_rotate_head_range_by_token_window_checked_strided(
        None,
        512,
        0,
        128,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        3,
        4,
        1,
    )[0] == ref.ROPE_Q16_ERR_NULL_PTR

    assert rope_q16_rotate_head_range_by_token_window_checked_strided(
        buf,
        512,
        0,
        -1,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        3,
        4,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM

    assert rope_q16_rotate_head_range_by_token_window_checked_strided(
        buf,
        -1,
        0,
        128,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        3,
        4,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_noop_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(256, 92)

    err, got = rope_q16_rotate_head_range_by_token_window_checked_strided(
        buf,
        256,
        17,
        96,
        11,
        2,
        24,
        16,
        2,
        base_q16,
        7,
        0,
        2,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf

    err, got = rope_q16_rotate_head_range_by_token_window_checked_strided(
        buf,
        256,
        17,
        96,
        11,
        0,
        24,
        16,
        2,
        base_q16,
        7,
        5,
        2,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf


def test_position_negative_in_window_rejected_no_partial_write() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(512, 93)

    err, got = rope_q16_rotate_head_range_by_token_window_checked_strided(
        buf,
        512,
        19,
        128,
        7,
        2,
        32,
        16,
        2,
        base_q16,
        4,
        5,
        -2,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM
    assert got == []


def test_matches_strided_position_composition_on_valid_inputs() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 4096

    inp = make_head_buffer(cap, 94)
    err, got = rope_q16_rotate_head_range_by_token_window_checked_strided(
        inp,
        cap,
        13,
        256,
        9,
        3,
        40,
        16,
        2,
        base_q16,
        75,
        6,
        3,
    )
    assert err == ref.ROPE_Q16_OK

    err_want, want = rope_q16_rotate_head_range_by_token_window_checked_strided_composed_reference(
        inp,
        cap,
        13,
        256,
        9,
        3,
        40,
        16,
        2,
        base_q16,
        75,
        6,
        3,
    )
    assert err_want == ref.ROPE_Q16_OK
    assert got == want


def test_randomized_parity_vs_strided_composition_for_valid_positions() -> None:
    rng = random.Random(2026041706)
    dims = [8, 16, 24, 32]
    freq_bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1500):
        head_dim = rng.choice(dims)
        pair_stride = rng.randint(2, 6)
        head_span = (head_dim // 2) * pair_stride + 1

        head_stride = rng.randint(max(1, head_span - 2), head_span + 14)
        head_count = rng.randint(0, 4)
        token_count = rng.randint(0, 6)
        token_stride = rng.randint(0, 256)
        position_start = rng.randint(0, 4096)
        position_step = rng.randint(0, 8)

        if head_count == 0:
            cap = rng.randint(max(1, head_span), max(2, head_span + 64))
            token_base = rng.randint(0, cap - 1)
            range_base = rng.randint(0, cap - 1)
        else:
            token_max = max(0, token_count - 1)
            token_span = token_max * token_stride
            needed = token_span + ((head_count - 1) * max(1, head_stride)) + head_span
            cap = rng.randint(max(needed, 1), max(needed + 96, 2))
            max_token_base = max(0, cap - needed)
            token_base = rng.randint(0, max_token_base)
            max_range_base = max(0, cap - token_base - token_span - ((head_count - 1) * max(1, head_stride)) - head_span)
            range_base = rng.randint(0, max_range_base)

        if rng.random() < 0.16:
            if rng.random() < 0.5:
                head_stride = 0
            else:
                token_count = -1

        freq_base_q16 = ref.q16_from_float(rng.choice(freq_bases))
        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err_got, got = rope_q16_rotate_head_range_by_token_window_checked_strided(
            inp,
            cap,
            token_base,
            token_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
        )

        err_want, want = rope_q16_rotate_head_range_by_token_window_checked_strided_composed_reference(
            inp,
            cap,
            token_base,
            token_stride,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
        )

        assert err_got == err_want
        if err_got == ref.ROPE_Q16_OK:
            assert got == want


def run() -> None:
    test_bad_param_contracts()
    test_noop_contracts()
    test_position_negative_in_window_rejected_no_partial_write()
    test_matches_strided_position_composition_on_valid_inputs()
    test_randomized_parity_vs_strided_composition_for_valid_positions()
    print("rope_q16_rotate_head_range_token_window_strided_reference_checks=ok")


if __name__ == "__main__":
    run()
