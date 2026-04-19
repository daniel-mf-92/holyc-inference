#!/usr/bin/env python3
"""Parity checks for RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_token_window_strided as base_ref


def rope_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    prod = lhs * rhs
    if prod > ref.I64_MAX or prod < ref.I64_MIN:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, prod


def rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(
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

    if head_cell_capacity == 0:
        return base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided(
            head_cells_q16,
            head_cell_capacity,
            token_base_index,
            token_stride_q16,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
        )

    err, stage_bytes = rope_try_mul_i64_checked(head_cell_capacity, 8)
    if err != ref.ROPE_Q16_OK:
        return err, []
    if stage_bytes <= 0:
        return ref.ROPE_Q16_ERR_OVERFLOW, []

    staged = list(head_cells_q16)
    err, staged_out = base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided(
        staged,
        head_cell_capacity,
        token_base_index,
        token_stride_q16,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        freq_base_q16,
        position_start,
        token_count,
        position_step,
    )
    if err != ref.ROPE_Q16_OK:
        return err, list(head_cells_q16)

    return ref.ROPE_Q16_OK, list(staged_out)


def explicit_staged_composition_reference(
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

    if head_cell_capacity == 0:
        return base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided(
            head_cells_q16,
            head_cell_capacity,
            token_base_index,
            token_stride_q16,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
        )

    err, stage_bytes = rope_try_mul_i64_checked(head_cell_capacity, 8)
    if err != ref.ROPE_Q16_OK:
        return err, []
    if stage_bytes <= 0:
        return ref.ROPE_Q16_ERR_OVERFLOW, []

    staged = list(head_cells_q16)
    err, staged_out = base_ref.rope_q16_rotate_head_range_by_token_window_checked_strided(
        staged,
        head_cell_capacity,
        token_base_index,
        token_stride_q16,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
        freq_base_q16,
        position_start,
        token_count,
        position_step,
    )
    if err != ref.ROPE_Q16_OK:
        return err, list(head_cells_q16)

    return ref.ROPE_Q16_OK, list(staged_out)


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_source_contains_wrapper_signature_and_delegate_call() -> None:
    source = Path("src/model/rope.HC").read_text(encoding="utf-8")
    assert "I32 RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedNoPartial(" in source

    body = source.split(
        "I32 RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedNoPartial(",
        1,
    )[1]
    assert "status = RoPEQ16RotateHeadRangeByTokenWindowCheckedStrided(staged_head_cells_q16," in body
    assert "for (cell_index = 0; cell_index < head_cell_capacity; cell_index++)" in body


def test_null_and_negative_capacity_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 711)

    err, out = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(
        None,
        64,
        0,
        16,
        0,
        2,
        16,
        8,
        2,
        base_q16,
        0,
        2,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_NULL_PTR
    assert out == []

    err, out = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(
        buf,
        -1,
        0,
        16,
        0,
        2,
        16,
        8,
        2,
        base_q16,
        0,
        2,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM
    assert out == []


def test_no_partial_on_error_paths_and_success_parity() -> None:
    base_q16 = ref.q16_from_float(10000.0)

    buf = make_head_buffer(96, 912)
    original = list(buf)
    err, out = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(
        buf,
        96,
        0,
        16,
        0,
        2,
        16,
        8,
        2,
        base_q16,
        0,
        3,
        -1,
    )
    assert err != ref.ROPE_Q16_OK
    assert out == original

    ok_buf = make_head_buffer(128, 993)
    args = (
        ok_buf,
        128,
        0,
        32,
        0,
        2,
        16,
        8,
        2,
        base_q16,
        1,
        3,
        1,
    )
    got_err, got = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(*args)
    want_err, want = explicit_staged_composition_reference(*args)
    assert got_err == want_err == ref.ROPE_Q16_OK
    assert got == want


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260419_519)
    base_q16 = ref.q16_from_float(10000.0)

    for _ in range(1200):
        head_cell_capacity = rng.randint(1, 320)
        token_count = rng.randint(0, 6)
        head_count = rng.randint(0, 4)
        head_dim = rng.choice([8, 16, 24, 32])
        pair_stride_cells = rng.choice([1, 2, 3, 4])
        head_stride_cells = max(head_dim * pair_stride_cells, 1)
        token_stride_q16 = max(head_stride_cells * max(head_count, 1), 0)

        max_base = max(0, head_cell_capacity - 1)
        token_base_index = rng.randint(0, max_base)
        range_base_index = rng.randint(0, min(32, max_base - token_base_index))
        position_start = rng.randint(0, 10)
        position_step = rng.randint(-1, 3)

        buf = make_head_buffer(head_cell_capacity, rng.randint(1, 1_000_000))

        args = (
            buf,
            head_cell_capacity,
            token_base_index,
            token_stride_q16,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            base_q16,
            position_start,
            token_count,
            position_step,
        )

        got_err, got = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(*args)
        want_err, want = explicit_staged_composition_reference(*args)

        assert got_err == want_err
        assert got == want


def run() -> None:
    test_source_contains_wrapper_signature_and_delegate_call()
    test_null_and_negative_capacity_contracts()
    test_no_partial_on_error_paths_and_success_parity()
    test_randomized_parity_vs_explicit_staged_composition()
    print("rope_q16_rotate_head_range_token_window_checked_strided_no_partial=ok")


if __name__ == "__main__":
    run()
