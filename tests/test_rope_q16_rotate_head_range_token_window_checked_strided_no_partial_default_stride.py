#!/usr/bin/env python3
"""Parity checks for RoPEQ16...NoPartialDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_token_window_checked_strided_no_partial as no_partial_ref


def rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_default_stride(
    head_cells_q16: list[int] | None,
    head_cell_capacity: int,
    token_base_index: int,
    head_cell_stride: int,
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
    token_stride_q16 = head_cell_stride
    return no_partial_ref.rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial(
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


def explicit_default_stride_composition_reference(
    head_cells_q16: list[int] | None,
    head_cell_capacity: int,
    token_base_index: int,
    head_cell_stride: int,
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
    token_stride_q16 = head_cell_stride
    return no_partial_ref.explicit_staged_composition_reference(
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


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_source_contains_default_stride_signature_and_delegate_shape() -> None:
    source = Path("src/model/rope.HC").read_text(encoding="utf-8")
    signature = "I32 RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedNoPartialDefaultStride("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "I64 token_stride_q16;" in body
    assert "token_stride_q16 = head_cell_stride;" in body
    assert "RoPEQ16RotateHeadRangeByTokenWindowCheckedStridedNoPartial(head_cells_q16," in body
    assert "token_stride_q16," in body


def test_boundary_and_overflow_fixtures() -> None:
    base_q16 = ref.q16_from_float(10000.0)

    err, out = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_default_stride(
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

    buf = make_head_buffer(64, 520)
    err, out = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_default_stride(
        buf,
        -1,
        16,
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

    err, out = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_default_stride(
        buf,
        64,
        ref.I64_MAX,
        2,
        2,
        1,
        8,
        8,
        2,
        base_q16,
        0,
        2,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW
    assert out == list(buf)


def test_randomized_parity_vs_explicit_default_stride_composition() -> None:
    rng = random.Random(20260419_520)
    base_q16 = ref.q16_from_float(10000.0)

    for _ in range(2400):
        head_cell_capacity = rng.randint(1, 384)
        token_count = rng.randint(0, 8)
        head_count = rng.randint(0, 4)
        head_dim = rng.choice([8, 16, 24, 32])
        pair_stride_cells = rng.choice([1, 2, 3, 4])
        head_stride_cells = max(head_dim * pair_stride_cells, 1)

        min_stride = max(head_stride_cells * max(head_count, 1), 1)
        head_cell_stride = rng.randint(min_stride, min_stride + 96)

        max_base = max(0, head_cell_capacity - 1)
        token_base_index = rng.randint(0, max_base)
        range_base_index = rng.randint(0, min(32, max_base - token_base_index))
        position_start = rng.randint(0, 12)
        position_step = rng.randint(-1, 4)

        buf = make_head_buffer(head_cell_capacity, rng.randint(1, 1_000_000))

        args = (
            buf,
            head_cell_capacity,
            token_base_index,
            head_cell_stride,
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

        got_err, got = rope_q16_rotate_head_range_by_token_window_checked_strided_no_partial_default_stride(*args)
        want_err, want = explicit_default_stride_composition_reference(*args)

        assert got_err == want_err
        assert got == want


def run() -> None:
    test_source_contains_default_stride_signature_and_delegate_shape()
    test_boundary_and_overflow_fixtures()
    test_randomized_parity_vs_explicit_default_stride_composition()
    print("rope_q16_rotate_head_range_token_window_checked_strided_no_partial_default_stride=ok")


if __name__ == "__main__":
    run()
