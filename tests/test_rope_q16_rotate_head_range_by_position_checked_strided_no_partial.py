#!/usr/bin/env python3
"""Parity harness for RoPEQ16RotateHeadRangeByPositionCheckedStridedNoPartial."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_position_strided as strided_ref


def rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
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
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if head_count == 0:
        return strided_ref.rope_q16_rotate_head_range_by_position_checked_strided(
            list(head_cells_q16),
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
            position,
        )

    staged = list(head_cells_q16)
    err, staged_out = strided_ref.rope_q16_rotate_head_range_by_position_checked_strided(
        staged,
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
        position,
    )
    if err != ref.ROPE_Q16_OK:
        return err, list(head_cells_q16)

    return ref.ROPE_Q16_OK, staged_out


def explicit_staged_composition(
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
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if head_count == 0:
        return strided_ref.rope_q16_rotate_head_range_by_position_checked_strided(
            list(head_cells_q16),
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
            position,
        )

    work = list(head_cells_q16)
    err, work = strided_ref.rope_q16_rotate_head_range_by_position_checked_strided(
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
        position,
    )
    if err != ref.ROPE_Q16_OK:
        return err, list(head_cells_q16)

    return ref.ROPE_Q16_OK, work


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_source_contains_wrapper_shape() -> None:
    source = pathlib.Path("src/model/rope.HC").read_text(encoding="utf-8")
    assert "RoPEQ16RotateHeadRangeByPositionCheckedStridedNoPartial" in source
    assert source.count("I32 RoPEQ16RotateHeadRangeByPositionCheckedStridedNoPartial(") == 2
    body = source.split(
        "I32 RoPEQ16RotateHeadRangeByPositionCheckedStridedNoPartial(", 1
    )[1]
    assert "RoPEQ16RotateHeadRangeByPositionCheckedStrided(scratch_cells_q16," in body
    assert "MAlloc(" in body
    assert "Free(scratch_cells_q16);" in body


def test_null_and_negative_capacity_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(256, 501)

    err, out = rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
        None,
        256,
        0,
        64,
        1,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        7,
    )
    assert err == ref.ROPE_Q16_ERR_NULL_PTR
    assert out == []

    err, out = rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
        buf,
        -1,
        0,
        64,
        1,
        5,
        2,
        32,
        16,
        2,
        base_q16,
        7,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM
    assert out == []


def test_no_partial_on_failure_surfaces_exact_error() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 320
    buf = make_head_buffer(cap, 502)
    sentinel = list(buf)

    err, got = rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
        buf,
        cap,
        40,
        128,
        2,
        27,
        3,
        64,
        16,
        2,
        base_q16,
        11,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM
    assert got == sentinel


def test_head_count_zero_matches_base_helper() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 512
    buf = make_head_buffer(cap, 503)

    args = dict(
        head_cell_capacity=cap,
        token_base_index=25,
        token_stride_q16=64,
        token_index=3,
        range_base_index=12,
        head_count=0,
        head_stride_cells=32,
        head_dim=16,
        pair_stride_cells=2,
        freq_base_q16=base_q16,
        position=13,
    )

    err_wrap, out_wrap = rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
        list(buf), **args
    )
    err_base, out_base = strided_ref.rope_q16_rotate_head_range_by_position_checked_strided(
        list(buf), **args
    )

    assert err_wrap == err_base == ref.ROPE_Q16_OK
    assert out_wrap == out_base == buf


def test_boundary_dims_and_randomized_parity_vs_explicit_staging() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    rng = random.Random(20260419_500)

    cases = [
        dict(
            cap=2048,
            token_base_index=7,
            token_stride_q16=256,
            token_index=3,
            range_base_index=5,
            head_count=4,
            head_stride_cells=32,
            head_dim=2,
            pair_stride_cells=2,
            position=31,
            seed=510,
        ),
        dict(
            cap=4096,
            token_base_index=11,
            token_stride_q16=384,
            token_index=5,
            range_base_index=23,
            head_count=3,
            head_stride_cells=40,
            head_dim=32,
            pair_stride_cells=2,
            position=67,
            seed=511,
        ),
    ]

    for case in cases:
        buf = make_head_buffer(case["cap"], case["seed"])

        err_wrap, out_wrap = rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
            buf,
            case["cap"],
            case["token_base_index"],
            case["token_stride_q16"],
            case["token_index"],
            case["range_base_index"],
            case["head_count"],
            case["head_stride_cells"],
            case["head_dim"],
            case["pair_stride_cells"],
            base_q16,
            case["position"],
        )
        err_ref, out_ref = explicit_staged_composition(
            buf,
            case["cap"],
            case["token_base_index"],
            case["token_stride_q16"],
            case["token_index"],
            case["range_base_index"],
            case["head_count"],
            case["head_stride_cells"],
            case["head_dim"],
            case["pair_stride_cells"],
            base_q16,
            case["position"],
        )
        assert err_wrap == err_ref == ref.ROPE_Q16_OK
        assert out_wrap == out_ref

    for _ in range(500):
        cap = rng.randint(256, 4096)
        token_stride_q16 = rng.randint(0, 256)
        token_index = rng.randint(0, 7)
        token_base_index = rng.randint(0, cap // 4)
        range_base_index = rng.randint(0, cap // 8)
        head_count = rng.randint(0, 4)
        head_stride_cells = rng.randint(1, 48)
        head_dim = rng.choice([2, 4, 8, 16, 32])
        pair_stride_cells = rng.randint(2, 4)
        position = rng.randint(0, 512)

        if head_count > 0:
            max_span = (head_count - 1) * head_stride_cells + (head_dim - 1) * pair_stride_cells + 1
            if token_base_index + token_index * token_stride_q16 + range_base_index + max_span >= cap:
                continue

        buf = make_head_buffer(cap, rng.randint(1, 10_000_000))
        err_wrap, out_wrap = rope_q16_rotate_head_range_by_position_checked_strided_no_partial(
            buf,
            cap,
            token_base_index,
            token_stride_q16,
            token_index,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            base_q16,
            position,
        )
        err_ref, out_ref = explicit_staged_composition(
            buf,
            cap,
            token_base_index,
            token_stride_q16,
            token_index,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
            base_q16,
            position,
        )

        assert err_wrap == err_ref
        if err_ref == ref.ROPE_Q16_OK:
            assert out_wrap == out_ref
        else:
            assert out_wrap == buf


def run() -> None:
    test_source_contains_wrapper_shape()
    test_null_and_negative_capacity_contracts()
    test_no_partial_on_failure_surfaces_exact_error()
    test_head_count_zero_matches_base_helper()
    test_boundary_dims_and_randomized_parity_vs_explicit_staging()
    print("rope_q16_rotate_head_range_by_position_checked_strided_no_partial=ok")


if __name__ == "__main__":
    run()
