#!/usr/bin/env python3
"""Reference checks for RoPEQ16 token-window rotation with precomputed head bases."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_compute_head_base_index as base_ref
import test_rope_q16_rotate_head_position as head_ref
import test_rope_q16_rotate_head_range_token_window as tw_ref
import test_rope_q16_validate_head_range_span_for_dim as span_ref


def rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
    head_cells_q16: list[int] | None,
    head_cell_capacity: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
    freq_base_q16: int,
    position_start: int,
    token_count: int,
    position_step: int,
    head_base_cache: list[int] | None,
    head_base_cache_capacity: int,
) -> tuple[int, list[int]]:
    if head_cells_q16 is None or head_base_cache is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []
    if token_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if head_base_cache_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    err, last_head_base = span_ref.cap_ref.rope_q16_validate_head_range_capacity_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, last_pair_y = span_ref.rope_q16_validate_head_range_span_for_dim_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    if head_count > head_base_cache_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if token_count == 0 or head_count == 0:
        return ref.ROPE_Q16_OK, list(head_cells_q16)

    cache = list(head_base_cache)
    work = list(head_cells_q16)

    for head_index in range(head_count):
        err, head_base = base_ref.rope_q16_compute_head_base_index_checked(
            range_base_index,
            head_index,
            head_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

        if head_base > last_head_base:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []

        cache[head_index] = head_base

    for token_index in range(token_count):
        err, token_offset = tw_ref.rope_try_mul_i64_checked(token_index, position_step)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position = tw_ref.rope_try_add_i64_checked(position_start, token_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        for head_index in range(head_count):
            head_base = cache[head_index]
            if head_base < 0 or head_base > last_head_base:
                return ref.ROPE_Q16_ERR_BAD_PARAM, []

            err, work = head_ref.rope_q16_rotate_head_by_position_checked(
                work,
                head_cell_capacity,
                head_base,
                head_dim,
                pair_stride_cells,
                freq_base_q16,
                token_position,
            )
            if err != ref.ROPE_Q16_OK:
                return err, []

    if last_pair_y >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    return ref.ROPE_Q16_OK, work


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(96, 51)
    cache = [0] * 8

    assert rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
        None,
        96,
        0,
        2,
        32,
        16,
        2,
        base_q16,
        4,
        3,
        1,
        cache,
        len(cache),
    )[0] == ref.ROPE_Q16_ERR_NULL_PTR

    assert rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
        buf,
        96,
        0,
        2,
        32,
        16,
        2,
        base_q16,
        4,
        3,
        1,
        None,
        len(cache),
    )[0] == ref.ROPE_Q16_ERR_NULL_PTR

    assert rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
        buf,
        96,
        0,
        2,
        32,
        16,
        2,
        base_q16,
        4,
        -1,
        1,
        cache,
        len(cache),
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM

    assert rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
        buf,
        96,
        0,
        2,
        32,
        16,
        2,
        base_q16,
        4,
        3,
        1,
        cache,
        -1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_cache_capacity_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(256, 52)

    err, _ = rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
        buf,
        256,
        8,
        4,
        48,
        16,
        2,
        base_q16,
        0,
        2,
        1,
        [0, 0, 0],
        3,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_head_count_noop_ignores_small_cache() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 53)

    err, got = rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
        buf,
        64,
        9,
        0,
        32,
        16,
        2,
        base_q16,
        9,
        4,
        1,
        [],
        0,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf


def test_parity_with_existing_preflighted_token_window() -> None:
    base_q16 = ref.q16_from_float(10000.0)

    for seed in range(20):
        rng = random.Random(20260417_149_000 + seed)

        head_count = rng.randint(1, 4)
        head_stride = rng.choice([32, 40, 48, 64])
        head_dim = rng.choice([8, 16, 24])
        pair_stride = 2

        cap = 1024
        max_base = cap - (head_count - 1) * head_stride - head_dim
        range_base = rng.randint(0, max(0, max_base))

        token_count = rng.randint(1, 6)
        position_start = rng.randint(0, 100)
        position_step = rng.randint(0, 5)

        inp = make_head_buffer(cap, 60 + seed)
        cache = [-777] * (head_count + 3)

        err_got, got = rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
            inp,
            cap,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            base_q16,
            position_start,
            token_count,
            position_step,
            cache,
            len(cache),
        )

        err_want, want = tw_ref.rope_q16_rotate_head_range_by_token_window_preflighted_checked(
            inp,
            cap,
            range_base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            base_q16,
            position_start,
            token_count,
            position_step,
        )

        assert err_got == err_want
        assert got == want


def run() -> None:
    test_bad_param_contracts()
    test_cache_capacity_contracts()
    test_zero_head_count_noop_ignores_small_cache()
    test_parity_with_existing_preflighted_token_window()
    print("rope_q16_rotate_head_range_token_window_preflighted_head_bases_reference_checks=ok")


if __name__ == "__main__":
    run()
