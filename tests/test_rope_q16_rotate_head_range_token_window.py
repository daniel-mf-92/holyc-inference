#!/usr/bin/env python3
"""Reference checks for RoPEQ16RotateHeadRangeByTokenWindowPreflightedChecked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_range_token_window_preflighted_head_bases as hb_ref
import test_rope_q16_compute_head_base_index as base_ref
import test_rope_q16_rotate_head_position as head_ref
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


def rope_q16_rotate_head_range_by_token_window_preflighted_checked(
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
) -> tuple[int, list[int]]:
    if head_cells_q16 is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []
    if token_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    err, last_head_base = preflight_ref.cap_ref.rope_q16_validate_head_range_capacity_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, last_pair_y = preflight_ref.rope_q16_validate_head_range_span_for_dim_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    if token_count == 0 or head_count == 0:
        return ref.ROPE_Q16_OK, list(head_cells_q16)

    work = list(head_cells_q16)
    for token_index in range(token_count):
        err, token_offset = rope_try_mul_i64_checked(token_index, position_step)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position = rope_try_add_i64_checked(position_start, token_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

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


def rope_q16_rotate_head_range_by_token_window_composed_reference(
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
) -> tuple[int, list[int]]:
    if head_cells_q16 is None:
        return ref.ROPE_Q16_ERR_NULL_PTR, []
    if token_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    err, _ = preflight_ref.rope_q16_validate_head_range_span_for_dim_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
        head_dim,
        pair_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    if token_count == 0 or head_count == 0:
        return ref.ROPE_Q16_OK, list(head_cells_q16)

    work = list(head_cells_q16)
    for token_index in range(token_count):
        err, token_offset = rope_try_mul_i64_checked(token_index, position_step)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, token_position = rope_try_add_i64_checked(position_start, token_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, work = preflight_ref.rope_q16_rotate_head_range_by_position_preflighted_checked(
            work,
            head_cell_capacity,
            range_base_index,
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


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(96, 11)

    assert rope_q16_rotate_head_range_by_token_window_preflighted_checked(
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
    )[0] == ref.ROPE_Q16_ERR_NULL_PTR

    assert rope_q16_rotate_head_range_by_token_window_preflighted_checked(
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
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_token_count_is_noop() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 22)

    err, got = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
        buf,
        64,
        8,
        2,
        24,
        16,
        2,
        base_q16,
        10,
        0,
        1,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf


def test_zero_head_count_is_noop_even_with_tokens() -> None:
    base_q16 = ref.q16_from_float(5000.0)
    buf = make_head_buffer(64, 23)

    err, got = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
        buf,
        64,
        8,
        0,
        24,
        16,
        2,
        base_q16,
        10,
        4,
        2,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf


def test_range_span_preflight_rejects_before_token_loop() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 64
    buf = make_head_buffer(cap, 33)

    err, _ = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
        buf,
        cap,
        20,
        2,
        16,
        32,
        2,
        base_q16,
        7,
        4,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_position_step_overflow_surfaces_err_overflow() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(96, 44)

    err, _ = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
        buf,
        96,
        0,
        2,
        32,
        16,
        2,
        base_q16,
        1,
        2,
        ref.I64_MAX,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_preflighted_token_window_matches_per_position_composition() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 512
    base = 29
    head_count = 3
    head_stride = 48
    head_dim = 16
    pair_stride = 2
    position_start = 73
    token_count = 5
    position_step = 3

    inp = make_head_buffer(cap, 55)

    err, got = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
        inp,
        cap,
        base,
        head_count,
        head_stride,
        head_dim,
        pair_stride,
        base_q16,
        position_start,
        token_count,
        position_step,
    )
    assert err == ref.ROPE_Q16_OK

    err_want, want = rope_q16_rotate_head_range_by_token_window_composed_reference(
        inp,
        cap,
        base,
        head_count,
        head_stride,
        head_dim,
        pair_stride,
        base_q16,
        position_start,
        token_count,
        position_step,
    )
    assert err_want == ref.ROPE_Q16_OK
    assert got == want


def test_randomized_contract_parity_vs_composed_reference() -> None:
    rng = random.Random(2026041703)
    dims = [8, 16, 24, 32]
    freq_bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1600):
        head_dim = rng.choice(dims)
        pair_stride = rng.randint(2, 6)
        head_span = (head_dim // 2) * pair_stride + 1

        head_stride = rng.randint(max(1, head_span - 2), head_span + 14)
        head_count = rng.randint(0, 4)
        token_count = rng.randint(0, 6)
        position_start = rng.randint(0, 4096)
        position_step = rng.randint(-8, 8)

        if head_count == 0:
            cap = rng.randint(max(1, head_span), max(2, head_span + 64))
            base = rng.randint(0, cap - 1)
        else:
            needed = ((head_count - 1) * max(1, head_stride)) + head_span
            cap = rng.randint(max(needed, 1), max(needed + 96, 2))
            max_base = cap - needed
            base = rng.randint(0, max(0, max_base))

        if rng.random() < 0.22:
            if rng.random() < 0.5:
                head_stride = 0
            else:
                token_count = -1

        freq_base_q16 = ref.q16_from_float(rng.choice(freq_bases))
        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err_got, got = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
            inp,
            cap,
            base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
        )

        err_want, want = rope_q16_rotate_head_range_by_token_window_composed_reference(
            inp,
            cap,
            base,
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


def test_preflighted_execution_path_parity_with_cached_head_bases_variant() -> None:
    rng = random.Random(2026041704)
    dims = [8, 16, 24, 32]
    freq_bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1400):
        head_dim = rng.choice(dims)
        pair_stride = rng.randint(2, 6)
        head_span = (head_dim // 2) * pair_stride + 1

        head_stride = rng.randint(max(1, head_span - 2), head_span + 14)
        head_count = rng.randint(0, 4)
        token_count = rng.randint(0, 6)
        position_start = rng.randint(-1024, 4096)
        position_step = rng.randint(-8, 8)

        if head_count == 0:
            cap = rng.randint(max(1, head_span), max(2, head_span + 64))
            base = rng.randint(0, cap - 1)
        else:
            needed = ((head_count - 1) * max(1, head_stride)) + head_span
            cap = rng.randint(max(needed, 1), max(needed + 96, 2))
            max_base = cap - needed
            base = rng.randint(0, max(0, max_base))

        if rng.random() < 0.22:
            if rng.random() < 0.5:
                head_stride = 0
            else:
                token_count = -1

        freq_base_q16 = ref.q16_from_float(rng.choice(freq_bases))
        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err_pre, got_pre = rope_q16_rotate_head_range_by_token_window_preflighted_checked(
            inp,
            cap,
            base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
        )

        cache_pad = rng.randint(0, 3)
        cache_cap = max(0, head_count) + cache_pad
        cache = [rng.randint(-1000, 1000) for _ in range(cache_cap)]

        err_hb, got_hb = hb_ref.rope_q16_rotate_head_range_by_token_window_preflighted_head_bases_checked(
            inp,
            cap,
            base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            position_start,
            token_count,
            position_step,
            cache,
            cache_cap,
        )

        assert err_pre == err_hb
        if err_pre == ref.ROPE_Q16_OK:
            assert got_pre == got_hb


def run() -> None:
    test_bad_param_contracts()
    test_zero_token_count_is_noop()
    test_zero_head_count_is_noop_even_with_tokens()
    test_range_span_preflight_rejects_before_token_loop()
    test_position_step_overflow_surfaces_err_overflow()
    test_preflighted_token_window_matches_per_position_composition()
    test_randomized_contract_parity_vs_composed_reference()
    test_preflighted_execution_path_parity_with_cached_head_bases_variant()
    print("rope_q16_rotate_head_range_token_window_reference_checks=ok")


if __name__ == "__main__":
    run()
