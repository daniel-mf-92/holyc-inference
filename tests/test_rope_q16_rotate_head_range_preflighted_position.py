#!/usr/bin/env python3
"""Reference checks for RoPEQ16RotateHeadRangeByPositionPreflightedChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_position as head_ref
import test_rope_q16_validate_head_range_capacity as cap_ref
import test_rope_q16_validate_head_span_for_dim as span_ref


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


def rope_q16_validate_head_range_span_for_dim_checked(
    head_cell_capacity: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
    head_dim: int,
    pair_stride_cells: int,
) -> tuple[int, int]:
    err, last_head_base = cap_ref.rope_q16_validate_head_range_capacity_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, 0

    if head_count == 0:
        return ref.ROPE_Q16_OK, range_base_index

    last_pair_y = range_base_index
    for head_index in range(head_count):
        err, head_offset = rope_try_mul_i64_checked(head_index, head_stride_cells)
        if err != ref.ROPE_Q16_OK:
            return err, 0

        err, head_base = rope_try_add_i64_checked(range_base_index, head_offset)
        if err != ref.ROPE_Q16_OK:
            return err, 0

        if head_base > last_head_base:
            return ref.ROPE_Q16_ERR_BAD_PARAM, 0

        err, last_pair_y = span_ref.rope_q16_validate_head_span_for_dim_checked(
            head_cell_capacity,
            head_base,
            head_dim,
            pair_stride_cells,
        )
        if err != ref.ROPE_Q16_OK:
            return err, 0

    return ref.ROPE_Q16_OK, last_pair_y


def rope_q16_rotate_head_range_by_position_preflighted_checked(
    head_cells_q16: list[int],
    head_cell_capacity: int,
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

    err, last_head_base = cap_ref.rope_q16_validate_head_range_capacity_checked(
        head_cell_capacity,
        range_base_index,
        head_count,
        head_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, []

    err, last_pair_y = rope_q16_validate_head_range_span_for_dim_checked(
        head_cell_capacity,
        range_base_index,
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

    for head_index in range(head_count):
        err, head_offset = rope_try_mul_i64_checked(head_index, head_stride_cells)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, head_base = rope_try_add_i64_checked(range_base_index, head_offset)
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
            position,
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
    buf = make_head_buffer(96, 1)

    assert rope_q16_rotate_head_range_by_position_preflighted_checked(
        buf,
        -1,
        0,
        2,
        32,
        16,
        2,
        base_q16,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_range_by_position_preflighted_checked(
        buf,
        96,
        -1,
        2,
        32,
        16,
        2,
        base_q16,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_range_by_position_preflighted_checked(
        buf,
        96,
        0,
        -1,
        32,
        16,
        2,
        base_q16,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_range_by_position_preflighted_checked(
        buf,
        96,
        0,
        2,
        0,
        16,
        2,
        base_q16,
        1,
    )[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_span_preflight_rejects_before_rotation() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 64
    buf = make_head_buffer(cap, 2)

    err, _ = rope_q16_rotate_head_range_by_position_preflighted_checked(
        buf,
        cap,
        20,
        2,
        16,
        32,
        2,
        base_q16,
        17,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_head_count_noop() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 3)

    err, got = rope_q16_rotate_head_range_by_position_preflighted_checked(
        buf,
        64,
        9,
        0,
        32,
        16,
        2,
        base_q16,
        9,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf


def test_parity_with_preflight_then_per_head_composition() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    cap = 512
    base = 23
    head_count = 3
    head_stride = 48
    head_dim = 16
    pair_stride = 2
    position = 77

    inp = make_head_buffer(cap, 4)

    err, got = rope_q16_rotate_head_range_by_position_preflighted_checked(
        inp,
        cap,
        base,
        head_count,
        head_stride,
        head_dim,
        pair_stride,
        base_q16,
        position,
    )
    assert err == ref.ROPE_Q16_OK

    want = list(inp)
    for head_index in range(head_count):
        head_base = base + (head_index * head_stride)
        err_head, want = head_ref.rope_q16_rotate_head_by_position_checked(
            want,
            cap,
            head_base,
            head_dim,
            pair_stride,
            base_q16,
            position,
        )
        assert err_head == ref.ROPE_Q16_OK

    assert got == want


def test_randomized_contract_parity() -> None:
    rng = random.Random(2026041702)
    dims = [8, 16, 24, 32]
    freq_bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1500):
        head_dim = rng.choice(dims)
        pair_stride = rng.randint(2, 6)
        head_span = (head_dim // 2) * pair_stride + 1

        head_stride = rng.randint(head_span, head_span + 12)
        head_count = rng.randint(0, 4)

        if head_count == 0:
            cap = rng.randint(head_span, head_span + 64)
            base = rng.randint(0, cap - 1)
        else:
            needed = ((head_count - 1) * head_stride) + head_span
            cap = rng.randint(needed, needed + 96)
            base = rng.randint(0, cap - needed)

        position = rng.randint(0, 4096)
        freq_base_q16 = ref.q16_from_float(rng.choice(freq_bases))
        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err_got, got = rope_q16_rotate_head_range_by_position_preflighted_checked(
            inp,
            cap,
            base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            position,
        )

        err_preflight, _ = rope_q16_validate_head_range_span_for_dim_checked(
            cap,
            base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
        )
        if err_preflight != ref.ROPE_Q16_OK:
            assert err_got == err_preflight
            continue

        want = list(inp)
        err_want = ref.ROPE_Q16_OK
        for head_index in range(head_count):
            head_base = base + (head_index * head_stride)
            err_head, want = head_ref.rope_q16_rotate_head_by_position_checked(
                want,
                cap,
                head_base,
                head_dim,
                pair_stride,
                freq_base_q16,
                position,
            )
            if err_head != ref.ROPE_Q16_OK:
                err_want = err_head
                want = []
                break

        assert err_got == err_want
        if err_got == ref.ROPE_Q16_OK:
            assert got == want


def run() -> None:
    test_bad_param_contracts()
    test_span_preflight_rejects_before_rotation()
    test_zero_head_count_noop()
    test_parity_with_preflight_then_per_head_composition()
    test_randomized_contract_parity()
    print("rope_q16_rotate_head_range_preflighted_position_reference_checks=ok")


if __name__ == "__main__":
    run()
