#!/usr/bin/env python3
"""Reference checks for RoPE Q16 rotate-head-range-by-position semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
import test_rope_q16_rotate_head_position as head_ref


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


def rope_q16_rotate_head_range_by_position_checked(
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

    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if range_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if head_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []
    if head_stride_cells <= 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    if head_count == 0:
        return ref.ROPE_Q16_OK, list(head_cells_q16)

    if range_base_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, []

    work = list(head_cells_q16)

    for head_index in range(head_count):
        err, head_offset = rope_try_mul_i64_checked(head_index, head_stride_cells)
        if err != ref.ROPE_Q16_OK:
            return err, []

        err, head_base_index = rope_try_add_i64_checked(range_base_index, head_offset)
        if err != ref.ROPE_Q16_OK:
            return err, []

        if head_base_index < 0 or head_base_index >= head_cell_capacity:
            return ref.ROPE_Q16_ERR_BAD_PARAM, []

        err, work = head_ref.rope_q16_rotate_head_by_position_checked(
            work,
            head_cell_capacity,
            head_base_index,
            head_dim,
            pair_stride_cells,
            freq_base_q16,
            position,
        )
        if err != ref.ROPE_Q16_OK:
            return err, []

    return ref.ROPE_Q16_OK, work


def make_head_buffer(capacity: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [ref.q16_from_float(rng.uniform(-2.0, 2.0)) for _ in range(capacity)]


def test_bad_param_contracts() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(96, 1)

    assert rope_q16_rotate_head_range_by_position_checked(buf, -1, 0, 2, 32, 16, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_range_by_position_checked(buf, 96, -1, 2, 32, 16, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_range_by_position_checked(buf, 96, 0, -1, 32, 16, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_rotate_head_range_by_position_checked(buf, 96, 0, 2, 0, 16, 2, base_q16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_head_count_is_noop() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(64, 2)

    err, got = rope_q16_rotate_head_range_by_position_checked(
        buf,
        64,
        10,
        0,
        32,
        16,
        2,
        base_q16,
        9,
    )
    assert err == ref.ROPE_Q16_OK
    assert got == buf


def test_range_capacity_bounds_guard() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(80, 3)

    err, _ = rope_q16_rotate_head_range_by_position_checked(
        buf,
        80,
        64,
        2,
        16,
        16,
        2,
        base_q16,
        3,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_huge_head_range_surfaces_bad_param_before_overflow() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    buf = make_head_buffer(8, 4)

    err, _ = rope_q16_rotate_head_range_by_position_checked(
        buf,
        8,
        0,
        ref.I64_MAX,
        ref.I64_MAX,
        8,
        2,
        base_q16,
        1,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_checked_index_primitives_report_overflow() -> None:
    err, _ = rope_try_mul_i64_checked(ref.I64_MAX, 2)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW

    err, _ = rope_try_add_i64_checked(ref.I64_MAX, 1)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_interleaved_head_range_matches_per_head_composition() -> None:
    base_q16 = ref.q16_from_float(10000.0)
    head_dim = 16
    pair_stride = 2
    head_stride = 48
    head_count = 3
    cap = 512
    base = 23
    pos = 77

    inp = make_head_buffer(cap, 5)

    err, got = rope_q16_rotate_head_range_by_position_checked(
        inp,
        cap,
        base,
        head_count,
        head_stride,
        head_dim,
        pair_stride,
        base_q16,
        pos,
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
            pos,
        )
        assert err_head == ref.ROPE_Q16_OK

    assert got == want


def test_randomized_contract_and_composition_parity() -> None:
    rng = random.Random(20260416138)
    dims = [8, 16, 24, 32]
    freq_bases = [5000.0, 10000.0, 50000.0]

    for _ in range(1000):
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

        pos = rng.randint(0, 4096)
        freq_base_q16 = ref.q16_from_float(rng.choice(freq_bases))
        inp = make_head_buffer(cap, rng.randint(0, 10_000_000))

        err_got, got = rope_q16_rotate_head_range_by_position_checked(
            inp,
            cap,
            base,
            head_count,
            head_stride,
            head_dim,
            pair_stride,
            freq_base_q16,
            pos,
        )

        want = list(inp)
        err_want = ref.ROPE_Q16_OK
        if head_count > 0:
            for head_index in range(head_count):
                head_base = base + (head_index * head_stride)
                err_head, want = head_ref.rope_q16_rotate_head_by_position_checked(
                    want,
                    cap,
                    head_base,
                    head_dim,
                    pair_stride,
                    freq_base_q16,
                    pos,
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
    test_zero_head_count_is_noop()
    test_range_capacity_bounds_guard()
    test_huge_head_range_surfaces_bad_param_before_overflow()
    test_checked_index_primitives_report_overflow()
    test_interleaved_head_range_matches_per_head_composition()
    test_randomized_contract_and_composition_parity()
    print("rope_q16_rotate_head_range_position_reference_checks=ok")


if __name__ == "__main__":
    run()
