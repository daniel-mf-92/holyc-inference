#!/usr/bin/env python3
"""Reference checks for RoPEQ16ValidateHeadRangeSpanForDimChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref
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


def expected_by_composition(
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


def test_bad_param_contracts() -> None:
    assert rope_q16_validate_head_range_span_for_dim_checked(-1, 0, 1, 16, 16, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_range_span_for_dim_checked(128, -1, 1, 16, 16, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_range_span_for_dim_checked(128, 0, -1, 16, 16, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_range_span_for_dim_checked(128, 0, 1, 0, 16, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_head_count_returns_base() -> None:
    err, last_pair_y = rope_q16_validate_head_range_span_for_dim_checked(128, 23, 0, 17, 16, 2)
    assert err == ref.ROPE_Q16_OK
    assert last_pair_y == 23


def test_known_value_case() -> None:
    err, last_pair_y = rope_q16_validate_head_range_span_for_dim_checked(256, 5, 3, 32, 16, 2)
    assert err == ref.ROPE_Q16_OK
    assert last_pair_y == 84


def test_span_failure_propagates() -> None:
    err, _ = rope_q16_validate_head_range_span_for_dim_checked(
        47,
        0,
        2,
        16,
        32,
        2,
    )
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_overflow_contracts() -> None:
    err, _ = rope_q16_validate_head_range_span_for_dim_checked(
        ref.I64_MAX,
        0,
        ref.I64_MAX,
        ref.I64_MAX,
        16,
        2,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW

    err, _ = rope_q16_validate_head_range_span_for_dim_checked(
        ref.I64_MAX,
        0,
        1,
        1,
        ref.I64_MAX - 1,
        ref.I64_MAX,
    )
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_randomized_reference_parity() -> None:
    rng = random.Random(2026041703)

    for _ in range(7000):
        mode = rng.randint(0, 9)

        if mode <= 6:
            head_cell_capacity = rng.randint(0, 500_000)
            range_base_index = rng.randint(0, 520_000)
            head_count = rng.randint(0, 2048)
            head_stride_cells = rng.randint(1, 2048)
            head_dim = rng.choice([2, 4, 8, 16, 32, 64, 128])
            pair_stride_cells = rng.randint(2, 32)
        else:
            head_cell_capacity = rng.choice([0, 1, ref.I64_MAX - 1, ref.I64_MAX])
            range_base_index = rng.choice([0, 1, ref.I64_MAX - 2, ref.I64_MAX - 1])
            head_count = rng.choice([0, 1, 2, 3, 4, 8, 16, 128])
            head_stride_cells = rng.choice([1, 2, ref.I64_MAX // 2, ref.I64_MAX])
            head_dim = rng.choice([2, 4, 8, 16, ref.I64_MAX - 1])
            pair_stride_cells = rng.choice([2, 3, ref.I64_MAX // 2, ref.I64_MAX])

        err_got, out_got = rope_q16_validate_head_range_span_for_dim_checked(
            head_cell_capacity,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
        )

        err_want, out_want = expected_by_composition(
            head_cell_capacity,
            range_base_index,
            head_count,
            head_stride_cells,
            head_dim,
            pair_stride_cells,
        )

        assert err_got == err_want
        if err_got == ref.ROPE_Q16_OK:
            assert out_got == out_want


def run() -> None:
    test_bad_param_contracts()
    test_zero_head_count_returns_base()
    test_known_value_case()
    test_span_failure_propagates()
    test_overflow_contracts()
    test_randomized_reference_parity()
    print("rope_q16_validate_head_range_span_for_dim_reference_checks=ok")


if __name__ == "__main__":
    run()
