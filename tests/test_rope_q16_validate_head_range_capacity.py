#!/usr/bin/env python3
"""Reference checks for RoPEQ16ValidateHeadRangeCapacityChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_rope_q16_angle_step as ref


def rope_try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > ref.I64_MAX - rhs:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < ref.I64_MIN - rhs:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, lhs + rhs


def rope_try_sub_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs == ref.I64_MIN:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return rope_try_add_i64_checked(lhs, -rhs)


def rope_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    prod = lhs * rhs
    if prod > ref.I64_MAX or prod < ref.I64_MIN:
        return ref.ROPE_Q16_ERR_OVERFLOW, 0
    return ref.ROPE_Q16_OK, prod


def rope_q16_compute_head_base_index_checked(
    range_base_index: int,
    head_index: int,
    head_stride_cells: int,
) -> tuple[int, int]:
    if range_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_stride_cells <= 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    err, head_offset = rope_try_mul_i64_checked(head_index, head_stride_cells)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, head_base = rope_try_add_i64_checked(range_base_index, head_offset)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    if head_base < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    return ref.ROPE_Q16_OK, head_base


def rope_q16_validate_head_range_capacity_checked(
    head_cell_capacity: int,
    range_base_index: int,
    head_count: int,
    head_stride_cells: int,
) -> tuple[int, int]:
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if range_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_count < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_stride_cells <= 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    if head_count == 0:
        return ref.ROPE_Q16_OK, range_base_index

    if range_base_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    err, last_head_index = rope_try_sub_i64_checked(head_count, 1)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, last_head_base = rope_q16_compute_head_base_index_checked(
        range_base_index,
        last_head_index,
        head_stride_cells,
    )
    if err != ref.ROPE_Q16_OK:
        return err, 0

    if last_head_base >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    return ref.ROPE_Q16_OK, last_head_base


def test_bad_param_contracts() -> None:
    assert rope_q16_validate_head_range_capacity_checked(-1, 0, 1, 16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_range_capacity_checked(64, -1, 1, 16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_range_capacity_checked(64, 0, -1, 16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_range_capacity_checked(64, 0, 1, 0)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_zero_head_count_returns_base() -> None:
    err, last_base = rope_q16_validate_head_range_capacity_checked(64, 23, 0, 16)
    assert err == ref.ROPE_Q16_OK
    assert last_base == 23


def test_known_value_cases() -> None:
    err, last_base = rope_q16_validate_head_range_capacity_checked(128, 8, 3, 24)
    assert err == ref.ROPE_Q16_OK
    assert last_base == 56

    err, _ = rope_q16_validate_head_range_capacity_checked(56, 8, 3, 24)
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_overflow_contracts() -> None:
    err, _ = rope_q16_validate_head_range_capacity_checked(ref.I64_MAX, 0, ref.I64_MAX, ref.I64_MAX)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_randomized_reference_parity() -> None:
    rng = random.Random(20260416141)

    for _ in range(5000):
        head_cell_capacity = rng.randint(0, 200_000)
        range_base_index = rng.randint(0, 220_000)
        head_count = rng.randint(0, 2000)
        head_stride_cells = rng.randint(1, 1024)

        err, last_base = rope_q16_validate_head_range_capacity_checked(
            head_cell_capacity,
            range_base_index,
            head_count,
            head_stride_cells,
        )

        if head_count == 0:
            assert err == ref.ROPE_Q16_OK
            assert last_base == range_base_index
            continue

        if range_base_index >= head_cell_capacity:
            assert err == ref.ROPE_Q16_ERR_BAD_PARAM
            continue

        wanted_last = range_base_index + ((head_count - 1) * head_stride_cells)
        if wanted_last > ref.I64_MAX:
            assert err == ref.ROPE_Q16_ERR_OVERFLOW
            continue

        if wanted_last >= head_cell_capacity:
            assert err == ref.ROPE_Q16_ERR_BAD_PARAM
            continue

        assert err == ref.ROPE_Q16_OK
        assert last_base == wanted_last


def run() -> None:
    test_bad_param_contracts()
    test_zero_head_count_returns_base()
    test_known_value_cases()
    test_overflow_contracts()
    test_randomized_reference_parity()
    print("rope_q16_validate_head_range_capacity_reference_checks=ok")


if __name__ == "__main__":
    run()
