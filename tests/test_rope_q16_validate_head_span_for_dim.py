#!/usr/bin/env python3
"""Reference checks for RoPEQ16ValidateHeadSpanForDimChecked semantics."""

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


def rope_q16_validate_head_span_for_dim_checked(
    head_cell_capacity: int,
    head_base_index: int,
    head_dim: int,
    pair_stride_cells: int,
) -> tuple[int, int]:
    if head_cell_capacity < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if head_dim <= 0 or (head_dim & 1):
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if pair_stride_cells < 2:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    if head_base_index >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    pair_count = head_dim >> 1

    err, last_pair_index = rope_try_sub_i64_checked(pair_count, 1)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, last_pair_offset = rope_try_mul_i64_checked(last_pair_index, pair_stride_cells)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, last_pair_x = rope_try_add_i64_checked(head_base_index, last_pair_offset)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    err, last_pair_y = rope_try_add_i64_checked(last_pair_x, 1)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    if last_pair_x < 0 or last_pair_y < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0
    if last_pair_y >= head_cell_capacity:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    return ref.ROPE_Q16_OK, last_pair_y


def test_bad_param_contracts() -> None:
    assert rope_q16_validate_head_span_for_dim_checked(-1, 0, 16, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_span_for_dim_checked(128, -1, 16, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_span_for_dim_checked(128, 0, 0, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_span_for_dim_checked(128, 0, 15, 2)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_validate_head_span_for_dim_checked(128, 0, 16, 1)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_known_value_case() -> None:
    err, last_y = rope_q16_validate_head_span_for_dim_checked(128, 7, 16, 2)
    assert err == ref.ROPE_Q16_OK
    assert last_y == 22


def test_capacity_guard() -> None:
    err, _ = rope_q16_validate_head_span_for_dim_checked(22, 7, 16, 2)
    assert err == ref.ROPE_Q16_ERR_BAD_PARAM


def test_overflow_contract() -> None:
    err, _ = rope_q16_validate_head_span_for_dim_checked(ref.I64_MAX, 0, ref.I64_MAX - 1, ref.I64_MAX)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW


def test_randomized_reference_parity() -> None:
    rng = random.Random(2026041701)

    for _ in range(5000):
        head_cell_capacity = rng.randint(0, 400_000)
        head_base_index = rng.randint(0, 420_000)
        head_dim = rng.choice([2, 4, 8, 16, 32, 64])
        pair_stride_cells = rng.randint(2, 64)

        err, last_y = rope_q16_validate_head_span_for_dim_checked(
            head_cell_capacity,
            head_base_index,
            head_dim,
            pair_stride_cells,
        )

        if head_base_index >= head_cell_capacity:
            assert err == ref.ROPE_Q16_ERR_BAD_PARAM
            continue

        pair_count = head_dim >> 1
        expected_last_y = head_base_index + ((pair_count - 1) * pair_stride_cells) + 1

        if expected_last_y > ref.I64_MAX:
            assert err == ref.ROPE_Q16_ERR_OVERFLOW
            continue

        if expected_last_y >= head_cell_capacity:
            assert err == ref.ROPE_Q16_ERR_BAD_PARAM
            continue

        assert err == ref.ROPE_Q16_OK
        assert last_y == expected_last_y


def run() -> None:
    test_bad_param_contracts()
    test_known_value_case()
    test_capacity_guard()
    test_overflow_contract()
    test_randomized_reference_parity()
    print("rope_q16_validate_head_span_for_dim_reference_checks=ok")


if __name__ == "__main__":
    run()
