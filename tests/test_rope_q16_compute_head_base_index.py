#!/usr/bin/env python3
"""Reference checks for RoPE Q16 head-base index helper semantics."""

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

    err, head_base_index = rope_try_add_i64_checked(range_base_index, head_offset)
    if err != ref.ROPE_Q16_OK:
        return err, 0

    if head_base_index < 0:
        return ref.ROPE_Q16_ERR_BAD_PARAM, 0

    return ref.ROPE_Q16_OK, head_base_index


def test_bad_param_contracts() -> None:
    assert rope_q16_compute_head_base_index_checked(-1, 0, 16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_compute_head_base_index_checked(0, -1, 16)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_compute_head_base_index_checked(0, 1, 0)[0] == ref.ROPE_Q16_ERR_BAD_PARAM
    assert rope_q16_compute_head_base_index_checked(0, 1, -7)[0] == ref.ROPE_Q16_ERR_BAD_PARAM


def test_known_value_cases() -> None:
    err, got = rope_q16_compute_head_base_index_checked(23, 0, 48)
    assert err == ref.ROPE_Q16_OK
    assert got == 23

    err, got = rope_q16_compute_head_base_index_checked(23, 3, 48)
    assert err == ref.ROPE_Q16_OK
    assert got == 167

    err, got = rope_q16_compute_head_base_index_checked(1024, 9, 64)
    assert err == ref.ROPE_Q16_OK
    assert got == 1600


def test_overflow_contracts() -> None:
    err, _ = rope_q16_compute_head_base_index_checked(0, ref.I64_MAX, 2)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW

    err, _ = rope_q16_compute_head_base_index_checked(ref.I64_MAX, 1, 1)
    assert err == ref.ROPE_Q16_ERR_OVERFLOW

    edge = ref.I64_MAX - 64
    err, got = rope_q16_compute_head_base_index_checked(edge, 1, 64)
    assert err == ref.ROPE_Q16_OK
    assert got == ref.I64_MAX


def test_randomized_matches_python_checked_math() -> None:
    rng = random.Random(20260416140)

    for _ in range(4000):
        range_base = rng.randint(0, ref.I64_MAX)
        head_index = rng.randint(0, 5_000_000)
        head_stride = rng.randint(1, 1_000_000)

        err, got = rope_q16_compute_head_base_index_checked(range_base, head_index, head_stride)

        want = range_base + (head_index * head_stride)
        if want > ref.I64_MAX:
            assert err == ref.ROPE_Q16_ERR_OVERFLOW
            continue

        assert err == ref.ROPE_Q16_OK
        assert got == want


def run() -> None:
    test_bad_param_contracts()
    test_known_value_cases()
    test_overflow_contracts()
    test_randomized_matches_python_checked_math()
    print("rope_q16_compute_head_base_index_reference_checks=ok")


if __name__ == "__main__":
    run()
