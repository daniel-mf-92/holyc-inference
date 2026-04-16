#!/usr/bin/env python3
"""Focused parity harness for checked Q8_0 matrix×vector row kernels."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_ERR_OVERFLOW,
    Q8_0_I64_MAX,
    Q8_0_OK,
    Q8_0_VALUES_PER_BLOCK,
    dot_rows_q16_matrix_vector,
    dot_rows_q16_matrix_vector_checked,
    half_bits,
    pack_signed,
    q8_0_dot_rows_q16_matrix_vector_checked,
)


def make_signed_block(rng: random.Random, *, scale: float | None = None) -> tuple[int, bytes]:
    if scale is None:
        scale = rng.uniform(-2.0, 2.0)
    return (
        half_bits(scale),
        pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]),
    )


def test_checked_matches_unchecked_randomized() -> None:
    rng = random.Random(2026041601)

    for _ in range(320):
        row_count = rng.randint(1, 9)
        vec_block_count = rng.randint(1, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)
        matrix_capacity = row_count * row_stride_blocks

        vec_blocks = [make_signed_block(rng) for _ in range(vec_block_count)]
        matrix_blocks = [make_signed_block(rng) for _ in range(matrix_capacity)]

        err_unchecked, rows_unchecked = dot_rows_q16_matrix_vector(
            matrix_blocks,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
        )
        assert err_unchecked == Q8_0_OK

        err_checked, rows_checked = dot_rows_q16_matrix_vector_checked(
            matrix_blocks,
            matrix_capacity,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
            vec_block_count,
        )
        assert err_checked == Q8_0_OK
        assert rows_checked == rows_unchecked


def test_checked_capacity_bounds_and_extents() -> None:
    rng = random.Random(2026041602)
    vec_blocks = [make_signed_block(rng) for _ in range(2)]
    matrix_blocks = [make_signed_block(rng) for _ in range(4)]

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, -1, 2, 2, vec_blocks, 2, 2)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 4, 2, 2, vec_blocks, -1, 2)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 4, 2, 1, vec_blocks, 2, 2)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 4, 2, 2, vec_blocks, 1, 2)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 3, 2, 2, vec_blocks, 2, 2)
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_checked_reports_matrix_extent_multiply_overflow() -> None:
    rng = random.Random(2026041603)
    vec_blocks = [make_signed_block(rng)]
    matrix_blocks = [make_signed_block(rng)]

    err, _ = dot_rows_q16_matrix_vector_checked(
        matrix_blocks,
        Q8_0_I64_MAX,
        Q8_0_I64_MAX,
        2,
        vec_blocks,
        1,
        1,
    )
    assert err == Q8_0_ERR_OVERFLOW


def test_checked_reports_row_accumulator_overflow_vectors() -> None:
    # Each block here is intentionally huge so that checked Q16 accumulation
    # overflows even with a small block count.
    max_pos = pack_signed([127] * Q8_0_VALUES_PER_BLOCK)
    max_neg = pack_signed([-128] * Q8_0_VALUES_PER_BLOCK)

    vec_blocks = [
        (half_bits(65504.0), max_pos),
        (half_bits(65504.0), max_pos),
    ]

    matrix_pos = [
        (half_bits(65504.0), max_pos),
        (half_bits(65504.0), max_pos),
    ]
    err, _ = dot_rows_q16_matrix_vector_checked(
        matrix_pos,
        2,
        1,
        2,
        vec_blocks,
        2,
        2,
    )
    assert err == Q8_0_ERR_OVERFLOW

    matrix_neg = [
        (half_bits(-65504.0), max_neg),
        (half_bits(-65504.0), max_neg),
    ]
    err, _ = dot_rows_q16_matrix_vector_checked(
        matrix_neg,
        2,
        1,
        2,
        vec_blocks,
        2,
        2,
    )
    assert err == Q8_0_ERR_OVERFLOW


def test_checked_wrapper_matches_core_randomized() -> None:
    rng = random.Random(2026041604)

    for _ in range(140):
        row_count = rng.randint(1, 7)
        vec_block_count = rng.randint(1, 4)
        row_stride_blocks = vec_block_count + rng.randint(0, 3)

        vec_blocks = [make_signed_block(rng) for _ in range(vec_block_count)]
        matrix_capacity = row_count * row_stride_blocks
        matrix_blocks = [make_signed_block(rng) for _ in range(matrix_capacity)]

        err_core, rows_core = dot_rows_q16_matrix_vector_checked(
            matrix_blocks,
            matrix_capacity,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
            vec_block_count,
        )
        err_wrapper, rows_wrapper = q8_0_dot_rows_q16_matrix_vector_checked(
            matrix_blocks,
            matrix_capacity,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
            vec_block_count,
        )

        assert err_core == err_wrapper
        assert rows_core == rows_wrapper


def run() -> None:
    test_checked_matches_unchecked_randomized()
    test_checked_capacity_bounds_and_extents()
    test_checked_reports_matrix_extent_multiply_overflow()
    test_checked_reports_row_accumulator_overflow_vectors()
    test_checked_wrapper_matches_core_randomized()
    print("q8_0_dot_matrix_vector_checked_checks=ok")


if __name__ == "__main__":
    run()
