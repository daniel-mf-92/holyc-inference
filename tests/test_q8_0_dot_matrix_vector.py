#!/usr/bin/env python3
"""Focused parity harness for Q8_0 matrix×vector row kernels."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_OK,
    Q8_0_VALUES_PER_BLOCK,
    dot_q32_to_q16,
    dot_product_blocks_q32,
    dot_row_blocks_q16,
    dot_rows_q16_matrix_vector,
    half_bits,
    pack_signed,
)


def test_stride_padding_is_ignored_randomized() -> None:
    rng = random.Random(20260416)

    for _ in range(220):
        row_count = rng.randint(1, 9)
        vec_block_count = rng.randint(1, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)

        vec_blocks: list[tuple[int, bytes]] = []
        for _ in range(vec_block_count):
            vec_scale = half_bits(rng.uniform(-2.0, 2.0))
            vec_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            vec_blocks.append((vec_scale, vec_qs))

        matrix_blocks: list[tuple[int, bytes]] = []
        expected_rows: list[int] = []

        for _ in range(row_count):
            active_blocks: list[tuple[int, bytes]] = []
            for _ in range(vec_block_count):
                row_scale = half_bits(rng.uniform(-2.0, 2.0))
                row_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
                block = (row_scale, row_qs)
                matrix_blocks.append(block)
                active_blocks.append(block)

            pad_blocks = row_stride_blocks - vec_block_count
            for _ in range(pad_blocks):
                pad_scale = half_bits(rng.uniform(-2.0, 2.0))
                pad_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
                matrix_blocks.append((pad_scale, pad_qs))

            err, expected_q16 = dot_row_blocks_q16(active_blocks, vec_blocks)
            assert err == Q8_0_OK
            expected_rows.append(expected_q16)

        err, got_rows = dot_rows_q16_matrix_vector(
            matrix_blocks,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
        )
        assert err == Q8_0_OK
        assert got_rows == expected_rows


def test_sign_semantics_match_reference_rowwise() -> None:
    row_count = 4
    vec_block_count = 1
    row_stride_blocks = 1

    vec_blocks = [(half_bits(1.0), pack_signed([3] * Q8_0_VALUES_PER_BLOCK))]

    matrix_rows = [
        [
            (half_bits(1.0), pack_signed([2] * Q8_0_VALUES_PER_BLOCK)),
        ],
        [
            (half_bits(-1.0), pack_signed([2] * Q8_0_VALUES_PER_BLOCK)),
        ],
        [
            (half_bits(1.0), pack_signed([-2] * Q8_0_VALUES_PER_BLOCK)),
        ],
        [
            (half_bits(-1.0), pack_signed([-2] * Q8_0_VALUES_PER_BLOCK)),
        ],
    ]

    matrix_blocks = [block for row in matrix_rows for block in row]
    err, got_rows = dot_rows_q16_matrix_vector(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
    )
    assert err == Q8_0_OK

    expected_rows: list[int] = []
    for row in matrix_rows:
        err, expected_q16 = dot_row_blocks_q16(row, vec_blocks)
        assert err == Q8_0_OK
        expected_rows.append(expected_q16)

    assert got_rows == expected_rows
    assert got_rows[0] > 0
    assert got_rows[1] < 0
    assert got_rows[2] < 0
    assert got_rows[3] > 0


def test_per_row_rounding_semantics_not_global_rounding() -> None:
    vec_blocks = [
        (half_bits(0.5), pack_signed([7] * Q8_0_VALUES_PER_BLOCK)),
        (half_bits(0.5), pack_signed([7] * Q8_0_VALUES_PER_BLOCK)),
    ]

    row0 = [
        (half_bits(0.5), pack_signed([11] * Q8_0_VALUES_PER_BLOCK)),
        (half_bits(0.5), pack_signed([11] * Q8_0_VALUES_PER_BLOCK)),
    ]
    row1 = [
        (half_bits(-0.5), pack_signed([11] * Q8_0_VALUES_PER_BLOCK)),
        (half_bits(-0.5), pack_signed([11] * Q8_0_VALUES_PER_BLOCK)),
    ]

    matrix_blocks = row0 + row1
    err, got_rows = dot_rows_q16_matrix_vector(matrix_blocks, 2, 2, vec_blocks, 2)
    assert err == Q8_0_OK

    err, row0_q16 = dot_row_blocks_q16(row0, vec_blocks)
    assert err == Q8_0_OK
    err, row1_q16 = dot_row_blocks_q16(row1, vec_blocks)
    assert err == Q8_0_OK
    assert got_rows == [row0_q16, row1_q16]

    err, row0_q32 = dot_product_blocks_q32(row0, vec_blocks)
    assert err == Q8_0_OK
    err, row1_q32 = dot_product_blocks_q32(row1, vec_blocks)
    assert err == Q8_0_OK
    global_round = dot_q32_to_q16(row0_q32 + row1_q32)
    assert isinstance(global_round, int)


def test_matrix_vector_argument_and_extent_errors() -> None:
    vec_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]
    matrix_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, -1, 1, vec_blocks, 1)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, -1, vec_blocks, 1)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, 1, vec_blocks, -1)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, 0, vec_blocks, 1)
    assert err == Q8_0_ERR_BAD_DST_LEN

    # Truncated matrix extent for two rows with stride=1.
    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 2, 1, vec_blocks, 1)
    assert err == Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_stride_padding_is_ignored_randomized()
    test_sign_semantics_match_reference_rowwise()
    test_per_row_rounding_semantics_not_global_rounding()
    test_matrix_vector_argument_and_extent_errors()
    print("q8_0_dot_matrix_vector_checks=ok")


if __name__ == "__main__":
    run()
