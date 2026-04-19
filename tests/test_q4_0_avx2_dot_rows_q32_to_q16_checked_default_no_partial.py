#!/usr/bin/env python3
"""Parity harness for Q4_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartial."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q4_0_avx2_dot_rows_q32_checked import (
    I64_MAX,
    Q4_0_AVX2_ERR_BAD_LEN,
    Q4_0_AVX2_ERR_NULL_PTR,
    Q4_0_AVX2_ERR_OVERFLOW,
    Q4_0_AVX2_OK,
    make_block,
)
from test_q4_0_avx2_dot_rows_q32_to_q16_checked_default import (
    q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr,
)


def q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    staged_out = {"rows": []}
    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
        out_holder=staged_out,
    )
    if err != Q4_0_AVX2_OK:
        return err

    out_holder["rows"] = staged_out["rows"]
    return Q4_0_AVX2_OK


def staged_composition_reference(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if out_holder is None:
        return Q4_0_AVX2_ERR_NULL_PTR

    staged_rows = []
    staged_out = {"rows": staged_rows}
    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_ptr(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
        out_holder=staged_out,
    )
    if err != Q4_0_AVX2_OK:
        return err

    out_holder["rows"] = staged_out["rows"]
    return Q4_0_AVX2_OK


def _make_random_matrix_and_vec(rng: random.Random, row_count: int, row_stride_blocks: int, vec_block_count: int):
    matrix = []
    for _ in range(row_count * row_stride_blocks):
        matrix.append(
            make_block(
                rng.choice([0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0]),
                [rng.randint(-8, 7) for _ in range(32)],
            )
        )

    vec = [
        make_block(
            rng.choice([0.125, 0.25, 0.5, 1.0, -0.25, -0.5, -1.0]),
            [rng.randint(-8, 7) for _ in range(32)],
        )
        for _ in range(vec_block_count)
    ]

    return matrix, vec


def test_no_partial_wrapper_matches_checked_default_success_and_errors() -> None:
    rng = random.Random(202604191483)

    for _ in range(360):
        row_count = rng.randint(0, 6)
        vec_block_count = rng.randint(0, 5)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)

        matrix, vec = _make_random_matrix_and_vec(rng, row_count, row_stride_blocks, vec_block_count)

        out_no_partial = {"rows": [101, 202, 303]}
        out_staged_ref = {"rows": [404, 505, 606]}

        err_no_partial = q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_no_partial,
        )
        err_ref = staged_composition_reference(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_staged_ref,
        )

        assert err_no_partial == err_ref
        if err_ref == Q4_0_AVX2_OK:
            assert out_no_partial["rows"] == out_staged_ref["rows"]
        else:
            assert out_no_partial["rows"] == [101, 202, 303]


def test_no_partial_wrapper_rejects_bad_len_and_null_without_commit() -> None:
    block = make_block(1.0, [0] * 32)
    out = {"rows": [9, 9]}

    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=out,
    )
    assert err == Q4_0_AVX2_ERR_BAD_LEN
    assert out["rows"] == [9, 9]

    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=None,
        vec_block_count=1,
        out_holder=out,
    )
    assert err == Q4_0_AVX2_ERR_NULL_PTR
    assert out["rows"] == [9, 9]


def test_no_partial_wrapper_preserves_output_on_late_failure() -> None:
    vec = [make_block(1.0, [1] * 32)]

    row0 = [make_block(1.0, [1] * 32)]
    # Malformed packed payload in row1 forces a checked BAD_LEN after row0
    # would otherwise succeed, validating all-or-nothing commit behavior.
    row1 = [(row0[0][0], row0[0][1][:15])]
    matrix = row0 + row1

    out_no_partial = {"rows": [7001, 7002]}
    out_ref = {"rows": [8001, 8002]}

    err_no_partial = q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=matrix,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_count=1,
        out_holder=out_no_partial,
    )
    err_ref = staged_composition_reference(
        matrix_blocks=matrix,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_count=1,
        out_holder=out_ref,
    )

    assert err_no_partial == err_ref == Q4_0_AVX2_ERR_BAD_LEN
    assert out_no_partial["rows"] == [7001, 7002]
    assert out_ref["rows"] == [8001, 8002]


def test_no_partial_wrapper_zero_rows_success_no_writes() -> None:
    out = {"rows": [1111, 2222, 3333]}

    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[],
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=[],
        vec_block_count=0,
        out_holder=out,
    )
    assert err == Q4_0_AVX2_OK
    assert out["rows"] == []


def test_default_capacity_overflow_no_partial() -> None:
    out = {"rows": [42]}

    err = q4_0_dot_rows_avx2_q32_to_q16_checked_default_no_partial_ptr(
        matrix_blocks=[],
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_count=0,
        out_holder=out,
    )
    assert err == Q4_0_AVX2_ERR_OVERFLOW
    assert out["rows"] == [42]


def test_source_contains_no_partial_wrapper() -> None:
    source = pathlib.Path("src/quant/q4_0_avx2.HC").read_text(encoding="utf-8")
    assert "I32 Q4_0DotRowsAVX2Q32ToQ16CheckedDefaultNoPartial(" in source
    assert "Q4_0DotRowsAVX2Q32ToQ16CheckedDefault(matrix_blocks," in source
    assert "staged_rows_q16" in source
    assert "out_rows_q16[row_index] = staged_rows_q16[row_index];" in source


if __name__ == "__main__":
    test_no_partial_wrapper_matches_checked_default_success_and_errors()
    test_no_partial_wrapper_rejects_bad_len_and_null_without_commit()
    test_no_partial_wrapper_preserves_output_on_late_failure()
    test_no_partial_wrapper_zero_rows_success_no_writes()
    test_default_capacity_overflow_no_partial()
    test_source_contains_no_partial_wrapper()
    print("q4_0_avx2_dot_rows_q32_to_q16_checked_default_no_partial_parity=ok")
