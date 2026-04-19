#!/usr/bin/env python3
"""Parity checks for Q8_0DotRowsAVX2Q32CheckedDefaultNoPartial semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_q8_0_avx2_blocks_q32 import (
    I64_MAX,
    Q8_0_AVX2_ERR_BAD_LEN,
    Q8_0_AVX2_ERR_NULL_PTR,
    Q8_0_AVX2_ERR_OVERFLOW,
    Q8_0_AVX2_OK,
    make_block,
)
from test_q8_0_avx2_dot_rows_q32_checked_default import (
    q8_0_dot_rows_avx2_q32_checked_default,
)


def q8_0_dot_rows_avx2_q32_checked_default_ptr(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if matrix_blocks is None or vec_blocks is None or out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    err, rows_q32 = q8_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["rows"] = rows_q32
    return Q8_0_AVX2_OK


def q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if matrix_blocks is None or vec_blocks is None or out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    err, staged_rows_q32 = q8_0_dot_rows_avx2_q32_checked_default(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
    )
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["rows"] = staged_rows_q32
    return Q8_0_AVX2_OK


def q8_0_dot_rows_avx2_q32_checked_default_no_partial_explicit_staged_ptr(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
    out_holder,
):
    if matrix_blocks is None or vec_blocks is None or out_holder is None:
        return Q8_0_AVX2_ERR_NULL_PTR

    if row_count < 0:
        return Q8_0_AVX2_ERR_BAD_LEN

    stage_rows = [0] * row_count
    stage_holder = {"rows": stage_rows}

    err = q8_0_dot_rows_avx2_q32_checked_default_ptr(
        matrix_blocks=matrix_blocks,
        row_count=row_count,
        row_stride_blocks=row_stride_blocks,
        vec_blocks=vec_blocks,
        vec_block_count=vec_block_count,
        out_holder=stage_holder,
    )
    if err != Q8_0_AVX2_OK:
        return err

    out_holder["rows"] = stage_holder["rows"]
    return Q8_0_AVX2_OK


def test_default_no_partial_matches_default_wrapper_on_success_and_error() -> None:
    rng = random.Random(2026041901)

    for _ in range(280):
        row_count = rng.randint(0, 7)
        vec_block_count = rng.randint(0, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)
        matrix_capacity = row_count * row_stride_blocks

        matrix = [
            make_block(
                rng.choice([0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0x4400, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(matrix_capacity)
        ]
        vec = [
            make_block(
                rng.choice([0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0x4400, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(vec_block_count)
        ]

        out_no_partial = {"rows": [111, 222, 333]}
        out_default = {"rows": [444, 555, 666]}

        err_no_partial = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_no_partial,
        )
        err_default = q8_0_dot_rows_avx2_q32_checked_default_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_default,
        )

        assert err_no_partial == err_default
        if err_no_partial == Q8_0_AVX2_OK:
            assert out_no_partial["rows"] == out_default["rows"]
        else:
            assert out_no_partial["rows"] == [111, 222, 333]


def test_default_no_partial_matches_explicit_staged_composition() -> None:
    rng = random.Random(2026041902)

    for _ in range(360):
        row_count = rng.randint(0, 8)
        vec_block_count = rng.randint(0, 7)

        # Include malformed row/stride relationships in the randomized corpus
        # so BAD_LEN gating parity is exercised heavily.
        if row_count == 0:
            row_stride_blocks = rng.randint(0, 2)
        else:
            row_stride_blocks = max(0, vec_block_count + rng.randint(-3, 3))

        matrix_capacity = row_count * max(row_stride_blocks, 0)

        matrix = [
            make_block(
                rng.choice([0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0x4400, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(matrix_capacity)
        ]
        vec = [
            make_block(
                rng.choice([0x0000, 0x1800, 0x2800, 0x3000, 0x3400, 0x3800, 0x3C00, 0x4000, 0x4400, 0xBC00]),
                [rng.randint(-128, 127) for _ in range(32)],
            )
            for _ in range(vec_block_count)
        ]

        out_no_partial = {"rows": [17, 23, 42]}
        out_staged = {"rows": [61, 62, 63]}

        err_no_partial = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_no_partial,
        )
        err_staged = q8_0_dot_rows_avx2_q32_checked_default_no_partial_explicit_staged_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_staged,
        )

        assert err_no_partial == err_staged
        if err_no_partial == Q8_0_AVX2_OK:
            assert out_no_partial["rows"] == out_staged["rows"]
        else:
            assert out_no_partial["rows"] == [17, 23, 42]


def test_default_no_partial_rejects_bad_len_and_overflow_without_commit() -> None:
    block = make_block(0x3C00, [1] * 32)
    out_holder = {"rows": [909, 808, 707]}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=-1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_ERR_BAD_LEN
    assert out_holder["rows"] == [909, 808, 707]

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
        matrix_blocks=[],
        row_count=I64_MAX,
        row_stride_blocks=2,
        vec_blocks=[],
        vec_block_count=0,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_ERR_OVERFLOW
    assert out_holder["rows"] == [909, 808, 707]


def test_default_no_partial_rejects_null_inputs_without_commit() -> None:
    block = make_block(0x3C00, [0] * 32)
    out_holder = {"rows": [77, 88]}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
        matrix_blocks=[block],
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=None,
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["rows"] == [77, 88]

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
        matrix_blocks=None,
        row_count=1,
        row_stride_blocks=1,
        vec_blocks=[block],
        vec_block_count=1,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_ERR_NULL_PTR
    assert out_holder["rows"] == [77, 88]


def test_default_no_partial_preserves_output_when_late_row_overflows() -> None:
    vec = [make_block(0x3C00, [127] * 32)]

    # Row 0 succeeds; row 1 uses fp16 Inf-scale and forces overflow.
    row0 = [make_block(0x3C00, [1] + [0] * 31)]
    row1 = [make_block(0x7C00, [127] * 32)]
    matrix = row0 + row1

    out_no_partial = {"rows": [501, 502]}
    out_default = {"rows": [601, 602]}

    err_no_partial = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
        matrix_blocks=matrix,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_count=1,
        out_holder=out_no_partial,
    )
    err_default = q8_0_dot_rows_avx2_q32_checked_default_ptr(
        matrix_blocks=matrix,
        row_count=2,
        row_stride_blocks=1,
        vec_blocks=vec,
        vec_block_count=1,
        out_holder=out_default,
    )

    assert err_no_partial == err_default == Q8_0_AVX2_ERR_OVERFLOW
    assert out_no_partial["rows"] == [501, 502]
    assert out_default["rows"] == [601, 602]


def test_default_no_partial_saturation_scale_fixtures_match_explicit_staged() -> None:
    rng = random.Random(2026041903)
    scales = [0x0000, 0x7C00, 0xFC00, 0x0400, 0x3C00, 0x4400, 0xBC00]

    for _ in range(120):
        row_count = rng.randint(1, 4)
        vec_block_count = rng.randint(1, 3)
        row_stride_blocks = vec_block_count + rng.randint(0, 2)

        matrix = []
        for _ in range(row_count * row_stride_blocks):
            matrix.append(
                make_block(
                    rng.choice(scales),
                    [rng.randint(-128, 127) for _ in range(32)],
                )
            )

        vec = []
        for _ in range(vec_block_count):
            vec.append(
                make_block(
                    rng.choice(scales),
                    [rng.randint(-128, 127) for _ in range(32)],
                )
            )

        out_no_partial = {"rows": [7001, 7002, 7003, 7004]}
        out_staged = {"rows": [8001, 8002, 8003, 8004]}

        err_no_partial = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_no_partial,
        )
        err_staged = q8_0_dot_rows_avx2_q32_checked_default_no_partial_explicit_staged_ptr(
            matrix_blocks=matrix,
            row_count=row_count,
            row_stride_blocks=row_stride_blocks,
            vec_blocks=vec,
            vec_block_count=vec_block_count,
            out_holder=out_staged,
        )

        assert err_no_partial == err_staged
        if err_no_partial == Q8_0_AVX2_OK:
            assert out_no_partial["rows"] == out_staged["rows"]
        else:
            assert out_no_partial["rows"] == [7001, 7002, 7003, 7004]


def test_default_no_partial_zero_rows_success_commits_empty() -> None:
    out_holder = {"rows": [11, 22, 33]}

    err = q8_0_dot_rows_avx2_q32_checked_default_no_partial_ptr(
        matrix_blocks=[],
        row_count=0,
        row_stride_blocks=0,
        vec_blocks=[],
        vec_block_count=0,
        out_holder=out_holder,
    )
    assert err == Q8_0_AVX2_OK
    assert out_holder["rows"] == []


def run() -> None:
    test_default_no_partial_matches_default_wrapper_on_success_and_error()
    test_default_no_partial_matches_explicit_staged_composition()
    test_default_no_partial_rejects_bad_len_and_overflow_without_commit()
    test_default_no_partial_rejects_null_inputs_without_commit()
    test_default_no_partial_preserves_output_when_late_row_overflows()
    test_default_no_partial_saturation_scale_fixtures_match_explicit_staged()
    test_default_no_partial_zero_rows_success_commits_empty()
    print("q8_0_avx2_dot_rows_q32_checked_default_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
