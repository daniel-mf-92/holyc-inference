#!/usr/bin/env python3
"""Parity harness for ...NoAllocRequiredBytesDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
from test_attention_q16_apply_score_scale_checked import (  # noqa: E402
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, stage_cell_capacity = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_default_capacity_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, stage_cell_capacity = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_noalloc_required_bytes_default_capacity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = AttentionTryMulI64Checked(row_count," in body
    assert "token_count," in body
    assert (
        "return AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes("
        in body
    )


def test_known_vectors_and_zero_case() -> None:
    row_count = 5
    token_count = 1

    required_cells = row_count * token_count
    in_scores = [7] * required_cells
    out_scores = [0] * required_cells

    out_stage_cells = [111]
    out_stage_bytes = [222]
    out_out_cells = [333]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        out_scores,
        len(out_scores),
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [required_cells]
    assert out_stage_bytes == [required_cells * 8]
    assert out_out_cells == [required_cells]

    out_stage_cells = [9]
    out_stage_bytes = [9]
    out_out_cells = [9]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        1,
        0,
        7,
        [0],
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [0]
    assert out_stage_bytes == [0]
    assert out_out_cells == [0]


def test_error_paths() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            [0], 1, 1, 1, [0], 1, None, out_stage_bytes, out_out_cells
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            None,
            1,
            1,
            1,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            [0],
            -1,
            1,
            1,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            [0],
            1,
            -1,
            1,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
        [0],
        I64_MAX,
        huge,
        huge,
        [0],
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_622)

    for _ in range(7000):
        row_count = rng.randint(0, 220)
        token_count = rng.randint(0, 220)

        if rng.random() < 0.06:
            row_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            token_count = -rng.randint(1, 80)

        if rng.random() < 0.02:
            row_count = (1 << 62) + rng.randint(0, 31)
            token_count = (1 << 62) + rng.randint(0, 31)

        huge_shape = row_count > 2000 or token_count > 2000
        if huge_shape:
            in_scores = [0]
            out_scores = [0]
            in_capacity = I64_MAX
            out_capacity = I64_MAX
        else:
            if row_count > 0 and token_count > 0:
                cell_count = (row_count - 1) * token_count + (token_count - 1) * token_count + 1
                cell_count = max(1, cell_count)
            else:
                cell_count = 1
            in_scores = [0] * cell_count
            out_scores = [0] * cell_count
            in_capacity = cell_count
            out_capacity = cell_count

        got_stage_cells = [0xAA]
        got_stage_bytes = [0xBB]
        got_out_cells = [0xCC]

        exp_stage_cells = [0xAA]
        exp_stage_bytes = [0xBB]
        exp_out_cells = [0xCC]

        err_new = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_default_capacity(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_ref = explicit_checked_default_capacity_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_new == err_ref
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes
        assert got_out_cells == exp_out_cells


if __name__ == "__main__":
    test_source_contains_noalloc_required_bytes_default_capacity_helper()
    test_known_vectors_and_zero_case()
    test_error_paths()
    test_randomized_parity_against_explicit_composition()
    print("ok")
