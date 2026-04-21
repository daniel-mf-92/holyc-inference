#!/usr/bin/env python3
"""Parity harness for AttentionQ16...NoAllocCommitCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_default_stride import (
    attention_q16_apply_score_scale_rows_checked_default_stride,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] > staging_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is in_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_default_stride(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_checked_commit_capacity_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] > staging_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is in_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_default_stride(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_noalloc_commit_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )
    assert "AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(" in body
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly("
        in body
    )
    assert "if (required_stage_cells > staged_scores_capacity)" in body
    assert "if (required_stage_bytes > staging_capacity_bytes)" in body
    assert "if (required_out_cells > out_scores_capacity)" in body
    assert "AttentionTryMulI64Checked(required_stage_cells," in body
    assert "if (required_stage_bytes != recomputed_required_stage_bytes)" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    row_count = 4
    token_count = 6
    score_scale_q16 = 23170

    required_cells = row_count * token_count
    in_capacity = required_cells
    out_capacity = required_cells
    stage_capacity = required_cells
    commit_stage_cell_capacity = required_cells
    commit_stage_byte_capacity = required_cells * 8

    in_scores = [((i * 17) - 91) << 14 for i in range(in_capacity)]

    out_a = [0x5151] * out_capacity
    out_b = [0x5151] * out_capacity
    stage_a = [0x3A3A] * stage_capacity
    stage_b = [0x3A3A] * stage_capacity

    err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        in_scores,
        in_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_a,
        out_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        stage_a,
        stage_capacity,
    )
    err_b = explicit_checked_commit_capacity_composition(
        in_scores,
        in_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_b,
        out_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        stage_b,
        stage_capacity,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_commit_capacity_rejection_is_no_partial() -> None:
    row_count = 3
    token_count = 5
    score_scale_q16 = 32768

    required = row_count * token_count
    in_scores = [7] * required
    out_scores = [0x4444] * required
    staged = [0x7777] * required

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_scores,
        required,
        required,
        required * 8,
        staged,
        required,
    )
    assert err == ATTN_Q16_OK

    out_before = [0xAAAA] * required
    out_after = out_before.copy()
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_after,
        required,
        required,
        required * 8,
        staged,
        required - 1,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_after == out_before


def test_out_capacity_rejection_is_no_partial() -> None:
    row_count = 2
    token_count = 7
    score_scale_q16 = 27500

    required = row_count * token_count
    in_scores = [((i * 13) - 17) << 12 for i in range(required)]
    staged = [0x7B7B] * required

    out_before = [0x5A5A] * (required - 1)
    out_after = out_before.copy()

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_after,
        required - 1,
        required,
        required * 8,
        staged,
        required,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_after == out_before


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_624)

    for _ in range(5000):
        row_count = rng.randint(0, 26)
        token_count = rng.randint(0, 26)
        score_scale_q16 = rng.randint(-65536, 65536)

        if rng.random() < 0.05:
            row_count = -rng.randint(1, 40)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 40)

        required = row_count * token_count if row_count >= 0 and token_count >= 0 else 0

        in_capacity = max(0, required + rng.randint(-20, 20))
        out_capacity = max(0, required + rng.randint(-20, 20))
        stage_capacity = max(0, required + rng.randint(-20, 20))

        if rng.random() < 0.08:
            in_capacity = -rng.randint(1, 30)
        if rng.random() < 0.08:
            out_capacity = -rng.randint(1, 30)
        if rng.random() < 0.08:
            stage_capacity = -rng.randint(1, 30)

        commit_stage_cell_capacity = max(0, required + rng.randint(-20, 20))
        if rng.random() < 0.08:
            commit_stage_cell_capacity = -rng.randint(1, 30)

        commit_stage_byte_capacity = max(0, (required + rng.randint(-20, 20)) * 8)
        if rng.random() < 0.08:
            commit_stage_byte_capacity = -rng.randint(1, 1000)

        in_scores = [rng.randint(-(1 << 40), (1 << 40) - 1) for _ in range(max(1, in_capacity))]
        out_a = [rng.randint(-20000, 20000) for _ in range(max(1, out_capacity))]
        out_b = list(out_a)
        stage_a = [rng.randint(-20000, 20000) for _ in range(max(1, stage_capacity))]
        stage_b = list(stage_a)

        in_ptr = in_scores if rng.random() > 0.03 else None
        out_ptr_a = out_a if rng.random() > 0.03 else None
        out_ptr_b = out_ptr_a
        stage_ptr_a = stage_a if rng.random() > 0.03 else None
        stage_ptr_b = stage_b if stage_ptr_a is not None else None

        if stage_ptr_a is not None and in_ptr is not None and rng.random() < 0.05:
            stage_ptr_a = in_ptr
            stage_ptr_b = in_ptr
        if stage_ptr_a is not None and out_ptr_a is not None and rng.random() < 0.05:
            stage_ptr_a = out_ptr_a
            stage_ptr_b = out_ptr_b

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
            in_ptr,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_ptr_a,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_ptr_a,
            stage_capacity,
        )
        err_b = explicit_checked_commit_capacity_composition(
            in_ptr,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_ptr_b,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_ptr_b,
            stage_capacity,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK and out_ptr_a is not None and out_ptr_b is not None:
            assert out_ptr_a == out_ptr_b

    huge = 1 << 62
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        [1],
        I64_MAX,
        huge,
        huge,
        123,
        [0],
        I64_MAX,
        I64_MAX,
        I64_MAX,
        [0],
        I64_MAX,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
