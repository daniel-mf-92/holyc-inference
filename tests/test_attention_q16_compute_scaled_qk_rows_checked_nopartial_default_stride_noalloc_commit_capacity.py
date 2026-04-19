#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity."""

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
)
from test_attention_q16_compute_scaled_qk_rows_checked import try_mul_i64_checked
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
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

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] > staging_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is q_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_checked_commit_capacity_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
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

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] > staging_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is q_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_noalloc_commit_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )
    assert "AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly("
        in body
    )
    assert "if (required_stage_cells > staged_scores_capacity)" in body
    assert "if (required_stage_bytes > staging_capacity_bytes)" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    query_row_count = 4
    token_count = 6
    head_dim = 8
    score_scale_q16 = 23170

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_cells = query_row_count * token_count
    stage_bytes = stage_cells * 8

    q_rows = [((i * 9) - 37) << 10 for i in range(q_capacity)]
    k_rows = [((41 - (i * 5)) << 11) for i in range(k_capacity)]

    out_a = [0x5A5A] * out_capacity
    out_b = [0x5A5A] * out_capacity
    stage_a = [0x3131] * stage_cells
    stage_b = [0x3131] * stage_cells

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
        stage_cells,
        stage_bytes,
        stage_a,
        stage_cells,
    )
    err_b = explicit_checked_commit_capacity_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
        stage_cells,
        stage_bytes,
        stage_b,
        stage_cells,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_commit_capacity_rejection_is_no_partial() -> None:
    query_row_count = 3
    token_count = 4
    head_dim = 5

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_cells = query_row_count * token_count
    required_bytes = stage_cells * 8

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0x7777] * out_capacity
    out_before = out_scores.copy()

    stage = [0x2222] * stage_cells
    stage_before = stage.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        1 << 16,
        out_scores,
        out_capacity,
        stage_cells,
        required_bytes - 8,
        stage,
        stage_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_scores == out_before
    assert stage == stage_before


def test_compute_overflow_preserves_output() -> None:
    query_row_count = 2
    token_count = 3
    head_dim = 4

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_cells = query_row_count * token_count

    q_rows = [1 << 16] * q_capacity
    k_rows = [1 << 16] * k_capacity
    q_rows[1] = I64_MIN

    out_scores = [0x6666] * out_capacity
    out_before = out_scores.copy()
    stage = [0x4444] * stage_cells
    stage_before = stage.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        1 << 16,
        out_scores,
        out_capacity,
        stage_cells,
        stage_cells * 8,
        stage,
        stage_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out_scores == out_before
    assert stage == stage_before


def test_error_paths() -> None:
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
            None, 1, 1, [0], 1, 1, 1, 1, [0], 1, 1, 8, [0], 1
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
            [0], 1, 1, [0], 1, 1, 1, 1, [0], 1, -1, 8, [0], 1
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
            [0], 1, 1, [0], 1, 1, 1, 1, [0], 1, 1, -1, [0], 1
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        1,
        [0],
        I64_MAX,
        I64_MAX,
        I64_MAX,
        [0],
        huge,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_611)

    for _ in range(4000):
        query_row_count = rng.randint(0, 20)
        token_count = rng.randint(0, 20)
        head_dim = rng.randint(0, 18)
        score_scale_q16 = rng.randint(-(1 << 16), 1 << 16)

        q_cells = query_row_count * head_dim
        k_cells = token_count * head_dim
        out_cells = query_row_count * token_count

        q_cap = q_cells + rng.randint(0, 6)
        k_cap = k_cells + rng.randint(0, 6)
        out_cap = out_cells + rng.randint(0, 8)

        q_rows = [rng.randint(-(1 << 20), 1 << 20) for _ in range(q_cap)]
        k_rows = [rng.randint(-(1 << 20), 1 << 20) for _ in range(k_cap)]

        stage_cap = out_cells + rng.randint(0, 8)
        stage_bytes_cap = stage_cap * 8
        if rng.random() < 0.35 and stage_bytes_cap >= 8:
            stage_bytes_cap -= 8

        staged_a = [rng.randint(-(1 << 12), 1 << 12) for _ in range(max(stage_cap, 1))]
        staged_b = staged_a.copy()

        out_a = [rng.randint(-(1 << 12), 1 << 12) for _ in range(max(out_cap, 1))]
        out_b = out_a.copy()

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
            q_rows,
            q_cap,
            query_row_count,
            k_rows,
            k_cap,
            token_count,
            head_dim,
            score_scale_q16,
            out_a,
            out_cap,
            stage_cap,
            stage_bytes_cap,
            staged_a,
            stage_cap,
        )
        err_b = explicit_checked_commit_capacity_composition(
            q_rows,
            q_cap,
            query_row_count,
            k_rows,
            k_cap,
            token_count,
            head_dim,
            score_scale_q16,
            out_b,
            out_cap,
            stage_cap,
            stage_bytes_cap,
            staged_b,
            stage_cap,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert out_a == out_b
        else:
            assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_noalloc_commit_capacity_wrapper()
    test_known_vectors_match_explicit_checked_composition()
    test_commit_capacity_rejection_is_no_partial()
    test_compute_overflow_preserves_output()
    test_error_paths()
    test_randomized_parity_against_explicit_composition()
    print("ok")
