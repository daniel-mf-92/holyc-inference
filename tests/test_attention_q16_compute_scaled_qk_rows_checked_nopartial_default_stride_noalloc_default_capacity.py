#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity."""

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

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
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

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

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
        default_stage_cell_capacity,
        staging_capacity_bytes,
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

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_checked_default_capacity_noalloc_composition(
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

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

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
        default_stage_cell_capacity,
        staging_capacity_bytes,
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

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_noalloc_default_capacity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = AttentionTryMulI64Checked(query_row_count," in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )
    assert "staging_out_q16" not in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(" in body


def test_known_vectors_and_stage_capacity_contract() -> None:
    query_row_count = 4
    token_count = 5
    head_dim = 8
    score_scale_q16 = 1 << 16

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_capacity = out_capacity

    q_rows = [((i * 5) - 37) << 4 for i in range(q_capacity)]
    k_rows = [((i * 3) - 19) << 3 for i in range(k_capacity)]

    out_new = [0x4444] * out_capacity
    out_ref = [0x4444] * out_capacity
    stage_new = [0x3333] * stage_capacity
    stage_ref = [0x3333] * stage_capacity

    err_new = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_new,
        out_capacity,
        stage_new,
        stage_capacity,
    )
    err_ref = explicit_checked_default_capacity_noalloc_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_ref,
        out_capacity,
        stage_ref,
        stage_capacity,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref

    out_fail = [0x7777] * out_capacity
    stage_fail = [0x8888] * (stage_capacity - 1)
    err_fail = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_fail,
        out_capacity,
        stage_fail,
        stage_capacity - 1,
    )
    assert err_fail == ATTN_Q16_ERR_BAD_PARAM
    assert out_fail == [0x7777] * out_capacity


def test_error_paths_and_overflow_passthrough() -> None:
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
            None, 0, 0, [0], 1, 0, 0, 0, [0], 1, [0], 1
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
            [0], 1, 1, [0], 1, 1, 1, 1 << 16, [0], 1, None, 1
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
            [0], -1, 1, [0], 1, 1, 1, 1 << 16, [0], 1, [0], 1
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        1 << 16,
        [0],
        I64_MAX,
        [0],
        I64_MAX,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_607)

    for _ in range(3500):
        query_row_count = rng.randint(0, 18)
        token_count = rng.randint(0, 20)
        head_dim = rng.randint(0, 24)
        score_scale_q16 = rng.randint(-(1 << 17), (1 << 17))

        q_required = query_row_count * head_dim
        k_required = token_count * head_dim
        out_required = query_row_count * token_count

        q_capacity = max(0, q_required + rng.randint(-3, 6))
        k_capacity = max(0, k_required + rng.randint(-3, 6))
        out_capacity = max(0, out_required + rng.randint(-3, 6))
        stage_capacity = max(0, out_required + rng.randint(-3, 6))

        if rng.random() < 0.03:
            q_capacity = -rng.randint(1, 8)
        if rng.random() < 0.03:
            k_capacity = -rng.randint(1, 8)
        if rng.random() < 0.03:
            out_capacity = -rng.randint(1, 8)
        if rng.random() < 0.03:
            stage_capacity = -rng.randint(1, 8)

        q_rows = None if rng.random() < 0.04 else [rng.randint(-(1 << 12), (1 << 12)) for _ in range(max(q_capacity, 0))]
        k_rows = None if rng.random() < 0.04 else [rng.randint(-(1 << 12), (1 << 12)) for _ in range(max(k_capacity, 0))]
        out_new = None if rng.random() < 0.04 else [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 0))]
        out_ref = None if out_new is None else out_new.copy()
        stage_new = None if rng.random() < 0.04 else [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(stage_capacity, 0))]
        stage_ref = None if stage_new is None else stage_new.copy()

        if stage_new is not None and q_rows is not None and rng.random() < 0.03:
            stage_new = q_rows
            stage_ref = q_rows
        if stage_new is not None and k_rows is not None and rng.random() < 0.03:
            stage_new = k_rows
            stage_ref = k_rows
        if stage_new is not None and out_new is not None and rng.random() < 0.03:
            stage_new = out_new
            stage_ref = out_ref

        err_new = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_new,
            out_capacity,
            stage_new,
            stage_capacity,
        )
        err_ref = explicit_checked_default_capacity_noalloc_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_ref,
            out_capacity,
            stage_ref,
            stage_capacity,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK and out_new is not None and out_ref is not None:
            assert out_new == out_ref


if __name__ == "__main__":
    test_source_contains_noalloc_default_capacity_helper()
    test_known_vectors_and_stage_capacity_contract()
    test_error_paths_and_overflow_passthrough()
    test_randomized_parity_against_explicit_composition()
    print("ok")
