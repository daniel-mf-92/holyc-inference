#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly."""

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
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    stage_cell_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or stage_cell_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    default_out_row_stride = token_count
    err, required_out_cells = try_mul_i64_checked(
        query_row_count - 1, default_out_row_stride
    )
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_checked_commit_composition(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    stage_cell_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or stage_cell_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_noalloc_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "required_stage_cells" in body
    assert "default_out_row_stride = token_count;" in body
    assert "if (required_stage_cells > stage_cell_capacity)" in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideCommitOnly(" in body


def test_known_vectors_and_stage_capacity_gate() -> None:
    query_row_count = 4
    token_count = 5
    total = query_row_count * token_count

    staged = [((i * 7) - 33) << 9 for i in range(total)]
    out_new = [0x1212] * total
    out_ref = [0x1212] * total

    err_new = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged,
        total,
        total,
        out_new,
        total,
    )
    err_ref = explicit_checked_commit_composition(
        query_row_count,
        token_count,
        staged,
        total,
        total,
        out_ref,
        total,
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref == staged

    out_fail = [0x8888] * total
    err_fail = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged,
        total,
        total - 1,
        out_fail,
        total,
    )
    assert err_fail == ATTN_Q16_ERR_BAD_PARAM
    assert out_fail == [0x8888] * total


def test_adversarial_error_contracts() -> None:
    sample_staged = [1, 2, 3, 4]
    sample_out = [0, 0, 0, 0]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
            1, 1, None, 1, 1, sample_out, 1
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
            1, 1, sample_staged, 1, 1, None, 1
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
            -1, 1, sample_staged, 4, 4, sample_out, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
            1, 1, sample_staged, -1, 4, sample_out, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        huge,
        huge,
        [1],
        (1 << 63) - 1,
        (1 << 63) - 1,
        [0],
        (1 << 63) - 1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_599)

    for _ in range(4500):
        query_row_count = rng.randint(0, 30)
        token_count = rng.randint(0, 30)
        required = query_row_count * token_count

        staged_capacity = max(0, required + rng.randint(-10, 10))
        stage_cell_capacity = max(0, required + rng.randint(-10, 10))
        out_capacity = max(0, required + rng.randint(-10, 10))

        if rng.random() < 0.05:
            staged_capacity = -rng.randint(1, 30)
        if rng.random() < 0.05:
            stage_cell_capacity = -rng.randint(1, 30)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 30)

        if rng.random() < 0.05:
            query_row_count = (1 << 62) + rng.randint(0, 8)
            token_count = (1 << 62) + rng.randint(0, 8)

        staged = (
            None
            if rng.random() < 0.03
            else [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(staged_capacity, 1))]
        )
        out_seed = [rng.randint(-(1 << 12), (1 << 12)) for _ in range(max(out_capacity, 1))]
        out_new = out_seed.copy()
        out_ref = out_seed.copy()

        err_new = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
            query_row_count,
            token_count,
            staged,
            staged_capacity,
            stage_cell_capacity,
            out_new,
            out_capacity,
        )
        err_ref = explicit_checked_commit_composition(
            query_row_count,
            token_count,
            staged,
            staged_capacity,
            stage_cell_capacity,
            out_ref,
            out_capacity,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert out_new == out_ref
        else:
            assert out_new == out_ref == out_seed


if __name__ == "__main__":
    test_source_contains_noalloc_commit_only_helper()
    test_known_vectors_and_stage_capacity_gate()
    test_adversarial_error_contracts()
    test_randomized_parity_against_explicit_composition()
    print("ok")
