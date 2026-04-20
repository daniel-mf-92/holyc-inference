#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityDefault."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_byte_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
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
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_required_bytes_commit_capacity_default_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_byte_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
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
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_noalloc_required_bytes_commit_capacity_default_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityDefault("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = AttentionTryMulI64Checked(query_row_count," in body
    assert "token_count," in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )


def test_known_vectors_and_capacity_rejection() -> None:
    query_row_count = 4
    token_count = 6
    head_dim = 5

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    required_stage_cells = query_row_count * token_count
    required_stage_bytes = required_stage_cells * 8

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    out_stage_cells = [111]
    out_stage_bytes = [222]
    out_out_cells = [333]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage_bytes,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [required_stage_cells]
    assert out_stage_bytes == [required_stage_bytes]
    assert out_out_cells == [out_capacity]

    out_stage_cells = [9]
    out_stage_bytes = [9]
    out_out_cells = [9]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage_bytes - 8,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [9]
    assert out_stage_bytes == [9]
    assert out_out_cells == [9]


def test_error_paths() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
            [0],
            1,
            1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            8,
            None,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
            [0],
            -1,
            1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            8,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
            [0],
            1,
            -1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            8,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
            [0],
            1,
            1,
            [0],
            1,
            1,
            1,
            [0],
            1,
            -1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        [0],
        I64_MAX,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_615)

    for _ in range(8000):
        query_row_count = rng.randint(0, 150)
        token_count = rng.randint(0, 150)
        head_dim = rng.randint(0, 96)

        if rng.random() < 0.05:
            query_row_count = -rng.randint(1, 80)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 80)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 80)

        if rng.random() < 0.02:
            query_row_count = (1 << 62) + rng.randint(0, 63)
            token_count = (1 << 62) + rng.randint(0, 63)

        if query_row_count >= 0 and token_count >= 0 and query_row_count < (1 << 31) and token_count < (1 << 31):
            required_cells = query_row_count * token_count
            required_bytes = required_cells * 8
            commit_stage_byte_capacity = max(0, required_bytes + rng.randint(-80, 176))
        else:
            commit_stage_byte_capacity = rng.randint(0, 1 << 20)

        if rng.random() < 0.04:
            commit_stage_byte_capacity = -rng.randint(1, 400)

        needs_huge_shape = (
            query_row_count > 2000
            or token_count > 2000
            or head_dim > 2000
        )

        if needs_huge_shape:
            q_rows = [0]
            k_rows = [0]
            out_scores = [0]
            q_rows_capacity = I64_MAX
            k_rows_capacity = I64_MAX
            out_scores_capacity = I64_MAX
        else:
            q_cells = max(1, query_row_count if query_row_count > 0 else 1)
            if head_dim > 0 and query_row_count > 0:
                q_cells = query_row_count * head_dim

            k_cells = max(1, token_count if token_count > 0 else 1)
            if head_dim > 0 and token_count > 0:
                k_cells = token_count * head_dim

            out_cells = max(1, query_row_count if query_row_count > 0 else 1)
            if token_count > 0 and query_row_count > 0:
                out_cells = query_row_count * token_count

            q_rows = [0] * q_cells
            k_rows = [0] * k_cells
            out_scores = [0] * out_cells

            q_rows_capacity = q_cells
            k_rows_capacity = k_cells
            out_scores_capacity = out_cells

        got_stage_cells = [0xAA]
        got_stage_bytes = [0xBB]
        got_out_cells = [0xCC]
        exp_stage_cells = [0xAA]
        exp_stage_bytes = [0xBB]
        exp_out_cells = [0xCC]

        err_new = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_default(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            commit_stage_byte_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_ref = explicit_checked_required_bytes_commit_capacity_default_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            commit_stage_byte_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_new == err_ref
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes
        assert got_out_cells == exp_out_cells


if __name__ == "__main__":
    test_source_contains_noalloc_required_bytes_commit_capacity_default_helper()
    test_known_vectors_and_capacity_rejection()
    test_error_paths()
    test_randomized_parity_against_explicit_composition()
    print("ok")
