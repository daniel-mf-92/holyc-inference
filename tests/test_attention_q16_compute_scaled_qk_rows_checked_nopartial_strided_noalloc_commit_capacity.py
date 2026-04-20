#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import (
    attention_q16_compute_scaled_qk_rows_checked,
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
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

    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    last_q = [0]
    last_k = [0]
    last_out = [0]
    required_q = [0]
    required_k = [0]
    required_out = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        last_q,
        last_k,
        last_out,
        required_q,
        required_k,
        required_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > commit_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes > commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes > staging_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    # Python lists have no interior-pointer views. Object identity is the
    # closest alias model to HolyC pointer-window overlap checks.
    if staged_scores_q32 is q_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
        token_count,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_out[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        err, out_row_base = try_mul_i64_checked(row_index, out_row_stride)
        if err != ATTN_Q16_OK:
            return err

        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, _ = try_add_i64_checked(out_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            err, _ = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

    for row_index in range(query_row_count):
        out_row_base = row_index * out_row_stride
        stage_row_base = row_index * token_count

        for token_index in range(token_count):
            out_scores_q32[out_row_base + token_index] = staged_scores_q32[
                stage_row_base + token_index
            ]

    return ATTN_Q16_OK


def explicit_checked_commit_capacity_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
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

    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    last_q = [0]
    last_k = [0]
    last_out = [0]
    required_q = [0]
    required_k = [0]
    required_out = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        last_q,
        last_k,
        last_out,
        required_q,
        required_k,
        required_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > commit_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes > commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes > staging_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is q_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
        token_count,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_out[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        err, out_row_base = try_mul_i64_checked(row_index, out_row_stride)
        if err != ATTN_Q16_OK:
            return err

        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(out_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            err, stage_index = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

            out_scores_q32[out_index] = staged_scores_q32[stage_index]

    return ATTN_Q16_OK


def test_source_contains_strided_noalloc_commit_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(" in body
    assert "AttentionQ16ComputeScaledQKRowsChecked(" in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(" in body
    assert "if (staged_begin < out_end && out_begin < staged_end)" in body


def test_known_vector_matches_explicit_checked_composition() -> None:
    query_row_count = 3
    token_count = 4
    head_dim = 8
    query_row_stride_q16 = 11
    k_row_stride_q16 = 9
    out_row_stride = 6
    score_scale_q16 = 65536

    q_rows = [
        101,
        -77,
        55,
        -33,
        22,
        -11,
        7,
        -5,
    ] * query_row_count
    q_rows += [0] * (query_row_count * query_row_stride_q16 - len(q_rows))

    k_rows = [
        13,
        -9,
        7,
        -5,
        4,
        -3,
        2,
        -1,
        0,
    ] * token_count

    out_capacity = (query_row_count - 1) * out_row_stride + token_count
    out_a = [777] * out_capacity
    out_b = out_a.copy()

    stage_cells = query_row_count * token_count
    stage_a = [0] * stage_cells
    stage_b = [0] * stage_cells

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
        out_row_stride,
        stage_cells,
        stage_cells * 8,
        stage_a,
        len(stage_a),
    )

    err_b = explicit_checked_commit_capacity_composition(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
        out_row_stride,
        stage_cells,
        stage_cells * 8,
        stage_b,
        len(stage_b),
    )

    assert err_a == ATTN_Q16_OK
    assert err_a == err_b
    assert out_a == out_b


def test_error_surfaces_and_no_partial_guarantee() -> None:
    q_rows = [1, 2, 3, 4]
    k_rows = [1, 2, 3, 4]
    out = [999, 999, 999, 999]
    out_before = out.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows,
        4,
        1,
        4,
        k_rows,
        4,
        1,
        4,
        4,
        65536,
        out,
        4,
        1,
        0,
        0,
        [0],
        1,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_before


def test_alias_window_rejected_by_overlap_guard() -> None:
    q_rows = [1, 2, 3, 4]
    k_rows = [5, 6, 7, 8]
    out = [999, 999, 999, 999]
    out_before = out.copy()

    # Identity alias maps to overlap in Python model and should reject.
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows,
        4,
        1,
        4,
        k_rows,
        4,
        1,
        4,
        4,
        65536,
        out,
        4,
        1,
        1,
        8,
        out,
        4,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_before

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows,
        4,
        1,
        4,
        k_rows,
        4,
        1,
        4,
        4,
        65536,
        out,
        4,
        1,
        1,
        8,
        out,
        4,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_before


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_630)

    for _ in range(3000):
        query_row_count = rng.randint(0, 10)
        token_count = rng.randint(0, 12)
        head_dim = rng.randint(0, 16)

        query_row_stride_q16 = rng.randint(0, 20)
        if query_row_count > 0 and head_dim > 0 and query_row_stride_q16 < head_dim:
            query_row_stride_q16 = head_dim + rng.randint(0, 4)

        k_row_stride_q16 = rng.randint(0, 20)
        if token_count > 0 and head_dim > 0 and k_row_stride_q16 < head_dim:
            k_row_stride_q16 = head_dim + rng.randint(0, 4)

        out_row_stride = rng.randint(0, 20)
        if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
            out_row_stride = token_count + rng.randint(0, 4)

        q_need = 0
        if query_row_count > 0:
            q_need = (query_row_count - 1) * query_row_stride_q16 + head_dim

        k_need = token_count * k_row_stride_q16

        out_need = 0
        if query_row_count > 0 and token_count > 0:
            out_need = (query_row_count - 1) * out_row_stride + token_count

        q_capacity = max(0, q_need + rng.randint(-2, 3))
        k_capacity = max(0, k_need + rng.randint(-2, 3))
        out_capacity = max(0, out_need + rng.randint(-2, 3))

        q_rows = [rng.randint(-20000, 20000) for _ in range(max(q_capacity, 1))]
        k_rows = [rng.randint(-20000, 20000) for _ in range(max(k_capacity, 1))]
        out_a = [rng.randint(-5000, 5000) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()

        stage_need = query_row_count * token_count
        staged_capacity = max(0, stage_need + rng.randint(-2, 3))
        commit_stage_cell_capacity = max(0, stage_need + rng.randint(-2, 3))

        # ~15% overflow-focused commit byte tests.
        if rng.random() < 0.15:
            commit_stage_byte_capacity = (1 << 63) - 1
            staged_capacity = (1 << 63) // 8 + rng.randint(1, 4)
        else:
            commit_stage_byte_capacity = max(0, stage_need * 8 + rng.randint(-16, 24))

        if staged_capacity > 100000:
            staged_a = [0]
            staged_b = [0]
        else:
            staged_a = [0] * max(staged_capacity, 1)
            staged_b = [0] * max(staged_capacity, 1)

        score_scale_q16 = rng.randint(-200000, 200000)

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_a,
            out_capacity,
            out_row_stride,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_a,
            staged_capacity,
        )

        err_b = explicit_checked_commit_capacity_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_b,
            out_capacity,
            out_row_stride,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_b,
            staged_capacity,
        )

        assert err_a in (
            ATTN_Q16_OK,
            ATTN_Q16_ERR_NULL_PTR,
            ATTN_Q16_ERR_BAD_PARAM,
            ATTN_Q16_ERR_OVERFLOW,
        )
        assert err_a == err_b

        if err_a == ATTN_Q16_OK:
            assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_strided_noalloc_commit_capacity_wrapper()
    test_known_vector_matches_explicit_checked_composition()
    test_error_surfaces_and_no_partial_guarantee()
    test_alias_window_rejected_by_overlap_guard()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
