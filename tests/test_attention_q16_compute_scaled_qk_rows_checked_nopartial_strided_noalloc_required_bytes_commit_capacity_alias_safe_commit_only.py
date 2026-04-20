#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesCommitCapacityAliasSafeCommitOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_commit_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    required_q_cells: int,
    required_k_cells: int,
    required_out_cells: int,
    required_stage_cells: int,
    required_stage_bytes: int,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
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
    if (
        required_q_cells < 0
        or required_k_cells < 0
        or required_out_cells < 0
        or required_stage_cells < 0
        or required_stage_bytes < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    commit_required_stage_cells = [0]
    commit_required_stage_bytes = [0]
    commit_required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        commit_required_stage_cells,
        commit_required_stage_bytes,
        commit_required_out_cells,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != ATTN_Q16_OK:
        return err

    canonical_required_q_cells = [0]
    canonical_required_k_cells = [0]
    canonical_required_out_cells = [0]
    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
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
        canonical_required_q_cells,
        canonical_required_k_cells,
        canonical_required_out_cells,
        canonical_required_stage_cells,
        canonical_required_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_q_cells != canonical_required_q_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_k_cells != canonical_required_k_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells != canonical_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_cells != canonical_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes != canonical_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if commit_required_stage_cells[0] != canonical_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_required_stage_bytes[0] != canonical_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_required_out_cells[0] != canonical_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        query_row_count,
        token_count,
        out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_checked_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    required_q_cells: int,
    required_k_cells: int,
    required_out_cells: int,
    required_stage_cells: int,
    required_stage_bytes: int,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_commit_only(
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
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        required_q_cells,
        required_k_cells,
        required_out_cells,
        required_stage_cells,
        required_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_required_bytes_commit_capacity_alias_safe_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafeCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafe(" in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(" in body
    assert "if (required_q_cells != canonical_required_q_cells)" in body
    assert "if (required_k_cells != canonical_required_k_cells)" in body
    assert "if (required_out_cells != canonical_required_out_cells)" in body
    assert "if (required_stage_cells != canonical_required_stage_cells)" in body
    assert "if (required_stage_bytes != canonical_required_stage_bytes)" in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitOnly(" in body


def test_known_vectors_and_parity() -> None:
    q_rows = [0] * 56
    k_rows = [0] * 64
    query_row_count = 3
    token_count = 5
    head_dim = 6
    query_row_stride_q16 = 8
    k_row_stride_q16 = 7
    out_row_stride = 9

    required_q = (query_row_count - 1) * query_row_stride_q16 + head_dim
    required_k = token_count * k_row_stride_q16
    required_out = (query_row_count - 1) * out_row_stride + token_count
    required_stage_cells = query_row_count * token_count
    required_stage_bytes = required_stage_cells * 8

    staged = [index * 7 - 20 for index in range(required_stage_cells)]
    out_a = [911] * required_out
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_commit_only(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        out_a,
        len(out_a),
        out_row_stride,
        required_stage_cells,
        len(staged) * 8,
        staged,
        len(staged),
        required_q,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )

    err_b = explicit_checked_composition(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        out_b,
        len(out_b),
        out_row_stride,
        required_stage_cells,
        len(staged) * 8,
        staged,
        len(staged),
        required_q,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )

    assert err_a == ATTN_Q16_OK
    assert err_a == err_b
    assert out_a == out_b


def test_rejects_diagnostic_mismatch_with_no_partial_output_writes() -> None:
    q_rows = [0] * 40
    k_rows = [0] * 48
    query_row_count = 2
    token_count = 4
    head_dim = 5
    query_row_stride_q16 = 10
    k_row_stride_q16 = 6
    out_row_stride = 7

    required_q = (query_row_count - 1) * query_row_stride_q16 + head_dim
    required_k = token_count * k_row_stride_q16
    required_out = (query_row_count - 1) * out_row_stride + token_count
    required_stage_cells = query_row_count * token_count
    required_stage_bytes = required_stage_cells * 8

    staged = [index + 3 for index in range(required_stage_cells)]
    out = [123456] * required_out
    before = out.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_commit_only(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        out,
        len(out),
        out_row_stride,
        required_stage_cells,
        len(staged) * 8,
        staged,
        len(staged),
        required_q + 1,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == before


def test_alias_rejection_propagates_without_output_mutation() -> None:
    q_rows = [0] * 24
    k_rows = [0] * 24
    out = [99] * 12
    staged = [1, 2, 3, 4, 5, 6]

    before = out.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_commit_only(
        q_rows,
        len(q_rows),
        2,
        6,
        k_rows,
        len(k_rows),
        3,
        6,
        6,
        out,
        len(out),
        6,
        6,
        len(staged) * 8,
        staged,
        len(staged),
        12,
        18,
        9,
        6,
        48,
        out_base_addr=0x500000,
        stage_base_addr=0x500000,
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == before


def test_randomized_parity() -> None:
    rng = random.Random(20260420_677)

    for _ in range(2000):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 9)
        head_dim = rng.randint(0, 10)
        query_row_stride_q16 = rng.randint(0, 12)
        k_row_stride_q16 = rng.randint(0, 12)
        out_row_stride = rng.randint(0, 12)

        if query_row_count > 0 and head_dim > query_row_stride_q16:
            query_row_stride_q16 = head_dim
        if token_count > 0 and head_dim > k_row_stride_q16:
            k_row_stride_q16 = head_dim
        if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
            out_row_stride = token_count + rng.randint(0, 3)

        required_q = 0
        if query_row_count > 0:
            required_q = (query_row_count - 1) * query_row_stride_q16 + head_dim

        required_k = token_count * k_row_stride_q16

        required_out = 0
        if query_row_count > 0 and token_count > 0:
            required_out = (query_row_count - 1) * out_row_stride + token_count

        required_stage_cells = query_row_count * token_count
        required_stage_bytes = required_stage_cells * 8

        q_capacity = max(0, required_q + rng.randint(-1, 3))
        k_capacity = max(0, required_k + rng.randint(-1, 3))
        out_capacity = max(0, required_out + rng.randint(-1, 3))
        stage_capacity = max(0, required_stage_cells + rng.randint(-1, 3))

        q_rows = [0] * q_capacity
        k_rows = [0] * k_capacity
        staged = [rng.randint(-50, 50) for _ in range(stage_capacity)]

        out_a = [rng.randint(-100, 100) for _ in range(out_capacity)]
        out_b = out_a.copy()

        commit_stage_cell_capacity = max(0, required_stage_cells + rng.randint(-1, 3))
        commit_stage_byte_capacity = max(0, stage_capacity * 8 + rng.randint(-8, 16))

        maybe_bad_required_q = required_q + (1 if rng.random() < 0.08 else 0)

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_commit_only(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_a,
            out_capacity,
            out_row_stride,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged,
            stage_capacity,
            maybe_bad_required_q,
            required_k,
            required_out,
            required_stage_cells,
            required_stage_bytes,
            q_base_addr=0x100000,
            k_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=0x400000,
        )

        err_b = explicit_checked_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_b,
            out_capacity,
            out_row_stride,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged,
            stage_capacity,
            maybe_bad_required_q,
            required_k,
            required_out,
            required_stage_cells,
            required_stage_bytes,
            q_base_addr=0x100000,
            k_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=0x400000,
        )

        assert err_a == err_b
        assert out_a == out_b
