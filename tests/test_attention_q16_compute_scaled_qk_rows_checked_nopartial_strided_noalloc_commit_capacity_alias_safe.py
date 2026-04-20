#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocCommitCapacityAliasSafe."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only,
)

I64_MAX = (1 << 63) - 1


def _ranges_overlap(a_base: int, a_end: int, b_base: int, b_end: int) -> bool:
    if a_base >= a_end or b_base >= b_end:
        return False
    return a_base < b_end and b_base < a_end


def _simulate_alias_safe_preflight(
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
    *,
    q_base_addr: int,
    k_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
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

    req_q = [0]
    req_k = [0]
    req_out = [0]
    req_stage_cells = [0]
    req_stage_bytes = [0]

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
        req_q,
        req_k,
        req_out,
        req_stage_cells,
        req_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if req_stage_cells[0] > commit_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if req_stage_bytes[0] > commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if req_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, stage_capacity_bytes = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err
    if req_stage_bytes[0] > stage_capacity_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    err, q_span_bytes = try_mul_i64_checked(req_q[0], 8)
    if err != ATTN_Q16_OK:
        return err
    err, k_span_bytes = try_mul_i64_checked(req_k[0], 8)
    if err != ATTN_Q16_OK:
        return err
    err, out_span_bytes = try_mul_i64_checked(req_out[0], 8)
    if err != ATTN_Q16_OK:
        return err

    q_end = q_base_addr + q_span_bytes
    k_end = k_base_addr + k_span_bytes
    out_end = out_base_addr + out_span_bytes
    stage_end = stage_base_addr + stage_capacity_bytes

    if _ranges_overlap(stage_base_addr, stage_end, q_base_addr, q_end):
        return ATTN_Q16_ERR_BAD_PARAM
    if _ranges_overlap(stage_base_addr, stage_end, k_base_addr, k_end):
        return ATTN_Q16_ERR_BAD_PARAM
    if _ranges_overlap(stage_base_addr, stage_end, out_base_addr, out_end):
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        65536,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
    )


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
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
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    _ = score_scale_q16
    return _simulate_alias_safe_preflight(
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
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def explicit_checked_alias_safe_composition(
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
    *,
    q_base_addr: int,
    k_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    _ = score_scale_q16
    return _simulate_alias_safe_preflight(
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
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_alias_safe_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(" in body
    assert "AttentionByteRangeEndChecked(" in body
    assert "AttentionByteRangesOverlap(" in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity(" in body


def test_alias_overlap_rejected_and_non_overlap_succeeds() -> None:
    q_rows = [1, 2, 3, 4, 5, 6]
    k_rows = [7, 8, 9, 10, 11, 12]
    out = [0] * 6
    stage = [0] * 6

    # Non-overlap synthetic base windows
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
        q_rows,
        6,
        2,
        3,
        k_rows,
        6,
        2,
        3,
        3,
        65536,
        out,
        6,
        3,
        6,
        48,
        stage,
        6,
        q_base_addr=0x100000,
        k_base_addr=0x200000,
        out_base_addr=0x300000,
        stage_base_addr=0x400000,
    )
    assert err == ATTN_Q16_OK

    # Interior-pointer-like overlap with q window
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
        q_rows,
        6,
        2,
        3,
        k_rows,
        6,
        2,
        3,
        3,
        65536,
        out,
        6,
        3,
        6,
        48,
        stage,
        6,
        q_base_addr=0x100000,
        k_base_addr=0x200000,
        out_base_addr=0x300000,
        stage_base_addr=0x100008,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_randomized_parity_vs_explicit_alias_checked_composition() -> None:
    rng = random.Random(20260420_645)

    for _ in range(600):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 8)
        head_dim = rng.randint(0, 10)

        query_row_stride_q16 = max(head_dim, head_dim + rng.randint(0, 3))
        k_row_stride_q16 = max(head_dim, head_dim + rng.randint(0, 3))
        out_row_stride = max(token_count, token_count + rng.randint(0, 3))

        q_need = 0 if query_row_count == 0 else (query_row_count - 1) * query_row_stride_q16 + head_dim
        k_need = 0 if token_count == 0 else (token_count - 1) * k_row_stride_q16 + head_dim
        out_need = 0 if (query_row_count == 0 or token_count == 0) else (query_row_count - 1) * out_row_stride + token_count
        stage_need = query_row_count * token_count

        q_capacity = max(0, q_need + rng.randint(-1, 2))
        k_capacity = max(0, k_need + rng.randint(-1, 2))
        out_capacity = max(0, out_need + rng.randint(-1, 2))
        staged_capacity = max(0, stage_need + rng.randint(-1, 2))

        q_rows = [rng.randint(-500, 500) for _ in range(max(1, q_capacity))]
        k_rows = [rng.randint(-500, 500) for _ in range(max(1, k_capacity))]
        out_a = [rng.randint(-100, 100) for _ in range(max(1, out_capacity))]
        out_b = out_a.copy()
        stage_a = [0 for _ in range(max(1, staged_capacity))]
        stage_b = [0 for _ in range(max(1, staged_capacity))]

        commit_stage_cell_capacity = max(0, stage_need + rng.randint(-1, 2))
        commit_stage_byte_capacity = max(0, staged_capacity * 8 + rng.randint(-8, 16))

        base = 0x100000
        q_base_addr = base + rng.randint(0, 512) * 8
        k_base_addr = base + 0x10000 + rng.randint(0, 512) * 8
        out_base_addr = base + 0x20000 + rng.randint(0, 512) * 8
        stage_base_addr = base + 0x30000 + rng.randint(0, 512) * 8

        # Force overlap in a subset to exercise alias rejection path.
        if rng.random() < 0.25:
            stage_base_addr = q_base_addr + rng.randint(0, 3) * 8

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            65536,
            out_a,
            out_capacity,
            out_row_stride,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_a,
            staged_capacity,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )
        err_b = explicit_checked_alias_safe_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            65536,
            out_b,
            out_capacity,
            out_row_stride,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_b,
            staged_capacity,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert out_a == out_b

    overflow_err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
        [0],
        1,
        2,
        I64_MAX,
        [0],
        1,
        1,
        1,
        1,
        65536,
        [0],
        1,
        1,
        1,
        8,
        [0],
        1,
    )
    assert overflow_err == ATTN_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_alias_safe_wrapper()
    test_alias_overlap_rejected_and_non_overlap_succeeds()
    test_randomized_parity_vs_explicit_alias_checked_composition()
    print("ok")
