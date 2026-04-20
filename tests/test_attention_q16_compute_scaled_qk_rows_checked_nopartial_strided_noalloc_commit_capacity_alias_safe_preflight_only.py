#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocCommitCapacityAliasSafePreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import (
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only,
)


def _ranges_overlap(a_base: int, a_end: int, b_base: int, b_end: int) -> bool:
    if a_base >= a_end or b_base >= b_end:
        return False
    return a_base < b_end and b_base < a_end


def _range_end_checked(base_addr: int, span_bytes: int) -> tuple[int, int]:
    if span_bytes < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0
    return try_add_i64_checked(base_addr, span_bytes)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_preflight_only(
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
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

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

    out_commit_required_stage_cells[0] = req_stage_cells[0]
    out_commit_required_stage_bytes[0] = req_stage_bytes[0]
    out_required_out_cells[0] = req_out[0]

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

    err, q_end = _range_end_checked(q_base_addr, q_span_bytes)
    if err != ATTN_Q16_OK:
        return err
    err, k_end = _range_end_checked(k_base_addr, k_span_bytes)
    if err != ATTN_Q16_OK:
        return err
    err, out_end = _range_end_checked(out_base_addr, out_span_bytes)
    if err != ATTN_Q16_OK:
        return err
    err, stage_end = _range_end_checked(stage_base_addr, stage_capacity_bytes)
    if err != ATTN_Q16_OK:
        return err

    if _ranges_overlap(stage_base_addr, stage_end, q_base_addr, q_end):
        return ATTN_Q16_ERR_BAD_PARAM
    if _ranges_overlap(stage_base_addr, stage_end, k_base_addr, k_end):
        return ATTN_Q16_ERR_BAD_PARAM
    if _ranges_overlap(stage_base_addr, stage_end, out_base_addr, out_end):
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def explicit_checked_preflight_composition(
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
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_preflight_only(
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
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_alias_safe_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafePreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "*out_commit_required_stage_cells = required_stage_cells;" in body
    assert "*out_commit_required_stage_bytes = required_stage_bytes;" in body
    assert "*out_required_out_cells = required_out_cells;" in body
    assert "AttentionByteRangesOverlap(stage_base_addr," in body

    alias_safe_sig = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe("
    )
    alias_safe_body = source.split(alias_safe_sig, 1)[1]
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafePreflightOnly(" in alias_safe_body


def test_known_vectors_and_alias_rejection() -> None:
    q_rows = [0] * 48
    k_rows = [0] * 64
    out_scores = [0] * 32
    staged_scores = [0] * 32

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out_cells = [666]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_preflight_only(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        4,
        8,
        6,
        out_scores,
        len(out_scores),
        6,
        12,
        96,
        staged_scores,
        len(staged_scores),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    err_exp = explicit_checked_preflight_composition(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        4,
        8,
        6,
        out_scores,
        len(out_scores),
        6,
        12,
        96,
        staged_scores,
        len(staged_scores),
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage_cells == exp_stage_cells == [12]
    assert got_stage_bytes == exp_stage_bytes == [96]
    assert got_out_cells == exp_out_cells == [16]

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_preflight_only(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        4,
        8,
        6,
        out_scores,
        len(out_scores),
        6,
        12,
        96,
        staged_scores,
        len(staged_scores),
        [0],
        [0],
        [0],
        stage_base_addr=0x100010,
    )
    assert err_alias == ATTN_Q16_ERR_BAD_PARAM


def test_randomized_parity() -> None:
    rng = random.Random(20260420_646)

    for _ in range(2500):
        query_row_count = rng.randint(0, 7)
        token_count = rng.randint(0, 9)
        head_dim = rng.randint(0, 11)
        query_row_stride_q16 = rng.randint(0, 14)
        k_row_stride_q16 = rng.randint(0, 14)
        out_row_stride = rng.randint(0, 14)

        if query_row_count > 0 and head_dim > 0 and query_row_stride_q16 < head_dim:
            query_row_stride_q16 = head_dim + rng.randint(0, 3)
        if token_count > 0 and head_dim > 0 and k_row_stride_q16 < head_dim:
            k_row_stride_q16 = head_dim + rng.randint(0, 3)
        if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
            out_row_stride = token_count + rng.randint(0, 4)

        q_need = 0 if query_row_count == 0 or head_dim == 0 else (query_row_count - 1) * query_row_stride_q16 + head_dim
        k_need = 0 if token_count == 0 or head_dim == 0 else (token_count - 1) * k_row_stride_q16 + head_dim
        out_need = 0 if query_row_count == 0 or token_count == 0 else (query_row_count - 1) * out_row_stride + token_count
        stage_need = query_row_count * token_count

        q_capacity = max(0, q_need + rng.randint(-2, 3))
        k_capacity = max(0, k_need + rng.randint(-2, 3))
        out_capacity = max(0, out_need + rng.randint(-2, 3))
        stage_capacity = max(0, stage_need + rng.randint(-2, 3))

        q_rows = [0] * q_capacity
        k_rows = [0] * k_capacity
        out_scores = [0] * out_capacity
        staged_scores = [0] * stage_capacity

        commit_stage_cells = max(0, stage_need + rng.randint(-2, 3))
        commit_stage_bytes = max(0, stage_capacity * 8 + rng.randint(-16, 24))

        # 30% overlap stress: stage overlaps q/k/out windows.
        if rng.randint(0, 9) < 3:
            stage_base = rng.choice([0x100000, 0x200000, 0x300000]) + rng.randint(0, 24)
        else:
            stage_base = 0x400000 + rng.randint(0, 256)

        got_stage_cells = [777]
        got_stage_bytes = [888]
        got_out_cells = [999]
        exp_stage_cells = [111]
        exp_stage_bytes = [222]
        exp_out_cells = [333]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_preflight_only(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            commit_stage_cells,
            commit_stage_bytes,
            staged_scores,
            stage_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
            q_base_addr=0x100000,
            k_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=stage_base,
        )

        err_exp = explicit_checked_preflight_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            commit_stage_cells,
            commit_stage_bytes,
            staged_scores,
            stage_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
            q_base_addr=0x100000,
            k_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=stage_base,
        )

        assert err_got == err_exp
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes
        assert got_out_cells == exp_out_cells


if __name__ == "__main__":
    test_source_contains_alias_safe_preflight_helper()
    test_known_vectors_and_alias_rejection()
    test_randomized_parity()
    print("ok")
