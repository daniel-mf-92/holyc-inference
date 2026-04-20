#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocRequiredBytesCommitCapacityAliasSafe."""

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
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only,
)

I64_MAX = (1 << 63) - 1


def _ranges_overlap(a_base: int, a_end: int, b_base: int, b_end: int) -> bool:
    if a_base >= a_end or b_base >= b_end:
        return False
    return a_base < b_end and b_base < a_end


def _range_end_checked(base_addr: int, span_bytes: int) -> tuple[int, int]:
    if span_bytes < 0:
        return ATTN_Q16_ERR_BAD_PARAM, 0
    return try_add_i64_checked(base_addr, span_bytes)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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

    if query_row_count != 0 and token_count != 0:
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

    out_commit_required_stage_cells[0] = req_stage_cells[0]
    out_commit_required_stage_bytes[0] = req_stage_bytes[0]
    out_required_out_cells[0] = req_out[0]
    return ATTN_Q16_OK


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
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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


def test_source_contains_required_bytes_commit_capacity_alias_safe_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafe("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly(" in body
    assert "if (required_stage_cells > commit_stage_cell_capacity)" in body
    assert "if (required_stage_bytes > commit_stage_byte_capacity)" in body
    assert "AttentionByteRangesOverlap(stage_base_addr," in body
    assert "*out_commit_required_stage_cells = required_stage_cells;" in body
    assert "*out_commit_required_stage_bytes = required_stage_bytes;" in body
    assert "*out_required_out_cells = required_out_cells;" in body


def test_known_vectors_and_alias_rejection() -> None:
    q_rows = [0] * 56
    k_rows = [0] * 64
    out_scores = [0] * 36
    staged_scores = [0] * 30

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out_cells = [666]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        5,
        7,
        6,
        out_scores,
        len(out_scores),
        8,
        15,
        120,
        staged_scores,
        len(staged_scores),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    err_exp = explicit_checked_composition(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        5,
        7,
        6,
        out_scores,
        len(out_scores),
        8,
        15,
        120,
        staged_scores,
        len(staged_scores),
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage_cells == exp_stage_cells == [15]
    assert got_stage_bytes == exp_stage_bytes == [120]
    assert got_out_cells == exp_out_cells == [21]

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        q_rows,
        len(q_rows),
        3,
        8,
        k_rows,
        len(k_rows),
        5,
        7,
        6,
        out_scores,
        len(out_scores),
        8,
        15,
        120,
        staged_scores,
        len(staged_scores),
        [0],
        [0],
        [0],
        stage_base_addr=0x100010,
    )
    assert err_alias == ATTN_Q16_ERR_BAD_PARAM


def test_diagnostics_no_partial_on_failure() -> None:
    q_rows = [0] * 32
    k_rows = [0] * 32
    out_scores = [0] * 32
    staged_scores = [0] * 32

    out_stage_cells = [7001]
    out_stage_bytes = [7002]
    out_out_cells = [7003]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        q_rows,
        len(q_rows),
        2,
        8,
        k_rows,
        len(k_rows),
        4,
        8,
        6,
        out_scores,
        len(out_scores),
        6,
        7,
        64,
        staged_scores,
        len(staged_scores),
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [7001]
    assert out_stage_bytes == [7002]
    assert out_out_cells == [7003]


def test_error_paths() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [2]
    out_out_cells = [3]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
            [0],
            1,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            1,
            1,
            8,
            [0],
            1,
            None,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
            [0],
            -1,
            1,
            1,
            [0],
            1,
            1,
            1,
            1,
            [0],
            1,
            1,
            1,
            8,
            [0],
            1,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        [0],
        I64_MAX,
        huge,
        1,
        [0],
        I64_MAX,
        3,
        1,
        1,
        [0],
        I64_MAX,
        3,
        I64_MAX,
        I64_MAX,
        [0],
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260420_648)

    for _ in range(3000):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 10)
        head_dim = rng.randint(0, 12)

        query_row_stride_q16 = rng.randint(max(1, head_dim), max(1, head_dim + 10))
        k_row_stride_q16 = rng.randint(max(1, head_dim), max(1, head_dim + 10))
        out_row_stride = rng.randint(max(1, token_count), max(1, token_count + 10))

        q_capacity = rng.randint(0, 120)
        k_capacity = rng.randint(0, 120)
        out_capacity = rng.randint(0, 120)
        staged_capacity = rng.randint(0, 120)

        q_rows = [rng.randint(-100, 100) for _ in range(max(1, q_capacity))]
        k_rows = [rng.randint(-100, 100) for _ in range(max(1, k_capacity))]
        out_scores = [rng.randint(-100, 100) for _ in range(max(1, out_capacity))]
        staged_scores = [rng.randint(-100, 100) for _ in range(max(1, staged_capacity))]

        commit_stage_cell_capacity = rng.randint(0, 160)
        commit_stage_byte_capacity = rng.randint(0, 1600)

        got_stage_cells = [101]
        got_stage_bytes = [102]
        got_out_cells = [103]
        exp_stage_cells = [201]
        exp_stage_bytes = [202]
        exp_out_cells = [203]

        q_base_addr = 0x100000 + rng.randint(0, 0x400)
        k_base_addr = 0x200000 + rng.randint(0, 0x400)
        out_base_addr = 0x300000 + rng.randint(0, 0x400)
        stage_base_addr = 0x400000 + rng.randint(0, 0x400)

        # Some iterations force adversarial overlap to stress alias guards.
        if rng.random() < 0.25:
            stage_base_addr = q_base_addr + rng.randint(0, 128)

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_scores,
            staged_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        err_exp = explicit_checked_composition(
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
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_scores,
            staged_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
            q_base_addr=q_base_addr,
            k_base_addr=k_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_stage_cells == [101]
            assert got_stage_bytes == [102]
            assert got_out_cells == [103]


if __name__ == "__main__":
    test_source_contains_required_bytes_commit_capacity_alias_safe_helper()
    test_known_vectors_and_alias_rejection()
    test_diagnostics_no_partial_on_failure()
    test_error_paths()
    test_randomized_parity_against_explicit_checked_composition()
    print("ok")
