#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAlloc."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_out_cells,
    )


def explicit_checked_wrapper_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_required_stage_cells,
        out_required_out_cells,
    )


def test_source_contains_preflight_only_default_stride_noalloc_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAlloc("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(" in body


def test_known_vectors_and_stage_capacity_gate() -> None:
    query_row_count = 5
    token_count = 7
    head_dim = 9

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    required_stage = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    got_stage = [111]
    got_out = [222]
    exp_stage = [333]
    exp_out = [444]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage,
        got_stage,
        got_out,
    )
    err_exp = explicit_checked_wrapper_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage,
        exp_stage,
        exp_out,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage == exp_stage == [required_stage]
    assert got_out == exp_out == [required_stage]

    got_stage = [7]
    got_out = [8]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage - 1,
        got_stage,
        got_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_stage == [7]
    assert got_out == [8]


def test_error_paths_preserve_outputs() -> None:
    out_stage = [31]
    out_out = [41]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
        None,
        0,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        1,
        out_stage,
        out_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out_stage == [31]
    assert out_out == [41]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
        [0],
        -1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        1,
        out_stage,
        out_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage == [31]
    assert out_out == [41]


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260420_577)

    for _ in range(2500):
        query_row_count = rng.randint(0, 80)
        token_count = rng.randint(0, 80)
        head_dim = rng.randint(0, 80)

        q_need = query_row_count * head_dim
        k_need = token_count * head_dim
        out_need = query_row_count * token_count

        q_rows_capacity = max(0, q_need + rng.randint(-8, 8))
        k_rows_capacity = max(0, k_need + rng.randint(-8, 8))
        out_scores_capacity = max(0, out_need + rng.randint(-8, 8))
        stage_cell_capacity = max(0, out_need + rng.randint(-8, 8))

        if rng.random() < 0.05:
            q_rows_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            k_rows_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            out_scores_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            stage_cell_capacity = -rng.randint(1, 9)

        if rng.random() < 0.05:
            query_row_count = -rng.randint(1, 9)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 9)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 9)

        q_rows = None if rng.random() < 0.03 else [0] * max(q_rows_capacity, 1)
        k_rows = None if rng.random() < 0.03 else [0] * max(k_rows_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_scores_capacity, 1)

        got_stage = [0x111]
        got_out = [0x222]
        exp_stage = [0x333]
        exp_out = [0x444]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            stage_cell_capacity,
            got_stage,
            got_out,
        )
        err_exp = explicit_checked_wrapper_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            stage_cell_capacity,
            exp_stage,
            exp_out,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage == exp_stage
            assert got_out == exp_out
        else:
            assert got_stage == [0x111]
            assert got_out == [0x222]


def test_overflow_passthrough() -> None:
    big = I64_MAX

    q_rows = [0]
    k_rows = [0]
    out_scores = [0]

    out_stage = [1]
    out_out = [1]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc(
        q_rows,
        big,
        2,
        k_rows,
        big,
        (big // 2) + 2,
        2,
        out_scores,
        big,
        big,
        out_stage,
        out_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out_stage == [1]
    assert out_out == [1]

