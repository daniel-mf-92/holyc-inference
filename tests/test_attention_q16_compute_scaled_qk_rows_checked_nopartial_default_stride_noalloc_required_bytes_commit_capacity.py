#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
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

    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
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
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_stage_bytes[0] > commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_commit_required_stage_cells[0] = required_stage_cells[0]
    out_commit_required_stage_bytes[0] = required_stage_bytes[0]
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def explicit_checked_required_bytes_commit_capacity_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
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

    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
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
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_stage_bytes[0] > commit_stage_byte_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_commit_required_stage_cells[0] = required_stage_cells[0]
    out_commit_required_stage_bytes[0] = required_stage_bytes[0]
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_noalloc_required_bytes_commit_capacity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes("
        in body
    )
    assert "if (commit_required_stage_bytes > commit_stage_byte_capacity)" in body


def test_known_vectors_and_capacity_rejection() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 7

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

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage_cells,
        required_stage_bytes,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert out_stage_cells == [required_stage_cells]
    assert out_stage_bytes == [required_stage_bytes]
    assert out_out_cells == [out_capacity]

    out_stage_cells = [7]
    out_stage_bytes = [7]
    out_out_cells = [7]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        required_stage_cells,
        required_stage_bytes - 8,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [7]
    assert out_stage_bytes == [7]
    assert out_out_cells == [7]


def test_error_paths_and_overflow_passthrough() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
            [0], 1, 1, [0], 1, 1, 1, [0], 1, 1, 8, None, out_stage_bytes, out_out_cells
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
            [0], 1, 1, [0], 1, 1, 1, [0], 1, -1, 8, out_stage_cells, out_stage_bytes, out_out_cells
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
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
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_606)

    for _ in range(5000):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 40)

        q_capacity = max(0, query_row_count * head_dim + rng.randint(-60, 60))
        k_capacity = max(0, token_count * head_dim + rng.randint(-60, 60))
        out_capacity = max(0, query_row_count * token_count + rng.randint(-60, 60))
        commit_stage_cell_capacity = max(
            0, query_row_count * token_count + rng.randint(-60, 60)
        )
        commit_stage_byte_capacity = max(
            0, commit_stage_cell_capacity * 8 + rng.randint(-128, 128)
        )

        if rng.random() < 0.06:
            query_row_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            token_count = -rng.randint(1, 80)
        if rng.random() < 0.06:
            head_dim = -rng.randint(1, 80)
        if rng.random() < 0.06:
            q_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            k_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            out_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            commit_stage_cell_capacity = -rng.randint(1, 80)
        if rng.random() < 0.06:
            commit_stage_byte_capacity = -rng.randint(1, 80)

        if rng.random() < 0.04:
            query_row_count = (1 << 62) + rng.randint(0, 16)
            token_count = (1 << 62) + rng.randint(0, 16)
            commit_stage_cell_capacity = I64_MAX
            commit_stage_byte_capacity = I64_MAX

        q_rows = None if rng.random() < 0.03 else [0] * max(q_capacity, 1)
        k_rows = None if rng.random() < 0.03 else [0] * max(k_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_capacity, 1)

        got_stage_cells = [0xAAAA]
        got_stage_bytes = [0xBBBB]
        got_out_cells = [0xCCCC]
        exp_stage_cells = [0xAAAA]
        exp_stage_bytes = [0xBBBB]
        exp_out_cells = [0xCCCC]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_required_bytes_commit_capacity_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_stage_cells == [0xAAAA]
            assert got_stage_bytes == [0xBBBB]
            assert got_out_cells == [0xCCCC]
            assert exp_stage_cells == [0xAAAA]
            assert exp_stage_bytes == [0xBBBB]
            assert exp_out_cells == [0xCCCC]
