#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    tmp_q = [0]
    tmp_k = [0]
    tmp_out = [0]
    tmp_stage_cells = [0]
    tmp_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        tmp_q,
        tmp_k,
        tmp_out,
        tmp_stage_cells,
        tmp_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if tmp_stage_cells[0] != default_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = tmp_q[0]
    out_required_k_cells[0] = tmp_k[0]
    out_required_out_cells[0] = tmp_out[0]
    out_required_stage_cells[0] = tmp_stage_cells[0]
    out_required_stage_bytes[0] = tmp_stage_bytes[0]
    return ATTN_Q16_OK


def explicit_checked_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    err, default_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    tmp_q = [0]
    tmp_k = [0]
    tmp_out = [0]
    tmp_stage_cells = [0]
    tmp_stage_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        tmp_q,
        tmp_k,
        tmp_out,
        tmp_stage_cells,
        tmp_stage_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if tmp_stage_cells[0] != default_stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = tmp_q[0]
    out_required_k_cells[0] = tmp_k[0]
    out_required_out_cells[0] = tmp_out[0]
    out_required_stage_cells[0] = tmp_stage_cells[0]
    out_required_stage_bytes[0] = tmp_stage_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_default_capacity_required_bytes_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnlyDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "token_count," in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnly(" in body
    assert "if (required_stage_cells != default_stage_cell_capacity)" in body


def test_known_vector_and_null_contract() -> None:
    q_rows = [0] * 48
    k_rows = [0] * 64
    out_scores = [0] * 24

    got_q = [11]
    got_k = [12]
    got_out = [13]
    got_stage_cells = [14]
    got_stage_bytes = [15]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        4,
        6,
        out_scores,
        len(out_scores),
        got_q,
        got_k,
        got_out,
        got_stage_cells,
        got_stage_bytes,
    )
    assert err == ATTN_Q16_OK
    assert got_q[0] == ((3 - 1) * 6) + 6
    assert got_k[0] == ((4 - 1) * 6) + 6
    assert got_out[0] == ((3 - 1) * 4) + 4
    assert got_stage_cells[0] == 12
    assert got_stage_bytes[0] == 96

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity(
        q_rows,
        len(q_rows),
        3,
        k_rows,
        len(k_rows),
        4,
        6,
        out_scores,
        len(out_scores),
        None,
        got_k,
        got_out,
        got_stage_cells,
        got_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR


def test_overflow_and_no_partial_output_commit() -> None:
    got_q = [101]
    got_k = [102]
    got_out = [103]
    got_stage_cells = [104]
    got_stage_bytes = [105]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity(
        [0],
        1,
        I64_MAX,
        [0],
        1,
        2,
        1,
        [0],
        1,
        got_q,
        got_k,
        got_out,
        got_stage_cells,
        got_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert got_q == [101]
    assert got_k == [102]
    assert got_out == [103]
    assert got_stage_cells == [104]
    assert got_stage_bytes == [105]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(725)

    for _ in range(600):
        query_row_count = rng.randint(0, 48)
        token_count = rng.randint(0, 48)
        head_dim = rng.randint(0, 48)

        q_capacity = rng.randint(0, 4096)
        k_capacity = rng.randint(0, 4096)
        out_capacity = rng.randint(0, 4096)

        q_rows = [0] * max(q_capacity, 1)
        k_rows = [0] * max(k_capacity, 1)
        out_scores = [0] * max(out_capacity, 1)

        got_q = [9001]
        got_k = [9002]
        got_out = [9003]
        got_stage_cells = [9004]
        got_stage_bytes = [9005]

        exp_q = [8001]
        exp_k = [8002]
        exp_out = [8003]
        exp_stage_cells = [8004]
        exp_stage_bytes = [8005]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            got_q,
            got_k,
            got_out,
            got_stage_cells,
            got_stage_bytes,
        )

        err_exp = explicit_checked_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            exp_q,
            exp_k,
            exp_out,
            exp_stage_cells,
            exp_stage_bytes,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_q[0] == exp_q[0]
            assert got_k[0] == exp_k[0]
            assert got_out[0] == exp_out[0]
            assert got_stage_cells[0] == exp_stage_cells[0]
            assert got_stage_bytes[0] == exp_stage_bytes[0]
        else:
            assert got_q == [9001]
            assert got_k == [9002]
            assert got_out == [9003]
            assert got_stage_cells == [9004]
            assert got_stage_bytes == [9005]
