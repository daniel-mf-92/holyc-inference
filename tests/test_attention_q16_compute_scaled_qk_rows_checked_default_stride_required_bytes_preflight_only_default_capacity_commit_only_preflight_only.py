#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityCommitOnlyPreflightOnly (IQ-811)."""

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
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only(
    q_rows_q16,
    query_row_count: int,
    k_rows_q16,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    if (
        out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_q_bytes is None
        or out_required_k_bytes is None
        or out_required_out_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        staged_last_q,
        staged_last_k,
        staged_last_out,
        staged_req_q,
        staged_req_k,
        staged_req_out,
        staged_req_q_bytes,
        staged_req_k_bytes,
        staged_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    out_last_q_base_index[0] = staged_last_q[0]
    out_last_k_base_index[0] = staged_last_k[0]
    out_last_out_base_index[0] = staged_last_out[0]
    out_required_q_cells[0] = staged_req_q[0]
    out_required_k_cells[0] = staged_req_k[0]
    out_required_out_cells[0] = staged_req_out[0]
    out_required_q_bytes[0] = staged_req_q_bytes[0]
    out_required_k_bytes[0] = staged_req_k_bytes[0]
    out_required_out_bytes[0] = staged_req_out_bytes[0]
    return ATTN_Q16_OK


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
    q_rows_q16,
    query_row_count: int,
    k_rows_q16,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    if (
        out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_q_bytes is None
        or out_required_k_bytes is None
        or out_required_out_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        staged_last_q,
        staged_last_k,
        staged_last_out,
        staged_req_q,
        staged_req_k,
        staged_req_out,
        staged_req_q_bytes,
        staged_req_k_bytes,
        staged_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_q_cells = try_mul_i64_checked(query_row_count, head_dim)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_k_cells = try_mul_i64_checked(token_count, head_dim)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_out_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_q_bytes = try_mul_i64_checked(recomputed_q_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_k_bytes = try_mul_i64_checked(recomputed_k_cells, 8)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_out_bytes = try_mul_i64_checked(recomputed_out_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    if query_row_count == 0:
        recomputed_last_q = 0
        recomputed_last_out = 0
    else:
        err, recomputed_last_q = try_mul_i64_checked(query_row_count - 1, head_dim)
        if err != ATTN_Q16_OK:
            return err
        err, recomputed_last_out = try_mul_i64_checked(query_row_count - 1, token_count)
        if err != ATTN_Q16_OK:
            return err

    if token_count == 0:
        recomputed_last_k = 0
    else:
        err, recomputed_last_k = try_mul_i64_checked(token_count - 1, head_dim)
        if err != ATTN_Q16_OK:
            return err

    if staged_req_q[0] != recomputed_q_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_k[0] != recomputed_k_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_out[0] != recomputed_out_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_q_bytes[0] != recomputed_q_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_k_bytes[0] != recomputed_k_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_out_bytes[0] != recomputed_out_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_q[0] != recomputed_last_q:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_k[0] != recomputed_last_k:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out[0] != recomputed_last_out:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_q_base_index[0] = staged_last_q[0]
    out_last_k_base_index[0] = staged_last_k[0]
    out_last_out_base_index[0] = staged_last_out[0]
    out_required_q_cells[0] = staged_req_q[0]
    out_required_k_cells[0] = staged_req_k[0]
    out_required_out_cells[0] = staged_req_out[0]
    out_required_q_bytes[0] = staged_req_q_bytes[0]
    out_required_k_bytes[0] = staged_req_k_bytes[0]
    out_required_out_bytes[0] = staged_req_out_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_default_capacity_commit_only_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyPreflightOnly("
    )
    assert sig in source
    body = source.split(sig, 1)[1]
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideRequiredBytesPreflightOnlyDefaultCapacityCommitOnly("
        in body
    )
    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(token_count," in body
    assert "AttentionTryMulI64Checked(recomputed_required_q_cells," in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body


def test_known_vector_expected_cells_bytes_and_last_indices() -> None:
    query_row_count = 5
    token_count = 4
    head_dim = 7
    q_rows = [0] * (query_row_count * head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (query_row_count * token_count)

    got_last_q = [101]
    got_last_k = [202]
    got_last_out = [303]
    got_req_q = [404]
    got_req_k = [505]
    got_req_out = [606]
    got_req_q_bytes = [707]
    got_req_k_bytes = [808]
    got_req_out_bytes = [909]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
        q_rows,
        query_row_count,
        k_rows,
        token_count,
        head_dim,
        out_scores,
        got_last_q,
        got_last_k,
        got_last_out,
        got_req_q,
        got_req_k,
        got_req_out,
        got_req_q_bytes,
        got_req_k_bytes,
        got_req_out_bytes,
    )

    assert err == ATTN_Q16_OK
    assert got_last_q == [(query_row_count - 1) * head_dim]
    assert got_last_k == [(token_count - 1) * head_dim]
    assert got_last_out == [(query_row_count - 1) * token_count]
    assert got_req_q == [query_row_count * head_dim]
    assert got_req_k == [token_count * head_dim]
    assert got_req_out == [query_row_count * token_count]
    assert got_req_q_bytes == [query_row_count * head_dim * 8]
    assert got_req_k_bytes == [token_count * head_dim * 8]
    assert got_req_out_bytes == [query_row_count * token_count * 8]


def test_error_paths_no_partial_output_publish() -> None:
    last_q = [11]
    last_k = [22]
    last_out = [33]
    req_q = [44]
    req_k = [55]
    req_out = [66]
    req_q_bytes = [77]
    req_k_bytes = [88]
    req_out_bytes = [99]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
        None,
        1,
        [0],
        1,
        1,
        [0],
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
        req_q_bytes,
        req_k_bytes,
        req_out_bytes,
    )

    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_q == [11]
    assert last_k == [22]
    assert last_out == [33]
    assert req_q == [44]
    assert req_k == [55]
    assert req_out == [66]
    assert req_q_bytes == [77]
    assert req_k_bytes == [88]
    assert req_out_bytes == [99]


def test_randomized_parity() -> None:
    rng = random.Random(20260421_811)

    for _ in range(1000):
        query_row_count = rng.randint(0, 40)
        token_count = rng.randint(0, 40)
        head_dim = rng.randint(0, 128)

        q_rows = [0] * max(1, rng.randint(1, 16384))
        k_rows = [0] * max(1, rng.randint(1, 16384))
        out_scores = [0] * max(1, rng.randint(1, 16384))

        got_last_q = [1001]
        got_last_k = [1002]
        got_last_out = [1003]
        got_req_q = [1004]
        got_req_k = [1005]
        got_req_out = [1006]
        got_req_q_bytes = [1007]
        got_req_k_bytes = [1008]
        got_req_out_bytes = [1009]

        err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
            q_rows,
            query_row_count,
            k_rows,
            token_count,
            head_dim,
            out_scores,
            got_last_q,
            got_last_k,
            got_last_out,
            got_req_q,
            got_req_k,
            got_req_out,
            got_req_q_bytes,
            got_req_k_bytes,
            got_req_out_bytes,
        )

        if err == ATTN_Q16_OK:
            assert got_last_q[0] == (0 if query_row_count == 0 else (query_row_count - 1) * head_dim)
            assert got_last_k[0] == (0 if token_count == 0 else (token_count - 1) * head_dim)
            assert got_last_out[0] == (
                0 if query_row_count == 0 else (query_row_count - 1) * token_count
            )
            assert got_req_q[0] == query_row_count * head_dim
            assert got_req_k[0] == token_count * head_dim
            assert got_req_out[0] == query_row_count * token_count
            assert got_req_q_bytes[0] == got_req_q[0] * 8
            assert got_req_k_bytes[0] == got_req_k[0] * 8
            assert got_req_out_bytes[0] == got_req_out[0] * 8


def test_overflow_head_dim_multiplication_propagates_error() -> None:
    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
        [0],
        huge,
        [0],
        3,
        huge,
        [0],
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
        [9],
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
