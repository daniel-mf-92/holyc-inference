#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityCommitOnlyParityCommitOnly (IQ-915)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only(
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

    if (
        out_last_q_base_index is out_last_k_base_index
        or out_last_q_base_index is out_last_out_base_index
        or out_last_q_base_index is out_required_q_cells
        or out_last_q_base_index is out_required_k_cells
        or out_last_q_base_index is out_required_out_cells
        or out_last_q_base_index is out_required_q_bytes
        or out_last_q_base_index is out_required_k_bytes
        or out_last_q_base_index is out_required_out_bytes
        or out_last_k_base_index is out_last_out_base_index
        or out_last_k_base_index is out_required_q_cells
        or out_last_k_base_index is out_required_k_cells
        or out_last_k_base_index is out_required_out_cells
        or out_last_k_base_index is out_required_q_bytes
        or out_last_k_base_index is out_required_k_bytes
        or out_last_k_base_index is out_required_out_bytes
        or out_last_out_base_index is out_required_q_cells
        or out_last_out_base_index is out_required_k_cells
        or out_last_out_base_index is out_required_out_cells
        or out_last_out_base_index is out_required_q_bytes
        or out_last_out_base_index is out_required_k_bytes
        or out_last_out_base_index is out_required_out_bytes
        or out_required_q_cells is out_required_k_cells
        or out_required_q_cells is out_required_out_cells
        or out_required_q_cells is out_required_q_bytes
        or out_required_q_cells is out_required_k_bytes
        or out_required_q_cells is out_required_out_bytes
        or out_required_k_cells is out_required_out_cells
        or out_required_k_cells is out_required_q_bytes
        or out_required_k_cells is out_required_k_bytes
        or out_required_k_cells is out_required_out_bytes
        or out_required_out_cells is out_required_q_bytes
        or out_required_out_cells is out_required_k_bytes
        or out_required_out_cells is out_required_out_bytes
        or out_required_q_bytes is out_required_k_bytes
        or out_required_q_bytes is out_required_out_bytes
        or out_required_k_bytes is out_required_out_bytes
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, snapshot_q_rows_capacity = try_mul_i64_checked(query_row_count, head_dim)
    if err != ATTN_Q16_OK:
        return err
    err, snapshot_k_rows_capacity = try_mul_i64_checked(token_count, head_dim)
    if err != ATTN_Q16_OK:
        return err
    err, snapshot_out_scores_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
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

    err, recomputed_q_rows_capacity = try_mul_i64_checked(query_row_count, head_dim)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_k_rows_capacity = try_mul_i64_checked(token_count, head_dim)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_out_scores_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    if snapshot_q_rows_capacity != recomputed_q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_k_rows_capacity != recomputed_k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != recomputed_out_scores_capacity:
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


def explicit_commit_only_composition(
    q_rows_q16,
    query_row_count: int,
    k_rows_q16,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_last_q_base_index: list[int],
    out_last_k_base_index: list[int],
    out_last_out_base_index: list[int],
    out_required_q_cells: list[int],
    out_required_k_cells: list[int],
    out_required_out_cells: list[int],
    out_required_q_bytes: list[int],
    out_required_k_bytes: list[int],
    out_required_out_bytes: list[int],
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        out_last_q_base_index,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
        out_required_q_bytes,
        out_required_k_bytes,
        out_required_out_bytes,
    )


def test_source_contains_commit_only_parity_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyParityCommitOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert "snapshot_q_rows_capacity =" in body
    assert "snapshot_k_rows_capacity =" in body
    assert "snapshot_out_scores_capacity =" in body


def test_known_vector_commit_only_outputs() -> None:
    query_row_count = 9
    token_count = 4
    head_dim = 6

    q_rows = [0] * (query_row_count * head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (query_row_count * token_count)

    got_last_q = [101]
    got_last_k = [102]
    got_last_out = [103]
    got_req_q = [104]
    got_req_k = [105]
    got_req_out = [106]
    got_req_q_bytes = [107]
    got_req_k_bytes = [108]
    got_req_out_bytes = [109]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only(
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


def test_randomized_commit_only_vs_explicit_composition() -> None:
    rng = random.Random(20260421_915)

    for _ in range(1000):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)
        head_dim = rng.randint(0, 96)

        q_rows = [0] * max(1, rng.randint(1, 8192))
        k_rows = [0] * max(1, rng.randint(1, 8192))
        out_scores = [0] * max(1, rng.randint(1, 8192))

        got_last_q = [701]
        got_last_k = [702]
        got_last_out = [703]
        got_req_q = [704]
        got_req_k = [705]
        got_req_out = [706]
        got_req_q_bytes = [707]
        got_req_k_bytes = [708]
        got_req_out_bytes = [709]

        exp_last_q = [801]
        exp_last_k = [802]
        exp_last_out = [803]
        exp_req_q = [804]
        exp_req_k = [805]
        exp_req_out = [806]
        exp_req_q_bytes = [807]
        exp_req_k_bytes = [808]
        exp_req_out_bytes = [809]

        err_new = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only(
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

        err_ref = explicit_commit_only_composition(
            q_rows,
            query_row_count,
            k_rows,
            token_count,
            head_dim,
            out_scores,
            exp_last_q,
            exp_last_k,
            exp_last_out,
            exp_req_q,
            exp_req_k,
            exp_req_out,
            exp_req_q_bytes,
            exp_req_k_bytes,
            exp_req_out_bytes,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert got_last_q == exp_last_q
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_q == exp_req_q
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
            assert got_req_q_bytes == exp_req_q_bytes
            assert got_req_k_bytes == exp_req_k_bytes
            assert got_req_out_bytes == exp_req_out_bytes
        else:
            assert got_last_q == [701]
            assert got_last_k == [702]
            assert got_last_out == [703]
            assert got_req_q == [704]
            assert got_req_k == [705]
            assert got_req_out == [706]
            assert got_req_q_bytes == [707]
            assert got_req_k_bytes == [708]
            assert got_req_out_bytes == [709]


def test_overflow_passthrough_preserves_outputs() -> None:
    huge = 1 << 62

    got_last_q = [11]
    got_last_k = [12]
    got_last_out = [13]
    got_req_q = [14]
    got_req_k = [15]
    got_req_out = [16]
    got_req_q_bytes = [17]
    got_req_k_bytes = [18]
    got_req_out_bytes = [19]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only(
        [0],
        huge,
        [0],
        3,
        3,
        [0],
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

    assert err == ATTN_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_commit_only_parity_commit_only_helper()
    test_known_vector_commit_only_outputs()
    test_randomized_commit_only_vs_explicit_composition()
    test_overflow_passthrough_preserves_outputs()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only=ok"
    )
