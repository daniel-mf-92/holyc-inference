#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityCommitOnlyParityCommitOnlyPreflightOnly (IQ-929)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only_preflight_only(
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

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only(
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

    if staged_last_q[0] != recomputed_last_q:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_k[0] != recomputed_last_k:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out[0] != recomputed_last_out:
        return ATTN_Q16_ERR_BAD_PARAM
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


def explicit_preflight_only_composition(
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
    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only(
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

    if staged_last_q[0] != recomputed_last_q:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_k[0] != recomputed_last_k:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out[0] != recomputed_last_out:
        return ATTN_Q16_ERR_BAD_PARAM
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


def test_source_contains_commit_only_parity_commit_only_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyParityCommitOnlyPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideRequiredBytesPreflightOnlyDefaultCapacityCommitOnlyParityCommitOnly("
        in body
    )
    assert "*out_last_q_base_index = staged_last_q_base_index;" in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body


def test_known_vector_outputs() -> None:
    query_row_count = 7
    token_count = 6
    head_dim = 5

    q_rows = [0] * (query_row_count * head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (query_row_count * token_count)

    got_last_q = [1]
    got_last_k = [2]
    got_last_out = [3]
    got_req_q = [4]
    got_req_k = [5]
    got_req_out = [6]
    got_req_q_bytes = [7]
    got_req_k_bytes = [8]
    got_req_out_bytes = [9]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only_preflight_only(
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


def test_error_paths_preserve_outputs() -> None:
    q_rows = [0] * 64
    k_rows = [0] * 64
    out_scores = [0] * 64

    got_last_q = [501]
    got_last_k = [502]
    got_last_out = [503]
    got_req_q = [504]
    got_req_k = [505]
    got_req_out = [506]
    got_req_q_bytes = [507]
    got_req_k_bytes = [508]
    got_req_out_bytes = [509]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only_preflight_only(
        q_rows,
        -1,
        k_rows,
        8,
        8,
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
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_last_q == [501]
    assert got_last_k == [502]
    assert got_last_out == [503]
    assert got_req_q == [504]
    assert got_req_k == [505]
    assert got_req_out == [506]
    assert got_req_q_bytes == [507]
    assert got_req_k_bytes == [508]
    assert got_req_out_bytes == [509]


def test_randomized_preflight_only_vs_explicit_composition() -> None:
    rng = random.Random(20260421_929)

    for _ in range(1000):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)
        head_dim = rng.randint(0, 96)

        q_rows = [0] * max(1, rng.randint(1, 8192))
        k_rows = [0] * max(1, rng.randint(1, 8192))
        out_scores = [0] * max(1, rng.randint(1, 8192))

        got_last_q = [601]
        got_last_k = [602]
        got_last_out = [603]
        got_req_q = [604]
        got_req_k = [605]
        got_req_out = [606]
        got_req_q_bytes = [607]
        got_req_k_bytes = [608]
        got_req_out_bytes = [609]

        exp_last_q = [701]
        exp_last_k = [702]
        exp_last_out = [703]
        exp_req_q = [704]
        exp_req_k = [705]
        exp_req_out = [706]
        exp_req_q_bytes = [707]
        exp_req_k_bytes = [708]
        exp_req_out_bytes = [709]

        err_new = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only_preflight_only(
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

        err_ref = explicit_preflight_only_composition(
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
            assert got_last_q == [601]
            assert got_last_k == [602]
            assert got_last_out == [603]
            assert got_req_q == [604]
            assert got_req_k == [605]
            assert got_req_out == [606]
            assert got_req_q_bytes == [607]
            assert got_req_k_bytes == [608]
            assert got_req_out_bytes == [609]


def test_overflow_passthrough_preserves_outputs() -> None:
    huge = 1 << 62

    got_last_q = [21]
    got_last_k = [22]
    got_last_out = [23]
    got_req_q = [24]
    got_req_k = [25]
    got_req_out = [26]
    got_req_q_bytes = [27]
    got_req_k_bytes = [28]
    got_req_out_bytes = [29]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only_preflight_only(
        [0],
        huge,
        [0],
        4,
        4,
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
    assert got_last_q == [21]
    assert got_last_k == [22]
    assert got_last_out == [23]
    assert got_req_q == [24]
    assert got_req_k == [25]
    assert got_req_out == [26]
    assert got_req_q_bytes == [27]
    assert got_req_k_bytes == [28]
    assert got_req_out_bytes == [29]


if __name__ == "__main__":
    test_source_contains_commit_only_parity_commit_only_preflight_only_helper()
    test_known_vector_outputs()
    test_error_paths_preserve_outputs()
    test_randomized_preflight_only_vs_explicit_composition()
    test_overflow_passthrough_preserves_outputs()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity_commit_only_preflight_only=ok"
    )
