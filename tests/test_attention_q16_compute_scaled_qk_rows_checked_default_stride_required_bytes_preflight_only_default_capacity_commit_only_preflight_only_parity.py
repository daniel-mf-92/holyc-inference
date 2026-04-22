#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyPreflightOnlyParity (IQ-833)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity,
)


I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def _try_mul_i64(a: int, b: int) -> tuple[int, int | None]:
    prod = a * b
    if prod < I64_MIN or prod > I64_MAX:
        return ATTN_Q16_ERR_OVERFLOW, None
    return ATTN_Q16_OK, prod


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
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

    status, q_rows_capacity = _try_mul_i64(query_row_count, head_dim)
    if status != ATTN_Q16_OK:
        return status
    status, k_rows_capacity = _try_mul_i64(token_count, head_dim)
    if status != ATTN_Q16_OK:
        return status
    status, out_scores_capacity = _try_mul_i64(query_row_count, token_count)
    if status != ATTN_Q16_OK:
        return ATTN_Q16_ERR_OVERFLOW

    snapshot_q_rows_capacity = q_rows_capacity
    snapshot_k_rows_capacity = k_rows_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    err_a_last_q = [0]
    err_a_last_k = [0]
    err_a_last_out = [0]
    err_a_req_q = [0]
    err_a_req_k = [0]
    err_a_req_out = [0]
    err_a_req_q_bytes = [0]
    err_a_req_k_bytes = [0]
    err_a_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        err_a_last_q,
        err_a_last_k,
        err_a_last_out,
        err_a_req_q,
        err_a_req_k,
        err_a_req_out,
        err_a_req_q_bytes,
        err_a_req_k_bytes,
        err_a_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    status, post_q_rows_capacity = _try_mul_i64(query_row_count, head_dim)
    if status != ATTN_Q16_OK:
        return status
    status, post_k_rows_capacity = _try_mul_i64(token_count, head_dim)
    if status != ATTN_Q16_OK:
        return status
    status, post_out_scores_capacity = _try_mul_i64(query_row_count, token_count)
    if status != ATTN_Q16_OK:
        return status

    if snapshot_q_rows_capacity != post_q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_k_rows_capacity != post_k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != post_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err_b_last_q = [0]
    err_b_last_k = [0]
    err_b_last_out = [0]
    err_b_req_q = [0]
    err_b_req_k = [0]
    err_b_req_out = [0]
    err_b_req_q_bytes = [0]
    err_b_req_k_bytes = [0]
    err_b_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_parity(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        err_b_last_q,
        err_b_last_k,
        err_b_last_out,
        err_b_req_q,
        err_b_req_k,
        err_b_req_out,
        err_b_req_q_bytes,
        err_b_req_k_bytes,
        err_b_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if err_a_last_q[0] != err_b_last_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_last_k[0] != err_b_last_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_last_out[0] != err_b_last_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_req_q[0] != err_b_req_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_req_k[0] != err_b_req_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_req_out[0] != err_b_req_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_req_q_bytes[0] != err_b_req_q_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_req_k_bytes[0] != err_b_req_k_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if err_a_req_out_bytes[0] != err_b_req_out_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_q_base_index[0] = err_a_last_q[0]
    out_last_k_base_index[0] = err_a_last_k[0]
    out_last_out_base_index[0] = err_a_last_out[0]
    out_required_q_cells[0] = err_a_req_q[0]
    out_required_k_cells[0] = err_a_req_k[0]
    out_required_out_cells[0] = err_a_req_out[0]
    out_required_q_bytes[0] = err_a_req_q_bytes[0]
    out_required_k_bytes[0] = err_a_req_k_bytes[0]
    out_required_out_bytes[0] = err_a_req_out_bytes[0]
    return ATTN_Q16_OK


def test_source_contains_commit_only_preflight_only_parity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyPreflightOnlyParity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_head_dim = head_dim;" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyPreflightOnly"
    ) in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacity("
    ) in body


def test_known_vector_geometry() -> None:
    query_row_count = 6
    token_count = 4
    head_dim = 7

    q_rows = [0] * (query_row_count * head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (query_row_count * token_count)

    got_last_q = [11]
    got_last_k = [22]
    got_last_out = [33]
    got_req_q = [44]
    got_req_k = [55]
    got_req_out = [66]
    got_req_q_bytes = [77]
    got_req_k_bytes = [88]
    got_req_out_bytes = [99]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
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


def test_alias_rejected_no_publish() -> None:
    shared = [123]
    req_out = [124]
    req_q_bytes = [125]
    req_k_bytes = [126]
    req_out_bytes = [127]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
        [0],
        1,
        [0],
        1,
        1,
        [0],
        shared,
        shared,
        [41],
        [42],
        [43],
        req_out,
        req_q_bytes,
        req_k_bytes,
        req_out_bytes,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert shared == [123]
    assert req_out == [124]
    assert req_q_bytes == [125]
    assert req_k_bytes == [126]
    assert req_out_bytes == [127]


def test_overflow_and_random_parity() -> None:
    limit = (1 << 63) - 1
    large = (limit // 8) + 1

    q_rows = [0]
    k_rows = [0]
    out_scores = [0]

    lq = [1]
    lk = [2]
    lo = [3]
    rq = [4]
    rk = [5]
    ro = [6]
    bq = [7]
    bk = [8]
    bo = [9]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
        q_rows,
        large,
        k_rows,
        1,
        1,
        out_scores,
        lq,
        lk,
        lo,
        rq,
        rk,
        ro,
        bq,
        bk,
        bo,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert lq == [1]
    assert lk == [2]
    assert lo == [3]

    rng = random.Random(833)
    for _ in range(150):
        query_row_count = rng.randint(0, 128)
        token_count = rng.randint(0, 128)
        head_dim = rng.randint(0, 96)

        q_rows = [0] * max(1, query_row_count * max(1, head_dim))
        k_rows = [0] * max(1, token_count * max(1, head_dim))
        out_scores = [0] * max(1, query_row_count * max(1, token_count))

        got_last_q = [9]
        got_last_k = [8]
        got_last_out = [7]
        got_req_q = [6]
        got_req_k = [5]
        got_req_out = [4]
        got_req_q_bytes = [3]
        got_req_k_bytes = [2]
        got_req_out_bytes = [1]

        exp_last_q = [101]
        exp_last_k = [202]
        exp_last_out = [303]
        exp_req_q = [404]
        exp_req_k = [505]
        exp_req_out = [606]
        exp_req_q_bytes = [707]
        exp_req_k_bytes = [808]
        exp_req_out_bytes = [909]

        got_err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity(
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
        exp_err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only(
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

        assert got_err == exp_err
        if got_err == ATTN_Q16_OK:
            assert got_last_q == exp_last_q
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_q == exp_req_q
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
            assert got_req_q_bytes == exp_req_q_bytes
            assert got_req_k_bytes == exp_req_k_bytes
            assert got_req_out_bytes == exp_req_out_bytes


if __name__ == "__main__":
    test_source_contains_commit_only_preflight_only_parity_helper()
    test_known_vector_geometry()
    test_alias_rejected_no_publish()
    test_overflow_and_random_parity()
    print("ok")
