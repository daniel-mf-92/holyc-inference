#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-835)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only,
)


def _try_mul_i64_checked(a: int, b: int) -> tuple[int, int]:
    limit = (1 << 63) - 1
    out = a * b
    if out < -limit - 1 or out > limit:
        return (ATTN_Q16_ERR_OVERFLOW, 0)
    return (ATTN_Q16_OK, out)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    snapshot_query_row_count = query_row_count
    snapshot_token_count = token_count
    snapshot_head_dim = head_dim

    err, q_rows_capacity = _try_mul_i64_checked(query_row_count, head_dim)
    if err != ATTN_Q16_OK:
        return err

    err, k_rows_capacity = _try_mul_i64_checked(token_count, head_dim)
    if err != ATTN_Q16_OK:
        return err

    err, out_scores_capacity = _try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    snapshot_q_rows_capacity = q_rows_capacity
    snapshot_k_rows_capacity = k_rows_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    staged_last_q = [0]
    staged_last_k = [0]
    staged_last_out = [0]
    staged_req_q = [0]
    staged_req_k = [0]
    staged_req_out = [0]
    staged_req_q_bytes = [0]
    staged_req_k_bytes = [0]
    staged_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only(
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

    canonical_last_q = [0]
    canonical_last_k = [0]
    canonical_last_out = [0]
    canonical_req_q = [0]
    canonical_req_k = [0]
    canonical_req_out = [0]
    canonical_req_q_bytes = [0]
    canonical_req_k_bytes = [0]
    canonical_req_out_bytes = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity(
        q_rows_q16,
        query_row_count,
        k_rows_q16,
        token_count,
        head_dim,
        out_scores_q32,
        canonical_last_q,
        canonical_last_k,
        canonical_last_out,
        canonical_req_q,
        canonical_req_k,
        canonical_req_out,
        canonical_req_q_bytes,
        canonical_req_k_bytes,
        canonical_req_out_bytes,
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot_query_row_count != query_row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_head_dim != head_dim:
        return ATTN_Q16_ERR_BAD_PARAM

    if snapshot_q_rows_capacity != q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_k_rows_capacity != k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_last_q[0] != canonical_last_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_k[0] != canonical_last_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out[0] != canonical_last_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_q[0] != canonical_req_q[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_k[0] != canonical_req_k[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_out[0] != canonical_req_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_q_bytes[0] != canonical_req_q_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_k_bytes[0] != canonical_req_k_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_req_out_bytes[0] != canonical_req_out_bytes[0]:
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


def test_source_contains_companion() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStride"
        "RequiredBytesPreflightOnlyDefaultCapacityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_q_rows_capacity = q_rows_capacity;" in body
    assert "staged_required_out_bytes != canonical_required_out_bytes" in body


def test_known_vector() -> None:
    q, t, h = 9, 5, 7
    out = [0] * (q * t)
    last_q = [10]
    last_k = [11]
    last_out = [12]
    req_q = [13]
    req_k = [14]
    req_out = [15]
    req_qb = [16]
    req_kb = [17]
    req_ob = [18]
    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only_preflight_only(
        [0] * (q * h),
        q,
        [0] * (t * h),
        t,
        h,
        out,
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
        req_qb,
        req_kb,
        req_ob,
    )
    assert err == ATTN_Q16_OK
    assert last_q == [(q - 1) * h]
    assert last_k == [(t - 1) * h]
    assert last_out == [(q - 1) * t]
    assert req_q == [q * h]
    assert req_k == [t * h]
    assert req_out == [q * t]
    assert req_qb == [q * h * 8]
    assert req_kb == [t * h * 8]
    assert req_ob == [q * t * 8]


def test_randomized_and_overflow() -> None:
    rng = random.Random(835)
    for _ in range(1000):
        q = rng.randint(0, 48)
        t = rng.randint(0, 48)
        h = rng.randint(0, 96)
        got = [[1000 + i] for i in range(9)]
        exp = [[2000 + i] for i in range(9)]
        err_g = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only_preflight_only(
            [0] * max(1, rng.randint(1, 4096)),
            q,
            [0] * max(1, rng.randint(1, 4096)),
            t,
            h,
            [0] * max(1, rng.randint(1, 4096)),
            got[0],
            got[1],
            got[2],
            got[3],
            got[4],
            got[5],
            got[6],
            got[7],
            got[8],
        )
        err_e = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only_preflight_only(
            [0] * max(1, rng.randint(1, 4096)),
            q,
            [0] * max(1, rng.randint(1, 4096)),
            t,
            h,
            [0] * max(1, rng.randint(1, 4096)),
            exp[0],
            exp[1],
            exp[2],
            exp[3],
            exp[4],
            exp[5],
            exp[6],
            exp[7],
            exp[8],
        )
        assert err_g == err_e
        if err_g != ATTN_Q16_OK:
            assert got == [[1000 + i] for i in range(9)]

    huge = 1 << 62
    vals = [[i] for i in range(1, 10)]
    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only_default_capacity_commit_only_preflight_only_parity_commit_only_preflight_only(
        [0],
        huge,
        [0],
        3,
        huge,
        [0],
        vals[0],
        vals[1],
        vals[2],
        vals[3],
        vals[4],
        vals[5],
        vals[6],
        vals[7],
        vals[8],
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert vals == [[i] for i in range(1, 10)]


if __name__ == "__main__":
    test_source_contains_companion()
    test_known_vector()
    test_randomized_and_overflow()
    print("ok")
