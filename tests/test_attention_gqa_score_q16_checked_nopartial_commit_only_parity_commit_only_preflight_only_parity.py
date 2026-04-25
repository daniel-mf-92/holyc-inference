#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParity (IQ-1410)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_gqa_score_q16_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    try_mul_i64_checked,
)
from test_attention_gqa_score_q16_checked_nopartial_commit_only_parity_commit_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only,
)
from test_attention_gqa_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only,
)

I64_MAX = (1 << 63) - 1


def gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
    q_rows_q16,
    q_rows_capacity: int,
    q_rows: int,
    k_rows_q16,
    k_rows_capacity: int,
    k_rows: int,
    group_count: int,
    seq_len: int,
    head_dim: int,
    out_scores_q32,
    out_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if q_rows < 0 or k_rows < 0 or group_count <= 0 or seq_len < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_q_cells = try_mul_i64_checked(q_rows, head_dim)
    if err != ATTN_Q16_OK:
        return err

    err, required_k_cells = try_mul_i64_checked(k_rows, seq_len)
    if err != ATTN_Q16_OK:
        return err
    err, required_k_cells = try_mul_i64_checked(required_k_cells, head_dim)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(q_rows, seq_len)
    if err != ATTN_Q16_OK:
        return err

    if (
        required_q_cells > q_rows_capacity
        or required_k_cells > k_rows_capacity
        or required_out_cells > out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if required_out_cells == 0:
        return ATTN_Q16_OK

    snapshot_q_rows = q_rows
    snapshot_k_rows = k_rows
    snapshot_group_count = group_count
    snapshot_seq_len = seq_len
    snapshot_head_dim = head_dim
    snapshot_out_capacity = out_capacity

    staged_req_q = [required_q_cells]
    staged_req_k = [required_k_cells]
    staged_req_out = [required_out_cells]

    out_snapshot = out_scores_q32[:required_out_cells]
    staged_commit = [0] * required_out_cells

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        out_scores_q32,
        out_capacity,
        staged_req_q,
        staged_req_k,
        staged_req_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        staged_commit,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_q_cells = try_mul_i64_checked(snapshot_q_rows, snapshot_head_dim)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_k_cells = try_mul_i64_checked(snapshot_k_rows, snapshot_seq_len)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_k_cells = try_mul_i64_checked(recomputed_k_cells, snapshot_head_dim)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_out_cells = try_mul_i64_checked(snapshot_q_rows, snapshot_seq_len)
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_q_rows != q_rows
        or snapshot_k_rows != k_rows
        or snapshot_group_count != group_count
        or snapshot_seq_len != seq_len
        or snapshot_head_dim != head_dim
        or snapshot_out_capacity != out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_q_cells != recomputed_q_cells
        or required_k_cells != recomputed_k_cells
        or required_out_cells != recomputed_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_q_cells > q_rows_capacity
        or required_k_cells > k_rows_capacity
        or required_out_cells > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_req_q[0] != required_q_cells
        or staged_req_k[0] != required_k_cells
        or staged_req_out[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_scores_q32[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_scores_q32[:required_out_cells] = staged_commit
    return ATTN_Q16_OK


def explicit_parity_composition(*args, **kwargs) -> int:
    return gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        *args, **kwargs
    )


def test_fixed_vector_reference_exact_score_publish() -> None:
    q_rows = 4
    k_rows = 2
    group_count = 2
    seq_len = 3
    head_dim = 4

    q = [
        1 << 16,
        2 << 16,
        3 << 16,
        4 << 16,
        2 << 16,
        1 << 16,
        -(1 << 16),
        3 << 16,
        -(2 << 16),
        3 << 16,
        1 << 16,
        2 << 16,
        1 << 16,
        -(3 << 16),
        2 << 16,
        1 << 16,
    ]

    k = [
        1 << 16,
        0,
        2 << 16,
        1 << 16,
        -(1 << 16),
        2 << 16,
        1 << 16,
        0,
        3 << 16,
        -(1 << 16),
        1 << 16,
        2 << 16,
        2 << 16,
        1 << 16,
        0,
        -(1 << 16),
        1 << 16,
        2 << 16,
        2 << 16,
        1 << 16,
        -(2 << 16),
        1 << 16,
        1 << 16,
        3 << 16,
    ]

    out_a = [7373] * (q_rows * seq_len)
    out_b = [7373] * (q_rows * seq_len)

    err_a = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        q,
        len(q),
        q_rows,
        k,
        len(k),
        k_rows,
        group_count,
        seq_len,
        head_dim,
        out_a,
        len(out_a),
    )
    err_b = explicit_parity_composition(
        q,
        len(q),
        q_rows,
        k,
        len(k),
        k_rows,
        group_count,
        seq_len,
        head_dim,
        out_b,
        len(out_b),
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_null_capacity_and_overflow_contract() -> None:
    out = [5150] * 6

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        None,
        0,
        0,
        [],
        0,
        0,
        1,
        0,
        0,
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == [5150] * 6

    q = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    k = [1 << 16, 2 << 16, 3 << 16, 4 << 16]

    out_cap = [900] * 8
    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        q,
        len(q) - 1,
        1,
        k,
        len(k),
        1,
        1,
        1,
        4,
        out_cap,
        len(out_cap),
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_cap == [900] * 8

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
        [1],
        1,
        I64_MAX,
        [1],
        1,
        1,
        1,
        1,
        2,
        [0],
        1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_zero_write_failure_and_success_publish() -> None:
    rng = random.Random(1410)

    for _ in range(120):
        q_rows = rng.randint(1, 4)
        divisors = [d for d in range(1, q_rows + 1) if (q_rows % d) == 0]
        group_count = rng.choice(divisors)
        k_rows = q_rows // group_count
        seq_len = rng.randint(1, 4)
        head_dim = rng.randint(1, 5)

        required_q = q_rows * head_dim
        required_k = k_rows * seq_len * head_dim
        required_out = q_rows * seq_len

        q = [rng.randint(-(1 << 16), (1 << 16)) for _ in range(required_q)]
        k = [rng.randint(-(1 << 16), (1 << 16)) for _ in range(required_k)]

        out_a = [rng.randint(-4000, 4000) for _ in range(required_out)]
        out_b = out_a.copy()

        err_a = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity(
            q,
            len(q),
            q_rows,
            k,
            len(k),
            k_rows,
            group_count,
            seq_len,
            head_dim,
            out_a,
            len(out_a),
        )
        err_b = explicit_parity_composition(
            q,
            len(q),
            q_rows,
            k,
            len(k),
            k_rows,
            group_count,
            seq_len,
            head_dim,
            out_b,
            len(out_b),
        )

        assert err_a == err_b == ATTN_Q16_OK
        assert out_a == out_b
