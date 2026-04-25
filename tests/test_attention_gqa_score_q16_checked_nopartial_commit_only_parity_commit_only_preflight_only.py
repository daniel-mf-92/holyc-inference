#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnly (IQ-1409)."""

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
from test_attention_gqa_score_q16_checked_nopartial_commit_only_parity import (
    gqa_attention_score_q16_checked_nopartial_commit_only_parity,
)
from test_attention_gqa_score_q16_checked_nopartial_commit_only_parity_commit_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only,
)

I64_MAX = (1 << 63) - 1


def gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
    out_required_q_cells,
    out_required_k_cells,
    out_required_out_cells,
) -> int:
    if (
        q_rows_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_q_cells is out_required_k_cells
        or out_required_q_cells is out_required_out_cells
        or out_required_k_cells is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if q_rows < 0 or k_rows < 0 or group_count <= 0 or seq_len < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_q_rows = q_rows
    snapshot_k_rows = k_rows
    snapshot_group_count = group_count
    snapshot_seq_len = seq_len
    snapshot_head_dim = head_dim
    snapshot_q_rows_capacity = q_rows_capacity
    snapshot_k_rows_capacity = k_rows_capacity
    snapshot_out_capacity = out_capacity
    snapshot_q_rows_q16 = q_rows_q16
    snapshot_k_rows_q16 = k_rows_q16
    snapshot_out_scores_q32 = out_scores_q32
    snapshot_required_q_slot = out_required_q_cells[0]
    snapshot_required_k_slot = out_required_k_cells[0]
    snapshot_required_out_slot = out_required_out_cells[0]

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

    if (
        out_required_q_cells is q_rows_q16
        or out_required_q_cells is k_rows_q16
        or out_required_q_cells is out_scores_q32
        or out_required_k_cells is q_rows_q16
        or out_required_k_cells is k_rows_q16
        or out_required_k_cells is out_scores_q32
        or out_required_out_cells is q_rows_q16
        or out_required_out_cells is k_rows_q16
        or out_required_out_cells is out_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

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

    if required_q_cells == 0:
        snapshot_q_digest = ()
    else:
        snapshot_q_digest = tuple(q_rows_q16[:required_q_cells])

    if required_k_cells == 0:
        snapshot_k_digest = ()
    else:
        snapshot_k_digest = tuple(k_rows_q16[:required_k_cells])

    if required_out_cells == 0:
        if (
            required_q_cells != recomputed_q_cells
            or required_k_cells != recomputed_k_cells
            or required_out_cells != recomputed_out_cells
        ):
            return ATTN_Q16_ERR_BAD_PARAM

        if (
            snapshot_q_rows != q_rows
            or snapshot_k_rows != k_rows
            or snapshot_group_count != group_count
            or snapshot_seq_len != seq_len
            or snapshot_head_dim != head_dim
            or snapshot_q_rows_capacity != q_rows_capacity
            or snapshot_k_rows_capacity != k_rows_capacity
            or snapshot_out_capacity != out_capacity
            or snapshot_q_rows_q16 is not q_rows_q16
            or snapshot_k_rows_q16 is not k_rows_q16
            or snapshot_out_scores_q32 is not out_scores_q32
        ):
            return ATTN_Q16_ERR_BAD_PARAM

        if (
            out_required_q_cells[0] != snapshot_required_q_slot
            or out_required_k_cells[0] != snapshot_required_k_slot
            or out_required_out_cells[0] != snapshot_required_out_slot
        ):
            return ATTN_Q16_ERR_BAD_PARAM
        return ATTN_Q16_OK

    if out_scores_q32 is q_rows_q16 or out_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_commit = [0] * required_out_cells
    staged_parity = [0] * required_out_cells
    out_snapshot = out_scores_q32[:required_out_cells]

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

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        staged_parity,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if (
        snapshot_q_rows != q_rows
        or snapshot_k_rows != k_rows
        or snapshot_group_count != group_count
        or snapshot_seq_len != seq_len
        or snapshot_head_dim != head_dim
        or snapshot_q_rows_capacity != q_rows_capacity
        or snapshot_k_rows_capacity != k_rows_capacity
        or snapshot_out_capacity != out_capacity
        or snapshot_q_rows_q16 is not q_rows_q16
        or snapshot_k_rows_q16 is not k_rows_q16
        or snapshot_out_scores_q32 is not out_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_q_cells != recomputed_q_cells
        or required_k_cells != recomputed_k_cells
        or required_out_cells != recomputed_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        required_q_cells > snapshot_q_rows_capacity
        or required_k_cells > snapshot_k_rows_capacity
        or required_out_cells > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if required_q_cells == 0:
        recomputed_q_digest = ()
    else:
        recomputed_q_digest = tuple(q_rows_q16[:required_q_cells])

    if required_k_cells == 0:
        recomputed_k_digest = ()
    else:
        recomputed_k_digest = tuple(k_rows_q16[:required_k_cells])

    if snapshot_q_digest != recomputed_q_digest or snapshot_k_digest != recomputed_k_digest:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_commit != staged_parity:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_q_cells[0] != snapshot_required_q_slot
        or out_required_k_cells[0] != snapshot_required_k_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_scores_q32[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def explicit_preflight_only_composition(*args, **kwargs) -> int:
    return gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        *args, **kwargs
    )


def test_fixed_vector_reference_zero_write_slots() -> None:
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

    seed_scores = [7373] * (q_rows * seq_len)
    out_a = seed_scores.copy()
    out_b = seed_scores.copy()
    req_q_a = [111]
    req_k_a = [222]
    req_out_a = [333]
    req_q_b = [111]
    req_k_b = [222]
    req_out_b = [333]

    err_a = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
        req_q_a,
        req_k_a,
        req_out_a,
    )
    err_b = explicit_preflight_only_composition(
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
        req_q_b,
        req_k_b,
        req_out_b,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b == seed_scores
    assert req_q_a == req_q_b == [111]
    assert req_k_a == req_k_b == [222]
    assert req_out_a == req_out_b == [333]


def test_null_alias_capacity_overflow_contract() -> None:
    out = [5150] * 6
    req_q = [10]
    req_k = [20]
    req_out = [30]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == [5150] * 6
    assert req_q == [10]
    assert req_k == [20]
    assert req_out == [30]

    q = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    k = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    out_alias = [3131] * 4
    req_alias = [99]
    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
        q,
        len(q),
        1,
        k,
        len(k),
        1,
        1,
        1,
        4,
        out_alias,
        len(out_alias),
        req_alias,
        req_alias,
        [77],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_alias == [3131] * 4
    assert req_alias == [99]

    out_cap = [900] * 8
    req_q_cap = [1]
    req_k_cap = [2]
    req_out_cap = [3]
    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
        req_q_cap,
        req_k_cap,
        req_out_cap,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_cap == [900] * 8
    assert req_q_cap == [1]
    assert req_k_cap == [2]
    assert req_out_cap == [3]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
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
        [7],
        [8],
        [9],
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_no_write_and_slot_preservation() -> None:
    rng = random.Random(1409)

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

        out_seed = [rng.randint(-4000, 4000) for _ in range(required_out)]
        out = out_seed.copy()
        req_q = [rng.randint(1, 1000)]
        req_k = [rng.randint(1, 1000)]
        req_out = [rng.randint(1, 1000)]
        req_q_seed = req_q[0]
        req_k_seed = req_k[0]
        req_out_seed = req_out[0]

        err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only(
            q,
            len(q),
            q_rows,
            k,
            len(k),
            k_rows,
            group_count,
            seq_len,
            head_dim,
            out,
            len(out),
            req_q,
            req_k,
            req_out,
        )

        assert err == ATTN_Q16_OK
        assert out == out_seed
        assert req_q == [req_q_seed]
        assert req_k == [req_k_seed]
        assert req_out == [req_out_seed]
