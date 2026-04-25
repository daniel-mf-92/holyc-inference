#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity (IQ-1439)."""

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
from test_attention_gqa_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only,
)
from test_attention_gqa_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only,
)

I64_MAX = (1 << 63) - 1


def gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    if q_rows_q16 is k_rows_q16 or q_rows_q16 is out_scores_q32 or k_rows_q16 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if q_rows < 0 or k_rows < 0 or group_count <= 0 or seq_len < 0 or head_dim < 0:
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

    snapshot_q_rows = q_rows
    snapshot_k_rows = k_rows
    snapshot_group_count = group_count
    snapshot_seq_len = seq_len
    snapshot_head_dim = head_dim
    snapshot_q_rows_capacity = q_rows_capacity
    snapshot_k_rows_capacity = k_rows_capacity
    snapshot_out_capacity = out_capacity
    snapshot_out_scores = out_scores_q32[:]
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

    staged_parity_req_q = [required_q_cells]
    staged_parity_req_k = [required_k_cells]
    staged_parity_req_out = [required_out_cells]
    staged_commit_req_q = [required_q_cells]
    staged_commit_req_k = [required_k_cells]
    staged_commit_req_out = [required_out_cells]

    if required_out_cells > 0:
        staged_parity_scores = [0] * required_out_cells
        staged_commit_scores = [0] * required_out_cells
        parity_out = staged_parity_scores
        parity_out_capacity = required_out_cells
        commit_out = staged_commit_scores
        commit_out_capacity = required_out_cells
    else:
        parity_out = out_scores_q32
        parity_out_capacity = out_capacity
        commit_out = out_scores_q32
        commit_out_capacity = out_capacity

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        parity_out,
        parity_out_capacity,
        staged_parity_req_q,
        staged_parity_req_k,
        staged_parity_req_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        commit_out,
        commit_out_capacity,
        staged_commit_req_q,
        staged_commit_req_k,
        staged_commit_req_out,
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
        or snapshot_q_rows_capacity != q_rows_capacity
        or snapshot_k_rows_capacity != k_rows_capacity
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
        staged_parity_req_q[0] != required_q_cells
        or staged_parity_req_k[0] != required_k_cells
        or staged_parity_req_out[0] != required_out_cells
        or staged_commit_req_q[0] != required_q_cells
        or staged_commit_req_k[0] != required_k_cells
        or staged_commit_req_out[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_req_q[0] != staged_commit_req_q[0]
        or staged_parity_req_k[0] != staged_commit_req_k[0]
        or staged_parity_req_out[0] != staged_commit_req_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_q_cells[0] != snapshot_required_q_slot
        or out_required_k_cells[0] != snapshot_required_k_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_scores_q32 != snapshot_out_scores:
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


def test_fixed_vector_zero_write_required_slots_preserved() -> None:
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

    out_scores = [5001] * (q_rows * seq_len)
    req_q = [111]
    req_k = [222]
    req_out = [333]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        q,
        len(q),
        q_rows,
        k,
        len(k),
        k_rows,
        group_count,
        seq_len,
        head_dim,
        out_scores,
        len(out_scores),
        req_q,
        req_k,
        req_out,
    )

    assert err == ATTN_Q16_OK
    assert out_scores == [5001] * (q_rows * seq_len)
    assert req_q == [111]
    assert req_k == [222]
    assert req_out == [333]


def test_bad_param_on_required_slot_alias() -> None:
    q = [1 << 16, 2 << 16]
    k = [3 << 16, 4 << 16]
    out = [0]
    req = [0]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        q, len(q), 1, k, len(k), 1, 1, 1, 2, out, len(out), req, req, [0]
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_bad_param_on_required_slot_input_alias() -> None:
    q = [1 << 16, 2 << 16]
    k = [3 << 16, 4 << 16]
    out = [0]
    req_k = [0]
    req_out = [0]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        q,
        len(q),
        1,
        k,
        len(k),
        1,
        1,
        1,
        2,
        out,
        len(out),
        q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_bad_param_on_q_k_out_alias() -> None:
    qk = [1 << 16, 2 << 16]
    req_q = [0]
    req_k = [1]
    req_out = [2]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        qk,
        len(qk),
        1,
        qk,
        len(qk),
        1,
        1,
        1,
        2,
        [0],
        1,
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_null_ptr_rejected() -> None:
    req_q = [0]
    req_k = [0]
    req_out = [0]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        None, 0, 0, [], 0, 0, 1, 0, 0, [], 0, req_q, req_k, req_out
    )
    assert err == ATTN_Q16_ERR_NULL_PTR


def test_capacity_and_overflow_vectors() -> None:
    q = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    k = [1 << 16, 0, 2 << 16, 1 << 16]
    out = [0, 0]
    req_q = [7]
    req_k = [8]
    req_out = [9]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        q, -1, 1, k, len(k), 1, 1, 2, 2, out, len(out), req_q, req_k, req_out
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert req_q == [7]
    assert req_k == [8]
    assert req_out == [9]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        q, len(q), I64_MAX, k, len(k), 1, 1, 1, 2, out, len(out), req_q, req_k, req_out
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert req_q == [7]
    assert req_k == [8]
    assert req_out == [9]


def test_random_vectors_keep_output_and_required_slots_immutable() -> None:
    rng = random.Random(1439)

    for _ in range(60):
        k_rows = rng.randint(1, 4)
        group_count = rng.randint(1, 4)
        q_rows = k_rows * group_count
        seq_len = rng.randint(1, 5)
        head_dim = rng.randint(1, 5)

        required_q = q_rows * head_dim
        required_k = k_rows * seq_len * head_dim
        required_out = q_rows * seq_len

        q = [rng.randint(-(2 << 16), 2 << 16) for _ in range(required_q)]
        k = [rng.randint(-(2 << 16), 2 << 16) for _ in range(required_k)]
        out = [rng.randint(-2000, 2000) for _ in range(required_out)]
        out_before = out.copy()

        req_q = [rng.randint(-500, 500)]
        req_k = [rng.randint(-500, 500)]
        req_out = [rng.randint(-500, 500)]
        req_snapshot = (req_q[0], req_k[0], req_out[0])

        err = gqa_attention_score_q16_checked_nopartial_commit_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
        assert out == out_before
        assert (req_q[0], req_k[0], req_out[0]) == req_snapshot


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 GQAAttentionScoreQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status = GQAAttentionScoreQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = GQAAttentionScoreQ16CheckedNoPartialCommitOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (*out_required_q_cells != snapshot_out_required_q_cells_value ||" in body
    assert "status = ATTN_Q16_OK;" in body


if __name__ == "__main__":
    test_fixed_vector_zero_write_required_slots_preserved()
    test_bad_param_on_required_slot_alias()
    test_bad_param_on_required_slot_input_alias()
    test_bad_param_on_q_k_out_alias()
    test_null_ptr_rejected()
    test_capacity_and_overflow_vectors()
    test_random_vectors_keep_output_and_required_slots_immutable()
    test_source_contract_markers()
    print("ok")
