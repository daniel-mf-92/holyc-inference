#!/usr/bin/env python3
"""Reference checks for ...PreflightOnlyParityCommitOnlyPreflightOnly (IQ-1377)."""

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
from test_attention_gqa_score_q16_checked_nopartial_commit_only_preflight_only_parity import (
    gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity,
)
from test_attention_gqa_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only,
)

I64_MAX = (1 << 63) - 1


def gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    if out_required_q_cells is None or out_required_k_cells is None or out_required_out_cells is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_q_cells is out_required_k_cells
        or out_required_q_cells is out_required_out_cells
        or out_required_k_cells is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

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

    snapshot_required_q_slot = out_required_q_cells[0]
    snapshot_required_k_slot = out_required_k_cells[0]
    snapshot_required_out_slot = out_required_out_cells[0]

    staged_commit_q = [required_q_cells]
    staged_commit_k = [required_k_cells]
    staged_commit_out = [required_out_cells]

    staged_parity_q = [required_q_cells]
    staged_parity_k = [required_k_cells]
    staged_parity_out = [required_out_cells]

    out_snapshot = out_scores_q32[:required_out_cells]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
        staged_commit_q,
        staged_commit_k,
        staged_commit_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity(
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
        staged_parity_q,
        staged_parity_k,
        staged_parity_out,
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
        staged_parity_q[0] != required_q_cells
        or staged_parity_k[0] != required_k_cells
        or staged_parity_out[0] != required_out_cells
        or staged_commit_q[0] != required_q_cells
        or staged_commit_k[0] != required_k_cells
        or staged_commit_out[0] != required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        staged_parity_q[0] != staged_commit_q[0]
        or staged_parity_k[0] != staged_commit_k[0]
        or staged_parity_out[0] != staged_commit_out[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_q_cells[0] != snapshot_required_q_slot
        or out_required_k_cells[0] != snapshot_required_k_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if out_scores_q32[:required_out_cells] != out_snapshot:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = staged_parity_q[0]
    out_required_k_cells[0] = staged_parity_k[0]
    out_required_out_cells[0] = staged_parity_out[0]
    return ATTN_Q16_OK


def test_fixed_vector_reference_publishes_required_tuple_without_touching_scores() -> None:
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

    out_scores = [7331] * (q_rows * seq_len)
    out_snapshot = out_scores.copy()
    req_q = [111]
    req_k = [222]
    req_out = [333]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    assert out_scores == out_snapshot
    assert req_q == [q_rows * head_dim]
    assert req_k == [k_rows * seq_len * head_dim]
    assert req_out == [q_rows * seq_len]


def test_error_contract_preserves_outputs() -> None:
    out = [5150] * 6
    req_q = [10]
    req_k = [20]
    req_out = [30]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        [1 << 16, 2 << 16],
        2,
        1,
        [3 << 16, 4 << 16],
        2,
        1,
        2,
        1,
        2,
        out,
        len(out),
        req_q,
        req_q,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == [5150] * 6
    assert req_q == [10]
    assert req_out == [30]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
        [1 << 16],
        1,
        I64_MAX,
        [1 << 16],
        1,
        1,
        1,
        1,
        2,
        out,
        len(out),
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == [5150] * 6
    assert req_q == [10]
    assert req_k == [20]
    assert req_out == [30]


def test_alias_output_slot_rejected() -> None:
    q = [1 << 16, 2 << 16]
    k = [1 << 16, 2 << 16]
    out = [7]
    req_k = [2]
    req_out = [1]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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


def test_randomized_parity_success() -> None:
    rng = random.Random(20260425_1377)

    for _ in range(220):
        k_rows = rng.randint(1, 4)
        group_count = rng.randint(1, 3)
        q_rows = k_rows * group_count
        seq_len = rng.randint(0, 6)
        head_dim = rng.randint(0, 10)

        required_q = q_rows * head_dim
        required_k = k_rows * seq_len * head_dim
        required_out = q_rows * seq_len

        q = [rng.randint(-(1 << 17), (1 << 17)) for _ in range(required_q)]
        k = [rng.randint(-(1 << 17), (1 << 17)) for _ in range(required_k)]
        out = [rng.randint(-5000, 5000) for _ in range(max(1, required_out))]
        out_snapshot = out.copy()

        req_q = [rng.randint(-999, 999)]
        req_k = [rng.randint(-999, 999)]
        req_out = [rng.randint(-999, 999)]

        err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
        assert out == out_snapshot
        assert req_q == [required_q]
        assert req_k == [required_k]
        assert req_out == [required_out]


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 GQAAttentionScoreQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = GQAAttentionScoreQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "status = GQAAttentionScoreQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "if (out_required_q_cells == q_rows_q16 ||" in body
    assert "if (*out_required_q_cells != snapshot_out_required_q_cells_value ||" in body
    assert "*out_required_q_cells = staged_parity_required_q_cells;" in body


if __name__ == "__main__":
    test_fixed_vector_reference_publishes_required_tuple_without_touching_scores()
    test_error_contract_preserves_outputs()
    test_alias_output_slot_rejected()
    test_randomized_parity_success()
    test_source_contract_markers()
    print(
        "gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_reference_checks=ok"
    )
