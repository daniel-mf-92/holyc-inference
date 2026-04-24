#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16CheckedNoPartialCommitOnlyPreflightOnly (IQ-1367)."""

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
from test_attention_gqa_score_q16_checked_nopartial import (
    gqa_attention_score_q16_checked_nopartial,
)
from test_attention_gqa_score_q16_checked_nopartial_commit_only import (
    gqa_attention_score_q16_checked_nopartial_commit_only,
)

I64_MAX = (1 << 63) - 1


def gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only(
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

    if required_out_cells == 0:
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
            required_q_cells != recomputed_q_cells
            or required_k_cells != recomputed_k_cells
            or required_out_cells != recomputed_out_cells
        ):
            return ATTN_Q16_ERR_BAD_PARAM

        if (
            out_required_q_cells[0] != snapshot_required_q_slot
            or out_required_k_cells[0] != snapshot_required_k_slot
            or out_required_out_cells[0] != snapshot_required_out_slot
        ):
            return ATTN_Q16_ERR_BAD_PARAM
        return ATTN_Q16_OK

    staged_commit = [0] * required_out_cells
    staged_canonical = [0] * required_out_cells
    snapshot_out_scores = out_scores_q32[:required_out_cells]

    err = gqa_attention_score_q16_checked_nopartial_commit_only(
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

    err = gqa_attention_score_q16_checked_nopartial(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        staged_canonical,
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
        required_q_cells > q_rows_capacity
        or required_k_cells > k_rows_capacity
        or required_out_cells > snapshot_out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_commit != staged_canonical:
        return ATTN_Q16_ERR_BAD_PARAM

    if out_scores_q32[:required_out_cells] != snapshot_out_scores:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_q_cells[0] != snapshot_required_q_slot
        or out_required_k_cells[0] != snapshot_required_k_slot
        or out_required_out_cells[0] != snapshot_required_out_slot
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    return ATTN_Q16_OK


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

    out_scores = [7331] * (q_rows * seq_len)
    req_q = [111]
    req_k = [222]
    req_out = [333]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only(
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
    assert out_scores == [7331] * (q_rows * seq_len)
    assert req_q == [111]
    assert req_k == [222]
    assert req_out == [333]


def test_error_contract_preserves_outputs() -> None:
    out = [5150] * 6
    req_q = [10]
    req_k = [20]
    req_out = [30]

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only(
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

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only(
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

    err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only(
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


def test_randomized_zero_write_success() -> None:
    rng = random.Random(20260424_1367)

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
        req_snapshot = (req_q[0], req_k[0], req_out[0])

        err = gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only(
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
        assert (req_q[0], req_k[0], req_out[0]) == req_snapshot


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 GQAAttentionScoreQ16CheckedNoPartialCommitOnlyPreflightOnly(" in source
    body = source.split("I32 GQAAttentionScoreQ16CheckedNoPartialCommitOnlyPreflightOnly(", 1)[1]
    assert "status = GQAAttentionScoreQ16CheckedNoPartialCommitOnly(" in body
    assert "status = GQAAttentionScoreQ16CheckedNoPartial(" in body
    assert "snapshot_out_required_q_cells_value = *out_required_q_cells;" in body
    assert "if (out_scores_q32[copy_index] != snapshot_out_scores_q32[copy_index])" in body


if __name__ == "__main__":
    test_fixed_vector_reference_zero_write_slots()
    test_error_contract_preserves_outputs()
    test_randomized_zero_write_success()
    test_source_contract_markers()
    print("gqa_attention_score_q16_checked_nopartial_commit_only_preflight_only_reference_checks=ok")
