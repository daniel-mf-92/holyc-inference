#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16CheckedNoPartialCommitOnly (IQ-1366)."""

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
    gqa_attention_score_q16_checked,
    try_mul_i64_checked,
)
from test_attention_gqa_score_q16_checked_nopartial import (
    gqa_attention_score_q16_checked_nopartial,
)


I64_MAX = (1 << 63) - 1


def gqa_attention_score_q16_checked_nopartial_commit_only(
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

    snapshot_q_rows = q_rows
    snapshot_k_rows = k_rows
    snapshot_group_count = group_count
    snapshot_seq_len = seq_len
    snapshot_head_dim = head_dim
    snapshot_out_capacity = out_capacity

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

    if out_scores_q32 is q_rows_q16 or out_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_scores = [0] * required_out_cells
    parity_scores = [0] * required_out_cells

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
        staged_scores,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err = gqa_attention_score_q16_checked(
        q_rows_q16,
        q_rows_capacity,
        q_rows,
        k_rows_q16,
        k_rows_capacity,
        k_rows,
        group_count,
        seq_len,
        head_dim,
        parity_scores,
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

    for idx in range(required_out_cells):
        if staged_scores[idx] != parity_scores[idx]:
            return ATTN_Q16_ERR_BAD_PARAM

    for idx in range(required_out_cells):
        out_scores_q32[idx] = staged_scores[idx]

    return ATTN_Q16_OK


def explicit_commit_only_composition(*args, **kwargs) -> int:
    return gqa_attention_score_q16_checked_nopartial_commit_only(*args, **kwargs)


def test_fixed_vector_reference() -> None:
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

    out_a = [333] * (q_rows * seq_len)
    out_b = out_a.copy()

    err_a = gqa_attention_score_q16_checked_nopartial_commit_only(
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
    err_b = explicit_commit_only_composition(
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


def test_error_contract_null_alias_bounds_overflow() -> None:
    seed = [9191] * 8

    out = seed.copy()
    err = gqa_attention_score_q16_checked_nopartial_commit_only(
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
    assert out == seed

    q_alias = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    k_alias = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    err = gqa_attention_score_q16_checked_nopartial_commit_only(
        q_alias,
        len(q_alias),
        1,
        k_alias,
        len(k_alias),
        1,
        1,
        1,
        4,
        q_alias,
        1,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert q_alias == [1 << 16, 2 << 16, 3 << 16, 4 << 16]

    out = seed.copy()
    err = gqa_attention_score_q16_checked_nopartial_commit_only(
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
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_score_q16_checked_nopartial_commit_only(
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
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == seed


def test_randomized_parity() -> None:
    rng = random.Random(20260424_1366)

    for _ in range(220):
        k_rows = rng.randint(1, 5)
        group_count = rng.randint(1, 4)
        q_rows = k_rows * group_count
        seq_len = rng.randint(0, 6)
        head_dim = rng.randint(0, 10)

        q = [rng.randint(-(1 << 17), (1 << 17)) for _ in range(q_rows * head_dim)]
        k = [rng.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows * seq_len * head_dim)]
        out_capacity = q_rows * seq_len
        out_a = [rng.randint(-999, 999) for _ in range(max(1, out_capacity))]
        out_b = out_a.copy()

        err_a = gqa_attention_score_q16_checked_nopartial_commit_only(
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
            out_capacity,
        )
        err_b = explicit_commit_only_composition(
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
            out_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


def test_source_contract_markers() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 GQAAttentionScoreQ16CheckedNoPartialCommitOnly(" in source
    body = source.split("I32 GQAAttentionScoreQ16CheckedNoPartialCommitOnly(", 1)[1]
    assert "status = GQAAttentionScoreQ16CheckedNoPartial(" in body
    assert "status = GQAAttentionScoreQ16Checked(" in body
    assert "required_q_cells != recomputed_required_q_cells" in body
    assert "staged_scores_q32[copy_index] != parity_scores_q32[copy_index]" in body


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_error_contract_null_alias_bounds_overflow()
    test_randomized_parity()
    test_source_contract_markers()
    print("gqa_attention_score_q16_checked_nopartial_commit_only_reference_checks=ok")
