#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16CheckedNoPartial semantics (IQ-1365)."""

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


def gqa_attention_score_q16_checked_nopartial(
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

    err, required_out_cells = try_mul_i64_checked(q_rows, seq_len)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells == 0:
        return ATTN_Q16_OK

    staged_scores = [0] * required_out_cells
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
        staged_scores,
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
        or snapshot_out_capacity != out_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if required_out_cells > snapshot_out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for idx in range(required_out_cells):
        out_scores_q32[idx] = staged_scores[idx]

    return ATTN_Q16_OK


def explicit_staged_composition(*args, **kwargs) -> int:
    return gqa_attention_score_q16_checked_nopartial(*args, **kwargs)


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

    out_a = [111] * (q_rows * seq_len)
    out_b = out_a.copy()

    err_a = gqa_attention_score_q16_checked_nopartial(
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
    err_b = explicit_staged_composition(
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


def test_failure_paths_preserve_output_no_partial() -> None:
    seed = [5151] * 6
    out = seed.copy()

    err = gqa_attention_score_q16_checked_nopartial(
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
    err = gqa_attention_score_q16_checked_nopartial(
        [1 << 16],
        1,
        1,
        [1 << 16],
        1,
        1,
        1,
        1,
        1,
        out,
        0,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    out = seed.copy()
    err = gqa_attention_score_q16_checked_nopartial(
        [((1 << 63) - 1)],
        1,
        1,
        [((1 << 63) - 1)],
        1,
        1,
        1,
        1,
        1,
        out,
        1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == seed

    err = gqa_attention_score_q16_checked_nopartial(
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


def test_randomized_parity() -> None:
    rng = random.Random(20260424_1365)

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

        err_a = gqa_attention_score_q16_checked_nopartial(
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
        err_b = explicit_staged_composition(
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
    assert "I32 GQAAttentionScoreQ16CheckedNoPartial(" in source
    body = source.split("I32 GQAAttentionScoreQ16CheckedNoPartial(", 1)[1]
    assert "snapshot_q_rows = q_rows;" in body
    assert "status = GQAAttentionScoreQ16Checked(" in body
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_failure_paths_preserve_output_no_partial()
    test_randomized_parity()
    test_source_contract_markers()
    print("gqa_attention_score_q16_checked_nopartial_reference_checks=ok")
