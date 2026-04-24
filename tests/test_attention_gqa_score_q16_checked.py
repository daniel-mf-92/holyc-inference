#!/usr/bin/env python3
"""Reference checks for GQAAttentionScoreQ16Checked semantics (IQ-1364)."""

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

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
ATTN_Q16_SHIFT = 16
GQA_ATTN_SCORE_CLAMP_Q32 = 32 << ATTN_Q16_SHIFT


def try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    value = lhs + rhs
    if value < I64_MIN or value > I64_MAX:
        return (ATTN_Q16_ERR_OVERFLOW, 0)
    return (ATTN_Q16_OK, value)


def try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    value = lhs * rhs
    if value < I64_MIN or value > I64_MAX:
        return (ATTN_Q16_ERR_OVERFLOW, 0)
    return (ATTN_Q16_OK, value)


def gqa_attention_score_q16_checked(
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

    if q_rows > 0 and k_rows <= 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if q_rows > 0 and (q_rows % group_count) != 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, qk_groups = try_mul_i64_checked(k_rows, group_count)
    if err != ATTN_Q16_OK:
        return err
    if qk_groups != q_rows:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_q_cells = try_mul_i64_checked(q_rows, head_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_q_cells > q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(k_rows, seq_len)
    if err != ATTN_Q16_OK:
        return err
    err, required_k_cells = try_mul_i64_checked(required_k_cells, head_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(q_rows, seq_len)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if required_out_cells == 0:
        return ATTN_Q16_OK

    scale_divisor = head_dim if head_dim > 0 else 1
    staged = [0] * required_out_cells

    for query_index in range(q_rows):
        err, q_base = try_mul_i64_checked(query_index, head_dim)
        if err != ATTN_Q16_OK:
            return err

        err, out_row_base = try_mul_i64_checked(query_index, seq_len)
        if err != ATTN_Q16_OK:
            return err

        kv_head_index = query_index // group_count
        err, kv_seq_base = try_mul_i64_checked(kv_head_index, seq_len)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(seq_len):
            err, k_base = try_add_i64_checked(kv_seq_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            err, k_base = try_mul_i64_checked(k_base, head_dim)
            if err != ATTN_Q16_OK:
                return err

            score_q32 = 0
            for lane_index in range(head_dim):
                err, product_q32 = try_mul_i64_checked(
                    q_rows_q16[q_base + lane_index],
                    k_rows_q16[k_base + lane_index],
                )
                if err != ATTN_Q16_OK:
                    return err

                err, score_q32 = try_add_i64_checked(score_q32, product_q32)
                if err != ATTN_Q16_OK:
                    return err

            scaled_score_q32 = score_q32 // scale_divisor
            if scaled_score_q32 > GQA_ATTN_SCORE_CLAMP_Q32:
                scaled_score_q32 = GQA_ATTN_SCORE_CLAMP_Q32
            if scaled_score_q32 < -GQA_ATTN_SCORE_CLAMP_Q32:
                scaled_score_q32 = -GQA_ATTN_SCORE_CLAMP_Q32

            staged[out_row_base + token_index] = scaled_score_q32

    for idx in range(required_out_cells):
        out_scores_q32[idx] = staged[idx]

    return ATTN_Q16_OK


def explicit_gqa_score_composition(*args, **kwargs) -> int:
    return gqa_attention_score_q16_checked(*args, **kwargs)


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

    out_a = [0] * (q_rows * seq_len)
    out_b = [0] * (q_rows * seq_len)

    err_a = gqa_attention_score_q16_checked(
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
    err_b = explicit_gqa_score_composition(
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


def test_error_contract_and_no_partial() -> None:
    seed = [777] * 8
    out = seed.copy()

    err = gqa_attention_score_q16_checked(
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

    err = gqa_attention_score_q16_checked(
        [1 << 16, 2 << 16, 3 << 16, 4 << 16],
        4,
        2,
        [5 << 16, 6 << 16],
        2,
        1,
        2,
        2,
        2,
        out,
        3,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == seed

    err = gqa_attention_score_q16_checked(
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


def test_overflow_path() -> None:
    q = [I64_MAX]
    k = [I64_MAX]
    out = [4242]

    err = gqa_attention_score_q16_checked(
        q,
        1,
        1,
        k,
        1,
        1,
        1,
        1,
        1,
        out,
        1,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert out == [4242]


def test_randomized_parity() -> None:
    rng = random.Random(2026042404)

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

        err_a = gqa_attention_score_q16_checked(
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
        err_b = explicit_gqa_score_composition(
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
    assert "I32 GQAAttentionScoreQ16Checked(" in source
    body = source.split("I32 GQAAttentionScoreQ16Checked(", 1)[1]
    assert "if (q_rows > 0 && (q_rows % group_count) != 0)" in body
    assert "kv_head_index = query_index / group_count;" in body
    assert "scaled_score_q32 = score_q32 / scale_divisor;" in body
    assert "if (scaled_score_q32 > GQA_ATTN_SCORE_CLAMP_Q32)" in body
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body


if __name__ == "__main__":
    test_fixed_vector_reference()
    test_error_contract_and_no_partial()
    test_overflow_path()
    test_randomized_parity()
    test_source_contract_markers()
    print("gqa_attention_score_q16_checked_reference_checks=ok")
