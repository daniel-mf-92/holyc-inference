#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
)
from test_attention_q16_compute_scaled_qk_rows_checked import q16_from_text
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    return attention_q16_compute_scaled_qk_rows_checked_nopartial(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
    )


def explicit_default_stride_nopartial_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        head_dim,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        token_count,
    )


def test_source_contains_default_stride_nopartial_rows_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStride(" in source
    body = source.split(
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStride(", 1
    )[1]
    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartial(" in body


def run_case(
    q_rows,
    query_row_count: int,
    k_rows,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
) -> None:
    out_capacity = 0
    if query_row_count > 0 and token_count > 0:
        out_capacity = (query_row_count - 1) * token_count + token_count

    out_a = [123456] * max(out_capacity, 1)
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride(
        q_rows,
        len(q_rows),
        query_row_count,
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
    )
    err_b = explicit_default_stride_nopartial_composition(
        q_rows,
        len(q_rows),
        query_row_count,
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
    )

    assert err_a == err_b
    assert out_a == out_b


def test_multilingual_batch_parity() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 14
    score_scale_q16 = 15811

    query_texts = ["γειά🙂世界", "مرحبا-abc", "שלום-123"]
    token_texts = ["こんにちは", "Привет", "bonjour", "नमस्ते", "hola"]

    q_rows = [0] * (query_row_count * head_dim)
    for row_index, text in enumerate(query_texts):
        row = q16_from_text(text, head_dim)
        base = row_index * head_dim
        for lane in range(head_dim):
            q_rows[base + lane] = row[lane]

    k_rows = [0] * (token_count * head_dim)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * head_dim
        for lane in range(head_dim):
            k_rows[base + lane] = row[lane]

    run_case(q_rows, query_row_count, k_rows, token_count, head_dim, score_scale_q16)


def test_error_parity_and_no_partial() -> None:
    q_rows = [1 << 16, 2 << 16, 3 << 16, 4 << 16, 5 << 16, 6 << 16]
    k_rows = [
        7 << 16,
        8 << 16,
        9 << 16,
        10 << 16,
        11 << 16,
        12 << 16,
        13 << 16,
        14 << 16,
        15 << 16,
    ]

    out_seed = [777] * 6
    out_a = out_seed.copy()
    out_b = out_seed.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride(
        q_rows,
        len(q_rows),
        2,
        k_rows,
        len(k_rows),
        3,
        3,
        1 << 16,
        out_a,
        5,
    )
    err_b = explicit_default_stride_nopartial_composition(
        q_rows,
        len(q_rows),
        2,
        k_rows,
        len(k_rows),
        3,
        3,
        1 << 16,
        out_b,
        5,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == out_seed

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride(
        None,
        0,
        1,
        k_rows,
        len(k_rows),
        1,
        3,
        1 << 16,
        out_a,
        len(out_a),
    )
    err_b = explicit_default_stride_nopartial_composition(
        None,
        0,
        1,
        k_rows,
        len(k_rows),
        1,
        3,
        1 << 16,
        out_b,
        len(out_b),
    )
    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR


def test_randomized_parity() -> None:
    random.seed(561)

    for _ in range(320):
        query_row_count = random.randint(0, 7)
        token_count = random.randint(0, 8)
        head_dim = random.randint(0, 12)

        q_rows_capacity = query_row_count * head_dim
        k_rows_capacity = token_count * head_dim

        q_rows = [random.randint(-(1 << 17), (1 << 17)) for _ in range(q_rows_capacity)]
        k_rows = [random.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows_capacity)]

        out_capacity = 0
        if query_row_count > 0 and token_count > 0:
            out_capacity = (query_row_count - 1) * token_count + token_count

        out_a = [random.randint(-9999, 9999) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()

        if query_row_count > 0 and token_count > 0 and random.random() < 0.24:
            out_capacity = max(0, out_capacity - 1)

        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride(
            q_rows,
            len(q_rows),
            query_row_count,
            k_rows,
            len(k_rows),
            token_count,
            head_dim,
            score_scale_q16,
            out_a,
            out_capacity,
        )
        err_b = explicit_default_stride_nopartial_composition(
            q_rows,
            len(q_rows),
            query_row_count,
            k_rows,
            len(k_rows),
            token_count,
            head_dim,
            score_scale_q16,
            out_b,
            out_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_rows_wrapper()
    test_multilingual_batch_parity()
    test_error_parity_and_no_partial()
    test_randomized_parity()
    print("ok")
