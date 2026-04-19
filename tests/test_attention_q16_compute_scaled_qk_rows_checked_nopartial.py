#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    try_add_i64_checked,
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked import (
    attention_q16_compute_scaled_qk_rows_checked,
    q16_from_text,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0:
        return ATTN_Q16_OK

    if token_count == 0:
        return attention_q16_compute_scaled_qk_rows_checked(
            q_rows_q16,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_scores_q32,
            out_scores_capacity,
            out_row_stride,
        )

    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, out_row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_scores = [0] * required_out_cells
    err = attention_q16_compute_scaled_qk_rows_checked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores,
        required_out_cells,
        out_row_stride,
    )
    if err != ATTN_Q16_OK:
        return err

    for row_index in range(query_row_count):
        row_base = row_index * out_row_stride
        for token_index in range(token_count):
            out_index = row_base + token_index
            out_scores_q32[out_index] = staged_scores[out_index]

    return ATTN_Q16_OK


def explicit_staged_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0:
        return ATTN_Q16_OK

    if token_count == 0:
        return attention_q16_compute_scaled_qk_rows_checked(
            q_rows_q16,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_scores_q32,
            out_scores_capacity,
            out_row_stride,
        )

    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, out_row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_scores = [0] * required_out_cells
    err = attention_q16_compute_scaled_qk_rows_checked(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores,
        required_out_cells,
        out_row_stride,
    )
    if err != ATTN_Q16_OK:
        return err

    for row_index in range(query_row_count):
        row_base = row_index * out_row_stride
        for token_index in range(token_count):
            out_index = row_base + token_index
            out_scores_q32[out_index] = staged_scores[out_index]

    return ATTN_Q16_OK


def test_source_contains_rows_nopartial_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartial(" in source
    body = source.split("I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartial(", 1)[1]
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body
    assert "status = AttentionQ16ComputeScaledQKRowsChecked(" in body


def test_multilingual_row_major_no_partial_parity() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 14
    query_row_stride_q16 = 16
    k_row_stride_q16 = 17
    out_row_stride = 7
    score_scale_q16 = 17312

    query_texts = ["γειά🙂世界", "مرحبا-Δabc", "hola-שלום"]
    token_texts = ["こんにちは", "Привет", "bonjour", "नमस्ते", "¡hola!"]

    q_rows_q16 = [0] * (query_row_count * query_row_stride_q16)
    for row_index, text in enumerate(query_texts):
        row = q16_from_text(text, head_dim)
        base = row_index * query_row_stride_q16
        for lane in range(head_dim):
            q_rows_q16[base + lane] = row[lane]

    k_rows_q16 = [0] * (token_count * k_row_stride_q16)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * k_row_stride_q16
        for lane in range(head_dim):
            k_rows_q16[base + lane] = row[lane]

    out_capacity = (query_row_count - 1) * out_row_stride + token_count
    out_a = [31337] * out_capacity
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial(
        q_rows_q16,
        len(q_rows_q16),
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        len(k_rows_q16),
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
        out_row_stride,
    )
    err_b = explicit_staged_composition(
        q_rows_q16,
        len(q_rows_q16),
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        len(k_rows_q16),
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
        out_row_stride,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_guard_failures_are_no_partial() -> None:
    q = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    k = [5 << 16, 6 << 16, 7 << 16, 8 << 16]

    out_a = [777] * 8
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial(
        None,
        len(q),
        1,
        4,
        k,
        len(k),
        1,
        4,
        4,
        1 << 16,
        out_a,
        len(out_a),
        2,
    )
    err_b = explicit_staged_composition(
        None,
        len(q),
        1,
        4,
        k,
        len(k),
        1,
        4,
        4,
        1 << 16,
        out_b,
        len(out_b),
        2,
    )
    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR
    assert out_a == out_b == [777] * 8

    out_a = [123] * 6
    out_b = out_a.copy()
    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial(
        q,
        len(q),
        2,
        4,
        k,
        len(k),
        2,
        4,
        4,
        1 << 16,
        out_a,
        5,
        2,
    )
    err_b = explicit_staged_composition(
        q,
        len(q),
        2,
        4,
        k,
        len(k),
        2,
        4,
        4,
        1 << 16,
        out_b,
        5,
        2,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == [123] * 6


def test_randomized_parity_success_and_errors() -> None:
    random.seed(556)

    for _ in range(280):
        query_row_count = random.randint(0, 6)
        token_count = random.randint(0, 7)
        head_dim = random.randint(0, 11)

        query_row_stride_q16 = head_dim + random.randint(0, 3)
        k_row_stride_q16 = head_dim + random.randint(0, 3)
        out_row_stride = token_count + random.randint(0, 3)

        q_rows_capacity = query_row_count * query_row_stride_q16
        q_rows_q16 = [random.randint(-(1 << 17), (1 << 17)) for _ in range(q_rows_capacity)]

        k_rows_capacity = token_count * k_row_stride_q16
        k_rows_q16 = [random.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows_capacity)]

        out_scores_capacity = (
            0
            if query_row_count == 0 or token_count == 0
            else (query_row_count - 1) * out_row_stride + token_count
        )
        out_a = [random.randint(-9999, 9999) for _ in range(max(out_scores_capacity, 1))]
        out_b = out_a.copy()

        if query_row_count > 0 and token_count > 0 and random.random() < 0.24:
            out_scores_capacity = max(0, out_scores_capacity - 1)

        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial(
            q_rows_q16,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_a,
            out_scores_capacity,
            out_row_stride,
        )
        err_b = explicit_staged_composition(
            q_rows_q16,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_b,
            out_scores_capacity,
            out_row_stride,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_rows_nopartial_helper()
    test_multilingual_row_major_no_partial_parity()
    test_guard_failures_are_no_partial()
    test_randomized_parity_success_and_errors()
    print("ok")
