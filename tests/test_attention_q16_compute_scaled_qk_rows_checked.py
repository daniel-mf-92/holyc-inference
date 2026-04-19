#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsChecked."""

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
from test_attention_q16_compute_scaled_qk_row_checked import (
    attention_q16_compute_scaled_qk_row_checked,
    q16_from_text,
)

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


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


def attention_q16_compute_scaled_qk_rows_checked(
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

    if query_row_count > 0 and query_row_stride_q16 < head_dim:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0:
        return ATTN_Q16_OK

    err, required_q_cells = try_mul_i64_checked(query_row_count - 1, query_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err

    err, required_q_cells = try_add_i64_checked(required_q_cells, head_dim)
    if err != ATTN_Q16_OK:
        return err

    if required_q_cells > q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err

    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count > 0:
        err, required_out_cells = try_mul_i64_checked(query_row_count - 1, out_row_stride)
        if err != ATTN_Q16_OK:
            return err

        err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
        if err != ATTN_Q16_OK:
            return err

        if required_out_cells > out_scores_capacity:
            return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        err, q_row_base = try_mul_i64_checked(row_index, query_row_stride_q16)
        if err != ATTN_Q16_OK:
            return err

        err, out_row_base = try_mul_i64_checked(row_index, out_row_stride)
        if err != ATTN_Q16_OK:
            return err

        row_out = [0] * max(token_count, 1)
        status = attention_q16_compute_scaled_qk_row_checked(
            q_rows_q16[q_row_base : q_row_base + head_dim],
            head_dim,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            row_out,
            token_count,
            1,
        )
        if status != ATTN_Q16_OK:
            return status

        for token_index in range(token_count):
            out_scores_q32[out_row_base + token_index] = row_out[token_index]

    return ATTN_Q16_OK


def explicit_rows_composition(
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

    if query_row_count > 0 and query_row_stride_q16 < head_dim:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0:
        return ATTN_Q16_OK

    err, required_q_cells = try_mul_i64_checked(query_row_count - 1, query_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    err, required_q_cells = try_add_i64_checked(required_q_cells, head_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_q_cells > q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count > 0:
        err, required_out_cells = try_mul_i64_checked(query_row_count - 1, out_row_stride)
        if err != ATTN_Q16_OK:
            return err
        err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
        if err != ATTN_Q16_OK:
            return err
        if required_out_cells > out_scores_capacity:
            return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        q_row_base = row_index * query_row_stride_q16
        out_row_base = row_index * out_row_stride

        row_out = [0] * max(token_count, 1)
        status = attention_q16_compute_scaled_qk_row_checked(
            q_rows_q16[q_row_base : q_row_base + head_dim],
            head_dim,
            k_rows_q16,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            row_out,
            token_count,
            1,
        )
        if status != ATTN_Q16_OK:
            return status

        for token_index in range(token_count):
            out_scores_q32[out_row_base + token_index] = row_out[token_index]

    return ATTN_Q16_OK


def test_source_contains_rows_helper_and_row_dispatch() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowsChecked(" in source
    body = source.split("I32 AttentionQ16ComputeScaledQKRowsChecked(", 1)[1]
    assert "AttentionQ16ComputeScaledQKRowChecked(" in body


def test_multilingual_row_major_parity() -> None:
    query_row_count = 3
    token_count = 4
    head_dim = 16
    query_row_stride_q16 = 19
    k_row_stride_q16 = 21
    out_row_stride = 6
    score_scale_q16 = 18204

    query_texts = ["γειά🙂世界", "مرحبا-Δabc", "hola-שלום"]
    token_texts = ["こんにちは", "Привет", "bonjour", "नमस्ते"]

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

    err_a = attention_q16_compute_scaled_qk_rows_checked(
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
    err_b = explicit_rows_composition(
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


def test_error_surface_and_no_spurious_writes_on_guard_failures() -> None:
    q = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    k = [5 << 16, 6 << 16, 7 << 16, 8 << 16]

    out_a = [777] * 8
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked(
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
        1,
    )
    err_b = explicit_rows_composition(
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
        1,
    )
    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR

    out_a = [123] * 6
    out_b = out_a.copy()
    err_a = attention_q16_compute_scaled_qk_rows_checked(
        q,
        len(q),
        2,
        1,
        k,
        len(k),
        1,
        4,
        4,
        1 << 16,
        out_a,
        len(out_a),
        3,
    )
    err_b = explicit_rows_composition(
        q,
        len(q),
        2,
        1,
        k,
        len(k),
        1,
        4,
        4,
        1 << 16,
        out_b,
        len(out_b),
        3,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == [123] * 6


def test_randomized_parity_success_and_errors() -> None:
    random.seed(555)

    for _ in range(260):
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

        if query_row_count > 0 and token_count > 0 and random.random() < 0.22:
            out_scores_capacity = max(0, out_scores_capacity - 1)

        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        err_a = attention_q16_compute_scaled_qk_rows_checked(
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
        err_b = explicit_rows_composition(
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
    test_source_contains_rows_helper_and_row_dispatch()
    test_multilingual_row_major_parity()
    test_error_surface_and_no_spurious_writes_on_guard_failures()
    test_randomized_parity_success_and_errors()
    print("ok")
