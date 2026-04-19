#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_compute_scaled_qk_row_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    attention_q16_compute_scaled_qk_row_checked,
    q16_from_text,
)
from test_attention_q16_apply_score_scale_checked import (
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_compute_scaled_qk_row_checked_nopartial(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
) -> int:
    if q_row_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_row_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0 or k_row_stride_q16 < 0 or head_dim < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        return ATTN_Q16_OK

    if out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_scores = [0] * token_count
    err = attention_q16_compute_scaled_qk_row_checked(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores,
        token_count,
        1,
    )
    if err != ATTN_Q16_OK:
        return err

    for token_index in range(token_count):
        err, out_base = try_mul_i64_checked(token_index, out_score_stride)
        if err != ATTN_Q16_OK:
            return err
        out_scores_q32[out_base] = staged_scores[token_index]

    return ATTN_Q16_OK


def explicit_staged_composition(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
) -> int:
    if q_row_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if q_row_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0 or k_row_stride_q16 < 0 or head_dim < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        return ATTN_Q16_OK

    if out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_scores = [0] * token_count
    err = attention_q16_compute_scaled_qk_row_checked(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        staged_scores,
        token_count,
        1,
    )
    if err != ATTN_Q16_OK:
        return err

    for token_index in range(token_count):
        err, out_base = try_mul_i64_checked(token_index, out_score_stride)
        if err != ATTN_Q16_OK:
            return err
        out_scores_q32[out_base] = staged_scores[token_index]

    return ATTN_Q16_OK


def test_source_contains_nopartial_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowCheckedNoPartial(" in source
    body = source.split("I32 AttentionQ16ComputeScaledQKRowCheckedNoPartial(", 1)[1]
    assert "staged_scores_q32 = MAlloc(stage_bytes);" in body
    assert "status = AttentionQ16ComputeScaledQKRowChecked(" in body


def run_case(
    q_row_q16: list[int],
    k_rows_q16: list[int],
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    score_scale_q16: int,
    out_score_stride: int,
) -> None:
    q_row_capacity = len(q_row_q16)
    k_rows_capacity = len(k_rows_q16)
    out_scores_capacity = 0 if token_count == 0 else (token_count - 1) * out_score_stride + 1

    out_a = [777] * max(out_scores_capacity, 1)
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_nopartial(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_a,
        out_scores_capacity,
        out_score_stride,
    )
    err_b = explicit_staged_composition(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_b,
        out_scores_capacity,
        out_score_stride,
    )

    assert err_a == err_b
    assert out_a == out_b


def test_multilingual_and_adversarial_vectors() -> None:
    head_dim = 18
    token_count = 5
    k_row_stride_q16 = 22
    out_score_stride = 3
    score_scale_q16 = 17476

    q_row = q16_from_text("γειά🙂世界-abc", head_dim)
    token_texts = [
        "مرحبا-Δ",
        "こんにちは🙂",
        "Привет!",
        "hola mundo",
        "שָׁלוֹם",
    ]

    k_rows = [0] * (token_count * k_row_stride_q16)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * k_row_stride_q16
        for lane in range(head_dim):
            k_rows[base + lane] = row[lane]

    run_case(
        q_row,
        k_rows,
        token_count,
        k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_score_stride,
    )


def test_no_partial_failure_paths() -> None:
    q = [1 << 16, 2 << 16, 3 << 16]
    k = [4 << 16, 5 << 16, 6 << 16]

    out_seed = [0x55AA] * 8
    out_a = out_seed.copy()
    out_b = out_seed.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_nopartial(
        q,
        3,
        k,
        3,
        2,
        3,
        3,
        1 << 16,
        out_a,
        1,
        1,
    )
    err_b = explicit_staged_composition(
        q,
        3,
        k,
        3,
        2,
        3,
        3,
        1 << 16,
        out_b,
        1,
        1,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == out_seed

    err_a = attention_q16_compute_scaled_qk_row_checked_nopartial(
        None,
        3,
        k,
        3,
        1,
        3,
        3,
        1 << 16,
        out_a,
        len(out_a),
        1,
    )
    err_b = explicit_staged_composition(
        None,
        3,
        k,
        3,
        1,
        3,
        3,
        1 << 16,
        out_b,
        len(out_b),
        1,
    )
    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR


def test_randomized_parity_and_no_partial() -> None:
    random.seed(553)

    for _ in range(260):
        token_count = random.randint(0, 8)
        head_dim = random.randint(0, 13)
        k_row_stride_q16 = head_dim + random.randint(0, 4)
        out_score_stride = random.randint(1, 4)

        q_row = [random.randint(-(1 << 17), (1 << 17)) for _ in range(head_dim)]
        q_row_capacity = head_dim

        k_rows_capacity = token_count * k_row_stride_q16
        k_rows = [
            random.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows_capacity)
        ]

        out_scores_capacity = 0 if token_count == 0 else (token_count - 1) * out_score_stride + 1
        out_a = [random.randint(-9999, 9999) for _ in range(max(out_scores_capacity, 1))]
        out_b = out_a.copy()

        if token_count > 0 and random.random() < 0.25:
            out_scores_capacity = max(0, out_scores_capacity - 1)

        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        err_a = attention_q16_compute_scaled_qk_row_checked_nopartial(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_a,
            out_scores_capacity,
            out_score_stride,
        )
        err_b = explicit_staged_composition(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            score_scale_q16,
            out_b,
            out_scores_capacity,
            out_score_stride,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_nopartial_helper()
    test_multilingual_and_adversarial_vectors()
    test_no_partial_failure_paths()
    test_randomized_parity_and_no_partial()
    print("ok")
