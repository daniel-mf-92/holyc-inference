#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowChecked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    attention_q16_apply_score_scale_checked,
)
from test_attention_q16_compute_qk_dot_row_checked import (
    attention_q16_compute_qk_dot_row_checked,
)


def attention_q16_compute_scaled_qk_row_checked(
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

    status = attention_q16_compute_qk_dot_row_checked(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
    )
    if status != ATTN_Q16_OK:
        return status

    return attention_q16_apply_score_scale_checked(
        out_scores_q32,
        out_scores_capacity,
        token_count,
        out_score_stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
    )


def explicit_scaled_qk_row_composition(
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
    status = attention_q16_compute_qk_dot_row_checked(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
    )
    if status != ATTN_Q16_OK:
        return status

    return attention_q16_apply_score_scale_checked(
        out_scores_q32,
        out_scores_capacity,
        token_count,
        out_score_stride,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        out_score_stride,
    )


def q16_from_text(text: str, width: int) -> list[int]:
    raw = text.encode("utf-8")
    vals: list[int] = []
    for b in raw:
        vals.append((b - 128) << 16)
        if len(vals) == width:
            break
    while len(vals) < width:
        vals.append(0)
    return vals


def test_source_contains_helper_and_two_stage_calls() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowChecked(" in source

    body = source.split("I32 AttentionQ16ComputeScaledQKRowChecked(", 1)[1]
    assert "AttentionQ16ComputeQKDotRowChecked(" in body
    assert "AttentionQ16ApplyScoreScaleChecked(" in body


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
    out_scores_capacity = 1 + ((token_count - 1) * out_score_stride) if token_count > 0 else 0

    out_a = [31337] * max(out_scores_capacity, 1)
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked(
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
    err_b = explicit_scaled_qk_row_composition(
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


def test_multilingual_adversarial_vectors_match_explicit_composition() -> None:
    head_dim = 24
    token_count = 5
    k_row_stride_q16 = 31
    out_score_stride = 3
    score_scale_q16 = 18919  # ~0.2888 in Q16

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


def test_adversarial_error_parity() -> None:
    q = [1 << 16, -(2 << 16), 3 << 16]
    k = [1 << 16, 2 << 16, 3 << 16]
    out = [123]

    err_a = attention_q16_compute_scaled_qk_row_checked(
        None, 3, k, 3, 1, 3, 3, 1 << 16, out, 1, 1
    )
    err_b = explicit_scaled_qk_row_composition(
        None, 3, k, 3, 1, 3, 3, 1 << 16, out, 1, 1
    )
    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR

    err_a = attention_q16_compute_scaled_qk_row_checked(
        q, 3, k, 3, -1, 3, 3, 1 << 16, out, 1, 1
    )
    err_b = explicit_scaled_qk_row_composition(
        q, 3, k, 3, -1, 3, 3, 1 << 16, out, 1, 1
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM

    # token_count=3 with stride 2 requires output cell index 4; capacity 4 is bad.
    out_a = [777] * 4
    out_b = out_a.copy()
    err_a = attention_q16_compute_scaled_qk_row_checked(
        q,
        3,
        [1 << 16, 2 << 16, 3 << 16, 4 << 16, 5 << 16, 6 << 16],
        6,
        3,
        3,
        3,
        1 << 16,
        out_a,
        4,
        2,
    )
    err_b = explicit_scaled_qk_row_composition(
        q,
        3,
        [1 << 16, 2 << 16, 3 << 16, 4 << 16, 5 << 16, 6 << 16],
        6,
        3,
        3,
        3,
        1 << 16,
        out_b,
        4,
        2,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == [777] * 4


def test_randomized_success_and_error_parity() -> None:
    random.seed(550)

    for _ in range(240):
        token_count = random.randint(0, 9)
        head_dim = random.randint(0, 13)
        k_row_stride_q16 = head_dim + random.randint(0, 4)
        out_score_stride = random.randint(1, 4)

        q_row = [random.randint(-(1 << 17), (1 << 17)) for _ in range(head_dim)]
        q_row_capacity = head_dim

        k_rows_capacity = token_count * k_row_stride_q16
        k_rows = [
            random.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows_capacity)
        ]

        out_scores_capacity = (
            0 if token_count == 0 else (token_count - 1) * out_score_stride + 1
        )
        out_a = [random.randint(-9999, 9999) for _ in range(max(out_scores_capacity, 1))]
        out_b = out_a.copy()

        # Sometimes force a capacity error to exercise error parity.
        if token_count > 0 and random.random() < 0.25:
            out_scores_capacity = max(0, out_scores_capacity - 1)

        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        err_a = attention_q16_compute_scaled_qk_row_checked(
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
        err_b = explicit_scaled_qk_row_composition(
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
    test_source_contains_helper_and_two_stage_calls()
    test_multilingual_adversarial_vectors_match_explicit_composition()
    test_adversarial_error_parity()
    test_randomized_success_and_error_parity()
    print("ok")
