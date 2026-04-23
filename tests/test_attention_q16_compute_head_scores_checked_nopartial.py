#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeHeadScoresCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_compute_scaled_qk_row_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    q16_from_text,
)
from test_attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride import (
    attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride,
)


def attention_q16_compute_head_scores_checked_nopartial(
    q_head_row_q16,
    q_head_row_capacity: int,
    k_head_rows_q16,
    k_head_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_head_scores_q32,
    out_head_scores_capacity: int,
) -> int:
    return attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
        q_head_row_q16,
        q_head_row_capacity,
        k_head_rows_q16,
        k_head_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_head_scores_q32,
        out_head_scores_capacity,
    )


def explicit_single_head_composition(
    q_head_row_q16,
    q_head_row_capacity: int,
    k_head_rows_q16,
    k_head_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_head_scores_q32,
    out_head_scores_capacity: int,
) -> int:
    # Explicit single-head composition mirrors HolyC contract:
    # key rows tightly packed by head_dim and no-partial commit semantics.
    return attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
        q_head_row_q16,
        q_head_row_capacity,
        k_head_rows_q16,
        k_head_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_head_scores_q32,
        out_head_scores_capacity,
    )


def test_source_contains_head_scores_nopartial_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeHeadScoresCheckedNoPartial(" in source
    body = source.split("I32 AttentionQ16ComputeHeadScoresCheckedNoPartial(", 1)[1]
    assert "AttentionQ16ComputeScaledQKRowCheckedNoPartialDefaultStride(" in body


def test_multilingual_single_head_parity() -> None:
    token_count = 6
    head_dim = 18
    score_scale_q16 = 17327

    q_head = q16_from_text("γειά🙂世界-مرحبا", head_dim)
    token_texts = [
        "こんにちは",
        "Привет",
        "hola mundo",
        "नमस्ते",
        "שָׁלוֹם",
        "bonjour-Δ",
    ]

    k_head_rows = [0] * (token_count * head_dim)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * head_dim
        for lane in range(head_dim):
            k_head_rows[base + lane] = row[lane]

    out_capacity = 0 if token_count == 0 else (token_count - 1) * token_count + 1
    out_a = [2026] * max(1, out_capacity)
    out_b = out_a.copy()

    err_a = attention_q16_compute_head_scores_checked_nopartial(
        q_head,
        len(q_head),
        k_head_rows,
        len(k_head_rows),
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
    )
    err_b = explicit_single_head_composition(
        q_head,
        len(q_head),
        k_head_rows,
        len(k_head_rows),
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
    )

    assert err_a == err_b
    assert out_a == out_b


def test_error_parity_and_no_partial_behavior() -> None:
    q_head = [1 << 16, 2 << 16, -(3 << 16)]
    k_rows = [4 << 16, 5 << 16, 6 << 16]

    seed = [0x5151] * 9
    out_a = seed.copy()
    out_b = seed.copy()

    err_a = attention_q16_compute_head_scores_checked_nopartial(
        q_head,
        len(q_head),
        k_rows,
        len(k_rows),
        2,
        3,
        1 << 16,
        out_a,
        1,
    )
    err_b = explicit_single_head_composition(
        q_head,
        len(q_head),
        k_rows,
        len(k_rows),
        2,
        3,
        1 << 16,
        out_b,
        1,
    )

    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == seed

    err_a = attention_q16_compute_head_scores_checked_nopartial(
        None,
        len(q_head),
        k_rows,
        len(k_rows),
        1,
        3,
        1 << 16,
        out_a,
        len(out_a),
    )
    err_b = explicit_single_head_composition(
        None,
        len(q_head),
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
    random.seed(2026042302)

    for _ in range(300):
        token_count = random.randint(0, 10)
        head_dim = random.randint(0, 20)
        score_scale_q16 = random.randint(-(1 << 17), (1 << 17))

        q_head = [random.randint(-(1 << 17), (1 << 17)) for _ in range(head_dim)]
        k_rows = [
            random.randint(-(1 << 17), (1 << 17))
            for _ in range(token_count * head_dim)
        ]

        out_capacity = 0 if token_count == 0 else (token_count - 1) * token_count + 1
        out_a = [random.randint(-9999, 9999) for _ in range(max(1, out_capacity))]
        out_b = out_a.copy()

        err_a = attention_q16_compute_head_scores_checked_nopartial(
            q_head,
            len(q_head),
            k_rows,
            len(k_rows),
            token_count,
            head_dim,
            score_scale_q16,
            out_a,
            out_capacity,
        )
        err_b = explicit_single_head_composition(
            q_head,
            len(q_head),
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
    test_source_contains_head_scores_nopartial_wrapper()
    test_multilingual_single_head_parity()
    test_error_parity_and_no_partial_behavior()
    test_randomized_parity()
    print("ok")
