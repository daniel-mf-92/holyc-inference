#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowCheckedDefaultStride.

IQ-564 updates this wrapper to bridge through the staged no-partial core.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_compute_scaled_qk_row_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    attention_q16_compute_scaled_qk_row_checked,
    q16_from_text,
)
from test_attention_q16_compute_scaled_qk_row_checked_nopartial import (
    attention_q16_compute_scaled_qk_row_checked_nopartial,
)


def attention_q16_compute_scaled_qk_row_checked_default_stride(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if q_row_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_row_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_k_row_stride_q16 = head_dim
    default_out_score_stride = token_count

    return attention_q16_compute_scaled_qk_row_checked_nopartial(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride,
    )


def explicit_default_stride_nopartial_bridge(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    return attention_q16_compute_scaled_qk_row_checked_nopartial(
        q_row_q16,
        q_row_capacity,
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


def test_source_contains_default_stride_nopartial_bridge() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStride(" in source
    body = source.split("I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStride(", 1)[1]
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_score_stride = token_count;" in body
    assert "return AttentionQ16ComputeScaledQKRowCheckedNoPartial(" in body


def run_case(
    q_row_q16: list[int],
    k_rows_q16: list[int],
    token_count: int,
    head_dim: int,
    score_scale_q16: int,
) -> None:
    q_row_capacity = len(q_row_q16)
    k_rows_capacity = len(k_rows_q16)
    out_scores_capacity = 0 if token_count == 0 else (token_count - 1) * token_count + 1

    out_a = [2468] * max(out_scores_capacity, 1)
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_default_stride(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_scores_capacity,
    )
    err_b = explicit_default_stride_nopartial_bridge(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_scores_capacity,
    )

    assert err_a == err_b
    assert out_a == out_b


def test_multilingual_and_adversarial_vectors() -> None:
    head_dim = 20
    token_count = 4
    score_scale_q16 = 15662

    q_row = q16_from_text("γειά🙂世界-abc", head_dim)
    token_texts = [
        "مرحبا",
        "こんにちは",
        "adversarial-\x00\n",
        "שלום🙂",
    ]

    k_rows = [0] * (token_count * head_dim)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * head_dim
        for lane in range(head_dim):
            k_rows[base + lane] = row[lane]

    run_case(q_row, k_rows, token_count, head_dim, score_scale_q16)


def test_error_parity_and_no_partial() -> None:
    out_seed = [111] * 8
    q = [1 << 16, 2 << 16, 3 << 16]
    k = [4 << 16, 5 << 16, 6 << 16]

    out_a = out_seed.copy()
    out_b = out_seed.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_default_stride(
        None,
        3,
        k,
        3,
        1,
        3,
        1 << 16,
        out_a,
        len(out_a),
    )
    err_b = explicit_default_stride_nopartial_bridge(
        None,
        3,
        k,
        3,
        1,
        3,
        1 << 16,
        out_b,
        len(out_b),
    )
    assert err_a == err_b == ATTN_Q16_ERR_NULL_PTR
    assert out_a == out_b == out_seed

    err_a = attention_q16_compute_scaled_qk_row_checked_default_stride(
        q,
        3,
        k,
        3,
        -1,
        3,
        1 << 16,
        out_a,
        len(out_a),
    )
    err_b = explicit_default_stride_nopartial_bridge(
        q,
        3,
        k,
        3,
        -1,
        3,
        1 << 16,
        out_b,
        len(out_b),
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == out_seed

    # token_count=3, out_stride=3 requires index 6 (7 cells). capacity 6 is invalid.
    k_big = [
        1 << 16,
        2 << 16,
        3 << 16,
        4 << 16,
        5 << 16,
        6 << 16,
        7 << 16,
        8 << 16,
        9 << 16,
    ]
    out_a = out_seed.copy()
    out_b = out_seed.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_default_stride(
        q,
        3,
        k_big,
        len(k_big),
        3,
        3,
        1 << 16,
        out_a,
        6,
    )
    err_b = explicit_default_stride_nopartial_bridge(
        q,
        3,
        k_big,
        len(k_big),
        3,
        3,
        1 << 16,
        out_b,
        6,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == out_seed


def test_randomized_bridge_and_checked_numeric_parity() -> None:
    random.seed(564)

    for _ in range(280):
        token_count = random.randint(0, 9)
        head_dim = random.randint(0, 12)
        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        q_row = [random.randint(-(1 << 17), (1 << 17)) for _ in range(head_dim)]
        q_row_capacity = head_dim

        k_rows_capacity = token_count * head_dim
        k_rows = [
            random.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows_capacity)
        ]

        out_scores_capacity = (
            0 if token_count == 0 else (token_count - 1) * token_count + 1
        )
        out_bridge = [
            random.randint(-999, 999) for _ in range(max(out_scores_capacity, 1))
        ]
        out_comp = out_bridge.copy()
        out_plain = out_bridge.copy()

        if token_count > 0 and random.random() < 0.3:
            out_scores_capacity = max(0, out_scores_capacity - 1)

        err_bridge = attention_q16_compute_scaled_qk_row_checked_default_stride(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_bridge,
            out_scores_capacity,
        )
        err_comp = explicit_default_stride_nopartial_bridge(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_comp,
            out_scores_capacity,
        )

        assert err_bridge == err_comp
        assert out_bridge == out_comp

        # When no-partial bridge succeeds, numeric outputs must match plain checked.
        if err_bridge == 0:
            err_plain = attention_q16_compute_scaled_qk_row_checked(
                q_row,
                q_row_capacity,
                k_rows,
                k_rows_capacity,
                token_count,
                head_dim,
                head_dim,
                score_scale_q16,
                out_plain,
                out_scores_capacity,
                token_count,
            )
            assert err_plain == 0
            assert out_plain == out_bridge


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_bridge()
    test_multilingual_and_adversarial_vectors()
    test_error_parity_and_no_partial()
    test_randomized_bridge_and_checked_numeric_parity()
    print("ok")
