#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowCheckedNoPartialDefaultStride."""

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


def attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
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


def explicit_default_stride_nopartial_composition(
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
    # Explicit composition mirrors the wrapper's contract:
    # default k stride = head_dim, default out stride = token_count,
    # staged no-partial implementation delegated to the strict core.
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


def test_source_contains_default_stride_nopartial_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert "I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialDefaultStride(" in source
    body = source.split(
        "I32 AttentionQ16ComputeScaledQKRowCheckedNoPartialDefaultStride(", 1
    )[1]
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

    out_a = [91357] * max(out_scores_capacity, 1)
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
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
    err_b = explicit_default_stride_nopartial_composition(
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
    head_dim = 18
    token_count = 5
    score_scale_q16 = 17476

    q_row = q16_from_text("γειά🙂世界-abc", head_dim)
    token_texts = [
        "مرحبا-Δ",
        "こんにちは🙂",
        "Привет!",
        "hola mundo",
        "שָׁלוֹם",
    ]

    k_rows = [0] * (token_count * head_dim)
    for token_index, text in enumerate(token_texts):
        row = q16_from_text(text, head_dim)
        base = token_index * head_dim
        for lane in range(head_dim):
            k_rows[base + lane] = row[lane]

    run_case(q_row, k_rows, token_count, head_dim, score_scale_q16)


def test_error_parity_and_no_partial() -> None:
    q = [1 << 16, 2 << 16, 3 << 16]
    k = [4 << 16, 5 << 16, 6 << 16]

    out_seed = [0x1234] * 8
    out_a = out_seed.copy()
    out_b = out_seed.copy()

    err_a = attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
        q,
        3,
        k,
        3,
        2,
        3,
        1 << 16,
        out_a,
        1,
    )
    err_b = explicit_default_stride_nopartial_composition(
        q,
        3,
        k,
        3,
        2,
        3,
        1 << 16,
        out_b,
        1,
    )
    assert err_a == err_b == ATTN_Q16_ERR_BAD_PARAM
    assert out_a == out_b == out_seed

    err_a = attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
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
    err_b = explicit_default_stride_nopartial_composition(
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


def test_randomized_parity_and_no_partial() -> None:
    random.seed(554)

    for _ in range(260):
        token_count = random.randint(0, 8)
        head_dim = random.randint(0, 13)

        q_row = [random.randint(-(1 << 17), (1 << 17)) for _ in range(head_dim)]
        q_row_capacity = head_dim

        k_rows_capacity = token_count * head_dim
        k_rows = [
            random.randint(-(1 << 17), (1 << 17)) for _ in range(k_rows_capacity)
        ]

        out_scores_capacity = 0 if token_count == 0 else (token_count - 1) * token_count + 1
        out_a = [random.randint(-9999, 9999) for _ in range(max(out_scores_capacity, 1))]
        out_b = out_a.copy()

        if token_count > 0 and random.random() < 0.25:
            out_scores_capacity = max(0, out_scores_capacity - 1)

        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        err_a = attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_a,
            out_scores_capacity,
        )
        err_b = explicit_default_stride_nopartial_composition(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_b,
            out_scores_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


def test_matches_scaled_row_checked_when_success_dense_output() -> None:
    # On successful dense default-stride layouts, no-partial wrapper should
    # produce the same score lanes as the non-no-partial checked default shape.
    random.seed(14554)

    for _ in range(64):
        token_count = random.randint(1, 7)
        head_dim = random.randint(1, 12)
        q_row = [random.randint(-(1 << 16), (1 << 16)) for _ in range(head_dim)]
        k_rows = [
            random.randint(-(1 << 16), (1 << 16))
            for _ in range(token_count * head_dim)
        ]
        score_scale_q16 = random.randint(-(1 << 16), (1 << 16))

        out_capacity = (token_count - 1) * token_count + 1
        out_no_partial = [0] * out_capacity
        out_checked = [0] * out_capacity

        err_np = attention_q16_compute_scaled_qk_row_checked_nopartial_default_stride(
            q_row,
            len(q_row),
            k_rows,
            len(k_rows),
            token_count,
            head_dim,
            score_scale_q16,
            out_no_partial,
            out_capacity,
        )
        err_checked = attention_q16_compute_scaled_qk_row_checked(
            q_row,
            len(q_row),
            k_rows,
            len(k_rows),
            token_count,
            head_dim,
            head_dim,
            score_scale_q16,
            out_checked,
            out_capacity,
            token_count,
        )

        assert err_np == err_checked == 0
        assert out_no_partial == out_checked


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_wrapper()
    test_multilingual_and_adversarial_vectors()
    test_error_parity_and_no_partial()
    test_randomized_parity_and_no_partial()
    test_matches_scaled_row_checked_when_success_dense_output()
    print("ok")
