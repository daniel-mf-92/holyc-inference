#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
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
    stage_cell_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    required_stage_cells = [0]
    required_out_cells = [0]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        required_stage_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if staged_scores_capacity < required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is q_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_staged_noalloc_composition(
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
    stage_cell_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
) -> int:
    required_stage_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        required_stage_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if staged_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if staged_scores_capacity < required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is q_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is k_rows_q16:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )


def q16_from_text(text: str, head_dim: int) -> list[int]:
    encoded = text.encode("utf-8")
    out = [0] * head_dim
    if head_dim == 0:
        return out

    for i in range(head_dim):
        b = encoded[i % len(encoded)] if encoded else 0
        out[i] = ((b - 128) << 9) + (i * 13)
    return out


def test_source_contains_noalloc_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAlloc("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly("
        in body
    )
    assert "AttentionQ16ComputeScaledQKRowsCheckedDefaultStride(" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly("
        in body
    )
    assert "staged_scores_q32 == q_rows_q16" in body
    assert "staged_scores_q32 == k_rows_q16" in body
    assert "staged_scores_q32 == out_scores_q32" in body


def test_known_vectors_match_explicit_composition() -> None:
    query_row_count = 4
    token_count = 5
    head_dim = 9
    score_scale_q16 = 17321

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_cells = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    for i in range(q_capacity):
        q_rows[i] = ((i * 7) - 39) << 11
    for i in range(k_capacity):
        k_rows[i] = (58 - (i * 3)) << 10

    out_a = [0x5151] * out_capacity
    out_b = [0x5151] * out_capacity
    stage_a = [0x1212] * stage_cells
    stage_b = [0x1212] * stage_cells

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_a,
        out_capacity,
        stage_cells,
        stage_a,
        stage_cells,
    )
    err_b = explicit_staged_noalloc_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        score_scale_q16,
        out_b,
        out_capacity,
        stage_cells,
        stage_b,
        stage_cells,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b


def test_no_partial_preserves_output_on_compute_failure() -> None:
    query_row_count = 2
    token_count = 3
    head_dim = 4

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [1 << 16] * q_capacity
    k_rows = [1 << 16] * k_capacity
    q_rows[2] = -(1 << 63)

    out = [0x7E7E] * out_capacity
    before = out.copy()
    stage = [0x0303] * out_capacity

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        1 << 16,
        out,
        out_capacity,
        out_capacity,
        stage,
        out_capacity,
    )

    assert err != ATTN_Q16_OK
    assert out == before


def test_staging_guards() -> None:
    q_rows = [0, 0, 0, 0]
    k_rows = [0, 0, 0, 0]
    out = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows, 4, 2, k_rows, 4, 2, 2, 1 << 16, out, 4, 4, None, 4
        )
        == ATTN_Q16_ERR_NULL_PTR
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows, 4, 2, k_rows, 4, 2, 2, 1 << 16, out, 4, 4, stage, 3
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows, 4, 2, k_rows, 4, 2, 2, 1 << 16, out, 4, 3, stage, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows, 4, 2, k_rows, 4, 2, 2, 1 << 16, out, 4, 4, q_rows, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows, 4, 2, k_rows, 4, 2, 2, 1 << 16, out, 4, 4, k_rows, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )
    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows, 4, 2, k_rows, 4, 2, 2, 1 << 16, out, 4, 4, out, 4
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_multilingual_and_randomized_parity() -> None:
    query_row_count = 3
    token_count = 4
    head_dim = 11
    score_scale_q16 = 14213

    q_texts = ["γειά🙂世界", "مرحبا-abc", "שלום-123"]
    k_texts = ["こんにちは", "Привет", "bonjour", "नमस्ते"]

    q_rows = [0] * (query_row_count * head_dim)
    k_rows = [0] * (token_count * head_dim)

    for row_index, text in enumerate(q_texts):
        row = q16_from_text(text, head_dim)
        base = row_index * head_dim
        for i in range(head_dim):
            q_rows[base + i] = row[i]

    for row_index, text in enumerate(k_texts):
        row = q16_from_text(text, head_dim)
        base = row_index * head_dim
        for i in range(head_dim):
            k_rows[base + i] = row[i]

    out_capacity = query_row_count * token_count
    stage_cells = out_capacity

    out_a = [0x6A6A] * out_capacity
    out_b = [0x6A6A] * out_capacity
    stage_a = [0x1919] * stage_cells
    stage_b = [0x1919] * stage_cells

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
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
        stage_cells,
        stage_a,
        stage_cells,
    )
    err_b = explicit_staged_noalloc_composition(
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
        stage_cells,
        stage_b,
        stage_cells,
    )

    assert err_a == err_b == ATTN_Q16_OK
    assert out_a == out_b

    rng = random.Random(20260420_604)
    for _ in range(300):
        query_row_count = rng.randint(0, 7)
        token_count = rng.randint(0, 8)
        head_dim = rng.randint(0, 10)
        score_scale_q16 = rng.randint(-(1 << 16), (1 << 16))

        q_capacity = query_row_count * head_dim
        k_capacity = token_count * head_dim
        out_capacity = query_row_count * token_count
        stage_cells = out_capacity

        q_rows = [rng.randint(-(1 << 17), (1 << 17)) for _ in range(q_capacity)]
        k_rows = [rng.randint(-(1 << 17), (1 << 17)) for _ in range(k_capacity)]

        if q_capacity > 0 and rng.random() < 0.15:
            q_rows[rng.randrange(q_capacity)] = -(1 << 63)

        out_a = [rng.randint(-9999, 9999) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()
        stage_a = [rng.randint(-2000, 2000) for _ in range(max(stage_cells, 1))]
        stage_b = stage_a.copy()

        if out_capacity > 0 and rng.random() < 0.2:
            out_capacity -= 1
        if stage_cells > 0 and rng.random() < 0.2:
            stage_cells -= 1

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_a,
            out_capacity,
            stage_cells,
            stage_a,
            len(stage_a),
        )
        err_b = explicit_staged_noalloc_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_b,
            out_capacity,
            stage_cells,
            stage_b,
            len(stage_b),
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_noalloc_wrapper()
    test_known_vectors_match_explicit_composition()
    test_no_partial_preserves_output_on_compute_failure()
    test_staging_guards()
    test_multilingual_and_randomized_parity()
    print("ok")
