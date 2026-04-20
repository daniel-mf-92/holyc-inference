#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnly (IQ-576)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only,
    explicit_checked_copy_loops,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
    )


def explicit_wrapper_composition(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    return explicit_checked_copy_loops(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_default_stride_nopartial_commit_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideCommitOnly("
        in body
    )


def test_known_vector_matches_explicit_wrapper_composition() -> None:
    query_row_count = 3
    token_count = 4
    total_cells = query_row_count * token_count

    staged = [
        7,
        -8,
        9,
        -10,
        11,
        -12,
        13,
        -14,
        15,
        -16,
        17,
        -18,
    ]
    out_new = [0x6666] * total_cells
    out_ref = out_new.copy()

    err_new = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only(
            query_row_count,
            token_count,
            staged,
            len(staged),
            out_new,
            len(out_new),
        )
    )
    err_ref = explicit_wrapper_composition(
        query_row_count,
        token_count,
        staged,
        len(staged),
        out_ref,
        len(out_ref),
    )

    assert err_new == err_ref == ATTN_Q16_OK
    assert out_new == out_ref == staged


def test_error_paths_preserve_output() -> None:
    staged = [1, 2, 3, 4]
    out_seed = [4242] * 4

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only(
            -1,
            4,
            staged,
            len(staged),
            out_new,
            len(out_new),
        )
    )
    err_ref = explicit_wrapper_composition(-1, 4, staged, len(staged), out_ref, len(out_ref))
    assert err_new == err_ref == ATTN_Q16_ERR_BAD_PARAM
    assert out_new == out_ref == out_seed

    out_new = out_seed.copy()
    out_ref = out_seed.copy()
    err_new = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only(
            1,
            4,
            None,
            len(staged),
            out_new,
            len(out_new),
        )
    )
    err_ref = explicit_wrapper_composition(1, 4, None, len(staged), out_ref, len(out_ref))
    assert err_new == err_ref == ATTN_Q16_ERR_NULL_PTR
    assert out_new == out_ref == out_seed


def test_randomized_parity_vs_explicit_wrapper() -> None:
    rng = random.Random(20260420_576)

    for _ in range(1000):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)
        required = query_row_count * token_count

        staged_cap = required + rng.randint(0, 6)
        out_cap = required + rng.randint(0, 6)

        staged = [rng.randint(-5000, 5000) for _ in range(max(staged_cap, 1))]
        out_new = [0x5A5A] * max(out_cap, 1)
        out_ref = out_new.copy()

        err_new = (
            attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only(
                query_row_count,
                token_count,
                staged,
                staged_cap,
                out_new,
                out_cap,
            )
        )
        err_ref = explicit_wrapper_composition(
            query_row_count,
            token_count,
            staged,
            staged_cap,
            out_ref,
            out_cap,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            live = required
            assert out_new[:live] == out_ref[:live]
            assert out_new[live:] == out_ref[live:]
        else:
            assert out_new == out_ref


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_commit_only_wrapper()
    test_known_vector_matches_explicit_wrapper_composition()
    test_error_paths_preserve_output()
    test_randomized_parity_vs_explicit_wrapper()
    print("attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only=ok")
