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
from test_attention_q16_apply_score_scale_checked import (
    try_add_i64_checked,
    try_mul_i64_checked,
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

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    default_out_row_stride = token_count

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_mul_i64_checked(
        query_row_count - 1, default_out_row_stride
    )
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        err, row_base = try_mul_i64_checked(row_index, default_out_row_stride)
        if err != ATTN_Q16_OK:
            return err
        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, _ = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            err, _ = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err

    for row_index in range(query_row_count):
        err, row_base = try_mul_i64_checked(row_index, default_out_row_stride)
        if err != ATTN_Q16_OK:
            return err
        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            err, stage_index = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            out_scores_q32[out_index] = staged_scores_q32[stage_index]

    return ATTN_Q16_OK


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

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        err, row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            err, stage_index = try_add_i64_checked(row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            out_scores_q32[out_index] = staged_scores_q32[stage_index]

    return ATTN_Q16_OK


def test_source_contains_default_stride_nopartial_commit_only_implementation() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "default_out_row_stride = token_count;" in body
    assert "required_stage_cells" in body
    assert "required_out_cells" in body
    assert "for (row_index = 0; row_index < query_row_count; row_index++)" in body
    assert "for (token_index = 0; token_index < token_count; token_index++)" in body
    assert "out_scores_q32[out_index] = staged_scores_q32[stage_index];" in body


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
    test_source_contains_default_stride_nopartial_commit_only_implementation()
    test_known_vector_matches_explicit_wrapper_composition()
    test_error_paths_preserve_output()
    test_randomized_parity_vs_explicit_wrapper()
    print("attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only=ok")
