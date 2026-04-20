#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import (
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
    query_row_count: int,
    token_count: int,
    out_row_stride: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0 or out_row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, out_row_stride)
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
        err, out_row_base = try_mul_i64_checked(row_index, out_row_stride)
        if err != ATTN_Q16_OK:
            return err

        err, stage_row_base = try_mul_i64_checked(row_index, token_count)
        if err != ATTN_Q16_OK:
            return err

        for token_index in range(token_count):
            err, out_index = try_add_i64_checked(out_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            if out_index < 0 or out_index >= out_scores_capacity:
                return ATTN_Q16_ERR_BAD_PARAM

            err, stage_index = try_add_i64_checked(stage_row_base, token_index)
            if err != ATTN_Q16_OK:
                return err
            if stage_index < 0 or stage_index >= staged_scores_capacity:
                return ATTN_Q16_ERR_BAD_PARAM

    for row_index in range(query_row_count):
        out_row_base = row_index * out_row_stride
        stage_row_base = row_index * token_count
        for token_index in range(token_count):
            out_scores_q32[out_row_base + token_index] = staged_scores_q32[
                stage_row_base + token_index
            ]

    return ATTN_Q16_OK


def explicit_commit_only_composition(
    query_row_count: int,
    token_count: int,
    out_row_stride: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        query_row_count,
        token_count,
        out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
    )


def test_source_contains_strided_noalloc_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitOnly("
    )
    assert signature in source

    commit_capacity_sig = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacity("
    )
    assert commit_capacity_sig in source
    commit_capacity_body = source.split(commit_capacity_sig, 1)[1]
    assert "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitOnly(" in commit_capacity_body


def test_known_vector_commit_only_parity() -> None:
    query_row_count = 3
    token_count = 4
    out_row_stride = 6

    stage_cells = query_row_count * token_count
    staged = [index * 3 - 7 for index in range(stage_cells)]
    out_a = [911] * ((query_row_count - 1) * out_row_stride + token_count)
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        query_row_count,
        token_count,
        out_row_stride,
        staged,
        len(staged),
        out_a,
        len(out_a),
    )

    err_b = explicit_commit_only_composition(
        query_row_count,
        token_count,
        out_row_stride,
        staged,
        len(staged),
        out_b,
        len(out_b),
    )

    assert err_a == ATTN_Q16_OK
    assert err_a == err_b
    assert out_a == out_b


def test_error_surfaces_and_no_partial_guarantee() -> None:
    staged = [1, 2, 3, 4]
    out = [111, 111, 111, 111]
    out_before = out.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        1,
        4,
        4,
        staged,
        len(staged),
        out,
        3,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_before

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        1,
        4,
        4,
        None,
        len(staged),
        out,
        len(out),
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out == out_before


def test_rejects_out_stride_smaller_than_token_count() -> None:
    staged = [5, 6, 7, 8]
    out = [303, 303, 303, 303]
    out_before = out.copy()

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        1,
        4,
        3,
        staged,
        len(staged),
        out,
        len(out),
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out == out_before


def test_randomized_parity() -> None:
    rng = random.Random(20260420_641)

    for _ in range(3000):
        query_row_count = rng.randint(0, 10)
        token_count = rng.randint(0, 12)
        out_row_stride = rng.randint(0, 20)

        if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
            out_row_stride = token_count + rng.randint(0, 4)

        stage_need = query_row_count * token_count
        out_need = 0
        if query_row_count > 0 and token_count > 0:
            out_need = (query_row_count - 1) * out_row_stride + token_count

        staged_capacity = max(0, stage_need + rng.randint(-2, 3))
        out_capacity = max(0, out_need + rng.randint(-2, 3))

        staged = [rng.randint(-2000, 2000) for _ in range(staged_capacity)]
        out_a = [rng.randint(-2000, 2000) for _ in range(out_capacity)]
        out_b = out_a.copy()

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
            query_row_count,
            token_count,
            out_row_stride,
            staged,
            staged_capacity,
            out_a,
            out_capacity,
        )

        err_b = explicit_commit_only_composition(
            query_row_count,
            token_count,
            out_row_stride,
            staged,
            staged_capacity,
            out_b,
            out_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


def test_commit_preserves_row_padding_cells() -> None:
    query_row_count = 2
    token_count = 3
    out_row_stride = 5

    staged = [10, 11, 12, 20, 21, 22]
    out = [777] * ((query_row_count - 1) * out_row_stride + out_row_stride)

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_only(
        query_row_count,
        token_count,
        out_row_stride,
        staged,
        len(staged),
        out,
        len(out),
    )

    assert err == ATTN_Q16_OK
    assert out[0:3] == [10, 11, 12]
    assert out[5:8] == [20, 21, 22]
    assert out[3] == 777 and out[4] == 777
    assert out[8] == 777 and out[9] == 777


if __name__ == "__main__":
    test_source_contains_strided_noalloc_commit_only_helper()
    test_known_vector_commit_only_parity()
    test_error_surfaces_and_no_partial_guarantee()
    test_rejects_out_stride_smaller_than_token_count()
    test_randomized_parity()
    test_commit_preserves_row_padding_cells()
    print("ok")
