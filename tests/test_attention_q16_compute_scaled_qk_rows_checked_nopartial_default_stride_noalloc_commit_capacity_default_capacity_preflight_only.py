#!/usr/bin/env python3
"""Parity harness for ...NoAllocCommitCapacityDefaultCapacityPreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import try_mul_i64_checked
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_preflight_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, commit_stage_cell_capacity = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, commit_stage_byte_capacity = try_mul_i64_checked(staged_scores_capacity, 8)
    if err != ATTN_Q16_OK:
        return err

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_default_capacity_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacityPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionTryMulI64Checked(query_row_count," in body
    assert "AttentionTryMulI64Checked(staged_scores_capacity," in body
    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )


def test_known_vectors_and_zero_case() -> None:
    query_row_count = 4
    token_count = 6
    head_dim = 8

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out_cells = [666]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        out_capacity,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )
    err_exp = explicit_checked_preflight_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        out_capacity,
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage_cells == exp_stage_cells == [out_capacity]
    assert got_stage_bytes == exp_stage_bytes == [out_capacity * 8]
    assert got_out_cells == exp_out_cells == [out_capacity]

    got_stage_cells = [9]
    got_stage_bytes = [9]
    got_out_cells = [9]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
        [0],
        1,
        0,
        [0],
        1,
        7,
        3,
        [0],
        1,
        1,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert got_stage_cells == [0]
    assert got_stage_bytes == [0]
    assert got_out_cells == [0]


def test_error_paths_preserve_outputs() -> None:
    out_stage_cells = [41]
    out_stage_bytes = [42]
    out_out_cells = [43]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
        None,
        1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert out_stage_cells == [41]
    assert out_stage_bytes == [42]
    assert out_out_cells == [43]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
        [0],
        1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        -1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [41]
    assert out_stage_bytes == [42]
    assert out_out_cells == [43]

    huge = 1 << 62
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
        [0],
        I64_MAX,
        huge,
        [0],
        I64_MAX,
        huge,
        1,
        [0],
        I64_MAX,
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
        [0],
        1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        huge,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_629)

    for _ in range(6000):
        query_row_count = rng.randint(0, 64)
        token_count = rng.randint(0, 64)
        head_dim = rng.randint(0, 64)

        q_need = query_row_count * head_dim
        k_need = token_count * head_dim
        out_need = query_row_count * token_count

        q_rows_capacity = max(0, q_need + rng.randint(-12, 12))
        k_rows_capacity = max(0, k_need + rng.randint(-12, 12))
        out_scores_capacity = max(0, out_need + rng.randint(-12, 12))
        staged_scores_capacity = max(0, out_need + rng.randint(-12, 12))

        if rng.random() < 0.05:
            q_rows_capacity = -rng.randint(1, 16)
        if rng.random() < 0.05:
            k_rows_capacity = -rng.randint(1, 16)
        if rng.random() < 0.05:
            out_scores_capacity = -rng.randint(1, 16)
        if rng.random() < 0.05:
            staged_scores_capacity = -rng.randint(1, 16)

        if rng.random() < 0.05:
            query_row_count = -rng.randint(1, 16)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 16)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 16)

        if rng.random() < 0.04:
            query_row_count = (1 << 62) + rng.randint(0, 32)
            token_count = (1 << 62) + rng.randint(0, 32)

        if rng.random() < 0.03:
            staged_scores_capacity = (1 << 62) + rng.randint(0, 32)

        q_rows = None if rng.random() < 0.03 else [0] * max(q_rows_capacity, 1)
        k_rows = None if rng.random() < 0.03 else [0] * max(k_rows_capacity, 1)
        out_scores = None if rng.random() < 0.03 else [0] * max(out_scores_capacity, 1)

        got_stage_cells = [101]
        got_stage_bytes = [102]
        got_out_cells = [103]
        exp_stage_cells = [201]
        exp_stage_bytes = [202]
        exp_out_cells = [203]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            staged_scores_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_preflight_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            staged_scores_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_stage_cells == [101]
            assert got_stage_bytes == [102]
            assert got_out_cells == [103]
            assert exp_stage_cells == [201]
            assert exp_stage_bytes == [202]
            assert exp_out_cells == [203]


if __name__ == "__main__":
    test_source_contains_default_capacity_preflight_only_wrapper()
    test_known_vectors_and_zero_case()
    test_error_paths_preserve_outputs()
    test_randomized_parity_against_explicit_composition()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity_preflight_only=ok"
    )
