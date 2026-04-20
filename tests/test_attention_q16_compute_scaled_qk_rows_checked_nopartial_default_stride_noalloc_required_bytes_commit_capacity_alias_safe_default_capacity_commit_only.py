#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityCommitOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    required_q_cells: int,
    required_k_cells: int,
    required_out_cells: int,
    required_stage_cells: int,
    required_stage_bytes: int,
    *,
    q_base_addr: int = 0x100000,
    k_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
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
    if (
        required_q_cells < 0
        or required_k_cells < 0
        or required_out_cells < 0
        or required_stage_cells < 0
        or required_stage_bytes < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        head_dim,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_q_cells,
        required_k_cells,
        required_out_cells,
        required_stage_cells,
        required_stage_bytes,
        q_base_addr=q_base_addr,
        k_base_addr=k_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def explicit_checked_composition(*args, **kwargs) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
        *args, **kwargs
    )


def test_source_contains_default_stride_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityCommitOnly(" in body


def test_known_vectors_and_parity() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 6

    required_q = (query_row_count - 1) * head_dim + head_dim
    required_k = token_count * head_dim
    required_out = (query_row_count - 1) * token_count + token_count
    required_stage_cells = query_row_count * token_count
    required_stage_bytes = required_stage_cells * 8

    q_rows = [0] * required_q
    k_rows = [0] * required_k
    staged = [index * 13 - 7 for index in range(required_stage_cells)]
    out_a = [1234] * required_out
    out_b = out_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
        q_rows,
        len(q_rows),
        query_row_count,
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        out_a,
        len(out_a),
        staged,
        len(staged),
        required_q,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )

    err_b = explicit_checked_composition(
        q_rows,
        len(q_rows),
        query_row_count,
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        out_b,
        len(out_b),
        staged,
        len(staged),
        required_q,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )

    assert err_a == ATTN_Q16_OK
    assert err_a == err_b
    assert out_a == out_b


def test_overflow_and_null_contracts() -> None:
    q_rows = [0] * 8
    k_rows = [0] * 8
    out_scores = [0] * 8

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
            q_rows,
            len(q_rows),
            (1 << 63) - 1,
            k_rows,
            len(k_rows),
            2,
            1,
            out_scores,
            len(out_scores),
            [0] * 8,
            8,
            0,
            0,
            0,
            0,
            0,
        )
        == ATTN_Q16_ERR_OVERFLOW
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
            None,
            0,
            0,
            [],
            0,
            0,
            0,
            [],
            0,
            [],
            0,
            0,
            0,
            0,
            0,
            0,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
            q_rows,
            len(q_rows),
            1,
            k_rows,
            len(k_rows),
            1,
            1,
            out_scores,
            len(out_scores),
            [0] * 8,
            8,
            1,
            1,
            1,
            1,
            -1,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity() -> None:
    rng = random.Random(20260420_679)

    for _ in range(2400):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 9)
        head_dim = rng.randint(0, 11)

        required_q = (
            0
            if query_row_count == 0
            else (query_row_count - 1) * head_dim + head_dim
        )
        required_k = token_count * head_dim
        required_out = (
            0
            if query_row_count == 0 or token_count == 0
            else (query_row_count - 1) * token_count + token_count
        )
        required_stage_cells = query_row_count * token_count
        required_stage_bytes = required_stage_cells * 8

        q_capacity = max(0, required_q + rng.randint(-2, 3))
        k_capacity = max(0, required_k + rng.randint(-2, 3))
        out_capacity = max(0, required_out + rng.randint(-2, 3))
        stage_capacity = max(0, required_stage_cells + rng.randint(-2, 3))

        q_rows = [0] * q_capacity
        k_rows = [0] * k_capacity
        out_a = [rng.randint(-1000, 1000) for _ in range(out_capacity)]
        out_b = out_a.copy()
        staged = [rng.randint(-2000, 2000) for _ in range(stage_capacity)]

        if rng.randint(0, 9) < 3:
            stage_base = rng.choice([0x100000, 0x200000, 0x300000]) + rng.randint(0, 24)
        else:
            stage_base = 0x400000 + rng.randint(0, 256)

        if rng.randint(0, 14) == 0:
            required_q_case = required_q + rng.randint(1, 2)
        else:
            required_q_case = required_q
        if rng.randint(0, 14) == 0:
            required_k_case = required_k + rng.randint(1, 2)
        else:
            required_k_case = required_k
        if rng.randint(0, 14) == 0:
            required_out_case = required_out + rng.randint(1, 2)
        else:
            required_out_case = required_out
        if rng.randint(0, 14) == 0:
            required_stage_cells_case = required_stage_cells + rng.randint(1, 2)
        else:
            required_stage_cells_case = required_stage_cells
        if rng.randint(0, 14) == 0:
            required_stage_bytes_case = required_stage_bytes + 8
        else:
            required_stage_bytes_case = required_stage_bytes

        q_base = 0x100000 + rng.randint(0, 256)
        k_base = 0x200000 + rng.randint(0, 256)
        out_base = 0x300000 + rng.randint(0, 256)

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_a,
            out_capacity,
            staged,
            stage_capacity,
            required_q_case,
            required_k_case,
            required_out_case,
            required_stage_cells_case,
            required_stage_bytes_case,
            q_base_addr=q_base,
            k_base_addr=k_base,
            out_base_addr=out_base,
            stage_base_addr=stage_base,
        )

        err_b = explicit_checked_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_b,
            out_capacity,
            staged,
            stage_capacity,
            required_q_case,
            required_k_case,
            required_out_case,
            required_stage_cells_case,
            required_stage_bytes_case,
            q_base_addr=q_base,
            k_base_addr=k_base,
            out_base_addr=out_base,
            stage_base_addr=stage_base,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_default_stride_required_bytes_commit_capacity_alias_safe_default_capacity_commit_only_helper()
    test_known_vectors_and_parity()
    test_overflow_and_null_contracts()
    test_randomized_parity()
    print("ok")
