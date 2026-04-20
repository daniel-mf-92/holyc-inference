#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocCommitCapacityAliasSafeDefaultCapacity."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
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
    staged_scores_q32,
    staged_scores_capacity: int,
    **addr,
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

    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        **addr,
    )


def explicit_checked_composition(
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
    staged_scores_q32,
    staged_scores_capacity: int,
    **addr,
) -> int:
    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe_default_capacity(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        staged_scores_q32,
        staged_scores_capacity,
        **addr,
    )


def test_source_contains_default_stride_alias_safe_default_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityAliasSafeDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacity("
        in body
    )


def test_known_vector_and_alias_overlap_path() -> None:
    q_rows = [1, 2, 3, 4, 5, 6, 7, 8]
    k_rows = [9, 10, 11, 12, 13, 14, 15, 16]
    out_a = [0] * 8
    out_b = [0] * 8
    stage_a = [0] * 8
    stage_b = [0] * 8

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
        q_rows,
        8,
        2,
        k_rows,
        8,
        2,
        4,
        65536,
        out_a,
        8,
        stage_a,
        8,
    )
    err_exp = explicit_checked_composition(
        q_rows,
        8,
        2,
        k_rows,
        8,
        2,
        4,
        65536,
        out_b,
        8,
        stage_b,
        8,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert out_a == out_b

    err_alias = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
        q_rows,
        8,
        2,
        k_rows,
        8,
        2,
        4,
        65536,
        out_a,
        8,
        stage_a,
        8,
        q_base_addr=0x100000,
        k_base_addr=0x200000,
        out_base_addr=0x300000,
        stage_base_addr=0x100008,
    )
    assert err_alias == ATTN_Q16_ERR_BAD_PARAM


def test_null_and_bad_param_contracts() -> None:
    q_rows = [0] * 16
    k_rows = [0] * 16
    out = [0] * 16
    stage = [0] * 16

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
            None,
            16,
            2,
            k_rows,
            16,
            2,
            4,
            65536,
            out,
            16,
            stage,
            16,
        )
        == ATTN_Q16_ERR_NULL_PTR
    )

    assert (
        attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
            q_rows,
            -1,
            2,
            k_rows,
            16,
            2,
            4,
            65536,
            out,
            16,
            stage,
            16,
        )
        == ATTN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity() -> None:
    rng = random.Random(20260420_672)

    for _ in range(350):
        query_row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 8)
        head_dim = rng.randint(0, 8)

        q_need = 0 if query_row_count == 0 else (query_row_count - 1) * head_dim + head_dim
        k_need = 0 if token_count == 0 else (token_count - 1) * head_dim + head_dim
        out_need = 0 if (query_row_count == 0 or token_count == 0) else (query_row_count - 1) * token_count + token_count
        stage_need = query_row_count * token_count

        q_capacity = max(0, q_need + rng.randint(-1, 2))
        k_capacity = max(0, k_need + rng.randint(-1, 2))
        out_capacity = max(0, out_need + rng.randint(-1, 2))
        stage_capacity = max(0, stage_need + rng.randint(-1, 2))

        q_rows = [rng.randint(-300, 300) for _ in range(max(1, q_capacity))]
        k_rows = [rng.randint(-300, 300) for _ in range(max(1, k_capacity))]
        out_got = [rng.randint(-100, 100) for _ in range(max(1, out_capacity))]
        out_exp = list(out_got)
        stage_got = [rng.randint(-100, 100) for _ in range(max(1, stage_capacity))]
        stage_exp = list(stage_got)

        addr = {
            "q_base_addr": 0x100000 + rng.randint(0, 256),
            "k_base_addr": 0x200000 + rng.randint(0, 256),
            "out_base_addr": 0x300000 + rng.randint(0, 256),
            "stage_base_addr": 0x400000 + rng.randint(0, 256),
        }
        pick = rng.random()
        if pick < 0.15:
            addr["stage_base_addr"] = addr["q_base_addr"]
        elif pick < 0.30:
            addr["stage_base_addr"] = addr["k_base_addr"]
        elif pick < 0.45:
            addr["stage_base_addr"] = addr["out_base_addr"]

        score_scale_q16 = rng.randint(-131072, 131072)

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_got,
            out_capacity,
            stage_got,
            stage_capacity,
            **addr,
        )
        err_exp = explicit_checked_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            score_scale_q16,
            out_exp,
            out_capacity,
            stage_exp,
            stage_capacity,
            **addr,
        )

        assert err_got == err_exp
        assert out_got == out_exp


if __name__ == "__main__":
    test_source_contains_default_stride_alias_safe_default_capacity_wrapper()
    test_known_vector_and_alias_overlap_path()
    test_null_and_bad_param_contracts()
    test_randomized_parity()
    print("ok")
