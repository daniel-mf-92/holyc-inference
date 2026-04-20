#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyDefaultStrideNoAllocRequiredBytes."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or stage_cell_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
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
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_wrapper_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    stage_cell_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if (
        q_rows_capacity < 0
        or k_rows_capacity < 0
        or out_scores_capacity < 0
        or stage_cell_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    return attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes(
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
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_preflight_only_default_stride_noalloc_required_bytes_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnlyDefaultStrideNoAllocRequiredBytes("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "return AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes("
        in body
    )


def test_known_vectors_and_zero_case() -> None:
    query_row_count = 4
    token_count = 3
    head_dim = 5

    q_capacity = query_row_count * head_dim
    k_capacity = token_count * head_dim
    out_capacity = query_row_count * token_count
    stage_capacity = query_row_count * token_count

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]
    exp_stage_cells = [444]
    exp_stage_bytes = [555]
    exp_out_cells = [666]

    err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        stage_capacity,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )
    err_exp = explicit_checked_wrapper_composition(
        q_rows,
        q_capacity,
        query_row_count,
        k_rows,
        k_capacity,
        token_count,
        head_dim,
        out_scores,
        out_capacity,
        stage_capacity,
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_stage_cells == exp_stage_cells == [stage_capacity]
    assert got_stage_bytes == exp_stage_bytes == [stage_capacity * 8]
    assert got_out_cells == exp_out_cells == [out_capacity]

    z_stage_cells = [9]
    z_stage_bytes = [9]
    z_out_cells = [9]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
        [0],
        1,
        0,
        [0],
        1,
        6,
        7,
        [0],
        1,
        0,
        z_stage_cells,
        z_stage_bytes,
        z_out_cells,
    )
    assert err == ATTN_Q16_OK
    assert z_stage_cells == [0]
    assert z_stage_bytes == [0]
    assert z_out_cells == [0]


def test_randomized_parity_and_overflow_behavior() -> None:
    random.seed(644)

    for _ in range(2500):
        query_row_count = random.randint(0, 80)
        token_count = random.randint(0, 80)
        head_dim = random.randint(0, 80)

        q_need = query_row_count * head_dim
        k_need = token_count * head_dim
        out_need = query_row_count * token_count

        q_rows_capacity = max(0, q_need + random.randint(-8, 8))
        k_rows_capacity = max(0, k_need + random.randint(-8, 8))
        out_scores_capacity = max(0, out_need + random.randint(-8, 8))
        stage_cell_capacity = max(0, out_need + random.randint(-8, 8))

        if random.random() < 0.05:
            q_rows_capacity = -random.randint(1, 9)
        if random.random() < 0.05:
            k_rows_capacity = -random.randint(1, 9)
        if random.random() < 0.05:
            out_scores_capacity = -random.randint(1, 9)
        if random.random() < 0.05:
            stage_cell_capacity = -random.randint(1, 9)

        if random.random() < 0.05:
            query_row_count = -random.randint(1, 9)
        if random.random() < 0.05:
            token_count = -random.randint(1, 9)
        if random.random() < 0.05:
            head_dim = -random.randint(1, 9)

        q_rows = None if random.random() < 0.03 else [0] * max(1, q_rows_capacity)
        k_rows = None if random.random() < 0.03 else [0] * max(1, k_rows_capacity)
        out_scores = None if random.random() < 0.03 else [0] * max(1, out_scores_capacity)

        got_stage_cells = [101]
        got_stage_bytes = [202]
        got_out_cells = [303]
        exp_stage_cells = [404]
        exp_stage_bytes = [505]
        exp_out_cells = [606]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            stage_cell_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_wrapper_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            stage_cell_capacity,
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
            assert got_stage_bytes == [202]
            assert got_out_cells == [303]
            assert exp_stage_cells == [404]
            assert exp_stage_bytes == [505]
            assert exp_out_cells == [606]

    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]
    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only_default_stride_noalloc_required_bytes(
        [0],
        I64_MAX,
        I64_MAX,
        [0],
        I64_MAX,
        2,
        0,
        [0],
        I64_MAX,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


if __name__ == "__main__":
    test_source_contains_preflight_only_default_stride_noalloc_required_bytes_wrapper()
    test_known_vectors_and_zero_case()
    test_randomized_parity_and_overflow_behavior()
    print("ok")
