#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnly."""

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
    try_mul_i64_checked,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
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
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_stage_cells is None
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

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        required_q_cells,
        required_k_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_q_cells[0] = required_q_cells[0]
    out_required_k_cells[0] = required_k_cells[0]
    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def explicit_composition(
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
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_stage_cells is None
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

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        required_q_cells,
        required_k_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > stage_cell_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_q_cells[0] = required_q_cells[0]
    out_required_k_cells[0] = required_k_cells[0]
    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly("
        in body
    )
    assert "status = AttentionTryMulI64Checked(query_row_count," in body
    assert "status = AttentionTryMulI64Checked(required_stage_cells," in body


def run_parity_case(
    query_row_count: int,
    token_count: int,
    head_dim: int,
    q_rows_q16: list[int],
    k_rows_q16: list[int],
    out_scores_q32: list[int],
    out_scores_capacity: int,
    stage_cell_capacity: int,
) -> None:
    out_q_a = [111]
    out_k_a = [222]
    out_stage_cells_a = [333]
    out_stage_bytes_a = [444]
    out_out_cells_a = [555]

    out_q_b = out_q_a.copy()
    out_k_b = out_k_a.copy()
    out_stage_cells_b = out_stage_cells_a.copy()
    out_stage_bytes_b = out_stage_bytes_a.copy()
    out_out_cells_b = out_out_cells_a.copy()

    err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        q_rows_q16,
        len(q_rows_q16),
        query_row_count,
        k_rows_q16,
        len(k_rows_q16),
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_q_a,
        out_k_a,
        out_stage_cells_a,
        out_stage_bytes_a,
        out_out_cells_a,
    )

    err_b = explicit_composition(
        q_rows_q16,
        len(q_rows_q16),
        query_row_count,
        k_rows_q16,
        len(k_rows_q16),
        token_count,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        stage_cell_capacity,
        out_q_b,
        out_k_b,
        out_stage_cells_b,
        out_stage_bytes_b,
        out_out_cells_b,
    )

    assert err_a == err_b
    assert out_q_a == out_q_b
    assert out_k_a == out_k_b
    assert out_stage_cells_a == out_stage_cells_b
    assert out_stage_bytes_a == out_stage_bytes_b
    assert out_out_cells_a == out_out_cells_b


def test_known_vector_and_zero_rows() -> None:
    query_row_count = 3
    token_count = 4
    head_dim = 6

    q_rows_q16 = [0] * (query_row_count * head_dim)
    k_rows_q16 = [0] * (token_count * head_dim)
    out_scores_q32 = [0] * (query_row_count * token_count)

    run_parity_case(
        query_row_count,
        token_count,
        head_dim,
        q_rows_q16,
        k_rows_q16,
        out_scores_q32,
        len(out_scores_q32),
        query_row_count * token_count,
    )

    run_parity_case(
        0,
        token_count,
        head_dim,
        [0],
        k_rows_q16,
        [0],
        1,
        0,
    )


def test_stage_capacity_failure_is_no_partial() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 2
    q_rows_q16 = [0] * (query_row_count * head_dim)
    k_rows_q16 = [0] * (token_count * head_dim)
    out_scores_q32 = [0] * (query_row_count * token_count)

    out_q = [71]
    out_k = [72]
    out_stage_cells = [73]
    out_stage_bytes = [74]
    out_out_cells = [75]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        q_rows_q16,
        len(q_rows_q16),
        query_row_count,
        k_rows_q16,
        len(k_rows_q16),
        token_count,
        head_dim,
        out_scores_q32,
        len(out_scores_q32),
        query_row_count * token_count - 1,
        out_q,
        out_k,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_q == [71]
    assert out_k == [72]
    assert out_stage_cells == [73]
    assert out_stage_bytes == [74]
    assert out_out_cells == [75]


def test_overflow_and_nullptr_paths() -> None:
    out_q = [1]
    out_k = [2]
    out_stage_cells = [3]
    out_stage_bytes = [4]
    out_out_cells = [5]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        None,
        0,
        0,
        [0],
        1,
        0,
        0,
        [0],
        1,
        0,
        out_q,
        out_k,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_default_stride_noalloc_required_bytes_preflight_only(
        [0],
        1,
        (1 << 62),
        [0],
        1,
        8,
        0,
        [0],
        1,
        (1 << 62),
        out_q,
        out_k,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err in (ATTN_Q16_ERR_OVERFLOW, ATTN_Q16_ERR_BAD_PARAM)


def test_randomized_parity() -> None:
    random.seed(724)

    for _ in range(220):
        query_row_count = random.randint(0, 7)
        token_count = random.randint(0, 8)
        head_dim = random.randint(0, 11)

        q_rows_q16 = [
            random.randint(-(1 << 17), (1 << 17))
            for _ in range(query_row_count * head_dim)
        ]
        k_rows_q16 = [
            random.randint(-(1 << 17), (1 << 17))
            for _ in range(token_count * head_dim)
        ]

        out_scores_capacity = query_row_count * token_count
        out_scores_q32 = [0] * max(out_scores_capacity, 1)

        stage_cell_capacity = query_row_count * token_count
        if random.random() < 0.25 and stage_cell_capacity > 0:
            stage_cell_capacity -= 1

        run_parity_case(
            query_row_count,
            token_count,
            head_dim,
            q_rows_q16,
            k_rows_q16,
            out_scores_q32,
            out_scores_capacity,
            stage_cell_capacity,
        )


if __name__ == "__main__":
    test_source_contains_required_bytes_preflight_only_wrapper()
    test_known_vector_and_zero_rows()
    test_stage_capacity_failure_is_no_partial()
    test_overflow_and_nullptr_paths()
    test_randomized_parity()
    print("ok")
