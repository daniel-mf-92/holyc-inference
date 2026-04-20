#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only,
)

I64_MAX = (1 << 63) - 1


def attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        [0],
        [0],
        [0],
        required_q_cells,
        required_k_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_q_cells[0] = required_q_cells[0]
    out_required_k_cells[0] = required_k_cells[0]
    out_required_out_cells[0] = required_out_cells[0]
    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    return ATTN_Q16_OK


def explicit_checked_strided_noalloc_preflight_only_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    query_row_stride_q16: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_row_stride: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
) -> int:
    if (
        out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    required_q_cells = [0]
    required_k_cells = [0]
    required_out_cells = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        out_row_stride,
        [0],
        [0],
        [0],
        required_q_cells,
        required_k_cells,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_q_cells[0] = required_q_cells[0]
    out_required_k_cells[0] = required_k_cells[0]
    out_required_out_cells[0] = required_out_cells[0]
    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    return ATTN_Q16_OK


def test_source_contains_strided_noalloc_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialStridedNoAllocPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(" in body
    assert "AttentionTryMulI64Checked(required_stage_cells," in body


def test_known_vectors_and_null_output_contract() -> None:
    query_row_count = 3
    token_count = 5
    head_dim = 8
    query_row_stride_q16 = 11
    k_row_stride_q16 = 10
    out_row_stride = 7

    q_capacity = query_row_count * query_row_stride_q16
    k_capacity = token_count * k_row_stride_q16
    out_capacity = query_row_count * out_row_stride

    q_rows = [0] * q_capacity
    k_rows = [0] * k_capacity
    out_scores = [0] * out_capacity

    required_q = [111]
    required_k = [222]
    required_out = [333]
    required_stage_cells = [444]
    required_stage_bytes = [555]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
        q_rows,
        q_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows,
        k_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        out_capacity,
        out_row_stride,
        required_q,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )
    assert err == ATTN_Q16_OK
    assert required_q[0] == (query_row_count - 1) * query_row_stride_q16 + head_dim
    assert required_k[0] == token_count * k_row_stride_q16
    assert required_out[0] == (query_row_count - 1) * out_row_stride + token_count
    assert required_stage_cells[0] == query_row_count * token_count
    assert required_stage_bytes[0] == (query_row_count * token_count) * 8

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
        q_rows,
        q_capacity,
        query_row_count,
        query_row_stride_q16,
        k_rows,
        k_capacity,
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        out_capacity,
        out_row_stride,
        None,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR


def test_stage_overflow_and_no_partial_output_commit() -> None:
    query_row_count = I64_MAX // 8 + 1
    token_count = 1
    q_capacity = query_row_count
    k_capacity = token_count
    out_capacity = query_row_count

    required_q = [9]
    required_k = [9]
    required_out = [9]
    required_stage_cells = [9]
    required_stage_bytes = [9]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
        [0],
        q_capacity,
        query_row_count,
        1,
        [0],
        k_capacity,
        token_count,
        1,
        1,
        [0],
        out_capacity,
        1,
        required_q,
        required_k,
        required_out,
        required_stage_cells,
        required_stage_bytes,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
    assert required_q[0] == 9
    assert required_k[0] == 9
    assert required_out[0] == 9
    assert required_stage_cells[0] == 9
    assert required_stage_bytes[0] == 9


def test_randomized_parity_vs_explicit_composition() -> None:
    random.seed(637)

    for _ in range(600):
        query_row_count = random.randint(0, 11)
        token_count = random.randint(0, 11)
        head_dim = random.randint(0, 11)

        query_row_stride_q16 = random.randint(max(head_dim, 0), max(head_dim, 0) + 6)
        k_row_stride_q16 = random.randint(max(head_dim, 0), max(head_dim, 0) + 6)
        out_row_stride = random.randint(max(token_count, 0), max(token_count, 0) + 6)

        q_capacity = query_row_count * query_row_stride_q16
        k_capacity = token_count * k_row_stride_q16
        out_capacity = query_row_count * out_row_stride

        if random.random() < 0.2 and q_capacity > 0:
            q_capacity -= random.randint(1, min(3, q_capacity))
        if random.random() < 0.2 and k_capacity > 0:
            k_capacity -= random.randint(1, min(3, k_capacity))
        if random.random() < 0.2 and out_capacity > 0:
            out_capacity -= random.randint(1, min(3, out_capacity))

        q_rows = [0] * max(0, q_capacity)
        k_rows = [0] * max(0, k_capacity)
        out_scores = [0] * max(0, out_capacity)

        a_required_q = [123]
        a_required_k = [234]
        a_required_out = [345]
        a_required_stage_cells = [456]
        a_required_stage_bytes = [567]

        b_required_q = [123]
        b_required_k = [234]
        b_required_out = [345]
        b_required_stage_cells = [456]
        b_required_stage_bytes = [567]

        err_a = attention_q16_compute_scaled_qk_rows_checked_nopartial_strided_noalloc_preflight_only(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            a_required_q,
            a_required_k,
            a_required_out,
            a_required_stage_cells,
            a_required_stage_bytes,
        )

        err_b = explicit_checked_strided_noalloc_preflight_only_composition(
            q_rows,
            q_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_capacity,
            out_row_stride,
            b_required_q,
            b_required_k,
            b_required_out,
            b_required_stage_cells,
            b_required_stage_bytes,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert a_required_q == b_required_q
            assert a_required_k == b_required_k
            assert a_required_out == b_required_out
            assert a_required_stage_cells == b_required_stage_cells
            assert a_required_stage_bytes == b_required_stage_bytes
        else:
            assert a_required_q == [123]
            assert a_required_k == [234]
            assert a_required_out == [345]
            assert a_required_stage_cells == [456]
            assert a_required_stage_bytes == [567]


if __name__ == "__main__":
    test_source_contains_strided_noalloc_preflight_only_helper()
    test_known_vectors_and_null_output_contract()
    test_stage_overflow_and_no_partial_output_commit()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
