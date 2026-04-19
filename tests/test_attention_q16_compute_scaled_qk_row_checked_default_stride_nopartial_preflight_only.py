#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowCheckedDefaultStrideNoPartialPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_compute_qk_dot_row_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
)
from test_attention_q16_compute_scaled_qk_row_checked_preflight_only import (
    attention_q16_compute_scaled_qk_row_checked_preflight_only,
)


def attention_q16_compute_scaled_qk_row_checked_default_stride_nopartial_preflight_only(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        q_row_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_k_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_row_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_k_row_stride_q16 = head_dim
    default_out_score_stride = token_count

    return attention_q16_compute_scaled_qk_row_checked_preflight_only(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_score_stride,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_k_cells,
        out_required_out_cells,
    )


def explicit_default_stride_composition(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    return attention_q16_compute_scaled_qk_row_checked_preflight_only(
        q_row_q16,
        q_row_capacity,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        head_dim,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        token_count,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_k_cells,
        out_required_out_cells,
    )


def test_source_contains_default_stride_nopartial_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowCheckedDefaultStrideNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_score_stride = token_count;" in body
    assert "return AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(" in body


def test_known_vector_outputs_expected_diagnostics() -> None:
    token_count = 5
    head_dim = 7

    q_row = [0] * head_dim
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (1 + (token_count - 1) * token_count)

    got_last_k = [13]
    got_last_out = [17]
    got_req_k = [19]
    got_req_out = [23]

    err = attention_q16_compute_scaled_qk_row_checked_default_stride_nopartial_preflight_only(
        q_row,
        len(q_row),
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        out_scores,
        len(out_scores),
        got_last_k,
        got_last_out,
        got_req_k,
        got_req_out,
    )
    assert err == ATTN_Q16_OK
    assert got_last_k == [(token_count - 1) * head_dim]
    assert got_last_out == [(token_count - 1) * token_count]
    assert got_req_k == [token_count * head_dim]
    assert got_req_out == [1 + (token_count - 1) * token_count]


def test_error_paths_preserve_output_diagnostics() -> None:
    last_k = [101]
    last_out = [202]
    req_k = [303]
    req_out = [404]

    err = attention_q16_compute_scaled_qk_row_checked_default_stride_nopartial_preflight_only(
        None,
        0,
        [0],
        1,
        1,
        1,
        [0],
        1,
        last_k,
        last_out,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_k == [101]
    assert last_out == [202]
    assert req_k == [303]
    assert req_out == [404]

    err = attention_q16_compute_scaled_qk_row_checked_default_stride_nopartial_preflight_only(
        [0],
        -1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        last_k,
        last_out,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert last_k == [101]
    assert last_out == [202]
    assert req_k == [303]
    assert req_out == [404]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260419_559)

    for _ in range(4200):
        token_count = rng.randint(0, 96)
        head_dim = rng.randint(0, 64)
        q_row_capacity = rng.randint(0, 80)

        q_row = [0] * max(q_row_capacity, 1)

        if token_count > 0 and head_dim > 0:
            base_k_need = token_count * head_dim
        else:
            base_k_need = 0
        k_rows_capacity = max(0, base_k_need + rng.randint(-8, 8))
        k_rows = [0] * max(k_rows_capacity, 1)

        if token_count > 0:
            base_out_need = 1 + (token_count - 1) * token_count
        else:
            base_out_need = 0
        out_scores_capacity = max(0, base_out_need + rng.randint(-8, 8))
        out_scores = [0] * max(out_scores_capacity, 1)

        if rng.random() < 0.05:
            q_row_capacity = -rng.randint(1, 12)
        if rng.random() < 0.05:
            k_rows_capacity = -rng.randint(1, 12)
        if rng.random() < 0.05:
            out_scores_capacity = -rng.randint(1, 12)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 12)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 12)

        got_last_k = [11]
        got_last_out = [22]
        got_req_k = [33]
        got_req_out = [44]

        exp_last_k = [55]
        exp_last_out = [66]
        exp_req_k = [77]
        exp_req_out = [88]

        err_got = attention_q16_compute_scaled_qk_row_checked_default_stride_nopartial_preflight_only(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            got_last_k,
            got_last_out,
            got_req_k,
            got_req_out,
        )
        err_exp = explicit_default_stride_composition(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            exp_last_k,
            exp_last_out,
            exp_req_k,
            exp_req_out,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
        else:
            assert got_last_k == [11]
            assert got_last_out == [22]
            assert got_req_k == [33]
            assert got_req_out == [44]


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_preflight_helper()
    test_known_vector_outputs_expected_diagnostics()
    test_error_paths_preserve_output_diagnostics()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
