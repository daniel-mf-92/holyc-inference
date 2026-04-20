#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedDefaultStridePreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_preflight_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        q_rows_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    # Mirrors HolyC IQ-575 contract: diagnostics route through shared checked
    # preflight helper with explicit default-stride expansion.
    scratch_last_q = [0]
    scratch_last_k = [0]
    scratch_last_out = [0]

    return attention_q16_compute_scaled_qk_rows_checked_preflight_only(
        q_rows_q16,
        q_rows_capacity,
        query_row_count,
        default_query_row_stride_q16,
        k_rows_q16,
        k_rows_capacity,
        token_count,
        default_k_row_stride_q16,
        head_dim,
        out_scores_q32,
        out_scores_capacity,
        default_out_row_stride,
        scratch_last_q,
        scratch_last_k,
        scratch_last_out,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
    )


def explicit_default_stride_preflight_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        q_rows_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    query_row_stride_q16 = head_dim
    k_row_stride_q16 = head_dim
    out_row_stride = token_count

    scratch_last_q = [0]
    scratch_last_k = [0]
    scratch_last_out = [0]

    return attention_q16_compute_scaled_qk_rows_checked_preflight_only(
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
        scratch_last_q,
        scratch_last_k,
        scratch_last_out,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
    )


def test_source_contains_default_stride_rows_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStridePreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body
    assert "return AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(" in body


def test_known_vector_expected_diagnostics() -> None:
    query_row_count = 4
    token_count = 5
    head_dim = 7

    q_rows = [0] * (((query_row_count - 1) * head_dim) + head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * (((query_row_count - 1) * token_count) + token_count)

    got_req_q = [111]
    got_req_k = [222]
    got_req_out = [333]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_preflight_only(
        q_rows,
        len(q_rows),
        query_row_count,
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        out_scores,
        len(out_scores),
        got_req_q,
        got_req_k,
        got_req_out,
    )

    assert err == ATTN_Q16_OK
    assert got_req_q == [query_row_count * head_dim]
    assert got_req_k == [token_count * head_dim]
    assert got_req_out == [query_row_count * token_count]


def test_error_paths_do_not_mutate_outputs() -> None:
    req_q = [101]
    req_k = [202]
    req_out = [303]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_preflight_only(
        None,
        0,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert req_q == [101]
    assert req_k == [202]
    assert req_out == [303]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_preflight_only(
        [0],
        -1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert req_q == [101]
    assert req_k == [202]
    assert req_out == [303]


def test_matches_explicit_composition_randomized() -> None:
    random.seed(575)

    for _ in range(400):
        query_row_count = random.randint(0, 16)
        token_count = random.randint(0, 16)
        head_dim = random.randint(0, 64)

        q_capacity = (
            0
            if query_row_count == 0
            else (query_row_count - 1) * head_dim + head_dim + random.randint(0, 8)
        )
        k_capacity = token_count * head_dim + random.randint(0, 8)
        out_capacity = (
            0
            if query_row_count == 0 or token_count == 0
            else (query_row_count - 1) * token_count + token_count + random.randint(0, 8)
        )

        if random.random() < 0.2 and q_capacity > 0:
            q_capacity = random.randint(0, q_capacity - 1)
        if random.random() < 0.2 and k_capacity > 0:
            k_capacity = random.randint(0, k_capacity - 1)
        if random.random() < 0.2 and out_capacity > 0:
            out_capacity = random.randint(0, out_capacity - 1)

        q_rows = [0] * max(q_capacity, 1)
        k_rows = [0] * max(k_capacity, 1)
        out_scores = [0] * max(out_capacity, 1)

        got_req_q = [1]
        got_req_k = [2]
        got_req_out = [3]

        exp_req_q = [1]
        exp_req_k = [2]
        exp_req_out = [3]

        got = attention_q16_compute_scaled_qk_rows_checked_default_stride_preflight_only(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            got_req_q,
            got_req_k,
            got_req_out,
        )
        exp = explicit_default_stride_preflight_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            exp_req_q,
            exp_req_k,
            exp_req_out,
        )

        assert got == exp
        assert got_req_q == exp_req_q
        assert got_req_k == exp_req_k
        assert got_req_out == exp_req_out



def test_overflow_surface_matches_core_helper() -> None:
    huge = (1 << 62)
    q_rows = [0]
    k_rows = [0]
    out_scores = [0]

    got_req_q = [9]
    got_req_k = [8]
    got_req_out = [7]
    exp_req_q = [9]
    exp_req_k = [8]
    exp_req_out = [7]

    got = attention_q16_compute_scaled_qk_rows_checked_default_stride_preflight_only(
        q_rows,
        1,
        2,
        k_rows,
        1,
        2,
        huge,
        out_scores,
        1,
        got_req_q,
        got_req_k,
        got_req_out,
    )
    exp = explicit_default_stride_preflight_composition(
        q_rows,
        1,
        2,
        k_rows,
        1,
        2,
        huge,
        out_scores,
        1,
        exp_req_q,
        exp_req_k,
        exp_req_out,
    )

    assert got == exp == ATTN_Q16_ERR_OVERFLOW
    assert got_req_q == exp_req_q == [9]
    assert got_req_k == exp_req_k == [8]
    assert got_req_out == exp_req_out == [7]


if __name__ == "__main__":
    test_source_contains_default_stride_rows_preflight_helper()
    test_known_vector_expected_diagnostics()
    test_error_paths_do_not_mutate_outputs()
    test_matches_explicit_composition_randomized()
    test_overflow_surface_matches_core_helper()
    print("ok")
