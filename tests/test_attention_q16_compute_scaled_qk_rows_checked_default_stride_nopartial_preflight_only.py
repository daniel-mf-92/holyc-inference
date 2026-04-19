#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked import (
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
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

    if query_row_count == 0:
        out_required_q_cells[0] = 0
        out_required_k_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    err, required_q_cells = try_mul_i64_checked(
        query_row_count - 1, default_query_row_stride_q16
    )
    if err != ATTN_Q16_OK:
        return err
    err, required_q_cells = try_add_i64_checked(required_q_cells, head_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_q_cells > q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, default_k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    required_out_cells = 0
    if token_count > 0:
        err, required_out_cells = try_mul_i64_checked(
            query_row_count - 1, default_out_row_stride
        )
        if err != ATTN_Q16_OK:
            return err
        err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
        if err != ATTN_Q16_OK:
            return err
        if required_out_cells > out_scores_capacity:
            return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = required_q_cells
    out_required_k_cells[0] = required_k_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def explicit_default_stride_nopartial_preflight_composition(
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

    if query_row_count == 0:
        out_required_q_cells[0] = 0
        out_required_k_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    err, required_q_cells = try_mul_i64_checked(query_row_count - 1, query_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    err, required_q_cells = try_add_i64_checked(required_q_cells, head_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_q_cells > q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    required_out_cells = 0
    if token_count > 0:
        err, required_out_cells = try_mul_i64_checked(query_row_count - 1, out_row_stride)
        if err != ATTN_Q16_OK:
            return err
        err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
        if err != ATTN_Q16_OK:
            return err
        if required_out_cells > out_scores_capacity:
            return ATTN_Q16_ERR_BAD_PARAM

    out_required_q_cells[0] = required_q_cells
    out_required_k_cells[0] = required_k_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def test_source_contains_default_stride_nopartial_rows_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body


def test_known_vector_expected_diagnostics() -> None:
    query_row_count = 4
    token_count = 5
    head_dim = 9

    q_rows_capacity = (query_row_count - 1) * head_dim + head_dim
    k_rows_capacity = token_count * head_dim
    out_scores_capacity = (query_row_count - 1) * token_count + token_count

    q_rows = [0] * q_rows_capacity
    k_rows = [0] * k_rows_capacity
    out_scores = [0] * out_scores_capacity

    got_required_q = [111]
    got_required_k = [222]
    got_required_out = [333]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        q_rows,
        q_rows_capacity,
        query_row_count,
        k_rows,
        k_rows_capacity,
        token_count,
        head_dim,
        out_scores,
        out_scores_capacity,
        got_required_q,
        got_required_k,
        got_required_out,
    )
    assert err == ATTN_Q16_OK
    assert got_required_q == [query_row_count * head_dim]
    assert got_required_k == [token_count * head_dim]
    assert got_required_out == [query_row_count * token_count]


def test_null_and_bad_param_preserve_outputs() -> None:
    got_required_q = [41]
    got_required_k = [42]
    got_required_out = [43]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        None,
        0,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        got_required_q,
        got_required_k,
        got_required_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert got_required_q == [41]
    assert got_required_k == [42]
    assert got_required_out == [43]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        [0],
        -1,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        got_required_q,
        got_required_k,
        got_required_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_required_q == [41]
    assert got_required_k == [42]
    assert got_required_out == [43]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260419_562)

    for _ in range(5000):
        query_row_count = rng.randint(0, 128)
        token_count = rng.randint(0, 128)
        head_dim = rng.randint(0, 128)

        q_base_need = query_row_count * head_dim
        k_base_need = token_count * head_dim
        out_base_need = query_row_count * token_count

        q_rows_capacity = max(0, q_base_need + rng.randint(-8, 8))
        k_rows_capacity = max(0, k_base_need + rng.randint(-8, 8))
        out_scores_capacity = max(0, out_base_need + rng.randint(-8, 8))

        q_rows = [0] * max(q_rows_capacity, 1)
        k_rows = [0] * max(k_rows_capacity, 1)
        out_scores = [0] * max(out_scores_capacity, 1)

        if rng.random() < 0.04:
            q_rows_capacity = -rng.randint(1, 9)
        if rng.random() < 0.04:
            k_rows_capacity = -rng.randint(1, 9)
        if rng.random() < 0.04:
            out_scores_capacity = -rng.randint(1, 9)
        if rng.random() < 0.04:
            query_row_count = -rng.randint(1, 9)
        if rng.random() < 0.04:
            token_count = -rng.randint(1, 9)
        if rng.random() < 0.04:
            head_dim = -rng.randint(1, 9)

        got_required_q = [7]
        got_required_k = [8]
        got_required_out = [9]

        exp_required_q = [17]
        exp_required_k = [18]
        exp_required_out = [19]

        err_got = (
            attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
                q_rows,
                q_rows_capacity,
                query_row_count,
                k_rows,
                k_rows_capacity,
                token_count,
                head_dim,
                out_scores,
                out_scores_capacity,
                got_required_q,
                got_required_k,
                got_required_out,
            )
        )
        err_exp = explicit_default_stride_nopartial_preflight_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            k_rows,
            k_rows_capacity,
            token_count,
            head_dim,
            out_scores,
            out_scores_capacity,
            exp_required_q,
            exp_required_k,
            exp_required_out,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_required_q == exp_required_q
            assert got_required_k == exp_required_k
            assert got_required_out == exp_required_out
        else:
            assert got_required_q == [7]
            assert got_required_k == [8]
            assert got_required_out == [9]


def test_overflow_surface_matches_checked_math() -> None:
    got_required_q = [5]
    got_required_k = [6]
    got_required_out = [7]

    huge = (1 << 63) - 1

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_preflight_only(
        [0],
        huge,
        huge,
        [0],
        huge,
        2,
        huge,
        [0],
        huge,
        got_required_q,
        got_required_k,
        got_required_out,
    )
    assert err in (ATTN_Q16_ERR_OVERFLOW, ATTN_Q16_ERR_BAD_PARAM)


if __name__ == "__main__":
    test_source_contains_default_stride_nopartial_rows_preflight_helper()
    test_known_vector_expected_diagnostics()
    test_null_and_bad_param_preserve_outputs()
    test_randomized_parity_vs_explicit_composition()
    test_overflow_surface_matches_checked_math()
    print("ok")
