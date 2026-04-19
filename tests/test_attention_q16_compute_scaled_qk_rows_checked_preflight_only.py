#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly."""

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


def attention_q16_compute_scaled_qk_rows_checked_preflight_only(
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
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        q_rows_q16 is None
        or k_rows_q16 is None
        or out_scores_q32 is None
        or out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        query_row_count < 0
        or query_row_stride_q16 < 0
        or token_count < 0
        or k_row_stride_q16 < 0
        or head_dim < 0
        or out_row_stride < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count > 0 and query_row_stride_q16 < head_dim:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count > 0 and token_count > 0 and out_row_stride < token_count:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0:
        out_last_q_base_index[0] = 0
        out_last_k_base_index[0] = 0
        out_last_out_base_index[0] = 0
        out_required_q_cells[0] = 0
        out_required_k_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    err, last_q_base_index = try_mul_i64_checked(query_row_count - 1, query_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err

    err, required_q_cells = try_add_i64_checked(last_q_base_index, head_dim)
    if err != ATTN_Q16_OK:
        return err
    if required_q_cells > q_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        out_last_q_base_index[0] = last_q_base_index
        out_last_k_base_index[0] = 0
        out_last_out_base_index[0] = 0
        out_required_q_cells[0] = required_q_cells
        out_required_k_cells[0] = required_k_cells
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    err, last_k_base_index = try_mul_i64_checked(token_count - 1, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err

    err, last_out_base_index = try_mul_i64_checked(query_row_count - 1, out_row_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base_index, token_count)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_q_base_index[0] = last_q_base_index
    out_last_k_base_index[0] = last_k_base_index
    out_last_out_base_index[0] = last_out_base_index
    out_required_q_cells[0] = required_q_cells
    out_required_k_cells[0] = required_k_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def explicit_checked_guard_composition(
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
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
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
        out_last_q_base_index,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
    )


def test_source_contains_rows_preflight_helper_and_row_wrappers_use_it() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "*out_last_q_base_index = last_q_base_index;" in body
    assert "*out_last_k_base_index = last_k_base_index;" in body
    assert "*out_last_out_base_index = last_out_base_index;" in body
    assert "*out_required_q_cells = required_q_cells;" in body
    assert "*out_required_k_cells = required_k_cells;" in body
    assert "*out_required_out_cells = required_out_cells;" in body

    rows_body = source.split("I32 AttentionQ16ComputeScaledQKRowsChecked(", 1)[1]
    assert "AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(" in rows_body



def test_known_vector_outputs_expected_diagnostics() -> None:
    query_row_count = 4
    query_row_stride_q16 = 17
    token_count = 5
    k_row_stride_q16 = 19
    head_dim = 13
    out_row_stride = 7

    q_rows = [0] * ((query_row_count - 1) * query_row_stride_q16 + head_dim)
    k_rows = [0] * (token_count * k_row_stride_q16)
    out_scores = [0] * ((query_row_count - 1) * out_row_stride + token_count)

    got_last_q = [111]
    got_last_k = [222]
    got_last_out = [333]
    got_req_q = [444]
    got_req_k = [555]
    got_req_out = [666]

    err = attention_q16_compute_scaled_qk_rows_checked_preflight_only(
        q_rows,
        len(q_rows),
        query_row_count,
        query_row_stride_q16,
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        len(out_scores),
        out_row_stride,
        got_last_q,
        got_last_k,
        got_last_out,
        got_req_q,
        got_req_k,
        got_req_out,
    )

    assert err == ATTN_Q16_OK
    assert got_last_q == [(query_row_count - 1) * query_row_stride_q16]
    assert got_last_k == [(token_count - 1) * k_row_stride_q16]
    assert got_last_out == [(query_row_count - 1) * out_row_stride]
    assert got_req_q == [((query_row_count - 1) * query_row_stride_q16) + head_dim]
    assert got_req_k == [token_count * k_row_stride_q16]
    assert got_req_out == [((query_row_count - 1) * out_row_stride) + token_count]



def test_error_paths_do_not_mutate_outputs() -> None:
    last_q = [101]
    last_k = [202]
    last_out = [303]
    req_q = [404]
    req_k = [505]
    req_out = [606]

    err = attention_q16_compute_scaled_qk_rows_checked_preflight_only(
        None,
        0,
        1,
        1,
        [0],
        1,
        1,
        1,
        1,
        [0],
        1,
        1,
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_q == [101]
    assert last_k == [202]
    assert last_out == [303]
    assert req_q == [404]
    assert req_k == [505]
    assert req_out == [606]

    err = attention_q16_compute_scaled_qk_rows_checked_preflight_only(
        [0],
        -1,
        1,
        1,
        [0],
        1,
        1,
        1,
        1,
        [0],
        1,
        1,
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert last_q == [101]
    assert last_k == [202]
    assert last_out == [303]
    assert req_q == [404]
    assert req_k == [505]
    assert req_out == [606]



def test_randomized_parity_vs_explicit_checked_guard_composition() -> None:
    rng = random.Random(20260419_567)

    for _ in range(5000):
        query_row_count = rng.randint(0, 80)
        token_count = rng.randint(0, 80)
        head_dim = rng.randint(0, 64)

        if query_row_count == 0:
            query_row_stride_q16 = rng.randint(0, 64)
        else:
            query_row_stride_q16 = rng.randint(max(0, head_dim - 2), head_dim + 16)

        if token_count == 0:
            k_row_stride_q16 = rng.randint(0, 64)
            out_row_stride = rng.randint(0, 64)
        else:
            k_row_stride_q16 = rng.randint(max(0, head_dim - 2), head_dim + 16)
            out_row_stride = rng.randint(max(0, token_count - 2), token_count + 16)

        q_need = 0
        if query_row_count > 0:
            q_need = (query_row_count - 1) * max(query_row_stride_q16, 0) + head_dim
        k_need = token_count * max(k_row_stride_q16, 0)
        out_need = 0
        if query_row_count > 0 and token_count > 0:
            out_need = (query_row_count - 1) * max(out_row_stride, 0) + token_count

        q_cap = max(0, q_need + rng.randint(-8, 8))
        k_cap = max(0, k_need + rng.randint(-8, 8))
        out_cap = max(0, out_need + rng.randint(-8, 8))

        if rng.random() < 0.02:
            q_cap = -rng.randint(1, 9)
        if rng.random() < 0.02:
            k_cap = -rng.randint(1, 9)
        if rng.random() < 0.02:
            out_cap = -rng.randint(1, 9)

        q_rows = [0] * max(q_cap, 1)
        k_rows = [0] * max(k_cap, 1)
        out_scores = [0] * max(out_cap, 1)

        got_last_q_a = [777]
        got_last_k_a = [888]
        got_last_out_a = [999]
        got_req_q_a = [111]
        got_req_k_a = [222]
        got_req_out_a = [333]

        got_last_q_b = got_last_q_a.copy()
        got_last_k_b = got_last_k_a.copy()
        got_last_out_b = got_last_out_a.copy()
        got_req_q_b = got_req_q_a.copy()
        got_req_k_b = got_req_k_a.copy()
        got_req_out_b = got_req_out_a.copy()

        err_a = attention_q16_compute_scaled_qk_rows_checked_preflight_only(
            q_rows,
            q_cap,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_cap,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_cap,
            out_row_stride,
            got_last_q_a,
            got_last_k_a,
            got_last_out_a,
            got_req_q_a,
            got_req_k_a,
            got_req_out_a,
        )

        err_b = explicit_checked_guard_composition(
            q_rows,
            q_cap,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_cap,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_cap,
            out_row_stride,
            got_last_q_b,
            got_last_k_b,
            got_last_out_b,
            got_req_q_b,
            got_req_k_b,
            got_req_out_b,
        )

        assert err_a == err_b
        assert got_last_q_a == got_last_q_b
        assert got_last_k_a == got_last_k_b
        assert got_last_out_a == got_last_out_b
        assert got_req_q_a == got_req_q_b
        assert got_req_k_a == got_req_k_b
        assert got_req_out_a == got_req_out_b


if __name__ == "__main__":
    test_source_contains_rows_preflight_helper_and_row_wrappers_use_it()
    test_known_vector_outputs_expected_diagnostics()
    test_error_paths_do_not_mutate_outputs()
    test_randomized_parity_vs_explicit_checked_guard_composition()
    print("ok")
