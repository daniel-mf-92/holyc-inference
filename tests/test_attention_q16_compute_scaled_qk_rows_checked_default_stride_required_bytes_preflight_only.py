#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideRequiredBytesPreflightOnly."""

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


def attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    if (
        out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_q_bytes is None
        or out_required_k_bytes is None
        or out_required_out_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_q16 is None or k_rows_q16 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if q_rows_capacity < 0 or k_rows_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if query_row_count < 0 or token_count < 0 or head_dim < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    tmp_last_q = [0]
    tmp_last_k = [0]
    tmp_last_out = [0]
    tmp_req_q = [0]
    tmp_req_k = [0]
    tmp_req_out = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_preflight_only(
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
        tmp_last_q,
        tmp_last_k,
        tmp_last_out,
        tmp_req_q,
        tmp_req_k,
        tmp_req_out,
    )
    if err != ATTN_Q16_OK:
        return err

    out_last_q_base_index[0] = tmp_last_q[0]
    out_last_k_base_index[0] = tmp_last_k[0]
    out_last_out_base_index[0] = tmp_last_out[0]
    out_required_q_cells[0] = tmp_req_q[0]
    out_required_k_cells[0] = tmp_req_k[0]
    out_required_out_cells[0] = tmp_req_out[0]

    out_required_q_bytes[0] = tmp_req_q[0] * 8
    out_required_k_bytes[0] = tmp_req_k[0] * 8
    out_required_out_bytes[0] = tmp_req_out[0] * 8
    return ATTN_Q16_OK


def explicit_checked_cell_to_byte_composition(
    q_rows_q16,
    q_rows_capacity: int,
    query_row_count: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_q_base_index: list[int] | None,
    out_last_k_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_q_cells: list[int] | None,
    out_required_k_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_q_bytes: list[int] | None,
    out_required_k_bytes: list[int] | None,
    out_required_out_bytes: list[int] | None,
) -> int:
    if (
        out_last_q_base_index is None
        or out_last_k_base_index is None
        or out_last_out_base_index is None
        or out_required_q_cells is None
        or out_required_k_cells is None
        or out_required_out_cells is None
        or out_required_q_bytes is None
        or out_required_k_bytes is None
        or out_required_out_bytes is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    default_query_row_stride_q16 = head_dim
    default_k_row_stride_q16 = head_dim
    default_out_row_stride = token_count

    err = attention_q16_compute_scaled_qk_rows_checked_preflight_only(
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
        out_last_q_base_index,
        out_last_k_base_index,
        out_last_out_base_index,
        out_required_q_cells,
        out_required_k_cells,
        out_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    out_required_q_bytes[0] = out_required_q_cells[0] * 8
    out_required_k_bytes[0] = out_required_k_cells[0] * 8
    out_required_out_bytes[0] = out_required_out_cells[0] * 8
    return ATTN_Q16_OK


def test_source_contains_default_stride_required_bytes_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideRequiredBytesPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "default_query_row_stride_q16 = head_dim;" in body
    assert "default_k_row_stride_q16 = head_dim;" in body
    assert "default_out_row_stride = token_count;" in body
    assert "AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(" in body
    assert "*out_required_q_bytes = required_q_bytes;" in body
    assert "*out_required_k_bytes = required_k_bytes;" in body
    assert "*out_required_out_bytes = required_out_bytes;" in body


def test_known_vector_expected_cells_bytes_and_last_indices() -> None:
    query_row_count = 4
    token_count = 5
    head_dim = 7

    q_rows = [0] * ((query_row_count - 1) * head_dim + head_dim)
    k_rows = [0] * (token_count * head_dim)
    out_scores = [0] * ((query_row_count - 1) * token_count + token_count)

    got_last_q = [11]
    got_last_k = [12]
    got_last_out = [13]
    got_req_q = [14]
    got_req_k = [15]
    got_req_out = [16]
    got_req_q_bytes = [17]
    got_req_k_bytes = [18]
    got_req_out_bytes = [19]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only(
        q_rows,
        len(q_rows),
        query_row_count,
        k_rows,
        len(k_rows),
        token_count,
        head_dim,
        out_scores,
        len(out_scores),
        got_last_q,
        got_last_k,
        got_last_out,
        got_req_q,
        got_req_k,
        got_req_out,
        got_req_q_bytes,
        got_req_k_bytes,
        got_req_out_bytes,
    )

    assert err == ATTN_Q16_OK
    assert got_last_q == [(query_row_count - 1) * head_dim]
    assert got_last_k == [(token_count - 1) * head_dim]
    assert got_last_out == [(query_row_count - 1) * token_count]
    assert got_req_q == [query_row_count * head_dim]
    assert got_req_k == [token_count * head_dim]
    assert got_req_out == [query_row_count * token_count]
    assert got_req_q_bytes == [query_row_count * head_dim * 8]
    assert got_req_k_bytes == [token_count * head_dim * 8]
    assert got_req_out_bytes == [query_row_count * token_count * 8]


def test_error_paths_no_partial_output_publish() -> None:
    last_q = [101]
    last_k = [202]
    last_out = [303]
    req_q = [404]
    req_k = [505]
    req_out = [606]
    req_q_bytes = [707]
    req_k_bytes = [808]
    req_out_bytes = [909]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only(
        None,
        0,
        1,
        [0],
        1,
        1,
        1,
        [0],
        1,
        last_q,
        last_k,
        last_out,
        req_q,
        req_k,
        req_out,
        req_q_bytes,
        req_k_bytes,
        req_out_bytes,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_q == [101]
    assert last_k == [202]
    assert last_out == [303]
    assert req_q == [404]
    assert req_k == [505]
    assert req_out == [606]
    assert req_q_bytes == [707]
    assert req_k_bytes == [808]
    assert req_out_bytes == [909]


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(806)

    for _ in range(500):
        query_row_count = rng.randint(0, 20)
        token_count = rng.randint(0, 20)
        head_dim = rng.randint(0, 48)

        q_capacity = rng.randint(0, 4096)
        k_capacity = rng.randint(0, 4096)
        out_capacity = rng.randint(0, 4096)

        q_rows = [0] * max(q_capacity, 1)
        k_rows = [0] * max(k_capacity, 1)
        out_scores = [0] * max(out_capacity, 1)

        got_last_q = [9001]
        got_last_k = [9002]
        got_last_out = [9003]
        got_req_q = [9004]
        got_req_k = [9005]
        got_req_out = [9006]
        got_req_q_bytes = [9007]
        got_req_k_bytes = [9008]
        got_req_out_bytes = [9009]

        exp_last_q = [8001]
        exp_last_k = [8002]
        exp_last_out = [8003]
        exp_req_q = [8004]
        exp_req_k = [8005]
        exp_req_out = [8006]
        exp_req_q_bytes = [8007]
        exp_req_k_bytes = [8008]
        exp_req_out_bytes = [8009]

        err_got = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            got_last_q,
            got_last_k,
            got_last_out,
            got_req_q,
            got_req_k,
            got_req_out,
            got_req_q_bytes,
            got_req_k_bytes,
            got_req_out_bytes,
        )

        err_exp = explicit_checked_cell_to_byte_composition(
            q_rows,
            q_capacity,
            query_row_count,
            k_rows,
            k_capacity,
            token_count,
            head_dim,
            out_scores,
            out_capacity,
            exp_last_q,
            exp_last_k,
            exp_last_out,
            exp_req_q,
            exp_req_k,
            exp_req_out,
            exp_req_q_bytes,
            exp_req_k_bytes,
            exp_req_out_bytes,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_last_q == exp_last_q
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_q == exp_req_q
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
            assert got_req_q_bytes == exp_req_q_bytes
            assert got_req_k_bytes == exp_req_k_bytes
            assert got_req_out_bytes == exp_req_out_bytes
        else:
            assert got_last_q == [9001]
            assert got_last_k == [9002]
            assert got_last_out == [9003]
            assert got_req_q == [9004]
            assert got_req_k == [9005]
            assert got_req_out == [9006]
            assert got_req_q_bytes == [9007]
            assert got_req_k_bytes == [9008]
            assert got_req_out_bytes == [9009]


def test_overflow_on_cell_to_byte_conversion_matches_explicit() -> None:
    huge = 1 << 62

    got_last_q = [1]
    got_last_k = [2]
    got_last_out = [3]
    got_req_q = [4]
    got_req_k = [5]
    got_req_out = [6]
    got_req_q_bytes = [7]
    got_req_k_bytes = [8]
    got_req_out_bytes = [9]

    exp_last_q = [10]
    exp_last_k = [11]
    exp_last_out = [12]
    exp_req_q = [13]
    exp_req_k = [14]
    exp_req_out = [15]
    exp_req_q_bytes = [16]
    exp_req_k_bytes = [17]
    exp_req_out_bytes = [18]

    err_got = attention_q16_compute_scaled_qk_rows_checked_default_stride_required_bytes_preflight_only(
        [0],
        1,
        2,
        [0],
        1,
        2,
        huge,
        [0],
        1,
        got_last_q,
        got_last_k,
        got_last_out,
        got_req_q,
        got_req_k,
        got_req_out,
        got_req_q_bytes,
        got_req_k_bytes,
        got_req_out_bytes,
    )

    err_exp = explicit_checked_cell_to_byte_composition(
        [0],
        1,
        2,
        [0],
        1,
        2,
        huge,
        [0],
        1,
        exp_last_q,
        exp_last_k,
        exp_last_out,
        exp_req_q,
        exp_req_k,
        exp_req_out,
        exp_req_q_bytes,
        exp_req_k_bytes,
        exp_req_out_bytes,
    )

    assert err_got == err_exp == ATTN_Q16_ERR_OVERFLOW
    assert got_last_q == [1]
    assert got_last_k == [2]
    assert got_last_out == [3]
    assert got_req_q == [4]
    assert got_req_k == [5]
    assert got_req_out == [6]
    assert got_req_q_bytes == [7]
    assert got_req_k_bytes == [8]
    assert got_req_out_bytes == [9]


if __name__ == "__main__":
    test_source_contains_default_stride_required_bytes_preflight_only_helper()
    test_known_vector_expected_cells_bytes_and_last_indices()
    test_error_paths_no_partial_output_publish()
    test_randomized_parity_vs_explicit_checked_composition()
    test_overflow_on_cell_to_byte_conversion_matches_explicit()
    print("ok")
