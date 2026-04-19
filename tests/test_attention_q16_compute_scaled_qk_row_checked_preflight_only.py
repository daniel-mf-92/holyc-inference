#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowCheckedPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_compute_qk_dot_row_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_compute_scaled_qk_row_checked_preflight_only(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
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
    if token_count < 0 or k_row_stride_q16 < 0 or head_dim < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if head_dim > q_row_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        out_last_k_base_index[0] = 0
        out_last_out_base_index[0] = 0
        out_required_k_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    if k_row_stride_q16 < head_dim:
        return ATTN_Q16_ERR_BAD_PARAM
    if out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_k_base_index = try_mul_i64_checked(token_count - 1, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err

    err, last_out_base_index = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base_index, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_k_base_index[0] = last_k_base_index
    out_last_out_base_index[0] = last_out_base_index
    out_required_k_cells[0] = required_k_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def explicit_checked_guard_composition(
    q_row_q16,
    q_row_capacity: int,
    k_rows_q16,
    k_rows_capacity: int,
    token_count: int,
    k_row_stride_q16: int,
    head_dim: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
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
    if token_count < 0 or k_row_stride_q16 < 0 or head_dim < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if head_dim > q_row_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        out_last_k_base_index[0] = 0
        out_last_out_base_index[0] = 0
        out_required_k_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    if k_row_stride_q16 < head_dim:
        return ATTN_Q16_ERR_BAD_PARAM
    if out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_k_cells = try_mul_i64_checked(token_count, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err
    if required_k_cells > k_rows_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_k_base_index = try_mul_i64_checked(token_count - 1, k_row_stride_q16)
    if err != ATTN_Q16_OK:
        return err

    err, last_out_base_index = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base_index, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_k_base_index[0] = last_k_base_index
    out_last_out_base_index[0] = last_out_base_index
    out_required_k_cells[0] = required_k_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def test_source_contains_preflight_helper_and_checked_row_uses_it() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowCheckedPreflightOnly("
    assert signature in source

    helper_body = source.split(signature, 1)[1]
    assert "if (head_dim > q_row_capacity)" in helper_body
    assert "*out_required_k_cells = required_k_cells;" in helper_body
    assert "*out_required_out_cells = required_out_cells;" in helper_body

    row_sig = "I32 AttentionQ16ComputeScaledQKRowChecked("
    row_body = source.split(row_sig, 1)[1]
    assert "AttentionQ16ComputeScaledQKRowCheckedPreflightOnly(" in row_body


def test_known_vector_outputs_expected_diagnostics() -> None:
    token_count = 4
    head_dim = 5
    k_row_stride_q16 = 9
    out_score_stride = 3

    q_row = [0] * head_dim
    k_rows = [0] * (token_count * k_row_stride_q16)
    out_scores = [0] * (1 + (token_count - 1) * out_score_stride)

    got_last_k = [123]
    got_last_out = [234]
    got_req_k = [345]
    got_req_out = [456]

    err = attention_q16_compute_scaled_qk_row_checked_preflight_only(
        q_row,
        len(q_row),
        k_rows,
        len(k_rows),
        token_count,
        k_row_stride_q16,
        head_dim,
        out_scores,
        len(out_scores),
        out_score_stride,
        got_last_k,
        got_last_out,
        got_req_k,
        got_req_out,
    )
    assert err == ATTN_Q16_OK
    assert got_last_k == [(token_count - 1) * k_row_stride_q16]
    assert got_last_out == [(token_count - 1) * out_score_stride]
    assert got_req_k == [token_count * k_row_stride_q16]
    assert got_req_out == [1 + (token_count - 1) * out_score_stride]


def test_error_paths_preserve_output_diagnostics() -> None:
    last_k = [101]
    last_out = [202]
    req_k = [303]
    req_out = [404]

    err = attention_q16_compute_scaled_qk_row_checked_preflight_only(
        None,
        0,
        [0],
        1,
        1,
        1,
        1,
        [0],
        1,
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

    err = attention_q16_compute_scaled_qk_row_checked_preflight_only(
        [0],
        1,
        [0],
        -1,
        1,
        1,
        1,
        [0],
        1,
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


def test_randomized_parity_vs_explicit_guard_composition() -> None:
    rng = random.Random(20260419_558)

    for _ in range(4000):
        token_count = rng.randint(0, 96)
        head_dim = rng.randint(0, 64)

        if token_count == 0:
            k_row_stride_q16 = rng.randint(0, 64)
            out_score_stride = rng.randint(0, 16)
        else:
            k_row_stride_q16 = rng.randint(0, 80)
            out_score_stride = rng.randint(0, 12)

        q_row_capacity = rng.randint(0, 80)
        q_row = [0] * max(q_row_capacity, 1)

        if token_count > 0 and k_row_stride_q16 > 0:
            base_k_need = token_count * k_row_stride_q16
        else:
            base_k_need = 0
        k_rows_capacity = max(0, base_k_need + rng.randint(-8, 8))
        k_rows = [0] * max(k_rows_capacity, 1)

        if token_count > 0 and out_score_stride > 0:
            base_out_need = 1 + (token_count - 1) * out_score_stride
        elif token_count == 0:
            base_out_need = 0
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
            k_row_stride_q16 = -rng.randint(1, 12)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 12)
        if rng.random() < 0.05:
            out_score_stride = -rng.randint(1, 12)

        got_last_k = [11]
        got_last_out = [22]
        got_req_k = [33]
        got_req_out = [44]

        exp_last_k = [55]
        exp_last_out = [66]
        exp_req_k = [77]
        exp_req_out = [88]

        err_got = attention_q16_compute_scaled_qk_row_checked_preflight_only(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_scores_capacity,
            out_score_stride,
            got_last_k,
            got_last_out,
            got_req_k,
            got_req_out,
        )
        err_exp = explicit_checked_guard_composition(
            q_row,
            q_row_capacity,
            k_rows,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_scores_capacity,
            out_score_stride,
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
    test_source_contains_preflight_helper_and_checked_row_uses_it()
    test_known_vector_outputs_expected_diagnostics()
    test_error_paths_preserve_output_diagnostics()
    test_randomized_parity_vs_explicit_guard_composition()
    print("ok")
