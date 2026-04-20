#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_preflight_only import (
    attention_q16_compute_scaled_qk_rows_checked_preflight_only,
)


def attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
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


def explicit_nopartial_guard_composition(
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


def test_source_contains_nopartial_rows_preflight_signature_and_delegate_shape() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "status = AttentionQ16ComputeScaledQKRowsCheckedPreflightOnly(" in body
    assert "recomputed_last_q_base_index" in body
    assert "recomputed_required_out_cells" in body
    assert "if (last_q_base_index != recomputed_last_q_base_index)" in body
    assert "if (required_out_cells != recomputed_required_out_cells)" in body
    assert "out_required_q_cells" in body
    assert "out_required_k_cells" in body
    assert "out_required_out_cells" in body

    core = source.split("I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartial(", 1)[1]
    assert "status = AttentionQ16ComputeScaledQKRowsCheckedNoPartialPreflightOnly(" in core


def test_known_vector_outputs_expected_diagnostics() -> None:
    query_row_count = 5
    query_row_stride_q16 = 21
    token_count = 4
    k_row_stride_q16 = 18
    head_dim = 11
    out_row_stride = 6

    q_rows = [0] * ((query_row_count - 1) * query_row_stride_q16 + head_dim)
    k_rows = [0] * (token_count * k_row_stride_q16)
    out_scores = [0] * ((query_row_count - 1) * out_row_stride + token_count)

    got_last_q = [101]
    got_last_k = [202]
    got_last_out = [303]
    got_req_q = [404]
    got_req_k = [505]
    got_req_out = [606]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
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


def test_error_paths_preserve_output_diagnostics() -> None:
    last_q = [17]
    last_k = [23]
    last_out = [29]
    req_q = [31]
    req_k = [37]
    req_out = [41]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
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
    assert last_q == [17]
    assert last_k == [23]
    assert last_out == [29]
    assert req_q == [31]
    assert req_k == [37]
    assert req_out == [41]

    err = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
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
    assert last_q == [17]
    assert last_k == [23]
    assert last_out == [29]
    assert req_q == [31]
    assert req_k == [37]
    assert req_out == [41]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260419_587)

    for _ in range(5000):
        query_row_count = rng.randint(0, 72)
        query_row_stride_q16 = rng.randint(0, 96)
        token_count = rng.randint(0, 88)
        k_row_stride_q16 = rng.randint(0, 96)
        head_dim = rng.randint(0, 64)
        out_row_stride = rng.randint(0, 96)

        if query_row_count > 0 and query_row_stride_q16 > 0:
            base_q_need = (query_row_count - 1) * query_row_stride_q16 + head_dim
        elif query_row_count > 0:
            base_q_need = head_dim
        else:
            base_q_need = 0

        if token_count > 0 and k_row_stride_q16 > 0:
            base_k_need = token_count * k_row_stride_q16
        else:
            base_k_need = 0

        if query_row_count > 0 and token_count > 0 and out_row_stride > 0:
            base_out_need = (query_row_count - 1) * out_row_stride + token_count
        elif query_row_count > 0 and token_count > 0:
            base_out_need = token_count
        else:
            base_out_need = 0

        q_rows_capacity = max(0, base_q_need + rng.randint(-8, 8))
        k_rows_capacity = max(0, base_k_need + rng.randint(-8, 8))
        out_scores_capacity = max(0, base_out_need + rng.randint(-8, 8))

        q_rows = [0] * max(q_rows_capacity, 1)
        k_rows = [0] * max(k_rows_capacity, 1)
        out_scores = [0] * max(out_scores_capacity, 1)

        if rng.random() < 0.05:
            q_rows_capacity = -rng.randint(1, 12)
        if rng.random() < 0.05:
            k_rows_capacity = -rng.randint(1, 12)
        if rng.random() < 0.05:
            out_scores_capacity = -rng.randint(1, 12)
        if rng.random() < 0.05:
            query_row_count = -rng.randint(1, 12)
        if rng.random() < 0.05:
            query_row_stride_q16 = -rng.randint(1, 12)
        if rng.random() < 0.05:
            token_count = -rng.randint(1, 12)
        if rng.random() < 0.05:
            k_row_stride_q16 = -rng.randint(1, 12)
        if rng.random() < 0.05:
            head_dim = -rng.randint(1, 12)
        if rng.random() < 0.05:
            out_row_stride = -rng.randint(1, 12)

        got_last_q = [11]
        got_last_k = [22]
        got_last_out = [33]
        got_req_q = [44]
        got_req_k = [55]
        got_req_out = [66]

        exp_last_q = [77]
        exp_last_k = [88]
        exp_last_out = [99]
        exp_req_q = [111]
        exp_req_k = [222]
        exp_req_out = [333]

        err_got = attention_q16_compute_scaled_qk_rows_checked_nopartial_preflight_only(
            q_rows,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_scores_capacity,
            out_row_stride,
            got_last_q,
            got_last_k,
            got_last_out,
            got_req_q,
            got_req_k,
            got_req_out,
        )

        err_exp = explicit_nopartial_guard_composition(
            q_rows,
            q_rows_capacity,
            query_row_count,
            query_row_stride_q16,
            k_rows,
            k_rows_capacity,
            token_count,
            k_row_stride_q16,
            head_dim,
            out_scores,
            out_scores_capacity,
            out_row_stride,
            exp_last_q,
            exp_last_k,
            exp_last_out,
            exp_req_q,
            exp_req_k,
            exp_req_out,
        )

        assert err_got == err_exp

        if err_got == ATTN_Q16_OK:
            assert got_last_q == exp_last_q
            assert got_last_k == exp_last_k
            assert got_last_out == exp_last_out
            assert got_req_q == exp_req_q
            assert got_req_k == exp_req_k
            assert got_req_out == exp_req_out
        else:
            assert got_last_q == [11]
            assert got_last_k == [22]
            assert got_last_out == [33]
            assert got_req_q == [44]
            assert got_req_k == [55]
            assert got_req_out == [66]
            assert exp_last_q == [77]
            assert exp_last_k == [88]
            assert exp_last_out == [99]
            assert exp_req_q == [111]
            assert exp_req_k == [222]
            assert exp_req_out == [333]


if __name__ == "__main__":
    test_source_contains_nopartial_rows_preflight_signature_and_delegate_shape()
    test_known_vector_outputs_expected_diagnostics()
    test_error_paths_preserve_output_diagnostics()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
