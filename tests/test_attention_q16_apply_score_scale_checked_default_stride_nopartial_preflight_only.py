#!/usr/bin/env python3
"""Parity checks for AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly."""

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
from test_attention_q16_apply_score_scale_checked_nopartial_preflight_only import (
    attention_q16_apply_score_scale_checked_nopartial_preflight_only,
)


def attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_in_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_in_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or out_last_in_base_index is None
        or out_last_out_base_index is None
        or out_required_in_cells is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    default_score_stride = token_count
    return attention_q16_apply_score_scale_checked_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        default_score_stride,
        out_scores_q32,
        out_scores_capacity,
        default_score_stride,
        out_last_in_base_index,
        out_last_out_base_index,
        out_required_in_cells,
        out_required_out_cells,
    )


def explicit_default_stride_composition(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_in_base_index: list[int] | None,
    out_last_out_base_index: list[int] | None,
    out_required_in_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    stride = token_count
    return attention_q16_apply_score_scale_checked_nopartial_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        token_count,
        stride,
        out_scores_q32,
        out_scores_capacity,
        stride,
        out_last_in_base_index,
        out_last_out_base_index,
        out_required_in_cells,
        out_required_out_cells,
    )


def test_source_contains_default_stride_nopartial_preflight_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleCheckedDefaultStrideNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "default_score_stride = token_count;" in body
    assert "return AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(" in body


def test_known_vector_matches_explicit_composition() -> None:
    token_count = 4
    stride = token_count

    in_capacity = 1 + (token_count - 1) * stride
    out_capacity = 1 + (token_count - 1) * stride

    in_scores = [0] * in_capacity
    seeds = [77, -123, 456, -789]
    for index, value in enumerate(seeds):
        in_scores[index * stride] = value

    out_scores = [0] * out_capacity

    got_last_in = [111]
    got_last_out = [222]
    got_req_in = [333]
    got_req_out = [444]

    exp_last_in = [555]
    exp_last_out = [666]
    exp_req_in = [777]
    exp_req_out = [888]

    err_got = attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only(
        in_scores,
        in_capacity,
        token_count,
        out_scores,
        out_capacity,
        got_last_in,
        got_last_out,
        got_req_in,
        got_req_out,
    )
    err_exp = explicit_default_stride_composition(
        in_scores,
        in_capacity,
        token_count,
        out_scores,
        out_capacity,
        exp_last_in,
        exp_last_out,
        exp_req_in,
        exp_req_out,
    )

    assert err_got == err_exp == ATTN_Q16_OK
    assert got_last_in == exp_last_in == [12]
    assert got_last_out == exp_last_out == [12]
    assert got_req_in == exp_req_in == [13]
    assert got_req_out == exp_req_out == [13]


def test_error_paths_no_output_mutation() -> None:
    last_in = [101]
    last_out = [202]
    req_in = [303]
    req_out = [404]

    err = attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only(
        None,
        0,
        0,
        [0],
        1,
        last_in,
        last_out,
        req_in,
        req_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_in == [101]
    assert last_out == [202]
    assert req_in == [303]
    assert req_out == [404]

    err = attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only(
        [0],
        -1,
        1,
        [0],
        1,
        last_in,
        last_out,
        req_in,
        req_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert last_in == [101]
    assert last_out == [202]
    assert req_in == [303]
    assert req_out == [404]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260419_557)

    for _ in range(3200):
        token_count = rng.randint(0, 64)

        stride = token_count
        if token_count == 0:
            in_capacity = rng.randint(0, 12)
            out_capacity = rng.randint(0, 12)
        else:
            need = 1 + (token_count - 1) * stride
            in_capacity = max(0, need + rng.randint(-4, 4))
            out_capacity = max(0, need + rng.randint(-4, 4))

        if rng.random() < 0.05:
            in_capacity = -rng.randint(1, 8)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 8)

        in_scores = [rng.randint(-(1 << 26), (1 << 26)) for _ in range(max(in_capacity, 1))]
        out_scores = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(out_capacity, 1))]

        got_last_in = [1]
        got_last_out = [2]
        got_req_in = [3]
        got_req_out = [4]

        exp_last_in = [5]
        exp_last_out = [6]
        exp_req_in = [7]
        exp_req_out = [8]

        err_got = attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only(
            in_scores,
            in_capacity,
            token_count,
            out_scores,
            out_capacity,
            got_last_in,
            got_last_out,
            got_req_in,
            got_req_out,
        )
        err_exp = explicit_default_stride_composition(
            in_scores,
            in_capacity,
            token_count,
            out_scores,
            out_capacity,
            exp_last_in,
            exp_last_out,
            exp_req_in,
            exp_req_out,
        )

        assert err_got == err_exp
        if err_got == ATTN_Q16_OK:
            assert got_last_in == exp_last_in
            assert got_last_out == exp_last_out
            assert got_req_in == exp_req_in
            assert got_req_out == exp_req_out
        else:
            assert got_last_in == [1]
            assert got_last_out == [2]
            assert got_req_in == [3]
            assert got_req_out == [4]
            assert exp_last_in == [5]
            assert exp_last_out == [6]
            assert exp_req_in == [7]
            assert exp_req_out == [8]


def run() -> None:
    test_source_contains_default_stride_nopartial_preflight_helper()
    test_known_vector_matches_explicit_composition()
    test_error_paths_no_output_mutation()
    test_randomized_parity_vs_explicit_composition()
    print("attention_q16_apply_score_scale_checked_default_stride_nopartial_preflight_only=ok")


if __name__ == "__main__":
    run()
