#!/usr/bin/env python3
"""Parity checks for AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly."""

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
    I64_MAX,
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_apply_score_scale_checked_nopartial_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    in_score_stride: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
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
    if token_count < 0 or in_score_stride < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        out_last_in_base_index[0] = 0
        out_last_out_base_index[0] = 0
        out_required_in_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    if in_score_stride < 1 or out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, _total_in_cells = try_mul_i64_checked(token_count, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, _total_out_cells = try_mul_i64_checked(token_count, out_score_stride)
    if err != ATTN_Q16_OK:
        return err

    err, last_in_base_index = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(last_in_base_index, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_out_base_index = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base_index, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_in_base_index[0] = last_in_base_index
    out_last_out_base_index[0] = last_out_base_index
    out_required_in_cells[0] = required_in_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def explicit_inline_checked_guards_reference(
    in_scores_q32,
    in_scores_capacity: int,
    token_count: int,
    in_score_stride: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_score_stride: int,
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
    if token_count < 0 or in_score_stride < 0 or out_score_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if token_count == 0:
        out_last_in_base_index[0] = 0
        out_last_out_base_index[0] = 0
        out_required_in_cells[0] = 0
        out_required_out_cells[0] = 0
        return ATTN_Q16_OK

    if in_score_stride < 1 or out_score_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, _ = try_mul_i64_checked(token_count, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, _ = try_mul_i64_checked(token_count, out_score_stride)
    if err != ATTN_Q16_OK:
        return err

    err, last_in_base_index = try_mul_i64_checked(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_in_cells = try_add_i64_checked(last_in_base_index, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_out_base_index = try_mul_i64_checked(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(last_out_base_index, 1)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_last_in_base_index[0] = last_in_base_index
    out_last_out_base_index[0] = last_out_base_index
    out_required_in_cells[0] = required_in_cells
    out_required_out_cells[0] = required_out_cells
    return ATTN_Q16_OK


def test_source_contains_preflight_signature_and_delegate_shape() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "if (!in_scores_q32 || !out_scores_q32 || !out_last_in_base_index ||" in body
    assert "status = AttentionTryMulI64Checked(token_count," in body
    assert "*out_last_in_base_index = last_in_base_index;" in body
    assert "*out_required_out_cells = required_out_cells;" in body

    wrapper = source.split("I32 AttentionQ16ApplyScoreScaleCheckedNoPartial(", 1)[1].split(
        signature,
        1,
    )[0]
    assert "status = AttentionQ16ApplyScoreScaleCheckedNoPartialPreflightOnly(" in wrapper


def test_error_paths_preserve_diagnostics_outputs() -> None:
    out1 = [111]
    out2 = [222]
    out3 = [333]
    out4 = [444]

    err = attention_q16_apply_score_scale_checked_nopartial_preflight_only(
        [1, 2, 3],
        -1,
        1,
        1,
        [0],
        1,
        1,
        out1,
        out2,
        out3,
        out4,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out1 == [111]
    assert out2 == [222]
    assert out3 == [333]
    assert out4 == [444]

    err = attention_q16_apply_score_scale_checked_nopartial_preflight_only(
        None,
        1,
        1,
        1,
        [0],
        1,
        1,
        out1,
        out2,
        out3,
        out4,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR


def test_known_vector_diagnostics() -> None:
    in_scores = [7, 0, 0, 11, 0, 0, -5]
    out_scores = [0] * 20

    last_in = [0]
    last_out = [0]
    req_in = [0]
    req_out = [0]

    err = attention_q16_apply_score_scale_checked_nopartial_preflight_only(
        in_scores,
        len(in_scores),
        3,
        3,
        out_scores,
        len(out_scores),
        4,
        last_in,
        last_out,
        req_in,
        req_out,
    )
    assert err == ATTN_Q16_OK
    assert last_in[0] == 6
    assert last_out[0] == 8
    assert req_in[0] == 7
    assert req_out[0] == 9


def test_randomized_parity_vs_inline_reference() -> None:
    rng = random.Random(20260419_549)

    for _ in range(3000):
        token_count = rng.randint(0, 64)
        in_stride = rng.randint(0, 8)
        out_stride = rng.randint(0, 8)

        if token_count == 0:
            in_capacity = rng.randint(0, 16)
            out_capacity = rng.randint(0, 16)
        else:
            in_capacity = max(0, 1 + (token_count - 1) * max(in_stride, 1) + rng.randint(-4, 4))
            out_capacity = max(0, 1 + (token_count - 1) * max(out_stride, 1) + rng.randint(-4, 4))

        if rng.random() < 0.05:
            in_capacity = -rng.randint(1, 8)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 8)

        if rng.random() < 0.02:
            token_count = I64_MAX
            in_stride = I64_MAX
        if rng.random() < 0.02:
            token_count = I64_MAX
            out_stride = I64_MAX

        in_scores = [rng.randint(-(1 << 40), (1 << 40)) for _ in range(max(in_capacity, 1))]
        out_scores = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(max(out_capacity, 1))]

        got_last_in = [777]
        got_last_out = [888]
        got_req_in = [999]
        got_req_out = [111]

        exp_last_in = [333]
        exp_last_out = [444]
        exp_req_in = [555]
        exp_req_out = [666]

        err_got = attention_q16_apply_score_scale_checked_nopartial_preflight_only(
            in_scores,
            in_capacity,
            token_count,
            in_stride,
            out_scores,
            out_capacity,
            out_stride,
            got_last_in,
            got_last_out,
            got_req_in,
            got_req_out,
        )
        err_exp = explicit_inline_checked_guards_reference(
            in_scores,
            in_capacity,
            token_count,
            in_stride,
            out_scores,
            out_capacity,
            out_stride,
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
            assert got_last_in == [777]
            assert got_last_out == [888]
            assert got_req_in == [999]
            assert got_req_out == [111]
            assert exp_last_in == [333]
            assert exp_last_out == [444]
            assert exp_req_in == [555]
            assert exp_req_out == [666]


def run() -> None:
    test_source_contains_preflight_signature_and_delegate_shape()
    test_error_paths_preserve_diagnostics_outputs()
    test_known_vector_diagnostics()
    test_randomized_parity_vs_inline_reference()
    print("attention_q16_apply_score_scale_checked_nopartial_preflight_only=ok")


if __name__ == "__main__":
    run()
