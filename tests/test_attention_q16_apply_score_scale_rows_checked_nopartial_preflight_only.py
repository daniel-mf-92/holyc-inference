#!/usr/bin/env python3
"""Parity harness for AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly."""

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

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def try_add_i64(lhs: int, rhs: int) -> tuple[int, int]:
    value = lhs + rhs
    if value < I64_MIN or value > I64_MAX:
        return ATTN_Q16_ERR_OVERFLOW, 0
    return ATTN_Q16_OK, value


def try_mul_i64(lhs: int, rhs: int) -> tuple[int, int]:
    value = lhs * rhs
    if value < I64_MIN or value > I64_MAX:
        return ATTN_Q16_ERR_OVERFLOW, 0
    return ATTN_Q16_OK, value


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    in_score_stride: int,
    out_score_stride: int,
    row_stride: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_row_base_index: list[int] | None,
    out_required_in_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
) -> int:
    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or out_last_row_base_index is None
        or out_required_in_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if in_score_stride < 0 or out_score_stride < 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count == 0 or token_count == 0:
        out_last_row_base_index[0] = 0
        out_required_in_cells[0] = 0
        out_required_out_cells[0] = 0
        out_required_stage_cells[0] = 0
        return ATTN_Q16_OK

    if in_score_stride < 1 or out_score_stride < 1 or row_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, in_row_cells = try_mul_i64(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, in_row_cells = try_add_i64(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    err, out_row_cells = try_mul_i64(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, out_row_cells = try_add_i64(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    if row_stride < in_row_cells or row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_row_base_index = try_mul_i64(row_count - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err

    err, required_in_cells = try_add_i64(last_row_base_index, in_row_cells)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_add_i64(last_row_base_index, out_row_cells)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_stage_cells = try_mul_i64(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    out_last_row_base_index[0] = last_row_base_index
    out_required_in_cells[0] = required_in_cells
    out_required_out_cells[0] = required_out_cells
    out_required_stage_cells[0] = required_stage_cells
    return ATTN_Q16_OK


def explicit_checked_guard_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    in_score_stride: int,
    out_score_stride: int,
    row_stride: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_last_row_base_index: list[int] | None,
    out_required_in_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
) -> int:
    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or out_last_row_base_index is None
        or out_required_in_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if in_score_stride < 0 or out_score_stride < 0 or row_stride < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count == 0 or token_count == 0:
        out_last_row_base_index[0] = 0
        out_required_in_cells[0] = 0
        out_required_out_cells[0] = 0
        out_required_stage_cells[0] = 0
        return ATTN_Q16_OK

    if in_score_stride < 1 or out_score_stride < 1 or row_stride < 1:
        return ATTN_Q16_ERR_BAD_PARAM

    err, in_row_cells = try_mul_i64(token_count - 1, in_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, in_row_cells = try_add_i64(in_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    err, out_row_cells = try_mul_i64(token_count - 1, out_score_stride)
    if err != ATTN_Q16_OK:
        return err
    err, out_row_cells = try_add_i64(out_row_cells, 1)
    if err != ATTN_Q16_OK:
        return err

    if row_stride < in_row_cells or row_stride < out_row_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_row_base_index = try_mul_i64(row_count - 1, row_stride)
    if err != ATTN_Q16_OK:
        return err

    err, required_in_cells = try_add_i64(last_row_base_index, in_row_cells)
    if err != ATTN_Q16_OK:
        return err
    if required_in_cells > in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_out_cells = try_add_i64(last_row_base_index, out_row_cells)
    if err != ATTN_Q16_OK:
        return err
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, required_stage_cells = try_mul_i64(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    out_last_row_base_index[0] = last_row_base_index
    out_required_in_cells[0] = required_in_cells
    out_required_out_cells[0] = required_out_cells
    out_required_stage_cells[0] = required_stage_cells
    return ATTN_Q16_OK


def test_source_contains_rows_nopartial_preflight_signature() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnly("
    assert signature in source

    body = source.split(signature, 1)[1]
    assert "status = AttentionTryMulI64Checked(row_count," in body
    assert "*out_required_stage_cells = required_stage_cells;" in body


def test_known_vector_outputs_expected_diagnostics() -> None:
    row_count = 4
    token_count = 3
    in_score_stride = 3
    out_score_stride = 5
    row_stride = 15

    in_row_cells = 1 + (token_count - 1) * in_score_stride
    out_row_cells = 1 + (token_count - 1) * out_score_stride
    required_in = (row_count - 1) * row_stride + in_row_cells
    required_out = (row_count - 1) * row_stride + out_row_cells

    in_scores = [0] * required_in
    out_scores = [0] * required_out

    last_row = [111]
    req_in = [222]
    req_out = [333]
    req_stage = [444]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        in_score_stride,
        out_score_stride,
        row_stride,
        out_scores,
        len(out_scores),
        last_row,
        req_in,
        req_out,
        req_stage,
    )

    assert err == ATTN_Q16_OK
    assert last_row == [(row_count - 1) * row_stride]
    assert req_in == [required_in]
    assert req_out == [required_out]
    assert req_stage == [row_count * token_count]


def test_error_paths_preserve_output_diagnostics() -> None:
    last_row = [17]
    req_in = [23]
    req_out = [29]
    req_stage = [31]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        None,
        0,
        1,
        1,
        1,
        1,
        1,
        [0],
        1,
        last_row,
        req_in,
        req_out,
        req_stage,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert last_row == [17]
    assert req_in == [23]
    assert req_out == [29]
    assert req_stage == [31]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
        [0],
        -1,
        1,
        1,
        1,
        1,
        1,
        [0],
        1,
        last_row,
        req_in,
        req_out,
        req_stage,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert last_row == [17]
    assert req_in == [23]
    assert req_out == [29]
    assert req_stage == [31]


def test_randomized_parity_vs_explicit_guard_composition() -> None:
    rng = random.Random(20260419_590)

    for _ in range(6000):
        row_count = rng.randint(0, 64)
        token_count = rng.randint(0, 64)
        in_score_stride = rng.randint(0, 64)
        out_score_stride = rng.randint(0, 64)
        row_stride = rng.randint(0, 160)

        if row_count == 0 or token_count == 0:
            base_in_need = 0
            base_out_need = 0
        else:
            base_in_need = (row_count - 1) * row_stride + (token_count - 1) * in_score_stride + 1
            base_out_need = (row_count - 1) * row_stride + (token_count - 1) * out_score_stride + 1

        in_capacity = max(0, base_in_need + rng.randint(-8, 8))
        out_capacity = max(0, base_out_need + rng.randint(-8, 8))

        if rng.random() < 0.04:
            row_count = -rng.randint(1, 8)
        if rng.random() < 0.04:
            token_count = -rng.randint(1, 8)
        if rng.random() < 0.04:
            in_score_stride = -rng.randint(1, 8)
        if rng.random() < 0.04:
            out_score_stride = -rng.randint(1, 8)
        if rng.random() < 0.04:
            row_stride = -rng.randint(1, 8)
        if rng.random() < 0.04:
            in_capacity = -rng.randint(1, 8)
        if rng.random() < 0.04:
            out_capacity = -rng.randint(1, 8)

        in_scores = [0] * max(in_capacity, 1)
        out_scores = [0] * max(out_capacity, 1)

        got_last = [rng.randint(-50, 50)]
        got_req_in = [rng.randint(-50, 50)]
        got_req_out = [rng.randint(-50, 50)]
        got_req_stage = [rng.randint(-50, 50)]

        ref_last = got_last.copy()
        ref_req_in = got_req_in.copy()
        ref_req_out = got_req_out.copy()
        ref_req_stage = got_req_stage.copy()

        err_new = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            in_score_stride,
            out_score_stride,
            row_stride,
            out_scores,
            out_capacity,
            got_last,
            got_req_in,
            got_req_out,
            got_req_stage,
        )

        err_ref = explicit_checked_guard_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            in_score_stride,
            out_score_stride,
            row_stride,
            out_scores,
            out_capacity,
            ref_last,
            ref_req_in,
            ref_req_out,
            ref_req_stage,
        )

        assert err_new == err_ref
        assert got_last == ref_last
        assert got_req_in == ref_req_in
        assert got_req_out == ref_req_out
        assert got_req_stage == ref_req_stage


if __name__ == "__main__":
    test_source_contains_rows_nopartial_preflight_signature()
    test_known_vector_outputs_expected_diagnostics()
    test_error_paths_preserve_output_diagnostics()
    test_randomized_parity_vs_explicit_guard_composition()
    print("ok")
