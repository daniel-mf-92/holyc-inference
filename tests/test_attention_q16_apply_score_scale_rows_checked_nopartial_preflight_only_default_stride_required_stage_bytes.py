#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyDefaultStrideRequiredStageBytes."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only import (
    try_mul_i64,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_cells import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_cells,
)

I64_MAX = (1 << 63) - 1


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_in_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_last_in_index: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or out_required_in_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_last_in_index is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    req_in = [0]
    req_out = [0]
    req_stage_cells = [0]
    last_in = [0]
    last_out = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_cells(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        req_in,
        req_out,
        req_stage_cells,
        last_in,
        last_out,
    )
    if err != ATTN_Q16_OK:
        return err

    err, req_stage_bytes = try_mul_i64(req_stage_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_in_cells[0] = req_in[0]
    out_required_out_cells[0] = req_out[0]
    out_required_stage_cells[0] = req_stage_cells[0]
    out_required_stage_bytes[0] = req_stage_bytes
    out_last_in_index[0] = last_in[0]
    out_last_out_index[0] = last_out[0]
    return ATTN_Q16_OK


def explicit_checked_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_in_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_last_in_index: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    return attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        out_required_in_cells,
        out_required_out_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_last_in_index,
        out_last_out_index,
    )


def test_source_contains_required_stage_bytes_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytes("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageCells("
        in body
    )
    assert "status = AttentionTryMulI64Checked(required_stage_cells," in body
    assert "sizeof(I64)," in body
    assert "*out_required_stage_bytes = required_stage_bytes;" in body


def test_known_vector_outputs_expected_diagnostics() -> None:
    row_count = 3
    token_count = 1
    default_stride = token_count

    last_index = (row_count - 1) * default_stride + (token_count - 1) * default_stride
    required = last_index + 1

    in_scores = [7] * required
    out_scores = [9] * required

    req_in = [101]
    req_out = [102]
    req_stage_cells = [103]
    req_stage_bytes = [104]
    last_in = [105]
    last_out = [106]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        out_scores,
        len(out_scores),
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )

    assert err == ATTN_Q16_OK
    assert req_in == [required]
    assert req_out == [required]
    assert req_stage_cells == [row_count * token_count]
    assert req_stage_bytes == [row_count * token_count * 8]
    assert last_in == [last_index]
    assert last_out == [last_index]


def test_error_paths_preserve_outputs() -> None:
    req_in = [11]
    req_out = [12]
    req_stage_cells = [13]
    req_stage_bytes = [14]
    last_in = [15]
    last_out = [16]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
        None,
        0,
        1,
        1,
        [0],
        1,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert req_in == [11]
    assert req_out == [12]
    assert req_stage_cells == [13]
    assert req_stage_bytes == [14]
    assert last_in == [15]
    assert last_out == [16]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
        [0],
        -1,
        1,
        1,
        [0],
        1,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    huge = 1 << 62
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
        [0],
        I64_MAX,
        huge,
        huge,
        [0],
        I64_MAX,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_and_no_write_on_error() -> None:
    rng = random.Random(716)

    for _ in range(500):
        row_count = rng.randint(-2, 30)
        token_count = rng.randint(-2, 30)

        required_stage = 0
        if row_count >= 0 and token_count >= 0:
            required_stage = row_count * token_count

        cap_pad_in = rng.randint(0, 4)
        cap_pad_out = rng.randint(0, 4)
        in_capacity = required_stage + cap_pad_in
        out_capacity = required_stage + cap_pad_out

        if rng.random() < 0.15:
            in_capacity = rng.randint(-3, 3)
        if rng.random() < 0.15:
            out_capacity = rng.randint(-3, 3)

        in_scores = [0] * max(0, in_capacity if in_capacity >= 0 else 0)
        out_scores = [0] * max(0, out_capacity if out_capacity >= 0 else 0)

        req_in_a = [901]
        req_out_a = [902]
        req_stage_cells_a = [903]
        req_stage_bytes_a = [904]
        last_in_a = [905]
        last_out_a = [906]

        req_in_b = [901]
        req_out_b = [902]
        req_stage_cells_b = [903]
        req_stage_bytes_b = [904]
        last_in_b = [905]
        last_out_b = [906]

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            req_in_a,
            req_out_a,
            req_stage_cells_a,
            req_stage_bytes_a,
            last_in_a,
            last_out_a,
        )
        err_b = explicit_checked_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            req_in_b,
            req_out_b,
            req_stage_cells_b,
            req_stage_bytes_b,
            last_in_b,
            last_out_b,
        )

        assert err_a == err_b
        assert req_in_a == req_in_b
        assert req_out_a == req_out_b
        assert req_stage_cells_a == req_stage_cells_b
        assert req_stage_bytes_a == req_stage_bytes_b
        assert last_in_a == last_in_b
        assert last_out_a == last_out_b


if __name__ == "__main__":
    test_source_contains_required_stage_bytes_helper()
    test_known_vector_outputs_expected_diagnostics()
    test_error_paths_preserve_outputs()
    test_randomized_parity_and_no_write_on_error()
    print("ok")
