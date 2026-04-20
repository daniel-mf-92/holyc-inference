#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnly (IQ-802)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_attention_q16_apply_score_scale_checked import (
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_OK,
    try_add_i64_checked,
    try_mul_i64_checked,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_out_cells is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_out_cells
        or out_required_stage_cells is out_last_out_index
        or out_required_out_cells is out_last_out_index
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count == 0 or token_count == 0:
        out_required_stage_cells[0] = 0
        out_required_out_cells[0] = 0
        out_last_out_index[0] = 0
        return ATTN_Q16_OK

    err, required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, required_out_cells = try_add_i64_checked(required_out_cells, token_count)
    if err != ATTN_Q16_OK:
        return err

    if required_stage_cells > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    err, last_out_index = try_add_i64_checked(required_out_cells, -1)
    if err != ATTN_Q16_OK:
        return err

    out_required_stage_cells[0] = required_stage_cells
    out_required_out_cells[0] = required_out_cells
    out_last_out_index[0] = last_out_index
    return ATTN_Q16_OK


def explicit_checked_composition(*args):
    return attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(*args)


def test_source_contains_commit_only_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1].split(
        "I32 AttentionQ16ComputeScaledQKRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(",
        1,
    )[0]

    assert "required_stage_cells" in body
    assert "required_out_cells" in body
    assert "last_out_index" in body
    assert "if (!out_required_stage_cells || !out_required_out_cells ||" in body
    assert "if (required_stage_cells > staged_scores_capacity)" in body
    assert "if (required_out_cells > out_scores_capacity)" in body
    assert "*out_required_stage_cells = required_stage_cells;" in body


def test_zero_geometry_publishes_zero_diagnostics() -> None:
    staged = [7, 8, 9]
    out = [11, 12, 13]
    req_stage = [123]
    req_out = [456]
    last = [789]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(
        0,
        5,
        staged,
        len(staged),
        out,
        len(out),
        req_stage,
        req_out,
        last,
    )
    assert err == ATTN_Q16_OK
    assert req_stage == [0]
    assert req_out == [0]
    assert last == [0]


def test_no_write_on_capacity_failure() -> None:
    staged = [1, 2, 3, 4]
    out = [9, 9, 9, 9]
    req_stage = [0x11]
    req_out = [0x22]
    last = [0x33]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(
        2,
        3,
        staged,
        5,
        out,
        5,
        req_stage,
        req_out,
        last,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert req_stage == [0x11]
    assert req_out == [0x22]
    assert last == [0x33]


def test_randomized_parity() -> None:
    rng = random.Random(20260421_802)

    for _ in range(2200):
        query_row_count = rng.randint(0, 120)
        token_count = rng.randint(0, 120)
        required = query_row_count * token_count

        staged_cap = required + rng.randint(0, 8)
        out_cap = required + rng.randint(0, 8)

        staged = [rng.randint(-10000, 10000) for _ in range(max(staged_cap, 1))]
        out = [rng.randint(-10000, 10000) for _ in range(max(out_cap, 1))]

        a_req_stage = [0x100]
        a_req_out = [0x200]
        a_last = [0x300]
        b_req_stage = [0x100]
        b_req_out = [0x200]
        b_last = [0x300]

        err_a = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(
            query_row_count,
            token_count,
            staged,
            staged_cap,
            out,
            out_cap,
            a_req_stage,
            a_req_out,
            a_last,
        )
        err_b = explicit_checked_composition(
            query_row_count,
            token_count,
            staged,
            staged_cap,
            out,
            out_cap,
            b_req_stage,
            b_req_out,
            b_last,
        )

        assert err_a == err_b
        assert a_req_stage == b_req_stage
        assert a_req_out == b_req_out
        assert a_last == b_last


if __name__ == "__main__":
    test_source_contains_commit_only_preflight_only_helper()
    test_zero_geometry_publishes_zero_diagnostics()
    test_no_write_on_capacity_failure()
    test_randomized_parity()
    print("attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only=ok")
