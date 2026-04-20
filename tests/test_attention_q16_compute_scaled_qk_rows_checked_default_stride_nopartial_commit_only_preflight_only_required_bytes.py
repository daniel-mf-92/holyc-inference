#!/usr/bin/env python3
"""Parity harness for AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytes (IQ-817)."""

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

    default_out_row_stride = token_count
    err, required_out_cells = try_mul_i64_checked(query_row_count - 1, default_out_row_stride)
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


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_out_cells
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_required_out_cells
        or out_required_stage_bytes is out_last_out_index
        or out_required_out_cells is out_last_out_index
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    staged_required_stage_cells = [0]
    staged_required_out_cells = [0]
    staged_last_out_index = [0]

    err = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(
            query_row_count,
            token_count,
            staged_scores_q32,
            staged_scores_capacity,
            out_scores_q32,
            out_scores_capacity,
            staged_required_stage_cells,
            staged_required_out_cells,
            staged_last_out_index,
        )
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(query_row_count, token_count)
    if err != ATTN_Q16_OK:
        return err
    err, recomputed_required_stage_bytes = try_mul_i64_checked(
        recomputed_required_stage_cells, 8
    )
    if err != ATTN_Q16_OK:
        return err

    if recomputed_required_stage_cells == 0:
        expected_last_out_index = 0
    else:
        err, expected_last_out_index = try_add_i64_checked(
            recomputed_required_stage_cells, -1
        )
        if err != ATTN_Q16_OK:
            return err

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != expected_last_out_index:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = recomputed_required_stage_bytes
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def explicit_checked_composition(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_last_out_index: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    staged_required_stage_cells = [0]
    staged_required_out_cells = [0]
    staged_last_out_index = [0]

    err = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only(
            query_row_count,
            token_count,
            staged_scores_q32,
            staged_scores_capacity,
            out_scores_q32,
            out_scores_capacity,
            staged_required_stage_cells,
            staged_required_out_cells,
            staged_last_out_index,
        )
    )
    if err != ATTN_Q16_OK:
        return err

    err, required_stage_bytes = try_mul_i64_checked(staged_required_stage_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytes("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnly(" in body
    assert "AttentionTryMulI64Checked(recomputed_required_stage_cells," in body
    assert "sizeof(I64)" in body
    assert "*out_required_stage_bytes = recomputed_required_stage_bytes;" in body


def test_known_vector_cells_bytes_and_last_index() -> None:
    query_row_count = 3
    token_count = 5
    total = query_row_count * token_count

    staged = [0] * total
    out = [0] * total

    req_stage_cells = [111]
    req_stage_bytes = [222]
    req_out_cells = [333]
    last_out = [444]

    err = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes(
            query_row_count,
            token_count,
            staged,
            len(staged),
            out,
            len(out),
            req_stage_cells,
            req_stage_bytes,
            req_out_cells,
            last_out,
        )
    )

    assert err == ATTN_Q16_OK
    assert req_stage_cells == [total]
    assert req_stage_bytes == [total * 8]
    assert req_out_cells == [total]
    assert last_out == [total - 1]


def test_error_paths_preserve_outputs() -> None:
    staged = [0] * 8
    out = [0] * 8

    req_stage_cells = [901]
    req_stage_bytes = [902]
    req_out_cells = [903]
    last_out = [904]

    err = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes(
            -1,
            4,
            staged,
            len(staged),
            out,
            len(out),
            req_stage_cells,
            req_stage_bytes,
            req_out_cells,
            last_out,
        )
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert req_stage_cells == [901]
    assert req_stage_bytes == [902]
    assert req_out_cells == [903]
    assert last_out == [904]

    err = (
        attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes(
            1,
            4,
            None,
            len(staged),
            out,
            len(out),
            req_stage_cells,
            req_stage_bytes,
            req_out_cells,
            last_out,
        )
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert req_stage_cells == [901]
    assert req_stage_bytes == [902]
    assert req_out_cells == [903]
    assert last_out == [904]


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260421_817)

    for _ in range(1000):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)

        required = query_row_count * token_count
        staged_cap = required + rng.randint(0, 8)
        out_cap = required + rng.randint(0, 8)

        staged = [0] * max(staged_cap, 1)
        out = [0] * max(out_cap, 1)

        got_stage_cells = [7001]
        got_stage_bytes = [7002]
        got_out_cells = [7003]
        got_last_out = [7004]

        exp_stage_cells = [8001]
        exp_stage_bytes = [8002]
        exp_out_cells = [8003]
        exp_last_out = [8004]

        err_new = (
            attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes(
                query_row_count,
                token_count,
                staged,
                staged_cap,
                out,
                out_cap,
                got_stage_cells,
                got_stage_bytes,
                got_out_cells,
                got_last_out,
            )
        )
        err_ref = explicit_checked_composition(
            query_row_count,
            token_count,
            staged,
            staged_cap,
            out,
            out_cap,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
            exp_last_out,
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
            assert got_last_out == exp_last_out
        else:
            assert got_stage_cells == [7001]
            assert got_stage_bytes == [7002]
            assert got_out_cells == [7003]
            assert got_last_out == [7004]


if __name__ == "__main__":
    test_source_contains_required_bytes_wrapper()
    test_known_vector_cells_bytes_and_last_index()
    test_error_paths_preserve_outputs()
    test_randomized_parity_vs_explicit_checked_composition()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes=ok"
    )
