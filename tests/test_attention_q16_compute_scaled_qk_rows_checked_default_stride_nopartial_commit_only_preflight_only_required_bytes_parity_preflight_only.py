#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesParityPreflightOnly (IQ-819)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_preflight_only(
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

    if (
        out_required_stage_cells is staged_scores_q32
        or out_required_stage_cells is out_scores_q32
        or out_required_stage_bytes is staged_scores_q32
        or out_required_stage_bytes is out_scores_q32
        or out_required_out_cells is staged_scores_q32
        or out_required_out_cells is out_scores_q32
        or out_last_out_index is staged_scores_q32
        or out_last_out_index is out_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if query_row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_query_row_count = query_row_count
    snapshot_token_count = token_count
    snapshot_staged_scores_q32 = staged_scores_q32
    snapshot_out_scores_q32 = out_scores_q32
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]
    staged_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
        staged_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(
        snapshot_query_row_count, snapshot_token_count
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64_checked(
        recomputed_required_stage_cells, 8
    )
    if err != ATTN_Q16_OK:
        return err

    if recomputed_required_stage_cells == 0:
        recomputed_last_out_index = 0
    else:
        err, recomputed_last_out_index = try_add_i64_checked(
            recomputed_required_stage_cells, -1
        )
        if err != ATTN_Q16_OK:
            return err

    if snapshot_query_row_count != query_row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_staged_scores_q32 is not staged_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_q32 is not out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_staged_scores_capacity != staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != recomputed_last_out_index:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def explicit_preflight_only_composition(
    query_row_count: int,
    token_count: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_scores_q32,
    out_scores_capacity: int,
    out_required_stage_cells: list[int],
    out_required_stage_bytes: list[int],
    out_required_out_cells: list[int],
    out_last_out_index: list[int],
) -> int:
    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]
    staged_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
        staged_last_out_index,
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
        recomputed_last_out_index = 0
    else:
        err, recomputed_last_out_index = try_add_i64_checked(
            recomputed_required_stage_cells, -1
        )
        if err != ATTN_Q16_OK:
            return err

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != recomputed_last_out_index:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_parity_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParity(" in body
    )
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_staged_scores_q32 = staged_scores_q32;" in body
    assert "snapshot_out_scores_q32 = out_scores_q32;" in body
    assert "snapshot_staged_scores_capacity = staged_scores_capacity;" in body
    assert "snapshot_out_scores_capacity = out_scores_capacity;" in body
    assert "if (snapshot_staged_scores_q32 != staged_scores_q32)" in body
    assert "if (snapshot_out_scores_q32 != out_scores_q32)" in body


def test_known_vector_outputs() -> None:
    query_row_count = 9
    token_count = 5
    total = query_row_count * token_count

    staged = [0] * total
    out = [0] * total

    got_stage_cells = [11]
    got_stage_bytes = [12]
    got_out_cells = [13]
    got_last_out = [14]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_preflight_only(
        query_row_count,
        token_count,
        staged,
        len(staged),
        out,
        len(out),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
        got_last_out,
    )

    assert err == ATTN_Q16_OK
    assert got_stage_cells == [total]
    assert got_stage_bytes == [total * 8]
    assert got_out_cells == [total]
    assert got_last_out == [total - 1]


def test_error_paths_preserve_outputs() -> None:
    staged = [0] * 4
    out = [0] * 4

    got_stage_cells = [901]
    got_stage_bytes = [902]
    got_out_cells = [903]
    got_last_out = [904]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_preflight_only(
        -1,
        4,
        staged,
        len(staged),
        out,
        len(out),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
        got_last_out,
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_stage_cells == [901]
    assert got_stage_bytes == [902]
    assert got_out_cells == [903]
    assert got_last_out == [904]




def test_alias_rejection_for_output_vs_input_buffers() -> None:
    staged = [0] * 8
    out = [0] * 8

    got_stage_bytes = [71]
    got_out_cells = [72]
    got_last_out = [73]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_preflight_only(
        1,
        8,
        staged,
        len(staged),
        out,
        len(out),
        staged,
        got_stage_bytes,
        got_out_cells,
        got_last_out,
    )

    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert got_stage_bytes == [71]
    assert got_out_cells == [72]
    assert got_last_out == [73]

def test_randomized_preflight_only_vs_explicit_composition() -> None:
    rng = random.Random(20260421_838)

    for _ in range(1200):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)

        required = query_row_count * token_count
        staged_cap = required + rng.randint(0, 8)
        out_cap = required + rng.randint(0, 8)

        staged = [0] * max(staged_cap, 1)
        out = [0] * max(out_cap, 1)

        got_stage_cells = [701]
        got_stage_bytes = [702]
        got_out_cells = [703]
        got_last_out = [704]

        exp_stage_cells = [801]
        exp_stage_bytes = [802]
        exp_out_cells = [803]
        exp_last_out = [804]

        err_new = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_preflight_only(
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
        err_ref = explicit_preflight_only_composition(
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
            assert got_stage_cells == [701]
            assert got_stage_bytes == [702]
            assert got_out_cells == [703]
            assert got_last_out == [704]


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_preflight_only_wrapper()
    test_known_vector_outputs()
    test_error_paths_preserve_outputs()
    test_alias_rejection_for_output_vs_input_buffers()
    test_randomized_preflight_only_vs_explicit_composition()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_preflight_only=ok"
    )
