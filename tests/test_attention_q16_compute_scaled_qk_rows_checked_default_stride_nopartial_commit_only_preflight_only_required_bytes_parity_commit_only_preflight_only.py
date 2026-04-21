#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesParityCommitOnlyPreflightOnly (IQ-841)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
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

    snapshot_query_row_count = query_row_count
    snapshot_token_count = token_count
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]
    staged_last_out_index = [0]

    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_required_out_cells = [0]
    canonical_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
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

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        canonical_required_stage_cells,
        canonical_required_stage_bytes,
        canonical_required_out_cells,
        canonical_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot_query_row_count != query_row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_staged_scores_capacity != staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] != canonical_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != canonical_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != canonical_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != canonical_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def explicit_commit_only_preflight_only_composition(
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

    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_required_out_cells = [0]
    canonical_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
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

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        canonical_required_stage_cells,
        canonical_required_stage_bytes,
        canonical_required_out_cells,
        canonical_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if staged_required_stage_cells[0] != canonical_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != canonical_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != canonical_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != canonical_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_parity_commit_only_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnly(" in body
    )
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParity(" in body
    )
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_staged_scores_capacity = staged_scores_capacity;" in body
    assert "snapshot_out_scores_capacity = out_scores_capacity;" in body


def test_known_vector_commit_only_preflight_only_outputs() -> None:
    query_row_count = 8
    token_count = 6
    total = query_row_count * token_count

    staged = [0] * total
    out = [0] * total

    got_stage_cells = [11]
    got_stage_bytes = [12]
    got_out_cells = [13]
    got_last_out = [14]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
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

    got_stage_cells = [501]
    got_stage_bytes = [502]
    got_out_cells = [503]
    got_last_out = [504]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
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
    assert got_stage_cells == [501]
    assert got_stage_bytes == [502]
    assert got_out_cells == [503]
    assert got_last_out == [504]


def test_randomized_commit_only_preflight_only_vs_explicit_composition() -> None:
    rng = random.Random(20260421_841)

    for _ in range(1200):
        query_row_count = rng.randint(0, 96)
        token_count = rng.randint(0, 96)

        required = query_row_count * token_count
        staged_cap = required + rng.randint(0, 8)
        out_cap = required + rng.randint(0, 8)

        staged = [0] * max(staged_cap, 1)
        out = [0] * max(out_cap, 1)

        got_stage_cells = [601]
        got_stage_bytes = [602]
        got_out_cells = [603]
        got_last_out = [604]

        exp_stage_cells = [701]
        exp_stage_bytes = [702]
        exp_out_cells = [703]
        exp_last_out = [704]

        err_new = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
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
        err_ref = explicit_commit_only_preflight_only_composition(
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
            assert got_stage_cells == [601]
            assert got_stage_bytes == [602]
            assert got_out_cells == [603]
            assert got_last_out == [604]


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_commit_only_preflight_only_wrapper()
    test_known_vector_commit_only_preflight_only_outputs()
    test_error_paths_preserve_outputs()
    test_randomized_commit_only_preflight_only_vs_explicit_composition()
    print(
        "attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only=ok"
    )
