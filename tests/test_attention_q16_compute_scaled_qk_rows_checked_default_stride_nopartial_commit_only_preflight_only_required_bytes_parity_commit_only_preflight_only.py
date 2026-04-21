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


def explicit_preflight_only_composition(*args) -> int:
    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]
    staged_last_out_index = [0]

    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_required_out_cells = [0]
    canonical_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        args[0], args[1], args[2], args[3], args[4], args[5],
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
        staged_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity(
        args[0], args[1], args[2], args[3], args[4], args[5],
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

    args[6][0] = staged_required_stage_cells[0]
    args[7][0] = staged_required_stage_bytes[0]
    args[8][0] = staged_required_out_cells[0]
    args[9][0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_required_bytes_parity_commit_only_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "RequiredBytesParityCommitOnly(" in body
    assert "RequiredBytesParity(" in body
    assert "if (staged_required_stage_cells != canonical_required_stage_cells)" in body
    assert "if (staged_required_stage_bytes != canonical_required_stage_bytes)" in body
    assert "if (staged_required_out_cells != canonical_required_out_cells)" in body
    assert "if (staged_last_out_index != canonical_last_out_index)" in body


def test_known_vector_outputs_and_error_no_publish() -> None:
    rows = 7
    tokens = 6
    total = rows * tokens
    staged = [0] * total
    out = [0] * total

    got_cells = [11]
    got_bytes = [12]
    got_out_cells = [13]
    got_last_out = [14]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        rows, tokens, staged, len(staged), out, len(out), got_cells, got_bytes, got_out_cells, got_last_out
    )
    assert err == ATTN_Q16_OK
    assert got_cells == [total]
    assert got_bytes == [total * 8]
    assert got_out_cells == [total]
    assert got_last_out == [total - 1]

    keep_cells = [901]
    keep_bytes = [902]
    keep_out_cells = [903]
    keep_last = [904]
    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
        -1, 4, staged, len(staged), out, len(out), keep_cells, keep_bytes, keep_out_cells, keep_last
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert keep_cells == [901]
    assert keep_bytes == [902]
    assert keep_out_cells == [903]
    assert keep_last == [904]


def test_randomized_preflight_only_vs_explicit_composition() -> None:
    rng = random.Random(20260421_841)
    for _ in range(1400):
        rows = rng.randint(0, 96)
        tokens = rng.randint(0, 96)
        required = rows * tokens
        staged_cap = required + rng.randint(0, 8)
        out_cap = required + rng.randint(0, 8)

        staged = [0] * max(staged_cap, 1)
        out = [0] * max(out_cap, 1)

        got_cells = [701]
        got_bytes = [702]
        got_out_cells = [703]
        got_last_out = [704]

        exp_cells = [801]
        exp_bytes = [802]
        exp_out_cells = [803]
        exp_last_out = [804]

        err_new = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_preflight_only(
            rows, tokens, staged, staged_cap, out, out_cap, got_cells, got_bytes, got_out_cells, got_last_out
        )
        err_ref = explicit_preflight_only_composition(
            rows, tokens, staged, staged_cap, out, out_cap, exp_cells, exp_bytes, exp_out_cells, exp_last_out
        )

        assert err_new == err_ref
        if err_new == ATTN_Q16_OK:
            assert got_cells == exp_cells
            assert got_bytes == exp_bytes
            assert got_out_cells == exp_out_cells
            assert got_last_out == exp_last_out
        else:
            assert got_cells == [701]
            assert got_bytes == [702]
            assert got_out_cells == [703]
            assert got_last_out == [704]


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_commit_only_preflight_only_wrapper()
    test_known_vector_outputs_and_error_no_publish()
    test_randomized_preflight_only_vs_explicit_composition()
    print("ok")
