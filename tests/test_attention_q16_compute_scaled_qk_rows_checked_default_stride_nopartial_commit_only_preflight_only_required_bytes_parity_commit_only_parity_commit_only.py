#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesParityCommitOnlyParityCommitOnly (IQ-846)."""

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
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only,
)
from test_attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity import (
    attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity,
)


def attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
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
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity

    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_required_out_cells = [0]
    parity_last_out_index = [0]

    commit_required_stage_cells = [0]
    commit_required_stage_bytes = [0]
    commit_required_out_cells = [0]
    commit_last_out_index = [0]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        parity_required_stage_cells,
        parity_required_stage_bytes,
        parity_required_out_cells,
        parity_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only(
        query_row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        out_scores_q32,
        out_scores_capacity,
        commit_required_stage_cells,
        commit_required_stage_bytes,
        commit_required_out_cells,
        commit_last_out_index,
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

    if parity_required_stage_cells[0] != commit_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_required_stage_bytes[0] != commit_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_required_out_cells[0] != commit_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_last_out_index[0] != commit_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_last_out_index[0] = parity_last_out_index[0]
    return ATTN_Q16_OK


def explicit_commit_only_parity_commit_only_composition(*args) -> int:
    return attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
        *args
    )


def test_source_contains_required_bytes_parity_commit_only_parity_commit_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyParityCommitOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnlyParity(" in body
    )
    assert (
        "AttentionQ16ComputeScaledQKRowsCheckedDefaultStrideNoPartialCommitOnlyPreflightOnlyRequiredBytesParityCommitOnly(" in body
    )
    assert "snapshot_query_row_count = query_row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_staged_scores_capacity = staged_scores_capacity;" in body
    assert "snapshot_out_scores_capacity = out_scores_capacity;" in body


def test_known_vector_outputs_and_alias_reject() -> None:
    query_row_count = 8
    token_count = 7
    total = query_row_count * token_count

    staged = [0] * total
    out = [0] * total

    got_stage_cells = [1001]
    got_stage_bytes = [1002]
    got_out_cells = [1003]
    got_last_out = [1004]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
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

    alias = [0xABCD]
    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
        query_row_count,
        token_count,
        staged,
        len(staged),
        out,
        len(out),
        alias,
        alias,
        [1],
        [2],
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_error_paths_preserve_outputs() -> None:
    staged = [0] * 24
    out = [0] * 24

    got_stage_cells = [801]
    got_stage_bytes = [802]
    got_out_cells = [803]
    got_last_out = [804]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
        -1,
        3,
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
    assert got_stage_cells == [801]
    assert got_stage_bytes == [802]
    assert got_out_cells == [803]
    assert got_last_out == [804]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
        3,
        3,
        None,
        9,
        out,
        len(out),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
        got_last_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR


def test_alias_safety_preserve_outputs() -> None:
    staged = [0] * 8
    out = [0] * 8

    got_stage_cells = [991]
    got_stage_bytes = [992]
    got_out_cells = [993]
    got_last_out = [994]

    err = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
        2,
        3,
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
    assert got_stage_bytes == [992]
    assert got_out_cells == [993]
    assert got_last_out == [994]


def test_fuzz_parity_vs_explicit_composition() -> None:
    random.seed(20260421_846)

    for i in range(2500):
        query_row_count = random.randint(0, 12)
        token_count = random.randint(0, 12)
        total = query_row_count * token_count

        pad = random.randint(0, 24)
        staged_scores_capacity = total + pad
        out_scores_capacity = total + pad

        staged_scores_q32 = [0] * staged_scores_capacity
        out_scores_q32_a = [0] * out_scores_capacity
        out_scores_q32_b = list(out_scores_q32_a)

        a_stage_cells = [0x11]
        a_stage_bytes = [0x22]
        a_out_cells = [0x33]
        a_last_out = [0x44]

        b_stage_cells = [0x11]
        b_stage_bytes = [0x22]
        b_out_cells = [0x33]
        b_last_out = [0x44]

        if i % 9 == 0 and staged_scores_capacity > 0:
            staged_scores_capacity = random.randint(0, max(0, total - 1))
        if i % 11 == 0 and out_scores_capacity > 0:
            out_scores_capacity = random.randint(0, max(0, total - 1))

        err_a = attention_q16_compute_scaled_qk_rows_checked_default_stride_nopartial_commit_only_preflight_only_required_bytes_parity_commit_only_parity_commit_only(
            query_row_count,
            token_count,
            staged_scores_q32,
            staged_scores_capacity,
            out_scores_q32_a,
            out_scores_capacity,
            a_stage_cells,
            a_stage_bytes,
            a_out_cells,
            a_last_out,
        )
        err_b = explicit_commit_only_parity_commit_only_composition(
            query_row_count,
            token_count,
            staged_scores_q32,
            staged_scores_capacity,
            out_scores_q32_b,
            out_scores_capacity,
            b_stage_cells,
            b_stage_bytes,
            b_out_cells,
            b_last_out,
        )

        assert err_a == err_b
        assert a_stage_cells == b_stage_cells
        assert a_stage_bytes == b_stage_bytes
        assert a_out_cells == b_out_cells
        assert a_last_out == b_last_out


if __name__ == "__main__":
    test_source_contains_required_bytes_parity_commit_only_parity_commit_only_wrapper()
    test_known_vector_outputs_and_alias_reject()
    test_error_paths_preserve_outputs()
    test_alias_safety_preserve_outputs()
    test_fuzz_parity_vs_explicit_composition()
    print("ok")
