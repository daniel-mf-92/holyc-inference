#!/usr/bin/env python3
"""Parity harness for ...NoAllocHardenedCommitOnlyPreflightOnly (IQ-792)."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only_commit_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only_commit_only,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_preflight_only(
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
        out_required_in_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_last_in_index is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_in_cells is out_required_out_cells
        or out_required_in_cells is out_required_stage_cells
        or out_required_in_cells is out_required_stage_bytes
        or out_required_in_cells is out_last_in_index
        or out_required_in_cells is out_last_out_index
        or out_required_out_cells is out_required_stage_cells
        or out_required_out_cells is out_required_stage_bytes
        or out_required_out_cells is out_last_in_index
        or out_required_out_cells is out_last_out_index
        or out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_last_in_index
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_last_in_index
        or out_required_stage_bytes is out_last_out_index
        or out_last_in_index is out_last_out_index
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot = (
        in_scores_capacity,
        out_scores_capacity,
        row_count,
        token_count,
    )

    commit_required_in_cells = [0]
    commit_required_out_cells = [0]
    commit_required_stage_cells = [0]
    commit_required_stage_bytes = [0]
    commit_last_in_index = [0]
    commit_last_out_index = [0]

    preflight_required_in_cells = [0]
    preflight_required_out_cells = [0]
    preflight_required_stage_cells = [0]
    preflight_required_stage_bytes = [0]
    preflight_last_in_index = [0]
    preflight_last_out_index = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_required_in_cells,
        commit_required_out_cells,
        commit_required_stage_cells,
        commit_required_stage_bytes,
        commit_last_in_index,
        commit_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only_commit_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        preflight_required_in_cells,
        preflight_required_out_cells,
        preflight_required_stage_cells,
        preflight_required_stage_bytes,
        preflight_last_in_index,
        preflight_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot != (
        in_scores_capacity,
        out_scores_capacity,
        row_count,
        token_count,
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if commit_required_in_cells[0] != preflight_required_in_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_required_out_cells[0] != preflight_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_required_stage_cells[0] != preflight_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_required_stage_bytes[0] != preflight_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_last_in_index[0] != preflight_last_in_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_last_out_index[0] != preflight_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = preflight_required_in_cells[0]
    out_required_out_cells[0] = preflight_required_out_cells[0]
    out_required_stage_cells[0] = preflight_required_stage_cells[0]
    out_required_stage_bytes[0] = preflight_required_stage_bytes[0]
    out_last_in_index[0] = preflight_last_in_index[0]
    out_last_out_index[0] = preflight_last_out_index[0]
    return ATTN_Q16_OK


def explicit_staged_composition(
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
        out_required_in_cells is None
        or out_required_out_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_last_in_index is None
        or out_last_out_index is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_in_cells is out_required_out_cells
        or out_required_in_cells is out_required_stage_cells
        or out_required_in_cells is out_required_stage_bytes
        or out_required_in_cells is out_last_in_index
        or out_required_in_cells is out_last_out_index
        or out_required_out_cells is out_required_stage_cells
        or out_required_out_cells is out_required_stage_bytes
        or out_required_out_cells is out_last_in_index
        or out_required_out_cells is out_last_out_index
        or out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_last_in_index
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_last_in_index
        or out_required_stage_bytes is out_last_out_index
        or out_last_in_index is out_last_out_index
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot = (
        in_scores_capacity,
        out_scores_capacity,
        row_count,
        token_count,
    )

    staged_required_in_cells = [0]
    staged_required_out_cells = [0]
    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_last_in_index = [0]
    staged_last_out_index = [0]

    parity_required_in_cells = [0]
    parity_required_out_cells = [0]
    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_last_in_index = [0]
    parity_last_out_index = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        staged_required_in_cells,
        staged_required_out_cells,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_last_in_index,
        staged_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only_commit_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        parity_required_in_cells,
        parity_required_out_cells,
        parity_required_stage_cells,
        parity_required_stage_bytes,
        parity_last_in_index,
        parity_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot != (
        in_scores_capacity,
        out_scores_capacity,
        row_count,
        token_count,
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_in_cells[0] != parity_required_in_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != parity_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_cells[0] != parity_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != parity_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_in_index[0] != parity_last_in_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != parity_last_out_index[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = parity_required_in_cells[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_last_in_index[0] = parity_last_in_index[0]
    out_last_out_index[0] = parity_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_commit_only_preflight_only() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert (
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStride"
        "RequiredStageBytesDefaultCapacityNoAllocHardenedCommitOnlyPreflightOnly("
    ) in source
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStride"
        "RequiredStageBytesDefaultCapacityNoAllocHardenedCommitOnly("
    ) in source
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStride"
        "RequiredStageBytesDefaultCapacityNoAllocHardenedPreflightOnlyCommitOnly("
    ) in source
    assert "snapshot_in_scores_capacity" in source
    assert "commit_required_stage_bytes" in source
    assert "preflight_required_stage_bytes" in source


def test_commit_only_preflight_only_nullptr_and_alias_contracts() -> None:
    out_a = [111]
    out_b = [222]
    out_c = [333]
    out_d = [444]
    out_e = [555]
    out_f = [666]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_preflight_only(
        [0],
        1,
        1,
        1,
        [0],
        1,
        None,
        out_b,
        out_c,
        out_d,
        out_e,
        out_f,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_preflight_only(
        [0],
        1,
        1,
        1,
        [0],
        1,
        out_a,
        out_a,
        out_c,
        out_d,
        out_e,
        out_f,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM


def test_commit_only_preflight_only_overflow_propagation() -> None:
    out_required_in_cells = [0]
    out_required_out_cells = [0]
    out_required_stage_cells = [0]
    out_required_stage_bytes = [0]
    out_last_in_index = [0]
    out_last_out_index = [0]

    huge = (1 << 62)
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_preflight_only(
        [0],
        huge,
        huge,
        3,
        [0],
        huge,
        out_required_in_cells,
        out_required_out_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_last_in_index,
        out_last_out_index,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_commit_only_preflight_only_matches_explicit_staged_composition_randomized() -> None:
    rng = random.Random(20260420_792)

    for _ in range(1000):
        row_count = rng.randint(0, 128)
        token_count = rng.randint(0, 128)
        required = row_count * token_count
        slack_in = rng.randint(0, 8)
        slack_out = rng.randint(0, 8)

        in_scores_capacity = required + slack_in
        out_scores_capacity = required + slack_out

        in_scores = [0] * max(in_scores_capacity, 1)
        out_scores = [0] * max(out_scores_capacity, 1)

        a_required_in = [777]
        a_required_out = [777]
        a_required_stage = [777]
        a_required_bytes = [777]
        a_last_in = [777]
        a_last_out = [777]

        b_required_in = [888]
        b_required_out = [888]
        b_required_stage = [888]
        b_required_bytes = [888]
        b_last_in = [888]
        b_last_out = [888]

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_preflight_only(
            in_scores,
            in_scores_capacity,
            row_count,
            token_count,
            out_scores,
            out_scores_capacity,
            a_required_in,
            a_required_out,
            a_required_stage,
            a_required_bytes,
            a_last_in,
            a_last_out,
        )
        err_b = explicit_staged_composition(
            in_scores,
            in_scores_capacity,
            row_count,
            token_count,
            out_scores,
            out_scores_capacity,
            b_required_in,
            b_required_out,
            b_required_stage,
            b_required_bytes,
            b_last_in,
            b_last_out,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK:
            assert a_required_in[0] == b_required_in[0]
            assert a_required_out[0] == b_required_out[0]
            assert a_required_stage[0] == b_required_stage[0]
            assert a_required_bytes[0] == b_required_bytes[0]
            assert a_last_in[0] == b_last_in[0]
            assert a_last_out[0] == b_last_out[0]
        else:
            assert a_required_in == [777]
            assert a_required_out == [777]
            assert a_required_stage == [777]
            assert a_required_bytes == [777]
            assert a_last_in == [777]
            assert a_last_out == [777]


if __name__ == "__main__":
    test_source_contains_commit_only_preflight_only()
    test_commit_only_preflight_only_nullptr_and_alias_contracts()
    test_commit_only_preflight_only_overflow_propagation()
    test_commit_only_preflight_only_matches_explicit_staged_composition_randomized()
    print(
        "attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_"
        "default_stride_required_stage_bytes_default_capacity_noalloc_hardened_"
        "commit_only_preflight_only=ok"
    )
