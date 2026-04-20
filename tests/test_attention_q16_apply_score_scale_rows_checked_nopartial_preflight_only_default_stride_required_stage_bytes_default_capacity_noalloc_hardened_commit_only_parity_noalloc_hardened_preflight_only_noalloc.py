#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyParityNoAllocHardenedPreflightOnlyNoAlloc."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only_noalloc(
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
        id(in_scores_q32),
        id(out_scores_q32),
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

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only(
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

    if snapshot != (
        id(in_scores_q32),
        id(out_scores_q32),
        in_scores_capacity,
        out_scores_capacity,
        row_count,
        token_count,
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = staged_required_in_cells[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_last_in_index[0] = staged_last_in_index[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def explicit_checked_composition(*args) -> int:
    (
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
    ) = args

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
        id(in_scores_q32),
        id(out_scores_q32),
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

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only(
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

    if snapshot != (
        id(in_scores_q32),
        id(out_scores_q32),
        in_scores_capacity,
        out_scores_capacity,
        row_count,
        token_count,
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = staged_required_in_cells[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_last_in_index[0] = staged_last_in_index[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_noalloc_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardenedCommitOnlyParityNoAllocHardenedPreflightOnlyNoAlloc("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardenedCommitOnlyParityNoAllocHardenedPreflightOnly(" in body
    assert "snapshot_in_scores_q32 != in_scores_q32" in body
    assert "snapshot_out_scores_q32 != out_scores_q32" in body
    assert "snapshot_in_scores_capacity != in_scores_capacity" in body
    assert "snapshot_out_scores_capacity != out_scores_capacity" in body
    assert "snapshot_row_count != row_count" in body
    assert "snapshot_token_count != token_count" in body


def test_null_overflow_capacity_sentinel_vectors() -> None:
    in_scores = [1] * 8
    out_scores = [2] * 8

    req_in = [777]
    req_out = [778]
    req_stage_cells = [779]
    req_stage_bytes = [780]
    last_in = [781]
    last_out = [782]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only_noalloc(
        None,
        len(in_scores),
        1,
        1,
        out_scores,
        len(out_scores),
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert req_in == [777]
    assert req_out == [778]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only_noalloc(
        in_scores,
        -1,
        1,
        1,
        out_scores,
        len(out_scores),
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert req_stage_cells == [779]
    assert req_stage_bytes == [780]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only_noalloc(
        in_scores,
        len(in_scores),
        1 << 62,
        4,
        out_scores,
        len(out_scores),
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err in (ATTN_Q16_ERR_BAD_PARAM, ATTN_Q16_ERR_OVERFLOW)
    assert last_in == [781]
    assert last_out == [782]


def test_randomized_parity() -> None:
    rng = random.Random(737)

    for _ in range(220):
        row_count = rng.randint(0, 24)
        token_count = rng.randint(0, 24)

        needed = row_count * token_count
        capacity = needed + rng.randint(0, 16)

        in_scores = [rng.randint(-200, 200) for _ in range(max(1, capacity))]
        out_scores = [rng.randint(-200, 200) for _ in range(max(1, capacity))]

        out_a = [[-1], [-2], [-3], [-4], [-5], [-6]]
        out_b = [x.copy() for x in out_a]

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity_noalloc_hardened_preflight_only_noalloc(
            in_scores,
            len(in_scores),
            row_count,
            token_count,
            out_scores,
            len(out_scores),
            out_a[0],
            out_a[1],
            out_a[2],
            out_a[3],
            out_a[4],
            out_a[5],
        )

        err_b = explicit_checked_composition(
            in_scores,
            len(in_scores),
            row_count,
            token_count,
            out_scores,
            len(out_scores),
            out_b[0],
            out_b[1],
            out_b[2],
            out_b[3],
            out_b[4],
            out_b[5],
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_noalloc_wrapper()
    test_null_overflow_capacity_sentinel_vectors()
    test_randomized_parity()
    print("ok")
