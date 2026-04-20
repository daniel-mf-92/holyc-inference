#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityNoAllocHardenedCommitOnlyParity."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity(
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

    canonical_required_in_cells = [0]
    canonical_required_out_cells = [0]
    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_last_in_index = [0]
    canonical_last_out_index = [0]

    parity_required_in_cells = [0]
    parity_required_out_cells = [0]
    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_last_in_index = [0]
    parity_last_out_index = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        canonical_required_in_cells,
        canonical_required_out_cells,
        canonical_required_stage_cells,
        canonical_required_stage_bytes,
        canonical_last_in_index,
        canonical_last_out_index,
    )
    if err != ATTN_Q16_OK:
        return err

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
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

    if (
        canonical_required_in_cells[0] != parity_required_in_cells[0]
        or canonical_required_out_cells[0] != parity_required_out_cells[0]
        or canonical_required_stage_cells[0] != parity_required_stage_cells[0]
        or canonical_required_stage_bytes[0] != parity_required_stage_bytes[0]
        or canonical_last_in_index[0] != parity_last_in_index[0]
        or canonical_last_out_index[0] != parity_last_out_index[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = parity_required_in_cells[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_last_in_index[0] = parity_last_in_index[0]
    out_last_out_index[0] = parity_last_out_index[0]
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

    canonical_required_in_cells = [0]
    canonical_required_out_cells = [0]
    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_last_in_index = [0]
    canonical_last_out_index = [0]

    parity_required_in_cells = [0]
    parity_required_out_cells = [0]
    parity_required_stage_cells = [0]
    parity_required_stage_bytes = [0]
    parity_last_in_index = [0]
    parity_last_out_index = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
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

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        canonical_required_in_cells,
        canonical_required_out_cells,
        canonical_required_stage_cells,
        canonical_required_stage_bytes,
        canonical_last_in_index,
        canonical_last_out_index,
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

    if (
        canonical_required_in_cells[0] != parity_required_in_cells[0]
        or canonical_required_out_cells[0] != parity_required_out_cells[0]
        or canonical_required_stage_cells[0] != parity_required_stage_cells[0]
        or canonical_required_stage_bytes[0] != parity_required_stage_bytes[0]
        or canonical_last_in_index[0] != parity_last_in_index[0]
        or canonical_last_out_index[0] != parity_last_out_index[0]
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = parity_required_in_cells[0]
    out_required_out_cells[0] = parity_required_out_cells[0]
    out_required_stage_cells[0] = parity_required_stage_cells[0]
    out_required_stage_bytes[0] = parity_required_stage_bytes[0]
    out_last_in_index[0] = parity_last_in_index[0]
    out_last_out_index[0] = parity_last_out_index[0]
    return ATTN_Q16_OK


def test_source_contains_default_capacity_noalloc_hardened_commit_only_parity() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardenedCommitOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardened(" in body
    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardenedPreflightOnly(" in body
    assert "snapshot_in_scores_capacity != in_scores_capacity" in body
    assert "snapshot_out_scores_capacity != out_scores_capacity" in body
    assert "snapshot_row_count != row_count" in body
    assert "snapshot_token_count != token_count" in body


def test_known_vector_and_immutability() -> None:
    in_scores = [7] * 64
    out_scores = [9] * 64

    req_in = [100]
    req_out = [101]
    req_stage_cells = [102]
    req_stage_bytes = [103]
    last_in = [104]
    last_out = [105]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity(
        in_scores,
        len(in_scores),
        3,
        5,
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
    assert req_in == [15]
    assert req_out == [15]
    assert req_stage_cells == [15]
    assert req_stage_bytes == [120]
    assert last_in == [14]
    assert last_out == [14]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity(
        in_scores,
        len(in_scores),
        -1,
        5,
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
    assert req_in == [15]
    assert req_out == [15]
    assert req_stage_cells == [15]
    assert req_stage_bytes == [120]
    assert last_in == [14]
    assert last_out == [14]


def test_randomized_parity() -> None:
    rng = random.Random(733)

    for _ in range(200):
        row_count = rng.randint(0, 24)
        token_count = rng.randint(0, 24)

        needed = row_count * token_count
        capacity = needed + rng.randint(0, 16)

        in_scores = [rng.randint(-200, 200) for _ in range(max(1, capacity))]
        out_scores = [rng.randint(-200, 200) for _ in range(max(1, capacity))]

        out_a = [[-1], [-2], [-3], [-4], [-5], [-6]]
        out_b = [[-11], [-12], [-13], [-14], [-15], [-16]]

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_commit_only_parity(
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
    test_source_contains_default_capacity_noalloc_hardened_commit_only_parity()
    test_known_vector_and_immutability()
    test_randomized_parity()
    print("ok")
