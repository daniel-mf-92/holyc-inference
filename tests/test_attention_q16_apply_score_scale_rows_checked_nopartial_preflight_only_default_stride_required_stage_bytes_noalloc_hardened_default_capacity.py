#!/usr/bin/env python3
"""Parity harness for ...RequiredStageBytesNoAllocHardenedDefaultCapacity (IQ-720)."""

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
    try_add_i64,
    try_mul_i64,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened_default_capacity(
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

    snapshot = (in_scores_capacity, out_scores_capacity, row_count, token_count)

    canonical_required_in_cells = [0]
    canonical_required_out_cells = [0]
    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_last_in_index = [0]
    canonical_last_out_index = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened(
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

    err, recomputed_required_stage_cells = try_mul_i64(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64(recomputed_required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    if recomputed_required_stage_cells == 0:
        expected_last_index = 0
    else:
        err, expected_last_index = try_add_i64(recomputed_required_stage_cells, -1)
        if err != ATTN_Q16_OK:
            return err

    if snapshot != (in_scores_capacity, out_scores_capacity, row_count, token_count):
        return ATTN_Q16_ERR_BAD_PARAM

    if canonical_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_required_in_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_last_in_index[0] != expected_last_index:
        return ATTN_Q16_ERR_BAD_PARAM
    if canonical_last_out_index[0] != expected_last_index:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = canonical_required_in_cells[0]
    out_required_out_cells[0] = canonical_required_out_cells[0]
    out_required_stage_cells[0] = canonical_required_stage_cells[0]
    out_required_stage_bytes[0] = canonical_required_stage_bytes[0]
    out_last_in_index[0] = canonical_last_in_index[0]
    out_last_out_index[0] = canonical_last_out_index[0]
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

    snapshot = (in_scores_capacity, out_scores_capacity, row_count, token_count)

    staged_required_in_cells = [0]
    staged_required_out_cells = [0]
    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_last_in_index = [0]
    staged_last_out_index = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened(
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

    err, recomputed_required_stage_cells = try_mul_i64(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64(recomputed_required_stage_cells, 8)
    if err != ATTN_Q16_OK:
        return err

    if recomputed_required_stage_cells == 0:
        expected_last_index = 0
    else:
        err, expected_last_index = try_add_i64(recomputed_required_stage_cells, -1)
        if err != ATTN_Q16_OK:
            return err

    if snapshot != (in_scores_capacity, out_scores_capacity, row_count, token_count):
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_in_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_in_index[0] != expected_last_index:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_last_out_index[0] != expected_last_index:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = staged_required_in_cells[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_last_in_index[0] = staged_last_in_index[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return ATTN_Q16_OK


def run_wrapper(fn, *, in_capacity, rows, tokens, out_capacity):
    in_scores = [0] * max(in_capacity, 0)
    out_scores = [0] * max(out_capacity, 0)
    out_required_in = [111]
    out_required_out = [222]
    out_required_stage = [333]
    out_required_stage_bytes = [444]
    out_last_in = [555]
    out_last_out = [666]

    status = fn(
        in_scores,
        in_capacity,
        rows,
        tokens,
        out_scores,
        out_capacity,
        out_required_in,
        out_required_out,
        out_required_stage,
        out_required_stage_bytes,
        out_last_in,
        out_last_out,
    )
    return (
        status,
        out_required_in,
        out_required_out,
        out_required_stage,
        out_required_stage_bytes,
        out_last_in,
        out_last_out,
    )


def test_source_contains_default_capacity_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    assert (
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesNoAllocHardenedDefaultCapacity("
        in source
    )
    assert "AttentionTryMulI64Checked(row_count," in source
    assert "snapshot_row_count = row_count;" in source
    assert "snapshot_token_count = token_count;" in source
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesNoAllocHardened("
        in source
    )


def test_default_capacity_wrapper_matches_explicit_checked_composition_randomized() -> None:
    rng = random.Random(20260420_720)

    for _ in range(1400):
        rows = rng.randint(0, 128)
        tokens = rng.randint(0, 128)

        err_cells, required_cells = try_mul_i64(rows, tokens)
        if err_cells != ATTN_Q16_OK:
            required_cells = 0

        in_capacity = required_cells + rng.randint(0, 3)
        out_capacity = required_cells + rng.randint(0, 3)

        got = run_wrapper(
            attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened_default_capacity,
            in_capacity=in_capacity,
            rows=rows,
            tokens=tokens,
            out_capacity=out_capacity,
        )

        expected = run_wrapper(
            explicit_checked_composition,
            in_capacity=in_capacity,
            rows=rows,
            tokens=tokens,
            out_capacity=out_capacity,
        )

        assert got == expected


def test_default_capacity_wrapper_error_and_no_partial_contracts() -> None:
    sentinel_in = [41]
    sentinel_out = [42]
    sentinel_stage = [43]
    sentinel_stage_bytes = [44]
    sentinel_last_in = [45]
    sentinel_last_out = [46]

    status = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened_default_capacity(
        [0] * 4,
        -1,
        2,
        2,
        [0] * 4,
        4,
        sentinel_in,
        sentinel_out,
        sentinel_stage,
        sentinel_stage_bytes,
        sentinel_last_in,
        sentinel_last_out,
    )
    assert status == ATTN_Q16_ERR_BAD_PARAM
    assert sentinel_in == [41]
    assert sentinel_out == [42]
    assert sentinel_stage == [43]
    assert sentinel_stage_bytes == [44]
    assert sentinel_last_in == [45]
    assert sentinel_last_out == [46]

    status = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened_default_capacity(
        [],
        (1 << 63) - 1,
        1,
        1 << 50,
        [],
        (1 << 63) - 1,
        sentinel_in,
        sentinel_out,
        sentinel_stage,
        sentinel_stage_bytes,
        sentinel_last_in,
        sentinel_last_out,
    )
    assert status == ATTN_Q16_ERR_OVERFLOW
    assert sentinel_in == [41]
    assert sentinel_out == [42]
    assert sentinel_stage == [43]
    assert sentinel_stage_bytes == [44]
    assert sentinel_last_in == [45]
    assert sentinel_last_out == [46]


def test_default_capacity_wrapper_null_ptr_classification() -> None:
    status = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened_default_capacity(
        None,
        0,
        0,
        0,
        [],
        0,
        [0],
        [0],
        [0],
        [0],
        [0],
        [0],
    )
    assert status == ATTN_Q16_ERR_NULL_PTR


if __name__ == "__main__":
    test_source_contains_default_capacity_wrapper()
    test_default_capacity_wrapper_matches_explicit_checked_composition_randomized()
    test_default_capacity_wrapper_error_and_no_partial_contracts()
    test_default_capacity_wrapper_null_ptr_classification()
    print(
        "attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_noalloc_hardened_default_capacity=ok"
    )
