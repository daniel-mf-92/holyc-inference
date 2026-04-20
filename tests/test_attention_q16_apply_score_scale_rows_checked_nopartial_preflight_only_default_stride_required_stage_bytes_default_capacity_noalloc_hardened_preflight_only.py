#!/usr/bin/env python3
"""Parity harness for ...RequiredStageBytesDefaultCapacityNoAllocHardenedPreflightOnly."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened import (
    attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened,
)

I64_SIZE = 8


def attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
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

    snapshot_in_scores_capacity = in_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity
    snapshot_row_count = row_count
    snapshot_token_count = token_count

    canonical_required_in_cells = [0]
    canonical_required_out_cells = [0]
    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_last_in_index = [0]
    canonical_last_out_index = [0]

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

    err, recomputed_required_stage_cells = try_mul_i64(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64(recomputed_required_stage_cells, I64_SIZE)
    if err != ATTN_Q16_OK:
        return err

    if recomputed_required_stage_cells == 0:
        expected_last_index = 0
    else:
        err, expected_last_index = try_add_i64(recomputed_required_stage_cells, -1)
        if err != ATTN_Q16_OK:
            return err

    if snapshot_in_scores_capacity != in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_row_count != row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
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

    snapshot_in_scores_capacity = in_scores_capacity
    snapshot_out_scores_capacity = out_scores_capacity
    snapshot_row_count = row_count
    snapshot_token_count = token_count

    canonical_required_in_cells = [0]
    canonical_required_out_cells = [0]
    canonical_required_stage_cells = [0]
    canonical_required_stage_bytes = [0]
    canonical_last_in_index = [0]
    canonical_last_out_index = [0]

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

    err, recomputed_required_stage_cells = try_mul_i64(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64(recomputed_required_stage_cells, I64_SIZE)
    if err != ATTN_Q16_OK:
        return err

    if recomputed_required_stage_cells == 0:
        expected_last_index = 0
    else:
        err, expected_last_index = try_add_i64(recomputed_required_stage_cells, -1)
        if err != ATTN_Q16_OK:
            return err

    if snapshot_in_scores_capacity != in_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_scores_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_row_count != row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
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


def test_source_contains_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    sig = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardenedPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialPreflightOnlyDefaultStrideRequiredStageBytesDefaultCapacityNoAllocHardened(" in body
    assert "status = AttentionTryMulI64Checked(row_count," in body
    assert "status = AttentionTryMulI64Checked(recomputed_required_stage_cells," in body
    assert "status = AttentionTryAddI64Checked(recomputed_required_stage_cells," in body
    assert "canonical_required_stage_bytes != recomputed_required_stage_bytes" in body
    assert "canonical_last_in_index != expected_last_index" in body
    assert "canonical_last_out_index != expected_last_index" in body
    assert "snapshot_in_scores_capacity != in_scores_capacity" in body
    assert "snapshot_out_scores_capacity != out_scores_capacity" in body
    assert "snapshot_row_count != row_count" in body
    assert "snapshot_token_count != token_count" in body
    assert "*out_required_stage_bytes = canonical_required_stage_bytes;" in body


def test_known_vector_outputs_expected_diagnostics() -> None:
    row_count = 3
    token_count = 1
    required_stage_cells = row_count * token_count
    default_score_stride = token_count
    default_row_stride = token_count
    expected_last = (row_count - 1) * default_row_stride + (token_count - 1) * default_score_stride
    required_in_out_cells = expected_last + 1

    in_scores = [7] * required_in_out_cells
    out_scores = [9] * required_in_out_cells

    req_in = [100]
    req_out = [101]
    req_stage_cells = [102]
    req_stage_bytes = [103]
    last_in = [104]
    last_out = [105]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
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
    assert req_in == [required_in_out_cells]
    assert req_out == [required_in_out_cells]
    assert req_stage_cells == [required_stage_cells]
    assert req_stage_bytes == [required_stage_cells * I64_SIZE]
    assert last_in == [expected_last]
    assert last_out == [expected_last]


def test_zero_shape_reports_zero_last_indices() -> None:
    req_in = [1]
    req_out = [2]
    req_stage_cells = [3]
    req_stage_bytes = [4]
    last_in = [5]
    last_out = [6]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
        [0],
        1,
        0,
        9,
        [0],
        1,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )

    assert err == ATTN_Q16_OK
    assert req_in == [0]
    assert req_out == [0]
    assert req_stage_cells == [0]
    assert req_stage_bytes == [0]
    assert last_in == [0]
    assert last_out == [0]


def test_error_paths_preserve_outputs() -> None:
    req_in = [11]
    req_out = [12]
    req_stage_cells = [13]
    req_stage_bytes = [14]
    last_in = [15]
    last_out = [16]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
        None,
        0,
        1,
        1,
        [0],
        1,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR
    assert req_in == [11]
    assert req_out == [12]
    assert req_stage_cells == [13]
    assert req_stage_bytes == [14]
    assert last_in == [15]
    assert last_out == [16]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
        [0],
        -1,
        1,
        1,
        [0],
        1,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    huge = 1 << 62
    err = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
        [0],
        (1 << 63) - 1,
        huge,
        huge,
        [0],
        (1 << 63) - 1,
        req_in,
        req_out,
        req_stage_cells,
        req_stage_bytes,
        last_in,
        last_out,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_and_no_write_on_error() -> None:
    rng = random.Random(721)

    for _ in range(500):
        row_count = rng.randint(-2, 40)
        token_count = rng.randint(-2, 40)

        required = 0
        if row_count >= 0 and token_count >= 0:
            required = row_count * token_count

        in_capacity = required + rng.randint(0, 5)
        out_capacity = required + rng.randint(0, 5)
        if rng.random() < 0.15:
            in_capacity = rng.randint(-3, 3)
        if rng.random() < 0.15:
            out_capacity = rng.randint(-3, 3)

        in_scores = [0] * max(0, in_capacity)
        out_scores = [0] * max(0, out_capacity)

        req_in_a = [901]
        req_out_a = [902]
        req_stage_cells_a = [903]
        req_stage_bytes_a = [904]
        last_in_a = [905]
        last_out_a = [906]

        req_in_b = [901]
        req_out_b = [902]
        req_stage_cells_b = [903]
        req_stage_bytes_b = [904]
        last_in_b = [905]
        last_out_b = [906]

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_preflight_only_default_stride_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            req_in_a,
            req_out_a,
            req_stage_cells_a,
            req_stage_bytes_a,
            last_in_a,
            last_out_a,
        )
        err_b = explicit_checked_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            out_scores,
            out_capacity,
            req_in_b,
            req_out_b,
            req_stage_cells_b,
            req_stage_bytes_b,
            last_in_b,
            last_out_b,
        )

        assert err_a == err_b
        assert req_in_a == req_in_b
        assert req_out_a == req_out_b
        assert req_stage_cells_a == req_stage_cells_b
        assert req_stage_bytes_a == req_stage_bytes_b
        assert last_in_a == last_in_b
        assert last_out_a == last_out_b


if __name__ == "__main__":
    test_source_contains_required_stage_bytes_default_capacity_noalloc_hardened_preflight_only_helper()
    test_known_vector_outputs_expected_diagnostics()
    test_zero_shape_reports_zero_last_indices()
    test_error_paths_preserve_outputs()
    test_randomized_parity_and_no_write_on_error()
    print("ok")
