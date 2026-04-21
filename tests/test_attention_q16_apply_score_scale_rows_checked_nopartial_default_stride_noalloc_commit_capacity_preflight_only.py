#!/usr/bin/env python3
"""Parity harness for ...NoAllocCommitCapacityPreflightOnly (IQ-860)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_attention_q16_apply_score_scale_checked import (  # noqa: E402
    ATTN_Q16_ERR_BAD_PARAM,
    ATTN_Q16_ERR_NULL_PTR,
    ATTN_Q16_ERR_OVERFLOW,
    ATTN_Q16_OK,
    I64_MAX,
    try_mul_i64_checked,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_required_out_cells
        or out_required_stage_bytes is out_required_out_cells
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR

    if in_scores_capacity < 0 or out_scores_capacity < 0 or staged_scores_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_stage_cells is in_scores_q32
        or out_required_stage_cells is out_scores_q32
        or out_required_stage_cells is staged_scores_q32
        or out_required_stage_bytes is in_scores_q32
        or out_required_stage_bytes is out_scores_q32
        or out_required_stage_bytes is staged_scores_q32
        or out_required_out_cells is in_scores_q32
        or out_required_out_cells is out_scores_q32
        or out_required_out_cells is staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_row_count = row_count
    snapshot_token_count = token_count
    snapshot_stage_capacity = staged_scores_capacity
    snapshot_out_capacity = out_scores_capacity

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(
        snapshot_row_count, snapshot_token_count
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64_checked(
        recomputed_required_stage_cells, 8
    )
    if err != ATTN_Q16_OK:
        return err

    if snapshot_row_count != row_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_token_count != token_count:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_stage_capacity != staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if snapshot_out_capacity != out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] > snapshot_stage_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > snapshot_out_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]

    _ = score_scale_q16
    return ATTN_Q16_OK


def explicit_preflight_only_composition(
    in_scores_q32,
    in_scores_capacity: int,
    row_count: int,
    token_count: int,
    score_scale_q16: int,
    out_scores_q32,
    out_scores_capacity: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staged_scores_q32,
    staged_scores_capacity: int,
    out_required_stage_cells: list[int],
    out_required_stage_bytes: list[int],
    out_required_out_cells: list[int],
) -> int:
    _ = score_scale_q16
    _ = staged_scores_q32

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_cells = try_mul_i64_checked(row_count, token_count)
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_stage_bytes = try_mul_i64_checked(
        recomputed_required_stage_cells, 8
    )
    if err != ATTN_Q16_OK:
        return err

    if staged_required_stage_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] != recomputed_required_stage_cells:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_commit_capacity_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(" in body
    )
    assert "snapshot_row_count = row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_stage_capacity = staged_scores_capacity;" in body
    assert "snapshot_out_capacity = out_scores_capacity;" in body


def test_known_vector_outputs() -> None:
    row_count = 11
    token_count = 7
    total = row_count * token_count

    in_scores = [0] * total
    out_scores = [123] * total
    staged = [0] * total

    got_stage_cells = [9]
    got_stage_bytes = [10]
    got_out_cells = [11]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        3 * 65536,
        out_scores,
        len(out_scores),
        total,
        total * 8,
        staged,
        len(staged),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    assert err == ATTN_Q16_OK
    assert got_stage_cells == [total]
    assert got_stage_bytes == [total * 8]
    assert got_out_cells == [total]
    assert out_scores == [123] * total


def test_adversarial_null_alias_and_bounds() -> None:
    in_scores = [1, 2, 3, 4]
    out_scores = [0, 0, 0, 0]
    staged = [0, 0, 0, 0]

    out_stage_cells = [0]
    out_stage_bytes = [0]
    out_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        None,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        4,
        32,
        staged,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        4,
        32,
        staged,
        4,
        out_stage_cells,
        out_stage_cells,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        4,
        32,
        staged,
        2,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores,
        4,
        I64_MAX,
        2,
        0,
        out_scores,
        4,
        4,
        32,
        staged,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(860)

    for _ in range(300):
        row_count = rng.randint(0, 48)
        token_count = rng.randint(0, 48)
        total = row_count * token_count

        in_capacity = total + rng.randint(0, 6)
        out_capacity = total + rng.randint(0, 6)
        stage_capacity = total + rng.randint(0, 6)

        in_scores = [rng.randint(-2000, 2000) for _ in range(in_capacity)]
        out_scores_a = [rng.randint(-2000, 2000) for _ in range(out_capacity)]
        out_scores_b = list(out_scores_a)
        staged = [0] * stage_capacity

        commit_stage_cell_capacity = total + rng.randint(0, 6)
        commit_stage_byte_capacity = (total * 8) + (8 * rng.randint(0, 6))

        if rng.random() < 0.2 and total > 0:
            stage_capacity = max(0, total - rng.randint(1, min(total, 3)))
            staged = [0] * stage_capacity

        got_stage_cells = [-1]
        got_stage_bytes = [-1]
        got_out_cells = [-1]

        exp_stage_cells = [-1]
        exp_stage_bytes = [-1]
        exp_out_cells = [-1]

        err_got = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            rng.randint(-8, 8) * 65536,
            out_scores_a,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged,
            stage_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )

        err_exp = explicit_preflight_only_composition(
            in_scores,
            in_capacity,
            row_count,
            token_count,
            0,
            out_scores_b,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged,
            stage_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        assert out_scores_a == out_scores_b

        if err_got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells


if __name__ == "__main__":
    test_source_contains_commit_capacity_preflight_only_wrapper()
    test_known_vector_outputs()
    test_adversarial_null_alias_and_bounds()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
