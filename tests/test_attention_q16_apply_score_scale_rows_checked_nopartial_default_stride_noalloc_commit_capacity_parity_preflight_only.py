#!/usr/bin/env python3
"""Parity harness for ...NoAllocCommitCapacityParityPreflightOnly (IQ-866)."""

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
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
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
    parity_out_scores_q32,
    parity_out_scores_capacity: int,
    parity_staged_scores_q32,
    parity_staged_scores_capacity: int,
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

    if (
        in_scores_q32 is None
        or out_scores_q32 is None
        or staged_scores_q32 is None
        or parity_out_scores_q32 is None
        or parity_staged_scores_q32 is None
    ):
        return ATTN_Q16_ERR_NULL_PTR

    if (
        in_scores_capacity < 0
        or out_scores_capacity < 0
        or staged_scores_capacity < 0
        or parity_out_scores_capacity < 0
        or parity_staged_scores_capacity < 0
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if row_count < 0 or token_count < 0:
        return ATTN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        out_required_stage_cells is in_scores_q32
        or out_required_stage_cells is out_scores_q32
        or out_required_stage_cells is staged_scores_q32
        or out_required_stage_cells is parity_out_scores_q32
        or out_required_stage_cells is parity_staged_scores_q32
        or out_required_stage_bytes is in_scores_q32
        or out_required_stage_bytes is out_scores_q32
        or out_required_stage_bytes is staged_scores_q32
        or out_required_stage_bytes is parity_out_scores_q32
        or out_required_stage_bytes is parity_staged_scores_q32
        or out_required_out_cells is in_scores_q32
        or out_required_out_cells is out_scores_q32
        or out_required_out_cells is staged_scores_q32
        or out_required_out_cells is parity_out_scores_q32
        or out_required_out_cells is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_row_count = row_count
    snapshot_token_count = token_count
    snapshot_score_scale_q16 = score_scale_q16
    snapshot_commit_stage_cell_capacity = commit_stage_cell_capacity
    snapshot_commit_stage_byte_capacity = commit_stage_byte_capacity
    snapshot_out_scores_capacity = out_scores_capacity
    snapshot_staged_scores_capacity = staged_scores_capacity
    snapshot_parity_out_scores_capacity = parity_out_scores_capacity
    snapshot_parity_staged_scores_capacity = parity_staged_scores_capacity

    preflight_required_stage_cells = [0]
    preflight_required_stage_bytes = [0]
    preflight_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        preflight_required_stage_cells,
        preflight_required_stage_bytes,
        preflight_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    parity_preflight_required_stage_cells = [0]
    parity_preflight_required_stage_bytes = [0]
    parity_preflight_required_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        parity_out_scores_q32,
        parity_out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
        parity_preflight_required_stage_cells,
        parity_preflight_required_stage_bytes,
        parity_preflight_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    required_required_stage_cells = [0]
    required_required_stage_bytes = [0]
    required_required_out_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        snapshot_row_count,
        snapshot_token_count,
        out_scores_q32,
        snapshot_out_scores_capacity,
        snapshot_commit_stage_cell_capacity,
        snapshot_commit_stage_byte_capacity,
        required_required_stage_cells,
        required_required_stage_bytes,
        required_required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    required_in_cells = [0]
    recomputed_required_out_cells = [0]
    recomputed_required_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        snapshot_row_count,
        snapshot_token_count,
        out_scores_q32,
        snapshot_out_scores_capacity,
        snapshot_commit_stage_cell_capacity,
        required_in_cells,
        recomputed_required_out_cells,
        recomputed_required_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, recomputed_required_bytes = try_mul_i64_checked(recomputed_required_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err

    if required_in_cells[0] != recomputed_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if (
        snapshot_row_count != row_count
        or snapshot_token_count != token_count
        or snapshot_score_scale_q16 != score_scale_q16
        or snapshot_commit_stage_cell_capacity != commit_stage_cell_capacity
        or snapshot_commit_stage_byte_capacity != commit_stage_byte_capacity
        or snapshot_out_scores_capacity != out_scores_capacity
        or snapshot_staged_scores_capacity != staged_scores_capacity
        or snapshot_parity_out_scores_capacity != parity_out_scores_capacity
        or snapshot_parity_staged_scores_capacity != parity_staged_scores_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    if preflight_required_stage_cells[0] != recomputed_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_out_cells[0] != recomputed_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_stage_bytes[0] != recomputed_required_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if parity_preflight_required_stage_cells[0] != recomputed_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_preflight_required_out_cells[0] != recomputed_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_preflight_required_stage_bytes[0] != recomputed_required_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if preflight_required_stage_cells[0] != parity_preflight_required_stage_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_stage_bytes[0] != parity_preflight_required_stage_bytes[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_out_cells[0] != parity_preflight_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if required_required_stage_cells[0] != recomputed_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_required_out_cells[0] != recomputed_required_out_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_required_stage_bytes[0] != recomputed_required_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if preflight_required_stage_cells[0] > snapshot_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_stage_cells[0] > snapshot_parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_out_cells[0] > snapshot_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if preflight_required_out_cells[0] > snapshot_parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    if snapshot_row_count != 0 and snapshot_token_count != 0:
        err, scratch_out_bytes = try_mul_i64_checked(preflight_required_out_cells[0], 8)
        if err != ATTN_Q16_OK:
            return err

        err, scratch_stage_bytes = try_mul_i64_checked(preflight_required_stage_cells[0], 8)
        if err != ATTN_Q16_OK:
            return err

        if scratch_out_bytes <= 0 or scratch_stage_bytes <= 0:
            return ATTN_Q16_ERR_OVERFLOW

        scratch_out_scores_q32 = [0] * preflight_required_out_cells[0]
        scratch_staged_scores_q32 = [0] * preflight_required_stage_cells[0]
        scratch_parity_out_scores_q32 = [0] * preflight_required_out_cells[0]
        scratch_parity_staged_scores_q32 = [0] * preflight_required_stage_cells[0]

        parity_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
            in_scores_q32,
            in_scores_capacity,
            snapshot_row_count,
            snapshot_token_count,
            snapshot_score_scale_q16,
            scratch_out_scores_q32,
            preflight_required_out_cells[0],
            snapshot_commit_stage_cell_capacity,
            snapshot_commit_stage_byte_capacity,
            scratch_staged_scores_q32,
            preflight_required_stage_cells[0],
            scratch_parity_out_scores_q32,
            preflight_required_out_cells[0],
            scratch_parity_staged_scores_q32,
            preflight_required_stage_cells[0],
        )
        if parity_status != ATTN_Q16_OK:
            return parity_status

    out_required_stage_cells[0] = preflight_required_stage_cells[0]
    out_required_stage_bytes[0] = preflight_required_stage_bytes[0]
    out_required_out_cells[0] = preflight_required_out_cells[0]
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
    parity_out_scores_q32,
    parity_out_scores_capacity: int,
    parity_staged_scores_q32,
    parity_staged_scores_capacity: int,
    out_required_stage_cells: list[int],
    out_required_stage_bytes: list[int],
    out_required_out_cells: list[int],
) -> int:
    staged_stage_cells = [0]
    staged_stage_bytes = [0]
    staged_out_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staged_scores_q32,
        staged_scores_capacity,
        staged_stage_cells,
        staged_stage_bytes,
        staged_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    parity_stage_cells = [0]
    parity_stage_bytes = [0]
    parity_out_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        parity_out_scores_q32,
        parity_out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
        parity_stage_cells,
        parity_stage_bytes,
        parity_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    base_required_in = [0]
    base_required_out = [0]
    base_required_cells = [0]
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        out_scores_q32,
        out_scores_capacity,
        commit_stage_cell_capacity,
        base_required_in,
        base_required_out,
        base_required_cells,
    )
    if err != ATTN_Q16_OK:
        return err

    err, base_required_bytes = try_mul_i64_checked(base_required_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err

    if base_required_in[0] != base_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if staged_stage_cells[0] != base_required_cells[0] or parity_stage_cells[0] != base_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_stage_bytes[0] != base_required_bytes or parity_stage_bytes[0] != base_required_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if staged_out_cells[0] != base_required_out[0] or parity_out_cells[0] != base_required_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    if required_stage_cells[0] != base_required_cells[0]:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] != base_required_bytes:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] != base_required_out[0]:
        return ATTN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_stage_cells[0]
    out_required_stage_bytes[0] = staged_stage_bytes[0]
    out_required_out_cells[0] = staged_out_cells[0]
    return ATTN_Q16_OK


def test_source_contains_parity_preflight_only_wrapper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = (
        "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityParityPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityPreflightOnly(" in body
    )
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(" in body
    )
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly(" in body
    )
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityParity(" in body
    )
    assert "snapshot_row_count = row_count;" in body
    assert "snapshot_token_count = token_count;" in body
    assert "snapshot_score_scale_q16 = score_scale_q16;" in body


def test_known_vector_outputs_and_no_write() -> None:
    row_count = 3
    token_count = 4
    total = row_count * token_count

    in_scores = [17 - i for i in range(total)]
    out_scores = [991] * total
    staged_scores = [992] * total
    parity_out_scores = [993] * total
    parity_staged_scores = [994] * total

    got_stage_cells = [101]
    got_stage_bytes = [102]
    got_out_cells = [103]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores,
        len(in_scores),
        row_count,
        token_count,
        2 * 65536,
        out_scores,
        len(out_scores),
        total,
        total * 8,
        staged_scores,
        len(staged_scores),
        parity_out_scores,
        len(parity_out_scores),
        parity_staged_scores,
        len(parity_staged_scores),
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    assert err == ATTN_Q16_OK
    assert got_stage_cells == [total]
    assert got_stage_bytes == [total * 8]
    assert got_out_cells == [total]
    assert out_scores == [991] * total
    assert staged_scores == [992] * total
    assert parity_out_scores == [993] * total
    assert parity_staged_scores == [994] * total


def test_adversarial_null_alias_bounds_and_overflow() -> None:
    in_scores = [1, 2, 3, 4]
    out_scores = [9, 9, 9, 9]
    staged_scores = [8, 8, 8, 8]
    parity_out_scores = [7, 7, 7, 7]
    parity_staged_scores = [6, 6, 6, 6]

    out_stage_cells = [0]
    out_stage_bytes = [0]
    out_out_cells = [0]

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        None,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        4,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        4,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        None,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_NULL_PTR

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        4,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_cells,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores,
        4,
        1,
        4,
        0,
        out_scores,
        4,
        -1,
        32,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
        in_scores,
        4,
        I64_MAX,
        2,
        0,
        out_scores,
        4,
        I64_MAX,
        I64_MAX,
        staged_scores,
        4,
        parity_out_scores,
        4,
        parity_staged_scores,
        4,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(866)

    for _ in range(220):
        row_count = rng.randint(0, 8)
        token_count = rng.randint(0, 8)

        err, total = try_mul_i64_checked(row_count, token_count)
        assert err == ATTN_Q16_OK

        in_len = total + rng.randint(0, 2)
        out_len = total + rng.randint(0, 2)
        staged_len = total + rng.randint(0, 2)
        parity_out_len = total + rng.randint(0, 2)
        parity_staged_len = total + rng.randint(0, 2)

        in_scores = [rng.randint(-500, 500) for _ in range(in_len)]
        out_scores_a = [rng.randint(-1000, 1000) for _ in range(out_len)]
        out_scores_b = list(out_scores_a)
        staged_a = [rng.randint(-1000, 1000) for _ in range(staged_len)]
        staged_b = list(staged_a)
        parity_out_a = [rng.randint(-1000, 1000) for _ in range(parity_out_len)]
        parity_out_b = list(parity_out_a)
        parity_staged_a = [rng.randint(-1000, 1000) for _ in range(parity_staged_len)]
        parity_staged_b = list(parity_staged_a)

        stage_cell_cap = total + rng.randint(0, 2)
        stage_byte_cap = stage_cell_cap * 8
        score_scale = rng.randint(-3, 3) * 65536

        got_stage_cells = [111]
        got_stage_bytes = [222]
        got_out_cells = [333]

        exp_stage_cells = [444]
        exp_stage_bytes = [555]
        exp_out_cells = [666]

        got = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity_preflight_only(
            in_scores,
            len(in_scores),
            row_count,
            token_count,
            score_scale,
            out_scores_a,
            len(out_scores_a),
            stage_cell_cap,
            stage_byte_cap,
            staged_a,
            len(staged_a),
            parity_out_a,
            len(parity_out_a),
            parity_staged_a,
            len(parity_staged_a),
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )

        expected = explicit_preflight_only_composition(
            in_scores,
            len(in_scores),
            row_count,
            token_count,
            score_scale,
            out_scores_b,
            len(out_scores_b),
            stage_cell_cap,
            stage_byte_cap,
            staged_b,
            len(staged_b),
            parity_out_b,
            len(parity_out_b),
            parity_staged_b,
            len(parity_staged_b),
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert got == expected
        if got == ATTN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        assert out_scores_a == out_scores_b
        assert staged_a == staged_b
        assert parity_out_a == parity_out_b
        assert parity_staged_a == parity_staged_b
