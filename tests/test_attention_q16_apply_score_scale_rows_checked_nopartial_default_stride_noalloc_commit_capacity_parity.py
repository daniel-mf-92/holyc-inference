#!/usr/bin/env python3
"""Parity harness for ...NoAllocCommitCapacityParity."""

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
from test_attention_q16_apply_score_scale_rows_checked_default_stride import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_default_stride,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only,
)
from test_attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (  # noqa: E402
    attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)


def attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
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
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if (
        staged_scores_q32 is None
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
        staged_scores_q32 is in_scores_q32
        or staged_scores_q32 is out_scores_q32
        or staged_scores_q32 is parity_out_scores_q32
        or staged_scores_q32 is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        parity_out_scores_q32 is in_scores_q32
        or parity_out_scores_q32 is out_scores_q32
        or parity_out_scores_q32 is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_staged_scores_q32 is in_scores_q32 or parity_staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

    snapshot_row_count = row_count
    snapshot_token_count = token_count
    snapshot_score_scale_q16 = score_scale_q16
    snapshot_commit_stage_cell_capacity = commit_stage_cell_capacity
    snapshot_commit_stage_byte_capacity = commit_stage_byte_capacity

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

    if required_stage_cells[0] < 0 or required_stage_bytes[0] < 0 or required_out_cells[0] < 0:
        return ATTN_Q16_ERR_BAD_PARAM

    err, recomputed_required_stage_bytes = try_mul_i64_checked(required_stage_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err
    if required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    canonical_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
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
    )
    if canonical_status != ATTN_Q16_OK:
        return canonical_status

    explicit_status = attention_q16_apply_score_scale_rows_checked_default_stride(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if explicit_status != canonical_status:
        return ATTN_Q16_ERR_BAD_PARAM
    if explicit_status != ATTN_Q16_OK:
        return explicit_status

    explicit_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
    )
    if explicit_status != canonical_status:
        return ATTN_Q16_ERR_BAD_PARAM

    for elem_index in range(required_out_cells[0]):
        if parity_staged_scores_q32[elem_index] != parity_out_scores_q32[elem_index]:
            return ATTN_Q16_ERR_BAD_PARAM

    if (
        snapshot_row_count != row_count
        or snapshot_token_count != token_count
        or snapshot_score_scale_q16 != score_scale_q16
        or snapshot_commit_stage_cell_capacity != commit_stage_cell_capacity
        or snapshot_commit_stage_byte_capacity != commit_stage_byte_capacity
    ):
        return ATTN_Q16_ERR_BAD_PARAM

    publish_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )
    if publish_status != canonical_status:
        return ATTN_Q16_ERR_BAD_PARAM

    for elem_index in range(required_out_cells[0]):
        if out_scores_q32[elem_index] != parity_out_scores_q32[elem_index]:
            return ATTN_Q16_ERR_BAD_PARAM

    return publish_status


def explicit_checked_parity_composition(
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
) -> int:
    if in_scores_q32 is None or out_scores_q32 is None:
        return ATTN_Q16_ERR_NULL_PTR
    if (
        staged_scores_q32 is None
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
        staged_scores_q32 is in_scores_q32
        or staged_scores_q32 is out_scores_q32
        or staged_scores_q32 is parity_out_scores_q32
        or staged_scores_q32 is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if (
        parity_out_scores_q32 is in_scores_q32
        or parity_out_scores_q32 is out_scores_q32
        or parity_out_scores_q32 is parity_staged_scores_q32
    ):
        return ATTN_Q16_ERR_BAD_PARAM
    if parity_staged_scores_q32 is in_scores_q32 or parity_staged_scores_q32 is out_scores_q32:
        return ATTN_Q16_ERR_BAD_PARAM

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

    err, recomputed_required_stage_bytes = try_mul_i64_checked(required_stage_cells[0], 8)
    if err != ATTN_Q16_OK:
        return err
    if required_stage_bytes[0] != recomputed_required_stage_bytes:
        return ATTN_Q16_ERR_BAD_PARAM

    if row_count == 0 or token_count == 0:
        return ATTN_Q16_OK

    if required_stage_cells[0] > staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_stage_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > parity_out_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM
    if required_out_cells[0] > parity_staged_scores_capacity:
        return ATTN_Q16_ERR_BAD_PARAM

    canonical_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
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
    )
    if canonical_status != ATTN_Q16_OK:
        return canonical_status

    explicit_status = attention_q16_apply_score_scale_rows_checked_default_stride(
        in_scores_q32,
        in_scores_capacity,
        row_count,
        token_count,
        score_scale_q16,
        staged_scores_q32,
        staged_scores_capacity,
    )
    if explicit_status != canonical_status:
        return ATTN_Q16_ERR_BAD_PARAM
    if explicit_status != ATTN_Q16_OK:
        return explicit_status

    explicit_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        parity_staged_scores_q32,
        parity_staged_scores_capacity,
    )
    if explicit_status != canonical_status:
        return ATTN_Q16_ERR_BAD_PARAM

    for elem_index in range(required_out_cells[0]):
        if parity_staged_scores_q32[elem_index] != parity_out_scores_q32[elem_index]:
            return ATTN_Q16_ERR_BAD_PARAM

    publish_status = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        token_count,
        staged_scores_q32,
        staged_scores_capacity,
        required_stage_cells[0],
        out_scores_q32,
        out_scores_capacity,
    )
    if publish_status != canonical_status:
        return ATTN_Q16_ERR_BAD_PARAM

    for elem_index in range(required_out_cells[0]):
        if out_scores_q32[elem_index] != parity_out_scores_q32[elem_index]:
            return ATTN_Q16_ERR_BAD_PARAM

    return publish_status


def test_source_contains_noalloc_commit_capacity_parity_helper() -> None:
    source = Path("src/model/attention.HC").read_text(encoding="utf-8")
    signature = "I32 AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityParity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )
    assert (
        "AttentionQ16ApplyScoreScaleRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity("
        in body
    )
    assert "AttentionQ16ApplyScoreScaleRowsCheckedDefaultStride(" in body
    assert "if (explicit_status != canonical_status)" in body
    assert "if (publish_status != canonical_status)" in body


def test_known_vectors_match_and_publish_parity() -> None:
    row_count = 4
    token_count = 1
    required = row_count * token_count
    score_scale_q16 = 23170

    in_scores = [((i * 19) - 93) << 13 for i in range(required)]
    out_a = [0x5151] * required
    out_b = [0x5151] * required
    stage_a = [0x3A3A] * required
    stage_b = [0x3A3A] * required
    parity_out_a = [0x7171] * required
    parity_out_b = [0x7171] * required
    parity_stage_a = [0x2424] * required
    parity_stage_b = [0x2424] * required

    err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_a,
        required,
        required,
        required * 8,
        stage_a,
        required,
        parity_out_a,
        required,
        parity_stage_a,
        required,
    )
    err_b = explicit_checked_parity_composition(
        in_scores,
        required,
        row_count,
        token_count,
        score_scale_q16,
        out_b,
        required,
        required,
        required * 8,
        stage_b,
        required,
        parity_out_b,
        required,
        parity_stage_b,
        required,
    )

    assert err_a == err_b
    if err_a == ATTN_Q16_OK:
        assert out_a == out_b


def test_alias_rejection_and_no_partial() -> None:
    row_count = 3
    token_count = 1
    required = row_count * token_count

    in_scores = [17] * required
    out_scores_before = [0xAAAA] * required
    out_scores_after = out_scores_before.copy()

    stage = [0x1111] * required
    parity_out = [0x2222] * required
    parity_stage = [0x3333] * required

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
        in_scores,
        required,
        row_count,
        token_count,
        32768,
        out_scores_after,
        required,
        required,
        required * 8,
        out_scores_after,
        required,
        parity_out,
        required,
        parity_stage,
        required,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_scores_after == out_scores_before

    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
        in_scores,
        required,
        row_count,
        token_count,
        32768,
        out_scores_after,
        required,
        required,
        required * 8,
        stage,
        required,
        out_scores_after,
        required,
        parity_stage,
        required,
    )
    assert err == ATTN_Q16_ERR_BAD_PARAM
    assert out_scores_after == out_scores_before


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260421_861)

    for _ in range(4000):
        row_count = rng.randint(0, 20)
        token_count = rng.randint(0, 20)
        score_scale_q16 = rng.randint(-65536, 65536)

        if rng.random() < 0.06:
            row_count = -rng.randint(1, 30)
        if rng.random() < 0.06:
            token_count = -rng.randint(1, 30)

        required_stage = row_count * token_count if row_count >= 0 and token_count >= 0 else 0
        required_out = 0
        if row_count > 0 and token_count > 0:
            required_out = (row_count - 1) * token_count + ((token_count - 1) * token_count + 1)

        in_capacity = max(0, required_stage + rng.randint(-15, 15))
        out_capacity = max(0, required_out + rng.randint(-15, 15))
        stage_capacity = max(0, required_stage + rng.randint(-15, 15))
        parity_out_capacity = max(0, required_out + rng.randint(-15, 15))
        parity_stage_capacity = max(0, max(required_stage, required_out) + rng.randint(-15, 15))

        if rng.random() < 0.06:
            in_capacity = -rng.randint(1, 30)
        if rng.random() < 0.06:
            out_capacity = -rng.randint(1, 30)
        if rng.random() < 0.06:
            stage_capacity = -rng.randint(1, 30)
        if rng.random() < 0.06:
            parity_out_capacity = -rng.randint(1, 30)
        if rng.random() < 0.06:
            parity_stage_capacity = -rng.randint(1, 30)

        commit_stage_cell_capacity = max(0, required_stage + rng.randint(-15, 15))
        commit_stage_byte_capacity = max(0, (required_stage + rng.randint(-15, 15)) * 8)
        if rng.random() < 0.08:
            commit_stage_cell_capacity = -rng.randint(1, 30)
        if rng.random() < 0.08:
            commit_stage_byte_capacity = -rng.randint(1, 500)

        in_scores = [rng.randint(-(1 << 40), (1 << 40) - 1) for _ in range(max(1, in_capacity))]
        out_a = [rng.randint(-20000, 20000) for _ in range(max(1, out_capacity))]
        out_b = list(out_a)
        stage_a = [rng.randint(-20000, 20000) for _ in range(max(1, stage_capacity))]
        stage_b = list(stage_a)
        parity_out_a = [rng.randint(-20000, 20000) for _ in range(max(1, parity_out_capacity))]
        parity_out_b = list(parity_out_a)
        parity_stage_a = [rng.randint(-20000, 20000) for _ in range(max(1, parity_stage_capacity))]
        parity_stage_b = list(parity_stage_a)

        in_ptr = in_scores if rng.random() > 0.03 else None
        out_ptr_a = out_a if rng.random() > 0.03 else None
        out_ptr_b = out_ptr_a
        stage_ptr_a = stage_a if rng.random() > 0.03 else None
        stage_ptr_b = stage_b if stage_ptr_a is not None else None
        parity_out_ptr_a = parity_out_a if rng.random() > 0.03 else None
        parity_out_ptr_b = parity_out_ptr_a
        parity_stage_ptr_a = parity_stage_a if rng.random() > 0.03 else None
        parity_stage_ptr_b = parity_stage_b if parity_stage_ptr_a is not None else None

        err_a = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
            in_ptr,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_ptr_a,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_ptr_a,
            stage_capacity,
            parity_out_ptr_a,
            parity_out_capacity,
            parity_stage_ptr_a,
            parity_stage_capacity,
        )
        err_b = explicit_checked_parity_composition(
            in_ptr,
            in_capacity,
            row_count,
            token_count,
            score_scale_q16,
            out_ptr_b,
            out_capacity,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_ptr_b,
            stage_capacity,
            parity_out_ptr_b,
            parity_out_capacity,
            parity_stage_ptr_b,
            parity_stage_capacity,
        )

        assert err_a == err_b
        if err_a == ATTN_Q16_OK and out_ptr_a is not None and out_ptr_b is not None:
            assert out_ptr_a == out_ptr_b

    huge = 1 << 62
    err = attention_q16_apply_score_scale_rows_checked_nopartial_default_stride_noalloc_commit_capacity_parity(
        [1],
        I64_MAX,
        huge,
        huge,
        123,
        [0],
        I64_MAX,
        I64_MAX,
        I64_MAX,
        [0],
        I64_MAX,
        [0],
        I64_MAX,
        [0],
        I64_MAX,
    )
    assert err == ATTN_Q16_ERR_OVERFLOW
