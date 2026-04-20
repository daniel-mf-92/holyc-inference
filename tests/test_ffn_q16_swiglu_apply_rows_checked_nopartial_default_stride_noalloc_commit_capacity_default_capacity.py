#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
from test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity import (
    ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity,
)

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    staging_out_q16,
    staging_out_capacity: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    err, default_commit_stage_cell_capacity = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, default_commit_stage_byte_capacity = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    return ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staging_out_q16,
        staging_out_capacity,
    )


def explicit_checked_default_capacity_commit_composition(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    staging_out_q16,
    staging_out_capacity: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    err, default_commit_stage_cell_capacity = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, default_commit_stage_byte_capacity = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    return ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        default_commit_stage_cell_capacity,
        default_commit_stage_byte_capacity,
        staging_out_q16,
        staging_out_capacity,
    )


def test_source_contains_default_capacity_commit_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNTryMulI64Checked(row_count," in body
    assert "FFNTryMulI64Checked(staging_out_capacity," in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity(" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    row_count = 6
    lane_count = 5
    required = row_count * lane_count

    gate = [0] * required
    up = [0] * required
    for i in range(required):
        gate[i] = ((i * 5) - 17) << 12
        up[i] = (31 - (i * 3)) << 11

    out_a = [0x6B6B] * required
    out_b = out_a.copy()
    stage_a = [0x1111] * (required + 4)
    stage_b = stage_a.copy()

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        gate,
        required,
        up,
        required,
        out_a,
        required,
        row_count,
        lane_count,
        stage_a,
        len(stage_a),
    )
    err_b = explicit_checked_default_capacity_commit_composition(
        gate,
        required,
        up,
        required,
        out_b,
        required,
        row_count,
        lane_count,
        stage_b,
        len(stage_b),
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b


def test_staging_capacity_shortfall_is_no_partial() -> None:
    row_count = 4
    lane_count = 7
    required = row_count * lane_count

    gate = [0] * required
    up = [0] * required
    out = [0x3C3C] * required
    out_before = out.copy()

    stage = [0x2222] * (required - 1)
    stage_before = stage.copy()

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        gate,
        required,
        up,
        required,
        out,
        required,
        row_count,
        lane_count,
        stage,
        len(stage),
    )

    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert stage == stage_before


def test_error_paths_and_overflow() -> None:
    gate = [0, 0, 0, 0]
    up = [0, 0, 0, 0]
    out = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            None, 4, up, 4, out, 4, 2, 2, stage, 4
        )
        == FFN_Q16_ERR_NULL_PTR
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            gate, 4, up, 4, out, 4, 2, 2, stage, -1
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        [0],
        I64_MAX,
        [0],
        I64_MAX,
        [0],
        I64_MAX,
        huge,
        huge,
        [0],
        I64_MAX,
    )
    assert err == FFN_Q16_ERR_OVERFLOW

    # Default byte-capacity derivation must overflow-check
    # `staging_out_capacity * sizeof(I64)` independently of row/lane geometry.
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        gate,
        4,
        up,
        4,
        out,
        4,
        1,
        1,
        stage,
        (I64_MAX // 8) + 1,
    )
    assert err == FFN_Q16_ERR_OVERFLOW


def test_alias_rejection_is_no_partial() -> None:
    row_count = 3
    lane_count = 4
    required = row_count * lane_count

    gate = [i << 10 for i in range(required)]
    up = [((i * 3) - 5) << 11 for i in range(required)]
    out = [0x4F4F] * required
    out_before = out.copy()

    # staging aliases out: must be rejected and preserve destination.
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
        gate,
        required,
        up,
        required,
        out,
        required,
        row_count,
        lane_count,
        out,
        len(out),
    )

    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260420_620)

    for _ in range(2000):
        row_count = rng.randint(0, 12)
        lane_count = rng.randint(0, 12)
        required = row_count * lane_count

        gate_capacity = required + rng.randint(0, 4)
        up_capacity = required + rng.randint(0, 4)
        out_capacity = required + rng.randint(0, 6)
        stage_capacity = max(0, required + rng.randint(-3, 8))

        gate = [rng.randint(-(8 << 16), (8 << 16)) for _ in range(max(gate_capacity, 1))]
        up = [rng.randint(-(8 << 16), (8 << 16)) for _ in range(max(up_capacity, 1))]

        out_a = [rng.randint(-(2 << 16), (2 << 16)) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()

        stage_a = [rng.randint(-(2 << 16), (2 << 16)) for _ in range(max(stage_capacity, 1))]
        stage_b = stage_a.copy()

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_a,
            out_capacity,
            row_count,
            lane_count,
            stage_a,
            stage_capacity,
        )
        err_b = explicit_checked_default_capacity_commit_composition(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_b,
            out_capacity,
            row_count,
            lane_count,
            stage_b,
            stage_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_default_capacity_commit_wrapper()
    test_known_vectors_match_explicit_checked_composition()
    test_staging_capacity_shortfall_is_no_partial()
    test_error_paths_and_overflow()
    test_alias_rejection_is_no_partial()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
