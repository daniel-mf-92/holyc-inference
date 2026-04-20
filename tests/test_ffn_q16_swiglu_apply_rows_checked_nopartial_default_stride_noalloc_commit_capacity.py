#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_default_stride as rows_default
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only as commit_only
from test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (
    ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staging_out_q16,
    staging_out_capacity: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if (
        gate_capacity < 0
        or up_capacity < 0
        or out_capacity < 0
        or staging_out_capacity < 0
    ):
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if required_out_cells[0] > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    if staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR
    if required_stage_cells[0] > staging_out_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] > staging_capacity_bytes:
        return FFN_Q16_ERR_BAD_PARAM

    if staging_out_q16 is gate_q16:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_q16 is up_q16:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_q16 is out_q16:
        return FFN_Q16_ERR_BAD_PARAM

    err = rows_default.ffn_q16_swiglu_apply_rows_checked_default_stride(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        staging_out_q16,
        staging_out_capacity,
        row_count,
        lane_count,
    )
    if err != FFN_Q16_OK:
        return err

    return commit_only.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells[0],
        out_q16,
        out_capacity,
    )


def explicit_checked_commit_capacity_composition(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staging_out_q16,
    staging_out_capacity: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if (
        gate_capacity < 0
        or up_capacity < 0
        or out_capacity < 0
        or staging_out_capacity < 0
    ):
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if required_out_cells[0] > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    if staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR
    if required_stage_cells[0] > staging_out_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_bytes[0] > staging_capacity_bytes:
        return FFN_Q16_ERR_BAD_PARAM

    if staging_out_q16 is gate_q16:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_q16 is up_q16:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_q16 is out_q16:
        return FFN_Q16_ERR_BAD_PARAM

    err = rows_default.ffn_q16_swiglu_apply_rows_checked_default_stride(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        staging_out_q16,
        staging_out_capacity,
        row_count,
        lane_count,
    )
    if err != FFN_Q16_OK:
        return err

    return commit_only.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells[0],
        out_q16,
        out_capacity,
    )


def test_source_contains_noalloc_commit_capacity_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert (
        "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
        in body
    )
    assert "FFNQ16SwiGLUApplyRowsCheckedDefaultStride(" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(" in body
    assert "if (required_out_cells > out_capacity)" in body
    assert "if (required_stage_cells > staging_out_capacity)" in body
    assert "if (required_stage_bytes > staging_capacity_bytes)" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    row_count = 5
    lane_count = 7

    required = row_count * lane_count
    stage_bytes = required * 8

    gate = [((i * 13) - 51) << 11 for i in range(required)]
    up = [((37 - (i * 7)) << 10) for i in range(required)]

    out_a = [0x5C5C] * required
    out_b = [0x5C5C] * required
    stage_a = [0x3A3A] * required
    stage_b = [0x3A3A] * required

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        gate,
        required,
        up,
        required,
        out_a,
        required,
        row_count,
        lane_count,
        required,
        stage_bytes,
        stage_a,
        required,
    )
    err_b = explicit_checked_commit_capacity_composition(
        gate,
        required,
        up,
        required,
        out_b,
        required,
        row_count,
        lane_count,
        required,
        stage_bytes,
        stage_b,
        required,
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b


def test_commit_capacity_rejection_is_no_partial() -> None:
    row_count = 4
    lane_count = 6
    required = row_count * lane_count
    required_bytes = required * 8

    gate = [1 << 16] * required
    up = [1 << 16] * required
    out = [0x7171] * required
    out_before = out.copy()
    stage = [0x2626] * required
    stage_before = stage.copy()

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        gate,
        required,
        up,
        required,
        out,
        required,
        row_count,
        lane_count,
        required,
        required_bytes - 8,
        stage,
        required,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert stage == stage_before


def test_out_capacity_rejection_is_no_partial_and_leaves_stage_untouched() -> None:
    row_count = 3
    lane_count = 5
    required = row_count * lane_count
    required_bytes = required * 8

    gate = [((i * 9) - 13) << 12 for i in range(required)]
    up = [((17 - i) << 11) for i in range(required)]
    out = [0x7B7B] * (required - 1)
    out_before = out.copy()
    stage = [0x2C2C] * required
    stage_before = stage.copy()

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        gate,
        required,
        up,
        required,
        out,
        required - 1,
        row_count,
        lane_count,
        required,
        required_bytes,
        stage,
        required,
    )

    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert stage == stage_before


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_612)

    for _ in range(3000):
        row_count = rng.randint(0, 14)
        lane_count = rng.randint(0, 14)
        required = row_count * lane_count

        gate_capacity = required
        up_capacity = required
        out_capacity = required
        staging_capacity = required

        if required > 0 and rng.random() < 0.25:
            gate_capacity = required - 1
        if required > 0 and rng.random() < 0.25:
            up_capacity = required - 1
        if required > 0 and rng.random() < 0.25:
            out_capacity = required - 1
        if required > 0 and rng.random() < 0.25:
            staging_capacity = required - 1

        commit_stage_cell_capacity = max(0, required + rng.randint(-40, 40))
        commit_stage_byte_capacity = max(0, (required * 8) + rng.randint(-320, 320))

        if rng.random() < 0.05:
            row_count = -rng.randint(1, 64)
        if rng.random() < 0.05:
            lane_count = -rng.randint(1, 64)
        if rng.random() < 0.05:
            commit_stage_cell_capacity = -rng.randint(1, 64)
        if rng.random() < 0.05:
            commit_stage_byte_capacity = -rng.randint(1, 64)

        live_cells = max(required, 1)
        gate_a = [rng.randint(-(10 << 16), (10 << 16)) for _ in range(live_cells)]
        up_a = [rng.randint(-(10 << 16), (10 << 16)) for _ in range(live_cells)]
        gate_b = gate_a.copy()
        up_b = up_a.copy()
        out_a = [0x4F4F] * live_cells
        out_b = out_a.copy()
        stage_a = [0x0E0E] * live_cells
        stage_b = stage_a.copy()

        if required > 0 and rng.random() < 0.16:
            bad_index = rng.randint(0, required - 1)
            gate_a[bad_index] = rows_core.I64_MIN
            gate_b[bad_index] = rows_core.I64_MIN

        alias_mode = rng.random()
        if alias_mode < 0.04:
            staged_a = None
            staged_b = None
        elif alias_mode < 0.14:
            staged_a = gate_a
            staged_b = gate_b
        elif alias_mode < 0.24:
            staged_a = up_a
            staged_b = up_b
        elif alias_mode < 0.34:
            staged_a = out_a
            staged_b = out_b
        else:
            staged_a = stage_a
            staged_b = stage_b

        err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
            gate_a,
            gate_capacity,
            up_a,
            up_capacity,
            out_a,
            out_capacity,
            row_count,
            lane_count,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_a,
            staging_capacity,
        )
        err_ref = explicit_checked_commit_capacity_composition(
            gate_b,
            gate_capacity,
            up_b,
            up_capacity,
            out_b,
            out_capacity,
            row_count,
            lane_count,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            staged_b,
            staging_capacity,
        )

        assert err_new == err_ref
        assert out_a == out_b


def test_overflow_passthrough() -> None:
    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity(
        [0],
        1,
        [0],
        1,
        [0],
        1,
        1,
        1,
        1,
        8,
        [0],
        huge,
    )
    assert err in (FFN_Q16_ERR_OVERFLOW, FFN_Q16_ERR_BAD_PARAM)


if __name__ == "__main__":
    test_source_contains_noalloc_commit_capacity_wrapper()
    test_known_vectors_match_explicit_checked_composition()
    test_commit_capacity_rejection_is_no_partial()
    test_out_capacity_rejection_is_no_partial_and_leaves_stage_untouched()
    test_randomized_parity_against_explicit_composition()
    test_overflow_passthrough()
    print("ok")
