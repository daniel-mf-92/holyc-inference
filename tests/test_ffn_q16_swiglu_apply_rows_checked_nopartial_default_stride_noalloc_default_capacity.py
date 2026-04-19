#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_default_stride as rows_default
from test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only import (
    ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only,
)
from test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity import (
    ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity,
)

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
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

    err, default_stage_cell_capacity = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        row_count,
        lane_count,
        default_stage_cell_capacity,
        staging_capacity_bytes,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

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

    return ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells[0],
        out_q16,
        out_capacity,
    )


def explicit_checked_default_capacity_noalloc_composition(
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

    err, default_stage_cell_capacity = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        row_count,
        lane_count,
        default_stage_cell_capacity,
        staging_capacity_bytes,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

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

    return ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only(
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        required_stage_cells[0],
        out_q16,
        out_capacity,
    )


def test_source_contains_noalloc_default_capacity_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = FFNTryMulI64Checked(row_count," in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity(" in body
    assert "if (staging_out_q16 == gate_q16)" in body
    assert "return FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(" in body


def test_known_vectors_and_stage_capacity_contract() -> None:
    row_count = 5
    lane_count = 7
    required = row_count * lane_count

    gate = [0] * required
    up = [0] * required
    for i in range(required):
        gate[i] = ((i - 11) * 3) << 13
        up[i] = (17 - i) << 12

    out_a = [0x6A6A] * required
    out_b = [0x6A6A] * required
    stage = [0x1111] * required

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
        gate,
        required,
        up,
        required,
        out_a,
        required,
        row_count,
        lane_count,
        stage,
        required,
    )
    err_b = explicit_checked_default_capacity_noalloc_composition(
        gate,
        required,
        up,
        required,
        out_b,
        required,
        row_count,
        lane_count,
        stage.copy(),
        required,
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b

    out_c = [0x2222] * required
    err_c = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
        gate,
        required,
        up,
        required,
        out_c,
        required,
        row_count,
        lane_count,
        [0] * (required - 1),
        required - 1,
    )
    assert err_c == FFN_Q16_ERR_BAD_PARAM


def test_error_paths_and_overflow() -> None:
    gate = [0, 0, 0, 0]
    up = [0, 0, 0, 0]
    out = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
            None, 4, up, 4, out, 4, 2, 2, stage, 4
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
            gate, 4, up, 4, out, 4, 2, 2, stage, -1
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
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


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260420_609)

    for _ in range(1200):
        row_count = rng.randint(0, 12)
        lane_count = rng.randint(0, 12)
        required = row_count * lane_count

        gate_capacity = required
        up_capacity = required
        out_capacity = required

        gate = [rng.randint(-(8 << 16), (8 << 16)) for _ in range(max(required, 1))]
        up = [rng.randint(-(8 << 16), (8 << 16)) for _ in range(max(required, 1))]
        out_a = [0x7575] * max(required, 1)
        out_b = [0x7575] * max(required, 1)

        stage_capacity = max(0, required + rng.randint(-3, 6))
        stage_a = [0x0909] * max(stage_capacity, 1)
        stage_b = stage_a.copy()

        if required > 0 and rng.random() < 0.12:
            gate[rng.randint(0, required - 1)] = rows_core.I64_MIN

        if rng.random() < 0.05:
            gate_capacity = -1
        if rng.random() < 0.05:
            out_capacity = -1
        if rng.random() < 0.05:
            row_count = -rng.randint(1, 8)

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_default_capacity(
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
        err_b = explicit_checked_default_capacity_noalloc_composition(
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
    test_source_contains_noalloc_default_capacity_helper()
    test_known_vectors_and_stage_capacity_contract()
    test_error_paths_and_overflow()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
