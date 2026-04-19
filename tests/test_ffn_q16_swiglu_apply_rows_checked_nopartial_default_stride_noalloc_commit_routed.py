#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc commit routing."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_default_stride as rows_default
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_only as commit_only
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only as noalloc_preflight

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_routed(
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

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = noalloc_preflight.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        staging_out_capacity,
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
    if staging_out_capacity < required_stage_cells[0]:
        return FFN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err
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


def explicit_commit_routed_composition(
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

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = noalloc_preflight.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        staging_out_capacity,
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
    if staging_out_capacity < required_stage_cells[0]:
        return FFN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err
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


def test_source_contains_commit_routed_noalloc_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "required_stage_bytes" in body
    assert "FFNTryMulI64Checked(staging_out_capacity," in body
    assert "required_stage_bytes > staging_capacity_bytes" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitOnly(" in body
    assert "required_stage_cells," in body


def test_known_vectors_wrapper_matches_commit_routed_composition() -> None:
    row_count = 6
    lane_count = 5
    required = row_count * lane_count

    gate = [0] * required
    up = [0] * required
    for i in range(required):
        gate[i] = ((i - 9) * 5) << 12
        up[i] = ((14 - i) * 3) << 12

    out_new = [0x3535] * required
    out_ref = out_new.copy()
    stage_new = [0x7A7A] * required
    stage_ref = [0x7A7A] * required

    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_routed(
        gate,
        required,
        up,
        required,
        out_new,
        required,
        row_count,
        lane_count,
        stage_new,
        required,
    )
    err_ref = explicit_commit_routed_composition(
        gate,
        required,
        up,
        required,
        out_ref,
        required,
        row_count,
        lane_count,
        stage_ref,
        required,
    )

    assert err_new == err_ref == FFN_Q16_OK
    assert out_new == out_ref


def test_commit_routed_error_paths_preserve_output() -> None:
    row_count = 4
    lane_count = 4
    required = row_count * lane_count

    gate = [1 << 16] * required
    up = [1 << 16] * required
    gate[5] = rows_core.I64_MIN

    out_new = [0x6262] * required
    out_ref = out_new.copy()
    stage_new = [0x5151] * required
    stage_ref = [0x5151] * required

    err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_routed(
        gate,
        required,
        up,
        required,
        out_new,
        required,
        row_count,
        lane_count,
        stage_new,
        required,
    )
    err_ref = explicit_commit_routed_composition(
        gate,
        required,
        up,
        required,
        out_ref,
        required,
        row_count,
        lane_count,
        stage_ref,
        required,
    )

    assert err_new == err_ref
    assert err_new != FFN_Q16_OK
    assert out_new == out_ref == [0x6262] * required


def test_randomized_parity_wrapper_vs_commit_routed_composition() -> None:
    rng = random.Random(0xFF603)

    for _ in range(1400):
        row_count = rng.randint(0, 11)
        lane_count = rng.randint(0, 11)
        required = row_count * lane_count

        gate = [0] * max(required, 1)
        up = [0] * max(required, 1)
        for i in range(required):
            gate[i] = rng.randint(-(8 << 16), (8 << 16))
            up[i] = rng.randint(-(8 << 16), (8 << 16))

        if required > 0 and rng.random() < 0.17:
            gate[rng.randint(0, required - 1)] = rows_core.I64_MIN

        out_new = [0x4E4E] * max(required, 1)
        out_ref = out_new.copy()
        stage_new = [0x0909] * max(required, 1)
        stage_ref = [0x0909] * max(required, 1)

        stage_capacity = required
        if required > 0 and rng.random() < 0.25:
            stage_capacity = required - 1

        err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_routed(
            gate,
            required,
            up,
            required,
            out_new,
            required,
            row_count,
            lane_count,
            stage_new,
            stage_capacity,
        )
        err_ref = explicit_commit_routed_composition(
            gate,
            required,
            up,
            required,
            out_ref,
            required,
            row_count,
            lane_count,
            stage_ref,
            stage_capacity,
        )

        assert err_new == err_ref
        assert out_new == out_ref
