#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked_default_stride as rows_default
import test_ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only as rows_preflight
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride as rows_nopartial_default

FFN_Q16_OK = rows_default.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_default.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_default.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
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

    default_row_stride = [lane_count]
    last_row_index = [0]
    required_gate_cells = [0]
    required_up_cells = [0]
    required_out_cells = [0]

    err = rows_preflight.ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        default_row_stride,
        last_row_index,
        required_gate_cells,
        required_up_cells,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR
    if staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_capacity < required_out_cells[0]:
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

    for row_index in range(row_count):
        row_base = row_index * lane_count
        for lane_index in range(lane_count):
            out_index = row_base + lane_index
            out_q16[out_index] = staging_out_q16[out_index]

    return FFN_Q16_OK


def test_source_contains_noalloc_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAlloc("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert (
        "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly("
        in body
    )
    assert "FFNQ16SwiGLUApplyRowsCheckedDefaultStride(gate_q16," in body
    assert "staging_out_q16 == gate_q16" in body
    assert "staging_out_q16 == up_q16" in body
    assert "staging_out_q16 == out_q16" in body


def test_known_vectors_match_explicit_staged_default_stride() -> None:
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

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
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
    err_b = rows_nopartial_default.explicit_staged_default_stride_composition(
        gate,
        required,
        up,
        required,
        out_b,
        required,
        row_count,
        lane_count,
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b


def test_failure_preserves_output_no_partial() -> None:
    row_count = 4
    lane_count = 6
    required = row_count * lane_count

    gate = [1 << 16] * required
    up = [1 << 16] * required
    out = [0x4242] * required
    stage = [0x1313] * required

    gate[3] = rows_default.rows_core.I64_MIN

    before = out.copy()
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
        gate,
        required,
        up,
        required,
        out,
        required,
        row_count,
        lane_count,
        stage,
        required,
    )

    assert err != FFN_Q16_OK
    assert out == before


def test_staging_guards() -> None:
    gate = [0, 0, 0, 0]
    up = [0, 0, 0, 0]
    out = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
            gate, 4, up, 4, out, 4, 2, 2, None, 4
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
            gate, 4, up, 4, out, 4, 2, 2, stage, -1
        )
        == FFN_Q16_ERR_BAD_PARAM
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
            gate, 4, up, 4, out, 4, 2, 2, stage, 3
        )
        == FFN_Q16_ERR_BAD_PARAM
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
            gate, 4, up, 4, out, 4, 2, 2, gate, 4
        )
        == FFN_Q16_ERR_BAD_PARAM
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
            gate, 4, up, 4, out, 4, 2, 2, up, 4
        )
        == FFN_Q16_ERR_BAD_PARAM
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
            gate, 4, up, 4, out, 4, 2, 2, out, 4
        )
        == FFN_Q16_ERR_BAD_PARAM
    )


def test_randomized_parity_vs_explicit_staged_default_stride() -> None:
    random.seed(0xFF586)

    for _ in range(320):
        row_count = random.randint(0, 10)
        lane_count = random.randint(0, 10)
        required = row_count * lane_count

        gate = [0] * max(required, 1)
        up = [0] * max(required, 1)
        for i in range(required):
            gate[i] = random.randint(-(8 << 16), (8 << 16))
            up[i] = random.randint(-(8 << 16), (8 << 16))

        if required > 0 and random.random() < 0.18:
            gate[random.randint(0, required - 1)] = rows_default.rows_core.I64_MIN

        out_a = [0x7575] * max(required, 1)
        out_b = [0x7575] * max(required, 1)
        stage = [0x0909] * max(required, 1)

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc(
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
        err_b = rows_nopartial_default.explicit_staged_default_stride_composition(
            gate,
            required,
            up,
            required,
            out_b,
            required,
            row_count,
            lane_count,
        )

        assert err_a == err_b
        assert out_a == out_b
