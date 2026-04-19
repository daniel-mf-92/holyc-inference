#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    out_default_row_stride: list[int] | None,
    out_last_row_index: list[int] | None,
    out_required_gate_cells: list[int] | None,
    out_required_up_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_default_row_stride is None
        or out_last_row_index is None
        or out_required_gate_cells is None
        or out_required_up_cells is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    default_row_stride = lane_count

    if row_count == 0 or lane_count == 0:
        out_default_row_stride[0] = default_row_stride
        out_last_row_index[0] = 0
        out_required_gate_cells[0] = 0
        out_required_up_cells[0] = 0
        out_required_out_cells[0] = 0
        return FFN_Q16_OK

    err, required_gate_cells = rows_core.i64_mul_checked(row_count - 1, default_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_gate_cells = rows_core.i64_add_checked(required_gate_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_up_cells = rows_core.i64_mul_checked(row_count - 1, default_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_up_cells = rows_core.i64_add_checked(required_up_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_out_cells = rows_core.i64_mul_checked(row_count - 1, default_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = rows_core.i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    if required_gate_cells > gate_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_up_cells > up_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_default_row_stride[0] = default_row_stride
    out_last_row_index[0] = row_count - 1
    out_required_gate_cells[0] = required_gate_cells
    out_required_up_cells[0] = required_up_cells
    out_required_out_cells[0] = required_out_cells
    return FFN_Q16_OK


def explicit_checked_guard_composition(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    out_default_row_stride: list[int] | None,
    out_last_row_index: list[int] | None,
    out_required_gate_cells: list[int] | None,
    out_required_up_cells: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_default_row_stride is None
        or out_last_row_index is None
        or out_required_gate_cells is None
        or out_required_up_cells is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    default_row_stride = lane_count

    if row_count == 0 or lane_count == 0:
        out_default_row_stride[0] = default_row_stride
        out_last_row_index[0] = 0
        out_required_gate_cells[0] = 0
        out_required_up_cells[0] = 0
        out_required_out_cells[0] = 0
        return FFN_Q16_OK

    required_gate_cells = (row_count - 1) * default_row_stride + lane_count
    required_up_cells = (row_count - 1) * default_row_stride + lane_count
    required_out_cells = (row_count - 1) * default_row_stride + lane_count

    if required_gate_cells > rows_core.I64_MAX or required_gate_cells < rows_core.I64_MIN:
        return rows_core.FFN_Q16_ERR_OVERFLOW
    if required_up_cells > rows_core.I64_MAX or required_up_cells < rows_core.I64_MIN:
        return rows_core.FFN_Q16_ERR_OVERFLOW
    if required_out_cells > rows_core.I64_MAX or required_out_cells < rows_core.I64_MIN:
        return rows_core.FFN_Q16_ERR_OVERFLOW

    if required_gate_cells > gate_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_up_cells > up_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_default_row_stride[0] = default_row_stride
    out_last_row_index[0] = row_count - 1
    out_required_gate_cells[0] = required_gate_cells
    out_required_up_cells[0] = required_up_cells
    out_required_out_cells[0] = required_out_cells
    return FFN_Q16_OK


def test_source_contains_preflight_helper_and_wrapper_usage() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly("
    assert signature in source

    helper_body = source.split(signature, 1)[1]
    assert "default_row_stride = lane_count;" in helper_body
    assert "*out_required_gate_cells = required_gate_cells;" in helper_body
    assert "*out_required_up_cells = required_up_cells;" in helper_body
    assert "*out_required_out_cells = required_out_cells;" in helper_body

    wrapper_body = source.split("I32 FFNQ16SwiGLUApplyRowsCheckedDefaultStride(", 1)[1]
    assert "FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly(" in wrapper_body


def test_known_vector_outputs_expected_diagnostics() -> None:
    row_count = 6
    lane_count = 5
    required_cells = (row_count - 1) * lane_count + lane_count

    got_stride = [77]
    got_last_row = [88]
    got_req_gate = [99]
    got_req_up = [111]
    got_req_out = [222]

    err = ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only(
        required_cells,
        required_cells,
        required_cells,
        row_count,
        lane_count,
        got_stride,
        got_last_row,
        got_req_gate,
        got_req_up,
        got_req_out,
    )

    assert err == FFN_Q16_OK
    assert got_stride == [lane_count]
    assert got_last_row == [row_count - 1]
    assert got_req_gate == [required_cells]
    assert got_req_up == [required_cells]
    assert got_req_out == [required_cells]


def test_error_paths_preserve_outputs() -> None:
    got_stride = [11]
    got_last_row = [22]
    got_req_gate = [33]
    got_req_up = [44]
    got_req_out = [55]

    err = ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only(
        -1,
        0,
        0,
        1,
        1,
        got_stride,
        got_last_row,
        got_req_gate,
        got_req_up,
        got_req_out,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert got_stride == [11]
    assert got_last_row == [22]
    assert got_req_gate == [33]
    assert got_req_up == [44]
    assert got_req_out == [55]


def test_null_output_pointer_guards() -> None:
    ok = [0]
    assert (
        ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only(
            0,
            0,
            0,
            0,
            0,
            None,
            ok,
            ok,
            ok,
            ok,
        )
        == FFN_Q16_ERR_NULL_PTR
    )


def test_randomized_parity_vs_explicit_guard_composition() -> None:
    random.seed(0xFF585)

    for _ in range(400):
        row_count = random.randint(0, 1 << 12)
        lane_count = random.randint(0, 1 << 12)

        if random.random() < 0.06:
            lane_count = rows_core.I64_MAX
            row_count = random.randint(2, 8)

        gate_capacity = random.randint(0, 1 << 20)
        up_capacity = random.randint(0, 1 << 20)
        out_capacity = random.randint(0, 1 << 20)

        if row_count > 0 and lane_count > 0 and random.random() < 0.35:
            required = (row_count - 1) * lane_count + lane_count
            if required <= rows_core.I64_MAX:
                tweak = random.randint(0, 8)
                gate_capacity = max(0, required - tweak)
                up_capacity = max(0, required - tweak)
                out_capacity = max(0, required - tweak)

        got_stride = [0x101]
        got_last_row = [0x202]
        got_req_gate = [0x303]
        got_req_up = [0x404]
        got_req_out = [0x505]

        exp_stride = got_stride.copy()
        exp_last_row = got_last_row.copy()
        exp_req_gate = got_req_gate.copy()
        exp_req_up = got_req_up.copy()
        exp_req_out = got_req_out.copy()

        got_err = ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only(
            gate_capacity,
            up_capacity,
            out_capacity,
            row_count,
            lane_count,
            got_stride,
            got_last_row,
            got_req_gate,
            got_req_up,
            got_req_out,
        )
        exp_err = explicit_checked_guard_composition(
            gate_capacity,
            up_capacity,
            out_capacity,
            row_count,
            lane_count,
            exp_stride,
            exp_last_row,
            exp_req_gate,
            exp_req_up,
            exp_req_out,
        )

        assert got_err == exp_err
        assert got_stride == exp_stride
        assert got_last_row == exp_last_row
        assert got_req_gate == exp_req_gate
        assert got_req_up == exp_req_up
        assert got_req_out == exp_req_out
