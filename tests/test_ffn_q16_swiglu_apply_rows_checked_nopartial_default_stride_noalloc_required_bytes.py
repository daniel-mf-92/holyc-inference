#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
    row_count: int,
    lane_count: int,
    stage_cell_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0 or stage_cell_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        out_required_stage_cells[0] = 0
        out_required_stage_bytes[0] = 0
        out_required_out_cells[0] = 0
        return FFN_Q16_OK

    err, required_stage_cells = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_stage_bytes = rows_core.i64_mul_checked(required_stage_cells, 8)
    if err != FFN_Q16_OK:
        return err

    default_row_stride = lane_count
    err, required_out_cells = rows_core.i64_mul_checked(row_count - 1, default_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = rows_core.i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells > stage_cell_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells
    return FFN_Q16_OK


def explicit_checked_required_bytes_composition(
    row_count: int,
    lane_count: int,
    stage_cell_capacity: int,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0 or stage_cell_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        out_required_stage_cells[0] = 0
        out_required_stage_bytes[0] = 0
        out_required_out_cells[0] = 0
        return FFN_Q16_OK

    err, required_stage_cells = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_stage_bytes = rows_core.i64_mul_checked(required_stage_cells, 8)
    if err != FFN_Q16_OK:
        return err

    err, required_out_cells = rows_core.i64_mul_checked(row_count - 1, lane_count)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = rows_core.i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells > stage_cell_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells
    return FFN_Q16_OK


def test_source_contains_noalloc_required_bytes_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "status = FFNTryMulI64Checked(row_count," in body
    assert "status = FFNTryMulI64Checked(required_stage_cells," in body
    assert "sizeof(I64)" in body
    assert "status = FFNTryMulI64Checked(row_count - 1," in body
    assert "if (required_stage_cells > stage_cell_capacity)" in body


def test_known_vectors_and_zero_case() -> None:
    out_stage_cells = [111]
    out_stage_bytes = [222]
    out_out_cells = [333]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        7,
        5,
        35,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_OK
    assert out_stage_cells == [35]
    assert out_stage_bytes == [35 * 8]
    assert out_out_cells == [35]

    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        0,
        5,
        0,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_OK
    assert out_stage_cells == [0]
    assert out_stage_bytes == [0]
    assert out_out_cells == [0]


def test_error_paths() -> None:
    out_stage_cells = [9]
    out_stage_bytes = [9]
    out_out_cells = [9]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
            1, 1, 1, None, out_stage_bytes, out_out_cells
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
            -1, 1, 1, out_stage_cells, out_stage_bytes, out_out_cells
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        2,
        3,
        5,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM

    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        huge,
        huge,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_600)

    for _ in range(5000):
        row_count = rng.randint(0, 250)
        lane_count = rng.randint(0, 250)
        required = row_count * lane_count

        stage_cell_capacity = max(0, required + rng.randint(-50, 50))

        if rng.random() < 0.06:
            row_count = -rng.randint(1, 100)
        if rng.random() < 0.06:
            lane_count = -rng.randint(1, 100)
        if rng.random() < 0.06:
            stage_cell_capacity = -rng.randint(1, 100)

        got_stage_cells = [0xAAAA]
        got_stage_bytes = [0xBBBB]
        got_out_cells = [0xCCCC]
        exp_stage_cells = [0xAAAA]
        exp_stage_bytes = [0xBBBB]
        exp_out_cells = [0xCCCC]

        err_new = (
            ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
                row_count,
                lane_count,
                stage_cell_capacity,
                got_stage_cells,
                got_stage_bytes,
                got_out_cells,
            )
        )
        err_ref = explicit_checked_required_bytes_composition(
            row_count,
            lane_count,
            stage_cell_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_new == err_ref
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes
        assert got_out_cells == exp_out_cells
