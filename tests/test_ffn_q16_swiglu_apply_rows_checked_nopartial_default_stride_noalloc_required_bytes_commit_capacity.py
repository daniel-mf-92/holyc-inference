#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
from test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes import (
    ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes,
)

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
    row_count: int,
    lane_count: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if required_stage_bytes[0] > commit_stage_byte_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_commit_required_stage_cells[0] = required_stage_cells[0]
    out_commit_required_stage_bytes[0] = required_stage_bytes[0]
    out_required_out_cells[0] = required_out_cells[0]
    return FFN_Q16_OK


def explicit_checked_required_bytes_commit_capacity_composition(
    row_count: int,
    lane_count: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes(
        row_count,
        lane_count,
        commit_stage_cell_capacity,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
    )
    if err != FFN_Q16_OK:
        return err

    if required_stage_bytes[0] > commit_stage_byte_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_commit_required_stage_cells[0] = required_stage_cells[0]
    out_commit_required_stage_bytes[0] = required_stage_bytes[0]
    out_required_out_cells[0] = required_out_cells[0]
    return FFN_Q16_OK


def test_source_contains_noalloc_required_bytes_commit_capacity_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytes(" in body
    assert "if (commit_required_stage_bytes > commit_stage_byte_capacity)" in body


def test_known_vectors_and_capacity_rejection() -> None:
    row_count = 4
    lane_count = 6
    required_stage_cells = row_count * lane_count
    required_stage_bytes = required_stage_cells * 8

    out_stage_cells = [111]
    out_stage_bytes = [222]
    out_out_cells = [333]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        row_count,
        lane_count,
        required_stage_cells,
        required_stage_bytes,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_OK
    assert out_stage_cells == [required_stage_cells]
    assert out_stage_bytes == [required_stage_bytes]
    assert out_out_cells == [required_stage_cells]

    out_stage_cells = [7]
    out_stage_bytes = [7]
    out_out_cells = [7]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        row_count,
        lane_count,
        required_stage_cells,
        required_stage_bytes - 8,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [7]
    assert out_stage_bytes == [7]
    assert out_out_cells == [7]


def test_error_paths_and_overflow_passthrough() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [1]
    out_out_cells = [1]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
            1,
            1,
            1,
            8,
            None,
            out_stage_bytes,
            out_out_cells,
        )
        == FFN_Q16_ERR_NULL_PTR
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
            1,
            1,
            -1,
            8,
            out_stage_cells,
            out_stage_bytes,
            out_out_cells,
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
        huge,
        huge,
        I64_MAX,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_ERR_OVERFLOW


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260420_608)

    for _ in range(5000):
        row_count = rng.randint(0, 250)
        lane_count = rng.randint(0, 250)

        required_cells = row_count * lane_count
        required_bytes = required_cells * 8

        commit_stage_cell_capacity = max(0, required_cells + rng.randint(-50, 50))
        commit_stage_byte_capacity = max(0, required_bytes + rng.randint(-400, 400))

        if rng.random() < 0.06:
            row_count = -rng.randint(1, 100)
        if rng.random() < 0.06:
            lane_count = -rng.randint(1, 100)
        if rng.random() < 0.06:
            commit_stage_cell_capacity = -rng.randint(1, 100)
        if rng.random() < 0.06:
            commit_stage_byte_capacity = -rng.randint(1, 100)

        got_stage_cells = [0xAAAA]
        got_stage_bytes = [0xBBBB]
        got_out_cells = [0xCCCC]
        exp_stage_cells = [0xAAAA]
        exp_stage_bytes = [0xBBBB]
        exp_out_cells = [0xCCCC]

        err_new = (
            ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity(
                row_count,
                lane_count,
                commit_stage_cell_capacity,
                commit_stage_byte_capacity,
                got_stage_cells,
                got_stage_bytes,
                got_out_cells,
            )
        )
        err_ref = explicit_checked_required_bytes_commit_capacity_composition(
            row_count,
            lane_count,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_new == err_ref
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes
        assert got_out_cells == exp_out_cells


if __name__ == "__main__":
    test_source_contains_noalloc_required_bytes_commit_capacity_helper()
    test_known_vectors_and_capacity_rejection()
    test_error_paths_and_overflow_passthrough()
    test_randomized_parity_against_explicit_composition()
    print("ok")
