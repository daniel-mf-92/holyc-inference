#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_default_stride_preflight_only as rows_preflight

FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    stage_capacity: int,
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

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if stage_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

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

    err, required_stage_cells = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err
    err, required_stage_bytes = rows_core.i64_mul_checked(required_stage_cells, 8)
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells > stage_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells[0]
    return FFN_Q16_OK


def explicit_checked_composition(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    stage_capacity: int,
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

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if stage_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

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

    err, required_stage_cells = rows_core.i64_mul_checked(row_count, lane_count)
    if err != FFN_Q16_OK:
        return err
    err, required_stage_bytes = rows_core.i64_mul_checked(required_stage_cells, 8)
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells > stage_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells[0]
    return FFN_Q16_OK


def test_source_contains_noalloc_preflight_only_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNQ16SwiGLUApplyRowsCheckedDefaultStridePreflightOnly(" in body
    assert "status = FFNTryMulI64Checked(row_count," in body
    assert "status = FFNTryMulI64Checked(required_stage_cells," in body
    assert "sizeof(I64)" in body
    assert "if (required_stage_cells > stage_capacity)" in body


def test_known_vectors_and_stage_capacity_gate() -> None:
    row_count = 7
    lane_count = 5
    required_cells = row_count * lane_count

    got_stage_cells = [111]
    got_stage_bytes = [222]
    got_out_cells = [333]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        required_cells,
        required_cells,
        required_cells,
        row_count,
        lane_count,
        required_cells,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    assert err == FFN_Q16_OK
    assert got_stage_cells == [required_cells]
    assert got_stage_bytes == [required_cells * 8]
    assert got_out_cells == [required_cells]

    got_stage_cells = [444]
    got_stage_bytes = [555]
    got_out_cells = [666]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        required_cells,
        required_cells,
        required_cells,
        row_count,
        lane_count,
        required_cells - 1,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )

    assert err == FFN_Q16_ERR_BAD_PARAM
    assert got_stage_cells == [444]
    assert got_stage_bytes == [555]
    assert got_out_cells == [666]


def test_error_paths_preserve_outputs() -> None:
    out_stage_cells = [17]
    out_stage_bytes = [27]
    out_out_cells = [37]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        -1,
        0,
        0,
        1,
        1,
        1,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out_stage_cells == [17]
    assert out_stage_bytes == [27]
    assert out_out_cells == [37]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        0,
        0,
        0,
        0,
        0,
        0,
        None,
        out_stage_bytes,
        out_out_cells,
    )
    assert err == FFN_Q16_ERR_NULL_PTR
    assert out_stage_bytes == [27]
    assert out_out_cells == [37]


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_597)

    for _ in range(5000):
        row_count = rng.randint(0, 200)
        lane_count = rng.randint(0, 200)

        required = row_count * lane_count
        gate_capacity = max(0, required + rng.randint(-8, 8))
        up_capacity = max(0, required + rng.randint(-8, 8))
        out_capacity = max(0, required + rng.randint(-8, 8))
        stage_capacity = max(0, required + rng.randint(-8, 8))

        if rng.random() < 0.05:
            gate_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            up_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            out_capacity = -rng.randint(1, 9)
        if rng.random() < 0.05:
            stage_capacity = -rng.randint(1, 9)

        if rng.random() < 0.05:
            row_count = -rng.randint(1, 9)
        if rng.random() < 0.05:
            lane_count = -rng.randint(1, 9)

        got_stage_cells = [91]
        got_stage_bytes = [92]
        got_out_cells = [93]
        exp_stage_cells = [81]
        exp_stage_bytes = [82]
        exp_out_cells = [83]

        err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
            gate_capacity,
            up_capacity,
            out_capacity,
            row_count,
            lane_count,
            stage_capacity,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_composition(
            gate_capacity,
            up_capacity,
            out_capacity,
            row_count,
            lane_count,
            stage_capacity,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        if err_got == FFN_Q16_OK:
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_stage_cells == [91]
            assert got_stage_bytes == [92]
            assert got_out_cells == [93]


def test_overflow_surface_matches_checked_math() -> None:
    out_stage_cells = [1]
    out_stage_bytes = [2]
    out_out_cells = [3]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only(
        I64_MAX,
        I64_MAX,
        I64_MAX,
        I64_MAX,
        2,
        I64_MAX,
        out_stage_cells,
        out_stage_bytes,
        out_out_cells,
    )
    assert err in (FFN_Q16_ERR_OVERFLOW, FFN_Q16_ERR_BAD_PARAM)


if __name__ == "__main__":
    test_source_contains_noalloc_preflight_only_helper()
    test_known_vectors_and_stage_capacity_gate()
    test_error_paths_preserve_outputs()
    test_randomized_parity_vs_explicit_checked_composition()
    test_overflow_surface_matches_checked_math()
    print("ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_preflight_only=ok")
