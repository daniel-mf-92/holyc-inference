#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity."""

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


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    in_row_stride_q16: int,
    out_row_stride_q16: int,
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
    if in_row_stride_q16 < 0 or out_row_stride_q16 < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    err, staging_capacity_bytes = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if in_row_stride_q16 < lane_count or out_row_stride_q16 < lane_count:
        return FFN_Q16_ERR_BAD_PARAM

    err, required_in_cells = rows_core.i64_mul_checked(row_count - 1, in_row_stride_q16)
    if err != FFN_Q16_OK:
        return err
    err, required_in_cells = rows_core.i64_add_checked(required_in_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_out_cells = rows_core.i64_mul_checked(row_count - 1, out_row_stride_q16)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = rows_core.i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err

    required_stage_cells = required_out_cells

    err, required_stage_bytes = rows_core.i64_mul_checked(required_stage_cells, 8)
    if err != FFN_Q16_OK:
        return err

    if required_in_cells > gate_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_in_cells > up_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    if required_stage_cells > commit_stage_cell_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_bytes > commit_stage_byte_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    if staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR
    if required_stage_cells > staging_out_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_stage_bytes > staging_capacity_bytes:
        return FFN_Q16_ERR_BAD_PARAM

    if staging_out_q16 is gate_q16:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_q16 is up_q16:
        return FFN_Q16_ERR_BAD_PARAM
    if staging_out_q16 is out_q16:
        return FFN_Q16_ERR_BAD_PARAM

    err = rows_core.ffn_q16_swiglu_apply_rows_checked(
        gate_q16,
        gate_capacity,
        in_row_stride_q16,
        up_q16,
        up_capacity,
        in_row_stride_q16,
        staging_out_q16,
        required_stage_cells,
        out_row_stride_q16,
        row_count,
        lane_count,
    )
    if err != FFN_Q16_OK:
        return err

    for row_index in range(row_count):
        out_row_base = row_index * out_row_stride_q16
        stage_row_base = row_index * out_row_stride_q16
        for lane_index in range(lane_count):
            out_index = out_row_base + lane_index
            stage_index = stage_row_base + lane_index

            if out_index < 0 or out_index >= required_out_cells:
                return FFN_Q16_ERR_BAD_PARAM
            if out_index >= out_capacity:
                return FFN_Q16_ERR_BAD_PARAM

            if stage_index < 0 or stage_index >= required_stage_cells:
                return FFN_Q16_ERR_BAD_PARAM
            if stage_index >= staging_out_capacity:
                return FFN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        out_row_base = row_index * out_row_stride_q16
        stage_row_base = row_index * out_row_stride_q16
        for lane_index in range(lane_count):
            out_index = out_row_base + lane_index
            stage_index = stage_row_base + lane_index
            out_q16[out_index] = staging_out_q16[stage_index]

    return FFN_Q16_OK


def explicit_checked_composition(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    in_row_stride_q16: int,
    out_row_stride_q16: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    staging_out_q16,
    staging_out_capacity: int,
) -> int:
    return ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        staging_out_q16,
        staging_out_capacity,
    )


def test_source_contains_strided_noalloc_commit_capacity_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNQ16SwiGLUApplyRowsChecked(gate_q16," in body
    assert "if (required_stage_cells > commit_stage_cell_capacity)" in body
    assert "if (required_stage_bytes > commit_stage_byte_capacity)" in body
    assert "if (staging_out_q16 == gate_q16)" in body
    assert "if (staging_out_q16 == up_q16)" in body
    assert "if (staging_out_q16 == out_q16)" in body


def test_known_vectors_match_explicit_composition() -> None:
    row_count = 4
    lane_count = 5
    in_row_stride_q16 = 8
    out_row_stride_q16 = 9

    in_required = (row_count - 1) * in_row_stride_q16 + lane_count
    out_required = (row_count - 1) * out_row_stride_q16 + lane_count
    stage_required = out_required

    gate = [((i * 11) - 37) << 11 for i in range(in_required)]
    up = [((91 - i * 5) << 10) for i in range(in_required)]

    out_a = [0x4A4A] * out_required
    out_b = [0x4A4A] * out_required
    stage_a = [0x2626] * stage_required
    stage_b = [0x2626] * stage_required

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
        gate,
        in_required,
        up,
        in_required,
        out_a,
        out_required,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        stage_required,
        stage_required * 8,
        stage_a,
        stage_required,
    )
    err_b = explicit_checked_composition(
        gate,
        in_required,
        up,
        in_required,
        out_b,
        out_required,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        stage_required,
        stage_required * 8,
        stage_b,
        stage_required,
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b

    for row_index in range(row_count):
        row_base = row_index * out_row_stride_q16
        for lane_index in range(lane_count, out_row_stride_q16):
            if row_base + lane_index < out_required:
                assert out_a[row_base + lane_index] == 0x4A4A


def test_commit_capacity_and_stride_failures_are_no_partial() -> None:
    row_count = 3
    lane_count = 4
    in_row_stride_q16 = 6
    out_row_stride_q16 = 7

    in_required = (row_count - 1) * in_row_stride_q16 + lane_count
    out_required = (row_count - 1) * out_row_stride_q16 + lane_count

    gate = [((i * 3) - 15) << 12 for i in range(in_required)]
    up = [((i * 7) - 21) << 11 for i in range(in_required)]

    before = [0x7B7B] * out_required
    out = before.copy()
    staging = [0x5555] * out_required

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
        gate,
        in_required,
        up,
        in_required,
        out,
        out_required,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        out_required,
        (out_required * 8) - 8,
        staging,
        out_required,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == before

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
        gate,
        in_required,
        up,
        in_required,
        out,
        out_required,
        row_count,
        lane_count,
        lane_count - 1,
        out_row_stride_q16,
        out_required,
        out_required * 8,
        staging,
        out_required,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == before


def test_adversarial_and_overflow_vectors() -> None:
    gate = [0]
    up = [0]
    out = [0x3131]
    stage = [0x2121]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
            None, 1, up, 1, out, 1, 1, 1, 1, 1, 1, 8, stage, 1
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
            gate, 1, up, 1, out, 1, 1, 1, 1, 1, -1, 8, stage, 1
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
            gate,
            1,
            up,
            1,
            out,
            1,
            2,
            1,
            I64_MAX,
            1,
            2,
            16,
            stage,
            2,
        )
        == FFN_Q16_ERR_OVERFLOW
    )


def test_randomized_parity_vs_explicit_composition() -> None:
    random.seed(0xFF631)

    for _ in range(300):
        row_count = random.randint(0, 8)
        lane_count = random.randint(0, 10)

        in_row_stride_q16 = random.randint(max(lane_count, 0), max(lane_count, 0) + 4)
        out_row_stride_q16 = random.randint(max(lane_count, 0), max(lane_count, 0) + 4)

        in_required = (
            0 if row_count == 0 or lane_count == 0 else (row_count - 1) * in_row_stride_q16 + lane_count
        )
        out_required = (
            0 if row_count == 0 or lane_count == 0 else (row_count - 1) * out_row_stride_q16 + lane_count
        )

        gate_capacity = in_required
        up_capacity = in_required
        out_capacity = out_required

        gate = [0] * max(in_required, 1)
        up = [0] * max(in_required, 1)

        for row_index in range(row_count):
            for lane_index in range(lane_count):
                idx = row_index * in_row_stride_q16 + lane_index
                gate[idx] = random.randint(-(8 << 16), (8 << 16))
                up[idx] = random.randint(-(8 << 16), (8 << 16))

        if row_count > 0 and lane_count > 0 and random.random() < 0.1:
            idx = random.randint(0, row_count - 1) * in_row_stride_q16 + random.randint(0, lane_count - 1)
            gate[idx] = rows_core.I64_MIN

        commit_stage_cell_capacity = out_required
        commit_stage_byte_capacity = out_required * 8
        stage_capacity = out_required

        out_a = [0x3D3D] * max(out_required, 1)
        out_b = [0x3D3D] * max(out_required, 1)
        stage_a = [0x2A2A] * max(stage_capacity, 1)
        stage_b = [0x2A2A] * max(stage_capacity, 1)

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_a,
            out_capacity,
            row_count,
            lane_count,
            in_row_stride_q16,
            out_row_stride_q16,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_a,
            stage_capacity,
        )
        err_b = explicit_checked_composition(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_b,
            out_capacity,
            row_count,
            lane_count,
            in_row_stride_q16,
            out_row_stride_q16,
            commit_stage_cell_capacity,
            commit_stage_byte_capacity,
            stage_b,
            stage_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_strided_noalloc_commit_capacity_wrapper()
    test_known_vectors_match_explicit_composition()
    test_commit_capacity_and_stride_failures_are_no_partial()
    test_adversarial_and_overflow_vectors()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
