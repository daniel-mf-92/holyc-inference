#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity as strided_commit


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
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

    if row_count == 0 or lane_count == 0:
        commit_stage_cell_capacity = 0
    else:
        err, commit_stage_cell_capacity = rows_core.i64_mul_checked(
            row_count - 1, out_row_stride_q16
        )
        if err != FFN_Q16_OK:
            return err
        err, commit_stage_cell_capacity = rows_core.i64_add_checked(
            commit_stage_cell_capacity, lane_count
        )
        if err != FFN_Q16_OK:
            return err

    err, commit_stage_byte_capacity = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    return strided_commit.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
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


def explicit_checked_default_capacity_commit_composition(
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

    if row_count == 0 or lane_count == 0:
        commit_stage_cell_capacity = 0
    else:
        err, commit_stage_cell_capacity = rows_core.i64_mul_checked(
            row_count - 1, out_row_stride_q16
        )
        if err != FFN_Q16_OK:
            return err
        err, commit_stage_cell_capacity = rows_core.i64_add_checked(
            commit_stage_cell_capacity, lane_count
        )
        if err != FFN_Q16_OK:
            return err

    err, commit_stage_byte_capacity = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    return strided_commit.explicit_checked_composition(
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


def test_source_contains_default_capacity_strided_commit_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = (
        "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNTryMulI64Checked(row_count - 1," in body
    assert "FFNTryAddI64Checked(commit_stage_cell_capacity," in body
    assert "FFNTryMulI64Checked(staging_out_capacity," in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity(" in body


def test_known_vectors_match_explicit_checked_composition() -> None:
    row_count = 5
    lane_count = 6
    in_stride = 9
    out_stride = 10

    needed_in = (row_count - 1) * in_stride + lane_count
    needed_out = (row_count - 1) * out_stride + lane_count

    gate = [0] * needed_in
    up = [0] * needed_in
    for row in range(row_count):
        for lane in range(lane_count):
            idx = row * in_stride + lane
            gate[idx] = ((row * 37 + lane * 11) - 33) << 11
            up[idx] = (91 - (row * 13 + lane * 7)) << 10

    out_a = [0x6161] * needed_out
    out_b = out_a.copy()
    stage_a = [0x1717] * (needed_out + 12)
    stage_b = stage_a.copy()

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        gate,
        len(gate),
        up,
        len(up),
        out_a,
        len(out_a),
        row_count,
        lane_count,
        in_stride,
        out_stride,
        stage_a,
        len(stage_a),
    )
    err_b = explicit_checked_default_capacity_commit_composition(
        gate,
        len(gate),
        up,
        len(up),
        out_b,
        len(out_b),
        row_count,
        lane_count,
        in_stride,
        out_stride,
        stage_b,
        len(stage_b),
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b


def test_staging_shortfall_and_alias_are_no_partial() -> None:
    row_count = 4
    lane_count = 5
    in_stride = 7
    out_stride = 8

    needed_in = (row_count - 1) * in_stride + lane_count
    needed_out = (row_count - 1) * out_stride + lane_count

    gate = [0] * needed_in
    up = [0] * needed_in

    out = [0x7A7A] * needed_out
    out_before = out.copy()
    too_small_stage = [0x2A2A] * (needed_out - 1)
    stage_before = too_small_stage.copy()

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        gate,
        len(gate),
        up,
        len(up),
        out,
        len(out),
        row_count,
        lane_count,
        in_stride,
        out_stride,
        too_small_stage,
        len(too_small_stage),
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert too_small_stage == stage_before

    out2 = [0x4242] * needed_out
    out2_before = out2.copy()
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        gate,
        len(gate),
        up,
        len(up),
        out2,
        len(out2),
        row_count,
        lane_count,
        in_stride,
        out_stride,
        out2,
        len(out2),
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out2 == out2_before


def test_error_paths_and_overflow() -> None:
    gate = [0, 0, 0, 0]
    up = [0, 0, 0, 0]
    out = [0, 0, 0, 0]
    stage = [0, 0, 0, 0]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
            None, 4, up, 4, out, 4, 1, 1, 1, 1, stage, 4
        )
        == FFN_Q16_ERR_NULL_PTR
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
            gate, 4, up, 4, out, 4, 1, 1, -1, 1, stage, 4
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    huge = 1 << 62
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        [0],
        I64_MAX,
        [0],
        I64_MAX,
        [0],
        I64_MAX,
        huge,
        2,
        2,
        huge,
        [0],
        I64_MAX,
    )
    assert err == FFN_Q16_ERR_OVERFLOW

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
        gate,
        4,
        up,
        4,
        out,
        4,
        1,
        1,
        1,
        1,
        stage,
        (I64_MAX // 8) + 1,
    )
    assert err == FFN_Q16_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(633)

    for _ in range(250):
        row_count = rng.randint(0, 7)
        lane_count = rng.randint(0, 8)
        in_stride = lane_count + rng.randint(0, 4)
        out_stride = lane_count + rng.randint(0, 5)

        required_in = 0 if row_count == 0 or lane_count == 0 else (row_count - 1) * in_stride + lane_count
        required_out = 0 if row_count == 0 or lane_count == 0 else (row_count - 1) * out_stride + lane_count

        gate_capacity = required_in + rng.randint(0, 6)
        up_capacity = required_in + rng.randint(0, 6)
        out_capacity = required_out + rng.randint(0, 6)
        stage_capacity = required_out + rng.randint(0, 6)

        gate = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(gate_capacity, 1))]
        up = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(up_capacity, 1))]
        out_a = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()
        stage_a = [rng.randint(-(1 << 18), (1 << 18)) for _ in range(max(stage_capacity, 1))]
        stage_b = stage_a.copy()

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_a,
            out_capacity,
            row_count,
            lane_count,
            in_stride,
            out_stride,
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
            in_stride,
            out_stride,
            stage_b,
            stage_capacity,
        )

        assert err_a == err_b
        assert out_a == out_b


def main() -> None:
    test_source_contains_default_capacity_strided_commit_wrapper()
    test_known_vectors_match_explicit_checked_composition()
    test_staging_shortfall_and_alias_are_no_partial()
    test_error_paths_and_overflow()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")


if __name__ == "__main__":
    main()
