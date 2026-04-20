#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocCommitCapacityAliasSafe."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity as commit_capacity
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe as required_alias_safe


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
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
    *,
    gate_base_addr: int = 0x100000,
    up_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None or staging_out_q16 is None:
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

    required_in_cells = [0]
    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = required_alias_safe.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
        required_in_cells,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells[0] != required_out_cells[0]:
        return FFN_Q16_ERR_BAD_PARAM

    err, expected_stage_bytes = rows_core.i64_mul_checked(required_stage_cells[0], 8)
    if err != FFN_Q16_OK:
        return err
    if expected_stage_bytes != required_stage_bytes[0]:
        return FFN_Q16_ERR_BAD_PARAM

    return commit_capacity.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
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


def explicit_checked_alias_safe_composition(
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
    *,
    gate_base_addr: int,
    up_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    required_in_cells = [0xA1]
    required_stage_cells = [0xB2]
    required_stage_bytes = [0xC3]
    required_out_cells = [0xD4]

    err = required_alias_safe.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
        required_in_cells,
        required_stage_cells,
        required_stage_bytes,
        required_out_cells,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != FFN_Q16_OK:
        return err

    if required_stage_cells[0] != required_out_cells[0]:
        return FFN_Q16_ERR_BAD_PARAM

    err, expected_stage_bytes = rows_core.i64_mul_checked(required_stage_cells[0], 8)
    if err != FFN_Q16_OK:
        return err
    if expected_stage_bytes != required_stage_bytes[0]:
        return FFN_Q16_ERR_BAD_PARAM

    return commit_capacity.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
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


def test_source_contains_strided_alias_safe_commit_capacity_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafe("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "if (!gate_q16 || !up_q16 || !out_q16 || !staging_out_q16)" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafe(" in body
    assert "required_stage_cells != required_out_cells" in body
    assert "expected_stage_bytes != required_stage_bytes" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity(" in body


def test_known_vectors_and_alias_rejection_no_partial() -> None:
    row_count = 5
    lane_count = 6
    in_row_stride_q16 = 9
    out_row_stride_q16 = 10
    required_in = (row_count - 1) * in_row_stride_q16 + lane_count
    required_out = (row_count - 1) * out_row_stride_q16 + lane_count

    gate = [((i * 7) - 13) << 11 for i in range(required_in + 6)]
    up = [((23 - i * 2) << 10) for i in range(required_in + 6)]

    out_a = [0x7171] * (required_out + 5)
    out_b = out_a.copy()
    stage_a = [0x1111] * (required_out + 11)
    stage_b = stage_a.copy()

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
        gate,
        len(gate),
        up,
        len(up),
        out_a,
        len(out_a),
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        required_out,
        (required_out + 11) * 8,
        stage_a,
        len(stage_a),
    )
    err_b = explicit_checked_alias_safe_composition(
        gate,
        len(gate),
        up,
        len(up),
        out_b,
        len(out_b),
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        required_out,
        (required_out + 11) * 8,
        stage_b,
        len(stage_b),
        gate_base_addr=0x100000,
        up_base_addr=0x200000,
        out_base_addr=0x300000,
        stage_base_addr=0x400000,
    )
    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b

    out = [0x3333] * (required_out + 2)
    out_before = out.copy()
    stage = [0x4444] * (required_out + 2)
    stage_before = stage.copy()
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
        gate,
        len(gate),
        up,
        len(up),
        out,
        len(out),
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        required_out,
        (required_out + 2) * 8,
        stage,
        len(stage),
        stage_base_addr=0x300000 + 8,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert stage == stage_before


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260420_681)

    for _ in range(360):
        row_count = rng.randint(0, 9)
        lane_count = rng.randint(0, 10)
        in_row_stride_q16 = rng.randint(0, 13)
        out_row_stride_q16 = rng.randint(0, 13)

        if row_count > 0 and lane_count > 0:
            if in_row_stride_q16 < lane_count:
                in_row_stride_q16 = lane_count + rng.randint(0, 3)
            if out_row_stride_q16 < lane_count:
                out_row_stride_q16 = lane_count + rng.randint(0, 3)

        required_in = (
            0
            if row_count == 0 or lane_count == 0
            else (row_count - 1) * in_row_stride_q16 + lane_count
        )
        required_out = (
            0
            if row_count == 0 or lane_count == 0
            else (row_count - 1) * out_row_stride_q16 + lane_count
        )

        gate_capacity = max(0, required_in + rng.randint(-2, 4))
        up_capacity = max(0, required_in + rng.randint(-2, 4))
        out_capacity = max(0, required_out + rng.randint(-2, 4))
        staging_out_capacity = max(0, required_out + rng.randint(-2, 4))

        if rng.random() < 0.12:
            commit_stage_cell_capacity = max(0, required_out - rng.randint(1, 3))
        else:
            commit_stage_cell_capacity = max(0, required_out + rng.randint(0, 3))

        if rng.random() < 0.12:
            commit_stage_byte_capacity = max(0, required_out * 8 - rng.randint(1, 16))
        else:
            commit_stage_byte_capacity = max(0, staging_out_capacity * 8 + rng.randint(0, 24))

        gate = [rng.randint(-4000, 4000) for _ in range(max(gate_capacity, 1))]
        up = [rng.randint(-4000, 4000) for _ in range(max(up_capacity, 1))]
        out_a = [rng.randint(-4000, 4000) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()
        stage_a = [rng.randint(-4000, 4000) for _ in range(max(staging_out_capacity, 1))]
        stage_b = stage_a.copy()

        gate_base_addr = 0x100000
        up_base_addr = 0x200000
        out_base_addr = 0x300000
        if rng.random() < 0.25:
            stage_base_addr = rng.choice([gate_base_addr, up_base_addr, out_base_addr]) + rng.randint(0, 16)
        else:
            stage_base_addr = 0x400000 + rng.randint(0, 128)

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
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
            staging_out_capacity,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )
        err_b = explicit_checked_alias_safe_composition(
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
            staging_out_capacity,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_a == err_b
        assert out_a == out_b


def test_null_and_bad_param_contracts() -> None:
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
            None,
            1,
            [0],
            1,
            [0],
            1,
            1,
            1,
            1,
            1,
            1,
            8,
            [0],
            1,
        )
        == FFN_Q16_ERR_NULL_PTR
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_alias_safe(
            [0],
            -1,
            [0],
            1,
            [0],
            1,
            1,
            1,
            1,
            1,
            1,
            8,
            [0],
            1,
        )
        == FFN_Q16_ERR_BAD_PARAM
    )


if __name__ == "__main__":
    test_source_contains_strided_alias_safe_commit_capacity_wrapper()
    test_known_vectors_and_alias_rejection_no_partial()
    test_randomized_parity_vs_explicit_composition()
    test_null_and_bad_param_contracts()
    print("ok")
