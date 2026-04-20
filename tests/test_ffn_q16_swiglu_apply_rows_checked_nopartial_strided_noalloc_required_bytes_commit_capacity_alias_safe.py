#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocRequiredBytesCommitCapacityAliasSafe."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity as required_bytes


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def i64_mul_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs * rhs
    if out < -(1 << 63) or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def byte_ranges_overlap(a_base: int, a_end: int, b_base: int, b_end: int) -> bool:
    if a_base >= a_end or b_base >= b_end:
        return False
    return a_base < b_end and b_base < a_end


def _alias_safe_required_bytes_compose(
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
    out_required_in_cells: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    gate_base_addr: int,
    up_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    if (
        out_required_in_cells is None
        or out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_q16 is None or up_q16 is None or out_q16 is None or staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    req_in = [0]
    req_stage_cells = [0]
    req_stage_bytes = [0]
    req_out = [0]

    err = required_bytes.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity(
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
        req_in,
        req_stage_cells,
        req_stage_bytes,
        req_out,
    )
    if err != FFN_Q16_OK:
        return err

    err, in_span_bytes = i64_mul_checked(req_in[0], 8)
    if err != FFN_Q16_OK:
        return err

    err, out_span_bytes = i64_mul_checked(req_out[0], 8)
    if err != FFN_Q16_OK:
        return err

    err, stage_capacity_bytes = i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    if req_stage_bytes[0] > stage_capacity_bytes:
        return FFN_Q16_ERR_BAD_PARAM

    stage_span_bytes = req_stage_bytes[0]

    gate_end_addr = gate_base_addr + in_span_bytes
    up_end_addr = up_base_addr + in_span_bytes
    out_end_addr = out_base_addr + out_span_bytes
    stage_end_addr = stage_base_addr + stage_span_bytes

    if byte_ranges_overlap(stage_base_addr, stage_end_addr, gate_base_addr, gate_end_addr):
        return FFN_Q16_ERR_BAD_PARAM
    if byte_ranges_overlap(stage_base_addr, stage_end_addr, up_base_addr, up_end_addr):
        return FFN_Q16_ERR_BAD_PARAM
    if byte_ranges_overlap(stage_base_addr, stage_end_addr, out_base_addr, out_end_addr):
        return FFN_Q16_ERR_BAD_PARAM

    out_required_in_cells[0] = req_in[0]
    out_commit_required_stage_cells[0] = req_stage_cells[0]
    out_commit_required_stage_bytes[0] = req_stage_bytes[0]
    out_required_out_cells[0] = req_out[0]
    return FFN_Q16_OK


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
    out_required_in_cells: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    gate_base_addr: int = 0x100000,
    up_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    return _alias_safe_required_bytes_compose(
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
        out_required_in_cells,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
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
    out_required_in_cells: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    *,
    gate_base_addr: int,
    up_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    return _alias_safe_required_bytes_compose(
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
        out_required_in_cells,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_alias_safe_required_bytes_commit_capacity_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = (
        "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafe("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(" in body
    assert "FFNByteRangeEndChecked(" in body
    assert "FFNByteRangesOverlap(" in body
    assert "required_stage_bytes > stage_capacity_bytes" in body


def test_known_vectors_non_overlap_and_overlap_rejection() -> None:
    gate = [1] * 64
    up = [2] * 64
    out = [0] * 64
    stage = [0] * 64

    got_in = [0]
    got_stage_cells = [0]
    got_stage_bytes = [0]
    got_out = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        gate,
        64,
        up,
        64,
        out,
        64,
        5,
        6,
        9,
        10,
        46,
        368,
        stage,
        64,
        got_in,
        got_stage_cells,
        got_stage_bytes,
        got_out,
    )
    assert err == FFN_Q16_OK
    assert got_in == [42]
    assert got_stage_cells == [46]
    assert got_stage_bytes == [368]
    assert got_out == [46]

    got_in = [11]
    got_stage_cells = [22]
    got_stage_bytes = [33]
    got_out = [44]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        gate,
        64,
        up,
        64,
        out,
        64,
        5,
        6,
        9,
        10,
        46,
        368,
        stage,
        64,
        got_in,
        got_stage_cells,
        got_stage_bytes,
        got_out,
        stage_base_addr=0x100010,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert got_in == [11]
    assert got_stage_cells == [22]
    assert got_stage_bytes == [33]
    assert got_out == [44]


def test_stage_capacity_guard_and_error_no_partial_writes() -> None:
    in_s = [0x11]
    stage_s = [0x22]
    bytes_s = [0x33]
    out_s = [0x44]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        [1] * 64,
        64,
        [2] * 64,
        64,
        [0] * 64,
        64,
        5,
        6,
        9,
        10,
        46,
        368,
        [0] * 45,
        45,
        in_s,
        stage_s,
        bytes_s,
        out_s,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert in_s == [0x11]
    assert stage_s == [0x22]
    assert bytes_s == [0x33]
    assert out_s == [0x44]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
        None,
        64,
        [2] * 64,
        64,
        [0] * 64,
        64,
        5,
        6,
        9,
        10,
        46,
        368,
        [0] * 64,
        64,
        in_s,
        stage_s,
        bytes_s,
        out_s,
    )
    assert err == FFN_Q16_ERR_NULL_PTR
    assert in_s == [0x11]
    assert stage_s == [0x22]
    assert bytes_s == [0x33]
    assert out_s == [0x44]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(649)

    for _ in range(350):
        row_count = rng.randint(0, 8)
        lane_count = rng.randint(0, 9)
        in_row_stride_q16 = rng.randint(lane_count if lane_count else 0, lane_count + 4)
        out_row_stride_q16 = rng.randint(lane_count if lane_count else 0, lane_count + 4)

        if row_count == 0 or lane_count == 0:
            required_in = 0
            required_out = 0
        else:
            required_in = (row_count - 1) * in_row_stride_q16 + lane_count
            required_out = (row_count - 1) * out_row_stride_q16 + lane_count

        required_stage_cells = row_count * lane_count
        required_stage_bytes = required_stage_cells * 8

        slack = rng.randint(0, 3)
        gate_capacity = required_in + slack
        up_capacity = required_in + rng.randint(0, 3)
        out_capacity = required_out + rng.randint(0, 3)

        commit_stage_cell_capacity = required_stage_cells + rng.randint(0, 3)
        commit_stage_byte_capacity = required_stage_bytes + rng.randint(0, 24)
        staging_out_capacity = required_stage_cells + rng.randint(0, 3)

        if rng.random() < 0.22 and required_stage_cells > 0:
            staging_out_capacity = rng.randint(0, required_stage_cells - 1)

        gate = [0] * max(gate_capacity, 1)
        up = [0] * max(up_capacity, 1)
        out_a = [0] * max(out_capacity, 1)
        out_b = [0] * max(out_capacity, 1)
        stage = [0] * max(staging_out_capacity, 1)

        got_in = [0xA1]
        got_stage_cells = [0xB2]
        got_stage_bytes = [0xC3]
        got_out = [0xD4]

        exp_in = [0xA1]
        exp_stage_cells = [0xB2]
        exp_stage_bytes = [0xC3]
        exp_out = [0xD4]

        gate_base_addr = 0x100000
        up_base_addr = 0x200000
        out_base_addr = 0x300000
        stage_base_addr = 0x400000

        if rng.random() < 0.25 and required_in > 0:
            stage_base_addr = gate_base_addr + rng.randint(0, required_in * 8 - 1)
        elif rng.random() < 0.25 and required_out > 0:
            stage_base_addr = out_base_addr + rng.randint(0, required_out * 8 - 1)

        err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe(
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
            stage,
            staging_out_capacity,
            got_in,
            got_stage_cells,
            got_stage_bytes,
            got_out,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        err_exp = explicit_checked_alias_safe_composition(
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
            stage,
            staging_out_capacity,
            exp_in,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_got == err_exp
        if err_got == FFN_Q16_OK:
            assert got_in == exp_in
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out == exp_out
        else:
            assert got_in == [0xA1]
            assert got_stage_cells == [0xB2]
            assert got_stage_bytes == [0xC3]
            assert got_out == [0xD4]


if __name__ == "__main__":
    test_source_contains_alias_safe_required_bytes_commit_capacity_wrapper()
    test_known_vectors_non_overlap_and_overlap_rejection()
    test_stage_capacity_guard_and_error_no_partial_writes()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
