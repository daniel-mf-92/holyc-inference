#!/usr/bin/env python3
"""Parity harness for ...StridedNoAllocCommitCapacityDefaultCapacityPreflightOnly."""

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


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_preflight_only(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    in_row_stride_q16: int,
    out_row_stride_q16: int,
    commit_stage_cell_capacity: int,
    commit_stage_byte_capacity: int,
    out_required_in_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_in_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if in_row_stride_q16 < 0 or out_row_stride_q16 < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if commit_stage_cell_capacity < 0 or commit_stage_byte_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        out_required_in_cells[0] = 0
        out_required_stage_cells[0] = 0
        out_required_stage_bytes[0] = 0
        out_required_out_cells[0] = 0
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

    out_required_in_cells[0] = required_in_cells
    out_required_stage_cells[0] = required_stage_cells
    out_required_stage_bytes[0] = required_stage_bytes
    out_required_out_cells[0] = required_out_cells
    return FFN_Q16_OK


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    in_row_stride_q16: int,
    out_row_stride_q16: int,
    staging_out_capacity: int,
    out_required_in_cells: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    if (
        out_required_in_cells is None
        or out_commit_required_stage_cells is None
        or out_commit_required_stage_bytes is None
        or out_required_out_cells is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
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

    return ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_preflight_only(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        commit_stage_cell_capacity,
        commit_stage_byte_capacity,
        out_required_in_cells,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_preflight_composition(
    gate_capacity: int,
    up_capacity: int,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    in_row_stride_q16: int,
    out_row_stride_q16: int,
    staging_out_capacity: int,
    out_required_in_cells: list[int] | None,
    out_commit_required_stage_cells: list[int] | None,
    out_commit_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    return ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        gate_capacity,
        up_capacity,
        out_capacity,
        row_count,
        lane_count,
        in_row_stride_q16,
        out_row_stride_q16,
        staging_out_capacity,
        out_required_in_cells,
        out_commit_required_stage_cells,
        out_commit_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_default_capacity_strided_preflight_only_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = (
        "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacityPreflightOnly("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNTryMulI64Checked(row_count - 1," in body
    assert "FFNTryMulI64Checked(staging_out_capacity," in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityPreflightOnly(" in body


def test_source_routes_commit_capacity_wrapper_through_preflight() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityPreflightOnly(" in body


def test_known_vectors_and_zero_case() -> None:
    got_in = [111]
    got_stage_cells = [222]
    got_stage_bytes = [333]
    got_out_cells = [444]

    exp_in = [555]
    exp_stage_cells = [666]
    exp_stage_bytes = [777]
    exp_out_cells = [888]

    err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        gate_capacity=128,
        up_capacity=128,
        out_capacity=160,
        row_count=5,
        lane_count=6,
        in_row_stride_q16=9,
        out_row_stride_q16=10,
        staging_out_capacity=80,
        out_required_in_cells=got_in,
        out_commit_required_stage_cells=got_stage_cells,
        out_commit_required_stage_bytes=got_stage_bytes,
        out_required_out_cells=got_out_cells,
    )
    err_exp = explicit_checked_preflight_composition(
        gate_capacity=128,
        up_capacity=128,
        out_capacity=160,
        row_count=5,
        lane_count=6,
        in_row_stride_q16=9,
        out_row_stride_q16=10,
        staging_out_capacity=80,
        out_required_in_cells=exp_in,
        out_commit_required_stage_cells=exp_stage_cells,
        out_commit_required_stage_bytes=exp_stage_bytes,
        out_required_out_cells=exp_out_cells,
    )

    assert err_got == err_exp == FFN_Q16_OK
    assert got_in == exp_in == [42]
    assert got_stage_cells == exp_stage_cells == [46]
    assert got_stage_bytes == exp_stage_bytes == [368]
    assert got_out_cells == exp_out_cells == [46]

    z_in = [9]
    z_stage_cells = [9]
    z_stage_bytes = [9]
    z_out_cells = [9]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        gate_capacity=0,
        up_capacity=0,
        out_capacity=0,
        row_count=0,
        lane_count=7,
        in_row_stride_q16=7,
        out_row_stride_q16=7,
        staging_out_capacity=0,
        out_required_in_cells=z_in,
        out_commit_required_stage_cells=z_stage_cells,
        out_commit_required_stage_bytes=z_stage_bytes,
        out_required_out_cells=z_out_cells,
    )
    assert err == FFN_Q16_OK
    assert z_in == [0]
    assert z_stage_cells == [0]
    assert z_stage_bytes == [0]
    assert z_out_cells == [0]


def test_randomized_parity_and_capacity_edges() -> None:
    random.seed(640)

    for _ in range(320):
        row_count = random.randint(0, 10)
        lane_count = random.randint(0, 10)
        in_stride = lane_count + random.randint(0, 6)
        out_stride = lane_count + random.randint(0, 6)

        if row_count == 0 or lane_count == 0:
            required_in = 0
            required_out = 0
            required_stage = 0
        else:
            required_in = (row_count - 1) * in_stride + lane_count
            required_out = (row_count - 1) * out_stride + lane_count
            required_stage = required_out

        gate_capacity = max(0, required_in + random.randint(-2, 4))
        up_capacity = max(0, required_in + random.randint(-2, 4))
        out_capacity = max(0, required_out + random.randint(-2, 4))
        staging_out_capacity = max(0, required_stage + random.randint(-2, 4))

        got_in = [0x101]
        got_stage_cells = [0x102]
        got_stage_bytes = [0x103]
        got_out_cells = [0x104]

        exp_in = [0x201]
        exp_stage_cells = [0x202]
        exp_stage_bytes = [0x203]
        exp_out_cells = [0x204]

        err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
            gate_capacity,
            up_capacity,
            out_capacity,
            row_count,
            lane_count,
            in_stride,
            out_stride,
            staging_out_capacity,
            got_in,
            got_stage_cells,
            got_stage_bytes,
            got_out_cells,
        )
        err_exp = explicit_checked_preflight_composition(
            gate_capacity,
            up_capacity,
            out_capacity,
            row_count,
            lane_count,
            in_stride,
            out_stride,
            staging_out_capacity,
            exp_in,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out_cells,
        )

        assert err_got == err_exp
        if err_got == FFN_Q16_OK:
            assert got_in == exp_in
            assert got_stage_cells == exp_stage_cells
            assert got_stage_bytes == exp_stage_bytes
            assert got_out_cells == exp_out_cells
        else:
            assert got_in == [0x101]
            assert got_stage_cells == [0x102]
            assert got_stage_bytes == [0x103]
            assert got_out_cells == [0x104]


def test_overflow_and_null_pointer_behavior() -> None:
    out_in = [1]
    out_stage_cells = [2]
    out_stage_bytes = [3]
    out_out_cells = [4]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        gate_capacity=16,
        up_capacity=16,
        out_capacity=16,
        row_count=2,
        lane_count=2,
        in_row_stride_q16=2,
        out_row_stride_q16=2,
        staging_out_capacity=2,
        out_required_in_cells=None,
        out_commit_required_stage_cells=out_stage_cells,
        out_commit_required_stage_bytes=out_stage_bytes,
        out_required_out_cells=out_out_cells,
    )
    assert err == FFN_Q16_ERR_NULL_PTR

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        gate_capacity=I64_MAX,
        up_capacity=I64_MAX,
        out_capacity=I64_MAX,
        row_count=2,
        lane_count=2,
        in_row_stride_q16=2,
        out_row_stride_q16=2,
        staging_out_capacity=(I64_MAX // 8) + 1,
        out_required_in_cells=out_in,
        out_commit_required_stage_cells=out_stage_cells,
        out_commit_required_stage_bytes=out_stage_bytes,
        out_required_out_cells=out_out_cells,
    )
    assert err == FFN_Q16_ERR_OVERFLOW

    assert out_in == [1]
    assert out_stage_cells == [2]
    assert out_stage_bytes == [3]
    assert out_out_cells == [4]


def test_matches_commit_capacity_path_on_live_buffers() -> None:
    row_count = 4
    lane_count = 5
    in_stride = 7
    out_stride = 8

    required_in = (row_count - 1) * in_stride + lane_count
    required_out = (row_count - 1) * out_stride + lane_count

    gate = [((i * 7) - 31) << 10 for i in range(required_in)]
    up = [((93 - i * 5)) << 10 for i in range(required_in)]

    out = [0x4A4A] * required_out
    stage = [0x2B2B] * required_out

    pre_req_in = [0]
    pre_req_stage_cells = [0]
    pre_req_stage_bytes = [0]
    pre_req_out = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity_default_capacity_preflight_only(
        gate_capacity=len(gate),
        up_capacity=len(up),
        out_capacity=len(out),
        row_count=row_count,
        lane_count=lane_count,
        in_row_stride_q16=in_stride,
        out_row_stride_q16=out_stride,
        staging_out_capacity=len(stage),
        out_required_in_cells=pre_req_in,
        out_commit_required_stage_cells=pre_req_stage_cells,
        out_commit_required_stage_bytes=pre_req_stage_bytes,
        out_required_out_cells=pre_req_out,
    )
    assert err == FFN_Q16_OK
    assert pre_req_stage_cells[0] == required_out

    err = strided_commit.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_commit_capacity(
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
        pre_req_stage_cells[0],
        pre_req_stage_bytes[0],
        stage,
        len(stage),
    )
    assert err == FFN_Q16_OK


if __name__ == "__main__":
    test_source_contains_default_capacity_strided_preflight_only_wrapper()
    test_source_routes_commit_capacity_wrapper_through_preflight()
    test_known_vectors_and_zero_case()
    test_randomized_parity_and_capacity_edges()
    test_overflow_and_null_pointer_behavior()
    test_matches_commit_capacity_path_on_live_buffers()
    print("ok")
