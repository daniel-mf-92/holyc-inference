#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity as req_commit


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
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
    staging_out_capacity: int,
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

    if gate_q16 is None or up_q16 is None or out_q16 is None:
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

    return req_commit.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity(
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
        out_required_in_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def explicit_checked_required_bytes_default_capacity_composition(
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
    staging_out_capacity: int,
    out_required_in_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
) -> int:
    return ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
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
        staging_out_capacity,
        out_required_in_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
    )


def test_source_contains_strided_noalloc_required_bytes_default_capacity_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = (
        "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesDefaultCapacity("
    )
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNTryMulI64Checked(row_count - 1," in body
    assert "FFNTryAddI64Checked(commit_stage_cell_capacity," in body
    assert "FFNTryMulI64Checked(staging_out_capacity," in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacity(" in body


def test_known_vectors_and_zero_case() -> None:
    got_in = [111]
    got_stage_cells = [222]
    got_stage_bytes = [333]
    got_out_cells = [444]

    exp_in = [555]
    exp_stage_cells = [666]
    exp_stage_bytes = [777]
    exp_out_cells = [888]

    err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
        [0] * 100,
        100,
        [0] * 100,
        100,
        [0] * 120,
        120,
        5,
        6,
        9,
        10,
        80,
        got_in,
        got_stage_cells,
        got_stage_bytes,
        got_out_cells,
    )
    err_exp = explicit_checked_required_bytes_default_capacity_composition(
        [0] * 100,
        100,
        [0] * 100,
        100,
        [0] * 120,
        120,
        5,
        6,
        9,
        10,
        80,
        exp_in,
        exp_stage_cells,
        exp_stage_bytes,
        exp_out_cells,
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

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
        [0],
        0,
        [0],
        0,
        [0],
        0,
        0,
        4,
        4,
        4,
        0,
        z_in,
        z_stage_cells,
        z_stage_bytes,
        z_out_cells,
    )
    assert err == FFN_Q16_OK
    assert z_in == [0]
    assert z_stage_cells == [0]
    assert z_stage_bytes == [0]
    assert z_out_cells == [0]


def test_errors_and_no_partial_diagnostic_writes() -> None:
    in_s = [0xAA11]
    stage_s = [0xBB22]
    byte_s = [0xCC33]
    out_s = [0xDD44]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
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
            in_s,
            stage_s,
            byte_s,
            out_s,
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert in_s == [0xAA11]
    assert stage_s == [0xBB22]
    assert byte_s == [0xCC33]
    assert out_s == [0xDD44]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
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
            in_s,
            stage_s,
            byte_s,
            out_s,
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
            [0] * 4,
            4,
            [0] * 4,
            4,
            [0] * 4,
            4,
            2,
            2,
            I64_MAX,
            2,
            4,
            in_s,
            stage_s,
            byte_s,
            out_s,
        )
        == FFN_Q16_ERR_OVERFLOW
    )
    assert in_s == [0xAA11]
    assert stage_s == [0xBB22]
    assert byte_s == [0xCC33]
    assert out_s == [0xDD44]


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    random.seed(0xFF647)

    for _ in range(500):
        row_count = random.randint(0, 9)
        lane_count = random.randint(0, 10)

        in_row_stride_q16 = random.randint(max(lane_count, 0), max(lane_count, 0) + 5)
        out_row_stride_q16 = random.randint(max(lane_count, 0), max(lane_count, 0) + 5)

        if row_count == 0 or lane_count == 0:
            required_in = 0
            required_out = 0
        else:
            required_in = (row_count - 1) * in_row_stride_q16 + lane_count
            required_out = (row_count - 1) * out_row_stride_q16 + lane_count

        gate_capacity = required_in + random.randint(0, 2)
        up_capacity = required_in + random.randint(0, 2)
        out_capacity = required_out + random.randint(0, 2)
        staging_out_capacity = required_out + random.randint(0, 2)

        gate = [random.randint(-(1 << 16), (1 << 16)) for _ in range(max(gate_capacity, 1))]
        up = [random.randint(-(1 << 16), (1 << 16)) for _ in range(max(up_capacity, 1))]
        out = [0x5A5A] * max(out_capacity, 1)

        got_in = [0x1010]
        got_stage_cells = [0x2020]
        got_stage_bytes = [0x3030]
        got_out_cells = [0x4040]

        exp_in = [0x5050]
        exp_stage_cells = [0x6060]
        exp_stage_bytes = [0x7070]
        exp_out_cells = [0x8080]

        err_got = (
            ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_default_capacity(
                gate,
                gate_capacity,
                up,
                up_capacity,
                out,
                out_capacity,
                row_count,
                lane_count,
                in_row_stride_q16,
                out_row_stride_q16,
                staging_out_capacity,
                got_in,
                got_stage_cells,
                got_stage_bytes,
                got_out_cells,
            )
        )

        err_exp = explicit_checked_required_bytes_default_capacity_composition(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out,
            out_capacity,
            row_count,
            lane_count,
            in_row_stride_q16,
            out_row_stride_q16,
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
            assert got_in == [0x1010]
            assert got_stage_cells == [0x2020]
            assert got_stage_bytes == [0x3030]
            assert got_out_cells == [0x4040]


if __name__ == "__main__":
    test_source_contains_strided_noalloc_required_bytes_default_capacity_wrapper()
    test_known_vectors_and_zero_case()
    test_errors_and_no_partial_diagnostic_writes()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
