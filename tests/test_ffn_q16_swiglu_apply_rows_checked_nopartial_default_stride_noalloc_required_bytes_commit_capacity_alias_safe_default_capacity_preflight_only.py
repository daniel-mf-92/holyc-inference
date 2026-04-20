#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityPreflightOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only as strided_preflight


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    staging_out_q16,
    staging_out_capacity: int,
    out_required_in_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_commit_stage_cell_capacity: list[int] | None,
    out_commit_stage_byte_capacity: list[int] | None,
    *,
    gate_base_addr: int = 0x100000,
    up_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if (
        out_required_in_cells is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_required_out_cells is None
        or out_commit_stage_cell_capacity is None
        or out_commit_stage_byte_capacity is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_q16 is None or up_q16 is None or out_q16 is None or staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    default_in_row_stride_q16 = lane_count
    default_out_row_stride_q16 = lane_count

    return strided_preflight.ffn_q16_swiglu_apply_rows_checked_nopartial_strided_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        default_in_row_stride_q16,
        default_out_row_stride_q16,
        staging_out_q16,
        staging_out_capacity,
        out_required_in_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
        out_commit_stage_cell_capacity,
        out_commit_stage_byte_capacity,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def explicit_checked_default_stride_preflight_composition(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
    staging_out_q16,
    staging_out_capacity: int,
    out_required_in_cells: list[int] | None,
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_required_out_cells: list[int] | None,
    out_commit_stage_cell_capacity: list[int] | None,
    out_commit_stage_byte_capacity: list[int] | None,
    *,
    gate_base_addr: int,
    up_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    return ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        gate_q16,
        gate_capacity,
        up_q16,
        up_capacity,
        out_q16,
        out_capacity,
        row_count,
        lane_count,
        staging_out_q16,
        staging_out_capacity,
        out_required_in_cells,
        out_required_stage_cells,
        out_required_stage_bytes,
        out_required_out_cells,
        out_commit_stage_cell_capacity,
        out_commit_stage_byte_capacity,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_default_stride_required_bytes_alias_safe_preflight_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityPreflightOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "if (!out_required_in_cells || !out_required_stage_cells ||" in body
    assert "if (!gate_q16 || !up_q16 || !out_q16 || !staging_out_q16)" in body
    assert "default_in_row_stride_q16 = lane_count;" in body
    assert "default_out_row_stride_q16 = lane_count;" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocRequiredBytesCommitCapacityAliasSafeDefaultCapacityPreflightOnly(" in body


def test_known_vectors_success_and_overlap_rejection() -> None:
    gate = [1] * 96
    up = [2] * 96
    out = [0] * 128
    stage = [0] * 128

    got_in = [0]
    got_stage_cells = [0]
    got_stage_bytes = [0]
    got_out = [0]
    got_commit_cells = [0]
    got_commit_bytes = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        gate,
        96,
        up,
        96,
        out,
        128,
        5,
        6,
        stage,
        64,
        got_in,
        got_stage_cells,
        got_stage_bytes,
        got_out,
        got_commit_cells,
        got_commit_bytes,
    )
    assert err == FFN_Q16_OK
    assert got_in == [30]
    assert got_stage_cells == [30]
    assert got_stage_bytes == [240]
    assert got_out == [30]
    assert got_commit_cells == [30]
    assert got_commit_bytes == [512]

    in_s = [0x11]
    stage_s = [0x22]
    bytes_s = [0x33]
    out_s = [0x44]
    commit_cells_s = [0x55]
    commit_bytes_s = [0x66]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
        gate,
        96,
        up,
        96,
        out,
        128,
        5,
        6,
        stage,
        64,
        in_s,
        stage_s,
        bytes_s,
        out_s,
        commit_cells_s,
        commit_bytes_s,
        stage_base_addr=0x100000 + 16,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert in_s == [0x11]
    assert stage_s == [0x22]
    assert bytes_s == [0x33]
    assert out_s == [0x44]
    assert commit_cells_s == [0x55]
    assert commit_bytes_s == [0x66]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(671)

    for _ in range(350):
        row_count = rng.randint(0, 10)
        lane_count = rng.randint(0, 10)

        if row_count == 0 or lane_count == 0:
            required = 0
        else:
            required = row_count * lane_count

        gate_capacity = required + rng.randint(0, 3)
        up_capacity = required + rng.randint(0, 3)
        out_capacity = required + rng.randint(0, 3)

        staging_out_capacity = required + rng.randint(0, 3)
        if rng.random() < 0.2 and required > 0:
            staging_out_capacity = rng.randint(0, required - 1)

        gate = [0] * max(gate_capacity, 1)
        up = [0] * max(up_capacity, 1)
        out_a = [0] * max(out_capacity, 1)
        out_b = [0] * max(out_capacity, 1)
        stage = [0] * max(staging_out_capacity, 1)

        got_in = [0xA1]
        got_stage_cells = [0xB2]
        got_stage_bytes = [0xC3]
        got_out = [0xD4]
        got_commit_cells = [0xE5]
        got_commit_bytes = [0xF6]

        exp_in = [0xA1]
        exp_stage_cells = [0xB2]
        exp_stage_bytes = [0xC3]
        exp_out = [0xD4]
        exp_commit_cells = [0xE5]
        exp_commit_bytes = [0xF6]

        gate_base_addr = 0x100000
        up_base_addr = 0x200000
        out_base_addr = 0x300000
        stage_base_addr = 0x400000

        if rng.random() < 0.2 and required > 0:
            stage_base_addr = gate_base_addr + rng.randint(0, required * 8 - 1)
        elif rng.random() < 0.2 and required > 0:
            stage_base_addr = out_base_addr + rng.randint(0, required * 8 - 1)

        err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_a,
            out_capacity,
            row_count,
            lane_count,
            stage,
            staging_out_capacity,
            got_in,
            got_stage_cells,
            got_stage_bytes,
            got_out,
            got_commit_cells,
            got_commit_bytes,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        err_exp = explicit_checked_default_stride_preflight_composition(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_b,
            out_capacity,
            row_count,
            lane_count,
            stage,
            staging_out_capacity,
            exp_in,
            exp_stage_cells,
            exp_stage_bytes,
            exp_out,
            exp_commit_cells,
            exp_commit_bytes,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_got == err_exp
        assert got_in == exp_in
        assert got_stage_cells == exp_stage_cells
        assert got_stage_bytes == exp_stage_bytes
        assert got_out == exp_out
        assert got_commit_cells == exp_commit_cells
        assert got_commit_bytes == exp_commit_bytes


if __name__ == "__main__":
    test_source_contains_default_stride_required_bytes_alias_safe_preflight_helper()
    test_known_vectors_success_and_overlap_rejection()
    test_randomized_parity_vs_explicit_composition()
    print("ok")
