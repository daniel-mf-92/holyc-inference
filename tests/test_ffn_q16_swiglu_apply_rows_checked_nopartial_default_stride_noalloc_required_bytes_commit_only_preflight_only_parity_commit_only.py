#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesCommitOnlyPreflightOnlyParityCommitOnly."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only as preflight_base


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity(
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
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_last_out_index: list[int] | None,
    *,
    gate_base_addr: int = 0x100000,
    up_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if (
        gate_q16 is None
        or up_q16 is None
        or out_q16 is None
        or staging_out_q16 is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_last_out_index is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_last_out_index
    ):
        return FFN_Q16_ERR_BAD_PARAM

    snapshot_tuple = (
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
    )

    staged_required_in_cells = [0]
    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_required_out_cells = [0]
    staged_commit_stage_cell_capacity = [0]
    staged_commit_stage_byte_capacity = [0]

    err = preflight_base.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
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
        staged_required_in_cells,
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_required_out_cells,
        staged_commit_stage_cell_capacity,
        staged_commit_stage_byte_capacity,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != FFN_Q16_OK:
        return err

    staged_b_required_stage_cells = [0]
    staged_b_required_stage_bytes = [0]
    staged_b_required_out_cells = [0]
    err = preflight_base.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_capacity_alias_safe_default_capacity_preflight_only(
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
        [0],
        staged_b_required_stage_cells,
        staged_b_required_stage_bytes,
        staged_b_required_out_cells,
        [0],
        [0],
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != FFN_Q16_OK:
        return err

    if snapshot_tuple != (
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
    ):
        return FFN_Q16_ERR_BAD_PARAM

    if (
        staged_required_stage_cells[0] != staged_b_required_stage_cells[0]
        or staged_required_stage_bytes[0] != staged_b_required_stage_bytes[0]
        or staged_required_out_cells[0] != staged_b_required_out_cells[0]
    ):
        return FFN_Q16_ERR_BAD_PARAM

    err, canonical_stage_bytes = rows_core.i64_mul_checked(staged_required_stage_cells[0], 8)
    if err != FFN_Q16_OK:
        return err
    if canonical_stage_bytes != staged_required_stage_bytes[0]:
        return FFN_Q16_ERR_BAD_PARAM

    if staged_required_out_cells[0] == 0:
        canonical_last_out_index = 0
    else:
        err, canonical_last_out_index = rows_core.i64_add_checked(
            staged_required_out_cells[0], -1
        )
        if err != FFN_Q16_OK:
            return err

    if row_count == 0 or lane_count == 0:
        canonical_commit_stage_cell_capacity = 0
    else:
        err, canonical_commit_stage_cell_capacity = rows_core.i64_mul_checked(
            row_count - 1, lane_count
        )
        if err != FFN_Q16_OK:
            return err
        err, canonical_commit_stage_cell_capacity = rows_core.i64_add_checked(
            canonical_commit_stage_cell_capacity, lane_count
        )
        if err != FFN_Q16_OK:
            return err

    err, canonical_commit_stage_byte_capacity = rows_core.i64_mul_checked(staging_out_capacity, 8)
    if err != FFN_Q16_OK:
        return err

    if (
        staged_commit_stage_cell_capacity[0] != canonical_commit_stage_cell_capacity
        or staged_commit_stage_byte_capacity[0] != canonical_commit_stage_byte_capacity
    ):
        return FFN_Q16_ERR_BAD_PARAM

    if staged_required_in_cells[0] != staged_required_out_cells[0]:
        return FFN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_last_out_index[0] = canonical_last_out_index
    return FFN_Q16_OK


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
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
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_last_out_index: list[int] | None,
    *,
    gate_base_addr: int = 0x100000,
    up_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if (
        gate_q16 is None
        or up_q16 is None
        or out_q16 is None
        or staging_out_q16 is None
        or out_required_stage_cells is None
        or out_required_stage_bytes is None
        or out_last_out_index is None
    ):
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if (
        out_required_stage_cells is out_required_stage_bytes
        or out_required_stage_cells is out_last_out_index
        or out_required_stage_bytes is out_last_out_index
    ):
        return FFN_Q16_ERR_BAD_PARAM

    snapshot_tuple = (
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
    )

    staged_required_stage_cells = [0]
    staged_required_stage_bytes = [0]
    staged_last_out_index = [0]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity(
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
        staged_required_stage_cells,
        staged_required_stage_bytes,
        staged_last_out_index,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )
    if err != FFN_Q16_OK:
        return err

    if snapshot_tuple != (
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
    ):
        return FFN_Q16_ERR_BAD_PARAM

    err, canonical_required_stage_bytes = rows_core.i64_mul_checked(
        staged_required_stage_cells[0], 8
    )
    if err != FFN_Q16_OK:
        return err
    if canonical_required_stage_bytes != staged_required_stage_bytes[0]:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        canonical_required_out_cells = 0
    else:
        err, canonical_required_out_cells = rows_core.i64_mul_checked(row_count, lane_count)
        if err != FFN_Q16_OK:
            return err

    if canonical_required_out_cells == 0:
        canonical_last_out_index = 0
    else:
        err, canonical_last_out_index = rows_core.i64_add_checked(canonical_required_out_cells, -1)
        if err != FFN_Q16_OK:
            return err

    if staged_last_out_index[0] != canonical_last_out_index:
        return FFN_Q16_ERR_BAD_PARAM

    out_required_stage_cells[0] = staged_required_stage_cells[0]
    out_required_stage_bytes[0] = staged_required_stage_bytes[0]
    out_last_out_index[0] = staged_last_out_index[0]
    return FFN_Q16_OK


def explicit_checked_commit_only_parity_composition(
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
    out_required_stage_cells: list[int] | None,
    out_required_stage_bytes: list[int] | None,
    out_last_out_index: list[int] | None,
    *,
    gate_base_addr: int,
    up_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    return ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
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
        out_required_stage_cells,
        out_required_stage_bytes,
        out_last_out_index,
        gate_base_addr=gate_base_addr,
        up_base_addr=up_base_addr,
        out_base_addr=out_base_addr,
        stage_base_addr=stage_base_addr,
    )


def test_source_contains_required_bytes_commit_only_preflight_parity_commit_only_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocRequiredBytesCommitOnlyPreflightOnlyParity(" in body
    assert "snapshot_gate_q16" in body
    assert "canonical_required_stage_bytes" in body
    assert "canonical_last_out_index" in body


def test_known_vectors_success_and_zero_shape() -> None:
    gate = [1] * 128
    up = [2] * 128
    out = [0] * 128
    stage = [0] * 128

    got_cells = [777]
    got_bytes = [888]
    got_last = [999]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
        gate,
        128,
        up,
        128,
        out,
        128,
        5,
        6,
        stage,
        128,
        got_cells,
        got_bytes,
        got_last,
    )
    assert err == FFN_Q16_OK
    assert got_cells == [30]
    assert got_bytes == [240]
    assert got_last == [29]

    got_cells = [11]
    got_bytes = [22]
    got_last = [33]
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
        gate,
        128,
        up,
        128,
        out,
        128,
        0,
        6,
        stage,
        128,
        got_cells,
        got_bytes,
        got_last,
    )
    assert err == FFN_Q16_OK
    assert got_cells == [0]
    assert got_bytes == [0]
    assert got_last == [0]


def test_error_paths_preserve_outputs() -> None:
    gate = [1] * 32
    up = [2] * 32
    out = [0] * 32
    stage = [0] * 32

    out_cells = [101]
    out_bytes = [202]
    out_last = [303]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
        gate,
        32,
        up,
        32,
        out,
        32,
        2,
        4,
        stage,
        32,
        out_cells,
        out_cells,
        out_last,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out_cells == [101]
    assert out_bytes == [202]
    assert out_last == [303]

    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
        None,
        32,
        up,
        32,
        out,
        32,
        2,
        4,
        stage,
        32,
        out_cells,
        out_bytes,
        out_last,
    )
    assert err == FFN_Q16_ERR_NULL_PTR
    assert out_cells == [101]
    assert out_bytes == [202]
    assert out_last == [303]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260422_1043)

    for _ in range(6000):
        row_count = rng.randint(0, 200)
        lane_count = rng.randint(0, 200)

        required = row_count * lane_count
        gate_capacity = max(0, required + rng.randint(-50, 100))
        up_capacity = max(0, required + rng.randint(-50, 100))
        out_capacity = max(0, required + rng.randint(-50, 100))
        staging_capacity = max(0, required + rng.randint(-50, 100))

        if rng.random() < 0.08:
            gate_capacity = -rng.randint(1, 20)
        if rng.random() < 0.08:
            up_capacity = -rng.randint(1, 20)
        if rng.random() < 0.08:
            out_capacity = -rng.randint(1, 20)
        if rng.random() < 0.08:
            staging_capacity = -rng.randint(1, 20)
        if rng.random() < 0.06:
            row_count = -rng.randint(1, 20)
        if rng.random() < 0.06:
            lane_count = -rng.randint(1, 20)

        gate = [1] * max(gate_capacity, 0)
        up = [2] * max(up_capacity, 0)
        out = [3] * max(out_capacity, 0)
        stage = [4] * max(staging_capacity, 0)

        got_cells = [0xA11]
        got_bytes = [0xB22]
        got_last = [0xC33]

        exp_cells = [0xA11]
        exp_bytes = [0xB22]
        exp_last = [0xC33]

        use_null = rng.random() < 0.03
        gate_arg = None if use_null else gate

        err_new = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_required_bytes_commit_only_preflight_only_parity_commit_only(
            gate_arg,
            gate_capacity,
            up,
            up_capacity,
            out,
            out_capacity,
            row_count,
            lane_count,
            stage,
            staging_capacity,
            got_cells,
            got_bytes,
            got_last,
        )

        err_ref = explicit_checked_commit_only_parity_composition(
            gate_arg,
            gate_capacity,
            up,
            up_capacity,
            out,
            out_capacity,
            row_count,
            lane_count,
            stage,
            staging_capacity,
            exp_cells,
            exp_bytes,
            exp_last,
            gate_base_addr=0x100000,
            up_base_addr=0x200000,
            out_base_addr=0x300000,
            stage_base_addr=0x400000,
        )

        assert err_new == err_ref
        assert got_cells == exp_cells
        assert got_bytes == exp_bytes
        assert got_last == exp_last


if __name__ == "__main__":
    raise SystemExit(
        __import__("pytest").main([__file__, "-q"])
    )
