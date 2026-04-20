#!/usr/bin/env python3
"""Parity harness for ...DefaultStrideNoAllocCommitCapacityAliasSafeDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity as default_commit
import test_ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only as default_preflight


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
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
    *,
    gate_base_addr: int = 0x100000,
    up_base_addr: int = 0x200000,
    out_base_addr: int = 0x300000,
    stage_base_addr: int = 0x400000,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None or staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    required_in_cells = [0]
    required_stage_cells = [0]
    required_stage_bytes = [0]
    required_out_cells = [0]

    err = default_preflight.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
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

    return default_commit.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
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


def explicit_checked_default_stride_alias_safe_default_capacity_composition(
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
    *,
    gate_base_addr: int,
    up_base_addr: int,
    out_base_addr: int,
    stage_base_addr: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None or staging_out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0 or staging_out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    required_in_cells = [0xA1]
    required_stage_cells = [0xB2]
    required_stage_bytes = [0xC3]
    required_out_cells = [0xD4]

    err = default_preflight.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity_preflight_only(
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

    return default_commit.ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_default_capacity(
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


def test_source_contains_default_stride_alias_safe_default_capacity_commit_wrapper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    signature = "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStrideNoAllocCommitCapacityAliasSafeDefaultCapacity("
    assert signature in source
    body = source.split(signature, 1)[1]

    assert "if (!gate_q16 || !up_q16 || !out_q16 || !staging_out_q16)" in body
    assert "default_in_row_stride_q16 = lane_count;" in body
    assert "default_out_row_stride_q16 = lane_count;" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityAliasSafeDefaultCapacityPreflightOnly(" in body
    assert "FFNQ16SwiGLUApplyRowsCheckedNoPartialStridedNoAllocCommitCapacityDefaultCapacity(" in body


def test_known_vectors_and_alias_rejection_no_partial() -> None:
    row_count = 5
    lane_count = 6
    required = row_count * lane_count

    gate = [((i * 7) - 13) << 11 for i in range(required)]
    up = [((23 - i * 2) << 10) for i in range(required)]

    out_a = [0x7171] * (required + 3)
    out_b = out_a.copy()
    stage_a = [0x1111] * (required + 9)
    stage_b = stage_a.copy()

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
        gate,
        required,
        up,
        required,
        out_a,
        len(out_a),
        row_count,
        lane_count,
        stage_a,
        len(stage_a),
    )
    err_b = explicit_checked_default_stride_alias_safe_default_capacity_composition(
        gate,
        required,
        up,
        required,
        out_b,
        len(out_b),
        row_count,
        lane_count,
        stage_b,
        len(stage_b),
        gate_base_addr=0x100000,
        up_base_addr=0x200000,
        out_base_addr=0x300000,
        stage_base_addr=0x400000,
    )
    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b

    out = [0x3333] * (required + 2)
    out_before = out.copy()
    stage = [0x4444] * (required + 2)
    stage_before = stage.copy()
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
        gate,
        required,
        up,
        required,
        out,
        len(out),
        row_count,
        lane_count,
        stage,
        len(stage),
        stage_base_addr=0x300000 + 8,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert stage == stage_before


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(673)

    for _ in range(320):
        row_count = rng.randint(0, 10)
        lane_count = rng.randint(0, 10)

        required = row_count * lane_count if row_count and lane_count else 0

        gate_capacity = required + rng.randint(0, 3)
        up_capacity = required + rng.randint(0, 3)
        out_capacity = required + rng.randint(0, 4)
        staging_out_capacity = required + rng.randint(0, 4)

        if rng.random() < 0.18 and required > 0:
            staging_out_capacity = rng.randint(0, required - 1)

        gate = [0] * max(gate_capacity, 1)
        up = [0] * max(up_capacity, 1)
        out_a = [rng.randint(-5000, 5000) for _ in range(max(out_capacity, 1))]
        out_b = out_a.copy()
        stage_a = [rng.randint(-5000, 5000) for _ in range(max(staging_out_capacity, 1))]
        stage_b = stage_a.copy()

        gate_base_addr = 0x100000
        up_base_addr = 0x200000
        out_base_addr = 0x300000
        stage_base_addr = 0x400000

        if rng.random() < 0.2 and required > 0:
            stage_base_addr = gate_base_addr + rng.randint(0, required * 8 - 1)
        elif rng.random() < 0.2 and required > 0:
            stage_base_addr = out_base_addr + rng.randint(0, required * 8 - 1)

        err_got = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_a,
            out_capacity,
            row_count,
            lane_count,
            stage_a,
            staging_out_capacity,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )
        err_exp = explicit_checked_default_stride_alias_safe_default_capacity_composition(
            gate,
            gate_capacity,
            up,
            up_capacity,
            out_b,
            out_capacity,
            row_count,
            lane_count,
            stage_b,
            staging_out_capacity,
            gate_base_addr=gate_base_addr,
            up_base_addr=up_base_addr,
            out_base_addr=out_base_addr,
            stage_base_addr=stage_base_addr,
        )

        assert err_got == err_exp
        assert out_a == out_b
        assert stage_a == stage_b


def test_error_paths_preserve_outputs() -> None:
    gate = [0, 0, 0, 0]
    up = [0, 0, 0, 0]
    out = [0x5151, 0x5252, 0x5353, 0x5454]
    stage = [0x6161, 0x6262, 0x6363, 0x6464]

    out_before = out.copy()
    stage_before = stage.copy()

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
            None, 4, up, 4, out, 4, 2, 2, stage, 4
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert out == out_before
    assert stage == stage_before

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride_noalloc_commit_capacity_alias_safe_default_capacity(
            gate, 4, up, 4, out, 4, 2, -1, stage, 4
        )
        == FFN_Q16_ERR_BAD_PARAM
    )
    assert out == out_before
    assert stage == stage_before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
