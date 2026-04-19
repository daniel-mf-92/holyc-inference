#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsCheckedDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_default_stride(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    default_row_stride = lane_count
    return rows_core.ffn_q16_swiglu_apply_rows_checked(
        gate_q16,
        gate_capacity,
        default_row_stride,
        up_q16,
        up_capacity,
        default_row_stride,
        out_q16,
        out_capacity,
        default_row_stride,
        row_count,
        lane_count,
    )


def explicit_default_stride_composition(
    gate_q16,
    gate_capacity: int,
    up_q16,
    up_capacity: int,
    out_q16,
    out_capacity: int,
    row_count: int,
    lane_count: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    return rows_core.ffn_q16_swiglu_apply_rows_checked(
        gate_q16,
        gate_capacity,
        lane_count,
        up_q16,
        up_capacity,
        lane_count,
        out_q16,
        out_capacity,
        lane_count,
        row_count,
        lane_count,
    )


def test_source_contains_default_stride_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyRowsCheckedDefaultStride(" in source
    body = source.split("I32 FFNQ16SwiGLUApplyRowsCheckedDefaultStride(", 1)[1]
    assert "default_row_stride = lane_count;" in body
    assert "return FFNQ16SwiGLUApplyRowsChecked(gate_q16," in body


def test_known_vectors_match_explicit_stride_rows() -> None:
    row_count = 4
    lane_count = 6
    gate_cap = row_count * lane_count
    up_cap = row_count * lane_count
    out_cap = row_count * lane_count

    gate = [0] * gate_cap
    up = [0] * up_cap

    for row_index in range(row_count):
        for lane_index in range(lane_count):
            idx = row_index * lane_count + lane_index
            gate[idx] = ((idx - 9) * 3) << 13
            up[idx] = (11 - idx) << 12

    out_a = [0x3131] * out_cap
    out_b = [0x3131] * out_cap

    err_a = ffn_q16_swiglu_apply_rows_checked_default_stride(
        gate,
        gate_cap,
        up,
        up_cap,
        out_a,
        out_cap,
        row_count,
        lane_count,
    )
    err_b = explicit_default_stride_composition(
        gate,
        gate_cap,
        up,
        up_cap,
        out_b,
        out_cap,
        row_count,
        lane_count,
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b


def test_adversarial_guard_parity() -> None:
    gate = [1]
    up = [1]
    out = [1]

    assert ffn_q16_swiglu_apply_rows_checked_default_stride(None, 1, up, 1, out, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked_default_stride(gate, 1, None, 1, out, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked_default_stride(gate, 1, up, 1, None, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR

    assert ffn_q16_swiglu_apply_rows_checked_default_stride(gate, 1, up, 1, out, 1, -1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked_default_stride(gate, 1, up, 1, out, 1, 1, -1) == FFN_Q16_ERR_BAD_PARAM


def test_overflow_parity_against_explicit_stride() -> None:
    gate = [0]
    up = [0]
    out_a = [0]
    out_b = [0]

    err_a = ffn_q16_swiglu_apply_rows_checked_default_stride(
        gate,
        1,
        up,
        1,
        out_a,
        1,
        2,
        I64_MAX,
    )
    err_b = explicit_default_stride_composition(
        gate,
        1,
        up,
        1,
        out_b,
        1,
        2,
        I64_MAX,
    )

    assert err_a == err_b


def test_randomized_parity_vs_explicit_stride() -> None:
    random.seed(0xFF583)

    for _ in range(320):
        row_count = random.randint(0, 9)
        lane_count = random.randint(0, 10)

        required_cells = 0 if row_count == 0 else (row_count - 1) * lane_count + lane_count

        gate_cap = required_cells
        up_cap = required_cells
        out_cap = required_cells

        gate = [0] * max(gate_cap, 1)
        up = [0] * max(up_cap, 1)

        for row_index in range(row_count):
            base = row_index * lane_count
            for lane_index in range(lane_count):
                gate[base + lane_index] = random.randint(-(8 << 16), (8 << 16))
                up[base + lane_index] = random.randint(-(8 << 16), (8 << 16))

        out_a = [0x5A5A] * max(out_cap, 1)
        out_b = out_a.copy()

        err_a = ffn_q16_swiglu_apply_rows_checked_default_stride(
            gate,
            gate_cap,
            up,
            up_cap,
            out_a,
            out_cap,
            row_count,
            lane_count,
        )
        err_b = explicit_default_stride_composition(
            gate,
            gate_cap,
            up,
            up_cap,
            out_b,
            out_cap,
            row_count,
            lane_count,
        )

        assert err_a == err_b
        assert out_a == out_b
