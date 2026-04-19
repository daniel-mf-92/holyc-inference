#!/usr/bin/env python3
"""No-partial default-stride parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStride."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked_nopartial as rows_np


FFN_Q16_OK = rows_np.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_np.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_np.FFN_Q16_ERR_BAD_PARAM


def ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(
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
    return rows_np.ffn_q16_swiglu_apply_rows_checked_nopartial(
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


def explicit_staged_default_stride_composition(
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

    return rows_np.explicit_staged_row_composition(
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


def test_source_contains_nopartial_default_stride_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStride(" in source
    body = source.split("I32 FFNQ16SwiGLUApplyRowsCheckedNoPartialDefaultStride(", 1)[1]
    assert "default_row_stride = lane_count;" in body
    assert "return FFNQ16SwiGLUApplyRowsCheckedNoPartial(gate_q16," in body


def test_known_vectors_match_explicit_staged_default_stride() -> None:
    row_count = 5
    lane_count = 4
    gate_cap = row_count * lane_count
    up_cap = row_count * lane_count
    out_cap = row_count * lane_count

    gate = [0] * gate_cap
    up = [0] * up_cap

    for row_idx in range(row_count):
        for lane_idx in range(lane_count):
            index = row_idx * lane_count + lane_idx
            gate[index] = ((index - 8) * 5) << 13
            up[index] = (14 - index) << 12

    out_a = [0x4747] * out_cap
    out_b = [0x4747] * out_cap

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(
        gate,
        gate_cap,
        up,
        up_cap,
        out_a,
        out_cap,
        row_count,
        lane_count,
    )
    err_b = explicit_staged_default_stride_composition(
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


def test_no_partial_preserved_on_failure() -> None:
    row_count = 3
    lane_count = 3
    capacity = row_count * lane_count

    gate = [1 << 16] * capacity
    up = [1 << 16] * capacity
    out = [0x6C6C] * capacity

    gate[2] = rows_np.rows_core.I64_MIN

    before = out.copy()
    err = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(
        gate,
        capacity,
        up,
        capacity,
        out,
        capacity,
        row_count,
        lane_count,
    )

    assert err != FFN_Q16_OK
    assert out == before


def test_guard_parity() -> None:
    gate = [1]
    up = [1]
    out = [1]

    assert ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(None, 1, up, 1, out, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(gate, 1, None, 1, out, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(gate, 1, up, 1, None, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(gate, 1, up, 1, out, 1, -1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(gate, 1, up, 1, out, 1, 1, -1) == FFN_Q16_ERR_BAD_PARAM


def test_randomized_parity_vs_explicit_staged_default_stride() -> None:
    random.seed(0xFF584)

    for _ in range(300):
        row_count = random.randint(0, 10)
        lane_count = random.randint(0, 10)

        capacity = 0 if row_count == 0 else row_count * lane_count

        gate = [0] * max(capacity, 1)
        up = [0] * max(capacity, 1)

        for row_idx in range(row_count):
            row_base = row_idx * lane_count
            for lane_idx in range(lane_count):
                gate[row_base + lane_idx] = random.randint(-(8 << 16), (8 << 16))
                up[row_base + lane_idx] = random.randint(-(8 << 16), (8 << 16))

        if row_count > 0 and lane_count > 0 and random.random() < 0.14:
            fail_row = random.randint(0, row_count - 1)
            fail_lane = random.randint(0, lane_count - 1)
            gate[fail_row * lane_count + fail_lane] = rows_np.rows_core.I64_MIN

        out_a = [0x2E2E] * max(capacity, 1)
        out_b = out_a.copy()

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial_default_stride(
            gate,
            capacity,
            up,
            capacity,
            out_a,
            capacity,
            row_count,
            lane_count,
        )
        err_b = explicit_staged_default_stride_composition(
            gate,
            capacity,
            up,
            capacity,
            out_b,
            capacity,
            row_count,
            lane_count,
        )

        assert err_a == err_b
        assert out_a == out_b
