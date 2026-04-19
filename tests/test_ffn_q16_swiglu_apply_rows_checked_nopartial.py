#!/usr/bin/env python3
"""No-partial parity harness for FFNQ16SwiGLUApplyRowsCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_rows_checked as rows_core


FFN_Q16_OK = rows_core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = rows_core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = rows_core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = rows_core.FFN_Q16_ERR_OVERFLOW
I64_MAX = rows_core.I64_MAX


def ffn_q16_swiglu_apply_rows_checked_nopartial(
    gate_q16,
    gate_capacity: int,
    gate_row_stride: int,
    up_q16,
    up_capacity: int,
    up_row_stride: int,
    out_q16,
    out_capacity: int,
    out_row_stride: int,
    row_count: int,
    lane_count: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if gate_row_stride < 0 or up_row_stride < 0 or out_row_stride < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if gate_row_stride < lane_count or up_row_stride < lane_count or out_row_stride < lane_count:
        return FFN_Q16_ERR_BAD_PARAM

    err, required_out_cells = rows_core.i64_mul_checked(row_count - 1, out_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = rows_core.i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    staged_out = [0] * required_out_cells
    err = rows_core.ffn_q16_swiglu_apply_rows_checked(
        gate_q16,
        gate_capacity,
        gate_row_stride,
        up_q16,
        up_capacity,
        up_row_stride,
        staged_out,
        required_out_cells,
        out_row_stride,
        row_count,
        lane_count,
    )
    if err != FFN_Q16_OK:
        return err

    for row_index in range(row_count):
        row_base = row_index * out_row_stride
        for lane_index in range(lane_count):
            out_index = row_base + lane_index
            out_q16[out_index] = staged_out[out_index]

    return FFN_Q16_OK


def explicit_staged_row_composition(
    gate_q16,
    gate_capacity: int,
    gate_row_stride: int,
    up_q16,
    up_capacity: int,
    up_row_stride: int,
    out_q16,
    out_capacity: int,
    out_row_stride: int,
    row_count: int,
    lane_count: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if gate_row_stride < 0 or up_row_stride < 0 or out_row_stride < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if row_count < 0 or lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if row_count == 0 or lane_count == 0:
        return FFN_Q16_OK

    if gate_row_stride < lane_count or up_row_stride < lane_count or out_row_stride < lane_count:
        return FFN_Q16_ERR_BAD_PARAM

    err, required_out_cells = rows_core.i64_mul_checked(row_count - 1, out_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_out_cells = rows_core.i64_add_checked(required_out_cells, lane_count)
    if err != FFN_Q16_OK:
        return err
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    staged_out = [0] * required_out_cells
    err = rows_core.ffn_q16_swiglu_apply_rows_checked(
        gate_q16,
        gate_capacity,
        gate_row_stride,
        up_q16,
        up_capacity,
        up_row_stride,
        staged_out,
        required_out_cells,
        out_row_stride,
        row_count,
        lane_count,
    )
    if err != FFN_Q16_OK:
        return err

    for row_index in range(row_count):
        row_base = row_index * out_row_stride
        for lane_index in range(lane_count):
            out_index = row_base + lane_index
            out_q16[out_index] = staged_out[out_index]

    return FFN_Q16_OK


def test_source_contains_rows_nopartial_helper() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyRowsCheckedNoPartial(" in source
    body = source.split("I32 FFNQ16SwiGLUApplyRowsCheckedNoPartial(", 1)[1]
    assert "staging_out_q16 = MAlloc(staging_bytes);" in body
    assert "status = FFNQ16SwiGLUApplyRowsChecked(gate_q16," in body


def test_known_vectors_match_explicit_staged_composition() -> None:
    row_count = 4
    lane_count = 5
    gate_row_stride = 7
    up_row_stride = 8
    out_row_stride = 9

    gate_cap = (row_count - 1) * gate_row_stride + lane_count
    up_cap = (row_count - 1) * up_row_stride + lane_count
    out_cap = (row_count - 1) * out_row_stride + lane_count + 4

    gate = [0] * gate_cap
    up = [0] * up_cap
    out_a = [0x5151] * out_cap
    out_b = out_a.copy()

    for r in range(row_count):
        for l in range(lane_count):
            gate[r * gate_row_stride + l] = ((r * 17 - l * 9) << 12)
            up[r * up_row_stride + l] = ((l * 13 - r * 5) << 11)

    err_a = ffn_q16_swiglu_apply_rows_checked_nopartial(
        gate,
        gate_cap,
        gate_row_stride,
        up,
        up_cap,
        up_row_stride,
        out_a,
        out_cap,
        out_row_stride,
        row_count,
        lane_count,
    )
    err_b = explicit_staged_row_composition(
        gate,
        gate_cap,
        gate_row_stride,
        up,
        up_cap,
        up_row_stride,
        out_b,
        out_cap,
        out_row_stride,
        row_count,
        lane_count,
    )

    assert err_a == err_b == FFN_Q16_OK
    assert out_a == out_b

    for r in range(row_count):
        row_base = r * out_row_stride
        for l in range(lane_count, out_row_stride):
            if row_base + l < out_cap:
                assert out_a[row_base + l] == 0x5151


def test_failures_preserve_output_no_partial() -> None:
    row_count = 2
    lane_count = 3
    gate_row_stride = 3
    up_row_stride = 3
    out_row_stride = 3

    gate_cap = (row_count - 1) * gate_row_stride + lane_count
    up_cap = (row_count - 1) * up_row_stride + lane_count
    out_cap = (row_count - 1) * out_row_stride + lane_count

    gate = [1 << 16] * gate_cap
    up = [1 << 16] * up_cap
    out = [0x7E7E] * out_cap

    gate[1] = rows_core.I64_MIN

    before = out.copy()
    err = ffn_q16_swiglu_apply_rows_checked_nopartial(
        gate,
        gate_cap,
        gate_row_stride,
        up,
        up_cap,
        up_row_stride,
        out,
        out_cap,
        out_row_stride,
        row_count,
        lane_count,
    )
    assert err == FFN_Q16_ERR_OVERFLOW
    assert out == before


def test_guard_and_overflow_parity() -> None:
    gate = [0]
    up = [0]
    out = [0x2222, 0x2222, 0x2222]

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            None, 1, 1, up, 1, 1, out, 1, 1, 1, 1
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate, 1, 1, None, 1, 1, out, 1, 1, 1, 1
        )
        == FFN_Q16_ERR_NULL_PTR
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate, 1, 1, up, 1, 1, None, 1, 1, 1, 1
        )
        == FFN_Q16_ERR_NULL_PTR
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate, -1, 1, up, 1, 1, out, 1, 1, 1, 1
        )
        == FFN_Q16_ERR_BAD_PARAM
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate, 1, 0, up, 1, 1, out, 1, 1, 1, 1
        )
        == FFN_Q16_ERR_BAD_PARAM
    )

    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate, 1, I64_MAX, up, 1, 1, out, 3, 1, 2, 1
        )
        == FFN_Q16_ERR_OVERFLOW
    )
    assert (
        ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate, 1, 1, up, 1, 1, out, 1, I64_MAX, 2, 1
        )
        == FFN_Q16_ERR_OVERFLOW
    )


def test_randomized_parity_and_no_partial() -> None:
    random.seed(0xFF582)

    for _ in range(260):
        row_count = random.randint(0, 9)
        lane_count = random.randint(0, 10)

        min_stride = max(lane_count, 1)
        gate_row_stride = random.randint(min_stride, min_stride + 4)
        up_row_stride = random.randint(min_stride, min_stride + 4)
        out_row_stride = random.randint(min_stride, min_stride + 4)

        gate_cap = 0 if row_count == 0 else (row_count - 1) * gate_row_stride + lane_count
        up_cap = 0 if row_count == 0 else (row_count - 1) * up_row_stride + lane_count
        out_cap = 0 if row_count == 0 else (row_count - 1) * out_row_stride + lane_count

        gate = [0] * max(gate_cap, 1)
        up = [0] * max(up_cap, 1)

        for r in range(row_count):
            for l in range(lane_count):
                gate[r * gate_row_stride + l] = random.randint(-(8 << 16), (8 << 16))
                up[r * up_row_stride + l] = random.randint(-(8 << 16), (8 << 16))

        if row_count > 0 and lane_count > 0 and random.random() < 0.16:
            bad_r = random.randint(0, row_count - 1)
            bad_l = random.randint(0, lane_count - 1)
            gate[bad_r * gate_row_stride + bad_l] = rows_core.I64_MIN

        out_a = [0x3B3B] * max(out_cap, 1)
        out_b = out_a.copy()

        err_a = ffn_q16_swiglu_apply_rows_checked_nopartial(
            gate,
            gate_cap,
            gate_row_stride,
            up,
            up_cap,
            up_row_stride,
            out_a,
            out_cap,
            out_row_stride,
            row_count,
            lane_count,
        )
        err_b = explicit_staged_row_composition(
            gate,
            gate_cap,
            gate_row_stride,
            up,
            up_cap,
            up_row_stride,
            out_b,
            out_cap,
            out_row_stride,
            row_count,
            lane_count,
        )

        assert err_a == err_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_rows_nopartial_helper()
    test_known_vectors_match_explicit_staged_composition()
    test_failures_preserve_output_no_partial()
    test_guard_and_overflow_parity()
    test_randomized_parity_and_no_partial()
    print("ok")
