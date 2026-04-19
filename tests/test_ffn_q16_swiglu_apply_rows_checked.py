#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyRowsChecked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_checked as core


FFN_Q16_OK = core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = core.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = core.FFN_Q16_ERR_OVERFLOW
I64_MAX = core.I64_MAX
I64_MIN = core.I64_MIN


def i64_mul_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs * rhs
    if out < I64_MIN or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def i64_add_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return FFN_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return FFN_Q16_ERR_OVERFLOW, 0
    out = lhs + rhs
    if out < I64_MIN or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def ffn_q16_swiglu_apply_rows_checked(
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

    err, required_gate = i64_mul_checked(row_count - 1, gate_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_gate = i64_add_checked(required_gate, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_up = i64_mul_checked(row_count - 1, up_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_up = i64_add_checked(required_up, lane_count)
    if err != FFN_Q16_OK:
        return err

    err, required_out = i64_mul_checked(row_count - 1, out_row_stride)
    if err != FFN_Q16_OK:
        return err
    err, required_out = i64_add_checked(required_out, lane_count)
    if err != FFN_Q16_OK:
        return err

    if required_gate > gate_capacity or required_up > up_capacity or required_out > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        gate_base = row_index * gate_row_stride
        up_base = row_index * up_row_stride
        out_base = row_index * out_row_stride

        row_out = [0] * lane_count
        err = core.ffn_q16_swiglu_apply_checked(
            gate_q16[gate_base : gate_base + lane_count],
            lane_count,
            1,
            up_q16[up_base : up_base + lane_count],
            lane_count,
            1,
            row_out,
            lane_count,
            1,
            lane_count,
        )
        if err != FFN_Q16_OK:
            return err

        for lane_idx in range(lane_count):
            out_q16[out_base + lane_idx] = row_out[lane_idx]

    return FFN_Q16_OK


def explicit_row_composition(
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

    required_gate = (row_count - 1) * gate_row_stride + lane_count
    required_up = (row_count - 1) * up_row_stride + lane_count
    required_out = (row_count - 1) * out_row_stride + lane_count
    if required_gate > gate_capacity or required_up > up_capacity or required_out > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    for row_index in range(row_count):
        gate_base = row_index * gate_row_stride
        up_base = row_index * up_row_stride
        out_base = row_index * out_row_stride

        row_out = [0] * lane_count
        err = core.ffn_q16_swiglu_apply_checked(
            gate_q16[gate_base : gate_base + lane_count],
            lane_count,
            1,
            up_q16[up_base : up_base + lane_count],
            lane_count,
            1,
            row_out,
            lane_count,
            1,
            lane_count,
        )
        if err != FFN_Q16_OK:
            return err

        for lane_idx in range(lane_count):
            out_q16[out_base + lane_idx] = row_out[lane_idx]

    return FFN_Q16_OK


def test_source_contains_helper_symbol() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyRowsChecked(" in source


def test_known_vectors_match_row_reference() -> None:
    row_count = 3
    lane_count = 5
    gate_row_stride = 7
    up_row_stride = 6
    out_row_stride = 8

    gate_cap = (row_count - 1) * gate_row_stride + lane_count
    up_cap = (row_count - 1) * up_row_stride + lane_count
    out_cap = (row_count - 1) * out_row_stride + lane_count

    gate = [0] * gate_cap
    up = [0] * up_cap

    for r in range(row_count):
        for l in range(lane_count):
            gate[r * gate_row_stride + l] = ((r * lane_count + l) - 7) * (1 << 14)
            up[r * up_row_stride + l] = (9 - (r * lane_count + l)) * (1 << 13)

    out_a = [0x1A1A] * out_cap
    out_b = [0x1A1A] * out_cap

    err_a = ffn_q16_swiglu_apply_rows_checked(
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
    err_b = explicit_row_composition(
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

    assert err_a == FFN_Q16_OK
    assert err_a == err_b
    assert out_a == out_b


def test_adversarial_param_guards() -> None:
    gate = [1]
    up = [2]
    out = [3]

    assert ffn_q16_swiglu_apply_rows_checked(None, 1, 1, up, 1, 1, out, 1, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, 1, None, 1, 1, out, 1, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, 1, up, 1, 1, None, 1, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR

    assert ffn_q16_swiglu_apply_rows_checked(gate, -1, 1, up, 1, 1, out, 1, 1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, -1, up, 1, 1, out, 1, 1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, 1, up, 1, -1, out, 1, 1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, 1, up, 1, 1, out, 1, -1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, 1, up, 1, 1, out, 1, 1, -1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_rows_checked(gate, 1, 1, up, 1, 1, out, 1, 1, 1, -1) == FFN_Q16_ERR_BAD_PARAM


def test_randomized_parity_vs_composition() -> None:
    random.seed(0xFF581)

    for _ in range(250):
        row_count = random.randint(0, 10)
        lane_count = random.randint(0, 12)

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

        if row_count > 0 and lane_count > 0 and random.random() < 0.12:
            bad_r = random.randint(0, row_count - 1)
            bad_l = random.randint(0, lane_count - 1)
            gate[bad_r * gate_row_stride + bad_l] = I64_MIN

        out_a = [0x4242] * max(out_cap, 1)
        out_b = list(out_a)

        err_a = ffn_q16_swiglu_apply_rows_checked(
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
        err_b = explicit_row_composition(
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


def test_row_stride_and_capacity_failures() -> None:
    row_count = 2
    lane_count = 4

    gate = [0] * 8
    up = [0] * 8
    out = [0x7777] * 8

    err = ffn_q16_swiglu_apply_rows_checked(
        gate,
        len(gate),
        3,
        up,
        len(up),
        4,
        out,
        len(out),
        4,
        row_count,
        lane_count,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM

    err = ffn_q16_swiglu_apply_rows_checked(
        gate,
        7,
        4,
        up,
        len(up),
        4,
        out,
        len(out),
        4,
        row_count,
        lane_count,
    )
    assert err == FFN_Q16_ERR_BAD_PARAM


def test_overflow_vectors_return_overflow() -> None:
    gate = [0]
    up = [0]
    out = [0]

    err = ffn_q16_swiglu_apply_rows_checked(
        gate,
        1,
        I64_MAX,
        up,
        1,
        1,
        out,
        1,
        1,
        2,
        1,
    )
    assert err == FFN_Q16_ERR_OVERFLOW

    err = ffn_q16_swiglu_apply_rows_checked(
        gate,
        1,
        1,
        up,
        1,
        I64_MAX,
        out,
        1,
        1,
        2,
        1,
    )
    assert err == FFN_Q16_ERR_OVERFLOW

    err = ffn_q16_swiglu_apply_rows_checked(
        gate,
        1,
        1,
        up,
        1,
        1,
        out,
        1,
        I64_MAX,
        2,
        1,
    )
    assert err == FFN_Q16_ERR_OVERFLOW


def test_row_padding_cells_are_preserved() -> None:
    row_count = 3
    lane_count = 4
    gate_row_stride = 7
    up_row_stride = 7
    out_row_stride = 9

    gate_cap = (row_count - 1) * gate_row_stride + lane_count
    up_cap = (row_count - 1) * up_row_stride + lane_count
    out_cap = (row_count - 1) * out_row_stride + lane_count + 3

    gate = [0] * gate_cap
    up = [0] * up_cap
    out = [0x6D6D] * out_cap

    for r in range(row_count):
        for l in range(lane_count):
            gate[r * gate_row_stride + l] = (r * 11 - l * 7) * (1 << 13)
            up[r * up_row_stride + l] = (l * 5 - r * 3) * (1 << 12)

    err = ffn_q16_swiglu_apply_rows_checked(
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
    assert err == FFN_Q16_OK

    for r in range(row_count):
        row_base = r * out_row_stride
        for l in range(lane_count, out_row_stride):
            if row_base + l < out_cap:
                assert out[row_base + l] == 0x6D6D


if __name__ == "__main__":
    test_source_contains_helper_symbol()
    test_known_vectors_match_row_reference()
    test_adversarial_param_guards()
    test_randomized_parity_vs_composition()
    test_row_stride_and_capacity_failures()
    test_overflow_vectors_return_overflow()
    test_row_padding_cells_are_preserved()
    print("ok")
