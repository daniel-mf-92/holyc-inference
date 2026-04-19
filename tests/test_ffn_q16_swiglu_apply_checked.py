#!/usr/bin/env python3
"""Parity and contract checks for FFNQ16SwiGLUApplyChecked."""

from __future__ import annotations

import math
import random
from pathlib import Path

FFN_Q16_OK = 0
FFN_Q16_ERR_NULL_PTR = 1
FFN_Q16_ERR_BAD_PARAM = 2
FFN_Q16_ERR_OVERFLOW = 4

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1
Q16_SHIFT = 16
Q16_ONE = 1 << Q16_SHIFT

EXP_Q16_MAX_INPUT = 10 << 16
EXP_Q16_MIN_INPUT = -(10 << 16)


def q16_round_half_up_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return (value + (1 << (shift - 1))) >> shift
    return -(((-value) + (1 << (shift - 1))) >> shift)


def i64_add_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return FFN_Q16_ERR_OVERFLOW, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return FFN_Q16_ERR_OVERFLOW, 0
    out = lhs + rhs
    if out < I64_MIN or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def i64_mul_checked(lhs: int, rhs: int) -> tuple[int, int]:
    out = lhs * rhs
    if out < I64_MIN or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def fp_q16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    err, prod = i64_mul_checked(a_q16, b_q16)
    if err != FFN_Q16_OK:
        return err, 0
    out = q16_round_half_up_signed(prod, Q16_SHIFT)
    if out < I64_MIN or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def fp_q16_div_checked(num_q16: int, den_q16: int) -> tuple[int, int]:
    if den_q16 == 0:
        return FFN_Q16_ERR_BAD_PARAM, 0

    sign_negative = (num_q16 < 0) ^ (den_q16 < 0)
    abs_num = abs(num_q16)
    abs_den = abs(den_q16)

    scaled_num = abs_num << Q16_SHIFT
    q, r = divmod(scaled_num, abs_den)
    if r >= ((abs_den + 1) >> 1):
        q += 1

    out = -q if sign_negative else q
    if out < I64_MIN or out > I64_MAX:
        return FFN_Q16_ERR_OVERFLOW, 0
    return FFN_Q16_OK, out


def fp_q16_exp_clamp_to_input_domain_checked(input_q16: int) -> tuple[int, int]:
    if input_q16 < EXP_Q16_MIN_INPUT:
        return FFN_Q16_OK, EXP_Q16_MIN_INPUT
    if input_q16 > EXP_Q16_MAX_INPUT:
        return FFN_Q16_OK, EXP_Q16_MAX_INPUT
    return FFN_Q16_OK, input_q16


def fp_q16_exp_from_clamped_input_checked(clamped_input_q16: int) -> tuple[int, int]:
    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        return FFN_Q16_ERR_BAD_PARAM, 0

    x = clamped_input_q16 / float(Q16_ONE)
    value = int(round(math.exp(x) * Q16_ONE))
    if value < 0:
        value = 0
    if value > I64_MAX:
        value = I64_MAX
    return FFN_Q16_OK, value


def ffn_q16_sigmoid_checked(gate_q16: int) -> tuple[int, int]:
    if gate_q16 == I64_MIN:
        return FFN_Q16_ERR_OVERFLOW, 0

    neg_gate_q16 = -gate_q16
    err, clamped_neg = fp_q16_exp_clamp_to_input_domain_checked(neg_gate_q16)
    if err != FFN_Q16_OK:
        return err, 0

    err, exp_neg = fp_q16_exp_from_clamped_input_checked(clamped_neg)
    if err != FFN_Q16_OK:
        return err, 0

    err, denom_q16 = i64_add_checked(Q16_ONE, exp_neg)
    if err != FFN_Q16_OK:
        return err, 0

    return fp_q16_div_checked(Q16_ONE, denom_q16)


def ffn_q16_silu_checked(gate_q16: int) -> tuple[int, int]:
    err, sigmoid_q16 = ffn_q16_sigmoid_checked(gate_q16)
    if err != FFN_Q16_OK:
        return err, 0
    return fp_q16_mul_checked(gate_q16, sigmoid_q16)


def ffn_q16_swiglu_lane_checked(gate_q16: int, up_q16: int) -> tuple[int, int]:
    err, silu_q16 = ffn_q16_silu_checked(gate_q16)
    if err != FFN_Q16_OK:
        return err, 0
    return fp_q16_mul_checked(silu_q16, up_q16)


def ffn_q16_swiglu_apply_checked(
    gate_q16,
    gate_capacity: int,
    gate_stride: int,
    up_q16,
    up_capacity: int,
    up_stride: int,
    out_q16,
    out_capacity: int,
    out_stride: int,
    lane_count: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR

    if gate_capacity < 0 or up_capacity < 0 or out_capacity < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if gate_stride < 0 or up_stride < 0 or out_stride < 0:
        return FFN_Q16_ERR_BAD_PARAM
    if lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        return FFN_Q16_OK

    if gate_stride < 1 or up_stride < 1 or out_stride < 1:
        return FFN_Q16_ERR_BAD_PARAM

    required_gate = (lane_count - 1) * gate_stride + 1
    required_up = (lane_count - 1) * up_stride + 1
    required_out = (lane_count - 1) * out_stride + 1

    if required_gate > gate_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_up > up_capacity:
        return FFN_Q16_ERR_BAD_PARAM
    if required_out > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    for lane_index in range(lane_count):
        gate_base = lane_index * gate_stride
        up_base = lane_index * up_stride
        out_base = lane_index * out_stride

        err, lane_out_q16 = ffn_q16_swiglu_lane_checked(gate_q16[gate_base], up_q16[up_base])
        if err != FFN_Q16_OK:
            return err

        out_q16[out_base] = lane_out_q16

    return FFN_Q16_OK


def scalar_reference(gate: list[int], up: list[int]) -> list[int]:
    out: list[int] = []
    for gate_lane, up_lane in zip(gate, up):
        err, lane = ffn_q16_swiglu_lane_checked(gate_lane, up_lane)
        assert err == FFN_Q16_OK
        out.append(lane)
    return out


def test_source_contains_helper_symbol() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyChecked(" in source


def test_known_vectors_match_reference() -> None:
    lane_count = 6
    gate_stride = 2
    up_stride = 3
    out_stride = 2

    gate = [0] * (1 + (lane_count - 1) * gate_stride)
    up = [0] * (1 + (lane_count - 1) * up_stride)

    gate_values = [-(3 << 16), -(1 << 16), -(1 << 14), (1 << 14), (1 << 16), (2 << 16)]
    up_values = [-(2 << 16), -(1 << 15), (1 << 16), (3 << 15), -(5 << 15), (7 << 14)]

    for i in range(lane_count):
        gate[i * gate_stride] = gate_values[i]
        up[i * up_stride] = up_values[i]

    out = [0x7777] * (1 + (lane_count - 1) * out_stride)
    err = ffn_q16_swiglu_apply_checked(
        gate,
        len(gate),
        gate_stride,
        up,
        len(up),
        up_stride,
        out,
        len(out),
        out_stride,
        lane_count,
    )
    assert err == FFN_Q16_OK

    expected = scalar_reference(gate_values, up_values)
    for i in range(lane_count):
        assert out[i * out_stride] == expected[i]


def test_randomized_parity() -> None:
    random.seed(0xFF5516)

    for _ in range(300):
        lane_count = random.randint(0, 48)
        gate_stride = random.randint(1, 6)
        up_stride = random.randint(1, 6)
        out_stride = random.randint(1, 6)

        gate_cap = 0 if lane_count == 0 else 1 + (lane_count - 1) * gate_stride
        up_cap = 0 if lane_count == 0 else 1 + (lane_count - 1) * up_stride
        out_cap = 0 if lane_count == 0 else 1 + (lane_count - 1) * out_stride

        gate = [0] * max(gate_cap, 1)
        up = [0] * max(up_cap, 1)
        out = [0x4444] * max(out_cap, 1)

        gate_values: list[int] = []
        up_values: list[int] = []

        for i in range(lane_count):
            gate_lane = random.randint(-(8 << 16), (8 << 16))
            up_lane = random.randint(-(8 << 16), (8 << 16))
            gate_values.append(gate_lane)
            up_values.append(up_lane)
            gate[i * gate_stride] = gate_lane
            up[i * up_stride] = up_lane

        err = ffn_q16_swiglu_apply_checked(
            gate,
            gate_cap,
            gate_stride,
            up,
            up_cap,
            up_stride,
            out,
            out_cap,
            out_stride,
            lane_count,
        )
        assert err == FFN_Q16_OK

        expected = scalar_reference(gate_values, up_values)
        for i in range(lane_count):
            assert out[i * out_stride] == expected[i]


def test_adversarial_error_contracts() -> None:
    gate = [1, 2, 3]
    up = [4, 5, 6]
    out = [7, 8, 9]

    assert ffn_q16_swiglu_apply_checked(None, 3, 1, up, 3, 1, out, 3, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, None, 3, 1, out, 3, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 1, None, 3, 1, 1) == FFN_Q16_ERR_NULL_PTR

    assert ffn_q16_swiglu_apply_checked(gate, -1, 1, up, 3, 1, out, 3, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, -1, up, 3, 1, out, 3, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, -1, 1, out, 3, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, -1, out, 3, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 1, out, -1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 1, out, 3, -1, 1) == FFN_Q16_ERR_BAD_PARAM

    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 1, out, 3, 1, -1) == FFN_Q16_ERR_BAD_PARAM

    assert ffn_q16_swiglu_apply_checked(gate, 3, 0, up, 3, 1, out, 3, 1, 2) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 0, out, 3, 1, 2) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 1, out, 3, 0, 2) == FFN_Q16_ERR_BAD_PARAM

    assert ffn_q16_swiglu_apply_checked(gate, 1, 1, up, 3, 1, out, 3, 1, 2) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 1, 1, out, 3, 1, 2) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked(gate, 3, 1, up, 3, 1, out, 1, 1, 2) == FFN_Q16_ERR_BAD_PARAM

    gate_overflow = [I64_MIN]
    up_overflow = [Q16_ONE]
    out_overflow = [0]
    assert (
        ffn_q16_swiglu_apply_checked(
            gate_overflow,
            1,
            1,
            up_overflow,
            1,
            1,
            out_overflow,
            1,
            1,
            1,
        )
        == FFN_Q16_ERR_OVERFLOW
    )


if __name__ == "__main__":
    test_source_contains_helper_symbol()
    test_known_vectors_match_reference()
    test_randomized_parity()
    test_adversarial_error_contracts()
    print("ok")
