#!/usr/bin/env python3
"""No-partial parity checks for FFNQ16SwiGLUApplyCheckedNoPartial."""

from __future__ import annotations

import random
import importlib.util
from pathlib import Path

_BASE = Path(__file__).resolve().parent / "test_ffn_q16_swiglu_apply_checked.py"
_SPEC = importlib.util.spec_from_file_location("ffn_q16_base", _BASE)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

FFN_Q16_OK = _MOD.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = _MOD.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = _MOD.FFN_Q16_ERR_BAD_PARAM
FFN_Q16_ERR_OVERFLOW = _MOD.FFN_Q16_ERR_OVERFLOW
I64_MIN = _MOD.I64_MIN
Q16_ONE = _MOD.Q16_ONE
ffn_q16_swiglu_apply_checked = _MOD.ffn_q16_swiglu_apply_checked


def ffn_q16_swiglu_apply_checked_nopartial(
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
    if out_stride < 1:
        return FFN_Q16_ERR_BAD_PARAM

    required_out_cells = (lane_count - 1) * out_stride + 1
    if required_out_cells > out_capacity:
        return FFN_Q16_ERR_BAD_PARAM

    staged = [0] * required_out_cells
    err = ffn_q16_swiglu_apply_checked(
        gate_q16,
        gate_capacity,
        gate_stride,
        up_q16,
        up_capacity,
        up_stride,
        staged,
        required_out_cells,
        out_stride,
        lane_count,
    )
    if err != FFN_Q16_OK:
        return err

    for lane_index in range(lane_count):
        out_base = lane_index * out_stride
        out_q16[out_base] = staged[out_base]
    return FFN_Q16_OK


def test_source_contains_helper_symbol() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyCheckedNoPartial(" in source


def test_success_matches_staged_composition() -> None:
    lane_count = 7
    gate_stride = 2
    up_stride = 3
    out_stride = 4

    gate = [0] * (1 + (lane_count - 1) * gate_stride)
    up = [0] * (1 + (lane_count - 1) * up_stride)
    for i in range(lane_count):
        gate[i * gate_stride] = (i - 3) * (1 << 15)
        up[i * up_stride] = (3 - i) * (1 << 15)

    out_nop = [0x5555] * (1 + (lane_count - 1) * out_stride)
    out_ref = list(out_nop)

    err_nop = ffn_q16_swiglu_apply_checked_nopartial(
        gate,
        len(gate),
        gate_stride,
        up,
        len(up),
        up_stride,
        out_nop,
        len(out_nop),
        out_stride,
        lane_count,
    )
    assert err_nop == FFN_Q16_OK

    required = (lane_count - 1) * out_stride + 1
    staged = [0] * required
    err_ref = ffn_q16_swiglu_apply_checked(
        gate,
        len(gate),
        gate_stride,
        up,
        len(up),
        up_stride,
        staged,
        required,
        out_stride,
        lane_count,
    )
    assert err_ref == FFN_Q16_OK

    for i in range(lane_count):
        out_ref[i * out_stride] = staged[i * out_stride]

    assert out_nop == out_ref


def test_error_keeps_output_unchanged() -> None:
    lane_count = 3
    gate_stride = 1
    up_stride = 1
    out_stride = 1

    gate = [0, I64_MIN, 0]
    up = [Q16_ONE, Q16_ONE, Q16_ONE]

    out_before = [0x1234, 0x2345, 0x3456]
    out_after = list(out_before)

    err = ffn_q16_swiglu_apply_checked_nopartial(
        gate,
        len(gate),
        gate_stride,
        up,
        len(up),
        up_stride,
        out_after,
        len(out_after),
        out_stride,
        lane_count,
    )
    assert err == FFN_Q16_ERR_OVERFLOW
    assert out_after == out_before


def test_adversarial_preflight_errors() -> None:
    gate = [1]
    up = [2]
    out = [3]

    assert ffn_q16_swiglu_apply_checked_nopartial(None, 1, 1, up, 1, 1, out, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, None, 1, 1, out, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, up, 1, 1, None, 1, 1, 1) == FFN_Q16_ERR_NULL_PTR

    assert ffn_q16_swiglu_apply_checked_nopartial(gate, -1, 1, up, 1, 1, out, 1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, -1, up, 1, 1, out, 1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, up, 1, -1, out, 1, 1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, up, 1, 1, out, 1, -1, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, up, 1, 1, out, 1, 1, -1) == FFN_Q16_ERR_BAD_PARAM

    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, up, 1, 1, out, 1, 0, 1) == FFN_Q16_ERR_BAD_PARAM
    assert ffn_q16_swiglu_apply_checked_nopartial(gate, 1, 1, up, 1, 1, out, 0, 1, 1) == FFN_Q16_ERR_BAD_PARAM


def test_randomized_success_and_error_parity() -> None:
    random.seed(0xFF579)

    for _ in range(200):
        lane_count = random.randint(0, 24)
        gate_stride = random.randint(1, 4)
        up_stride = random.randint(1, 4)
        out_stride = random.randint(1, 4)

        gate_cap = 0 if lane_count == 0 else 1 + (lane_count - 1) * gate_stride
        up_cap = 0 if lane_count == 0 else 1 + (lane_count - 1) * up_stride
        out_cap = 0 if lane_count == 0 else 1 + (lane_count - 1) * out_stride

        gate = [0] * max(gate_cap, 1)
        up = [0] * max(up_cap, 1)

        for i in range(lane_count):
            gate[i * gate_stride] = random.randint(-(8 << 16), (8 << 16))
            up[i * up_stride] = random.randint(-(8 << 16), (8 << 16))

        if lane_count > 0 and random.random() < 0.25:
            bad_lane = random.randint(0, lane_count - 1)
            gate[bad_lane * gate_stride] = I64_MIN

        out_a = [0x1111] * max(out_cap, 1)
        out_b = list(out_a)

        err_a = ffn_q16_swiglu_apply_checked_nopartial(
            gate,
            gate_cap,
            gate_stride,
            up,
            up_cap,
            up_stride,
            out_a,
            out_cap,
            out_stride,
            lane_count,
        )

        required_out = 0 if lane_count == 0 else 1 + (lane_count - 1) * out_stride
        if required_out > out_cap:
            err_b = FFN_Q16_ERR_BAD_PARAM
        else:
            staged = [0] * max(required_out, 1)
            err_b = ffn_q16_swiglu_apply_checked(
                gate,
                gate_cap,
                gate_stride,
                up,
                up_cap,
                up_stride,
                staged,
                required_out,
                out_stride,
                lane_count,
            )
            if err_b == FFN_Q16_OK:
                for i in range(lane_count):
                    out_b[i * out_stride] = staged[i * out_stride]

        assert err_a == err_b
        if err_a == FFN_Q16_OK:
            assert out_a == out_b
        else:
            assert out_a == [0x1111] * max(out_cap, 1)


if __name__ == "__main__":
    test_source_contains_helper_symbol()
    test_success_matches_staged_composition()
    test_error_keeps_output_unchanged()
    test_adversarial_preflight_errors()
    test_randomized_success_and_error_parity()
    print("ok")
