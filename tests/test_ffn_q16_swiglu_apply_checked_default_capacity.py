#!/usr/bin/env python3
"""Parity harness for FFNQ16SwiGLUApplyCheckedDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))
import test_ffn_q16_swiglu_apply_checked as core


FFN_Q16_OK = core.FFN_Q16_OK
FFN_Q16_ERR_NULL_PTR = core.FFN_Q16_ERR_NULL_PTR
FFN_Q16_ERR_BAD_PARAM = core.FFN_Q16_ERR_BAD_PARAM
I64_MIN = core.I64_MIN


def ffn_q16_swiglu_apply_checked_default_capacity(
    gate_q16,
    up_q16,
    out_q16,
    lane_count: int,
) -> int:
    if gate_q16 is None or up_q16 is None or out_q16 is None:
        return FFN_Q16_ERR_NULL_PTR
    if lane_count < 0:
        return FFN_Q16_ERR_BAD_PARAM
    return core.ffn_q16_swiglu_apply_checked(
        gate_q16,
        lane_count,
        1,
        up_q16,
        lane_count,
        1,
        out_q16,
        lane_count,
        1,
        lane_count,
    )


def test_source_contains_helper_symbol() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    assert "I32 FFNQ16SwiGLUApplyCheckedDefaultCapacity(" in source


def test_zero_lane_is_ok() -> None:
    gate = [0x11]
    up = [0x22]
    out = [0x33]
    err = ffn_q16_swiglu_apply_checked_default_capacity(gate, up, out, 0)
    assert err == FFN_Q16_OK
    assert out == [0x33]


def test_null_and_negative_guards() -> None:
    gate = [1]
    up = [2]
    out = [3]

    assert ffn_q16_swiglu_apply_checked_default_capacity(None, up, out, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked_default_capacity(gate, None, out, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked_default_capacity(gate, up, None, 1) == FFN_Q16_ERR_NULL_PTR
    assert ffn_q16_swiglu_apply_checked_default_capacity(gate, up, out, -1) == FFN_Q16_ERR_BAD_PARAM


def test_known_vector_matches_explicit_capacity_core() -> None:
    lane_count = 8
    gate = [0] * lane_count
    up = [0] * lane_count
    for i in range(lane_count):
        gate[i] = (i - 4) * (1 << 15)
        up[i] = (4 - i) * (1 << 14)

    out_default = [0x7777] * lane_count
    out_explicit = [0x7777] * lane_count

    err_default = ffn_q16_swiglu_apply_checked_default_capacity(gate, up, out_default, lane_count)
    err_explicit = core.ffn_q16_swiglu_apply_checked(
        gate,
        lane_count,
        1,
        up,
        lane_count,
        1,
        out_explicit,
        lane_count,
        1,
        lane_count,
    )

    assert err_default == FFN_Q16_OK
    assert err_default == err_explicit
    assert out_default == out_explicit


def test_randomized_parity_vs_explicit_capacity_core() -> None:
    random.seed(0xFF580)

    for _ in range(300):
        lane_count = random.randint(0, 64)

        gate = [0] * max(lane_count, 1)
        up = [0] * max(lane_count, 1)

        for i in range(lane_count):
            gate[i] = random.randint(-(8 << 16), (8 << 16))
            up[i] = random.randint(-(8 << 16), (8 << 16))

        if lane_count > 0 and random.random() < 0.15:
            gate[random.randint(0, lane_count - 1)] = I64_MIN

        out_default = [0x5151] * max(lane_count, 1)
        out_explicit = list(out_default)

        err_default = ffn_q16_swiglu_apply_checked_default_capacity(
            gate,
            up,
            out_default,
            lane_count,
        )
        err_explicit = core.ffn_q16_swiglu_apply_checked(
            gate,
            lane_count,
            1,
            up,
            lane_count,
            1,
            out_explicit,
            lane_count,
            1,
            lane_count,
        )

        assert err_default == err_explicit
        if err_default == FFN_Q16_OK:
            assert out_default == out_explicit


def test_wrapper_uses_lane_count_for_all_capacities() -> None:
    source = Path("src/model/ffn.HC").read_text(encoding="utf-8")
    marker = "I32 FFNQ16SwiGLUApplyCheckedDefaultCapacity("
    assert marker in source

    section = source.split(marker, 1)[1].split(
        "I32 FFNQ16SwiGLUApplyCheckedNoPartial(",
        1,
    )[0]
    assert "FFNQ16SwiGLUApplyChecked(gate_q16," in section
    assert section.count("lane_count") >= 6


if __name__ == "__main__":
    test_source_contains_helper_symbol()
    test_zero_lane_is_ok()
    test_null_and_negative_guards()
    test_known_vector_matches_explicit_capacity_core()
    test_randomized_parity_vs_explicit_capacity_core()
    test_wrapper_uses_lane_count_for_all_capacities()
    print("ok")
