#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayCheckedDefault wrapper."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_intexp_exp_array_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
    fpq16_exp_array_checked,
)


def fpq16_exp_array_checked_default(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1100,
    synthetic_output_addr: int = 0x2200,
) -> int:
    return fpq16_exp_array_checked(
        input_q16,
        output_q16,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=synthetic_output_addr,
    )[0]


def test_source_contains_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayCheckedDefault" in source
    assert "return FPQ16ExpArrayChecked(input_q16," in source


def test_null_pointer_and_negative_count_surfaces() -> None:
    assert fpq16_exp_array_checked_default(None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_default([0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_default([0], [0], -1) == FP_Q16_ERR_BAD_PARAM


def test_zero_count_no_write() -> None:
    out = [17, 29, 41]
    err = fpq16_exp_array_checked_default([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert out == [17, 29, 41]


def test_pointer_span_overflow_parity() -> None:
    inp = [0, 1]
    out_a = [77, 88]
    out_b = out_a.copy()

    huge_count = (I64_MAX_VALUE >> 3) + 2
    err_a = fpq16_exp_array_checked_default(inp, out_a, huge_count)
    err_b, _ = fpq16_exp_array_checked(inp, out_b, huge_count)
    assert err_a == err_b == FP_Q16_ERR_OVERFLOW
    assert out_a == out_b == [77, 88]

    near_end = U64_MAX_VALUE - (((len(inp) - 1) << 3)) + 1
    err_a = fpq16_exp_array_checked_default(
        inp,
        out_a,
        len(inp),
        synthetic_input_addr=near_end,
    )
    err_b, _ = fpq16_exp_array_checked(
        inp,
        out_b,
        len(inp),
        synthetic_input_addr=near_end,
    )
    assert err_a == err_b == FP_Q16_ERR_OVERFLOW


def test_boundary_and_randomized_wrapper_equivalence() -> None:
    boundary_inputs = [
        -(1 << 20),
        -655360,
        -655359,
        -1,
        0,
        1,
        655359,
        655360,
        (1 << 20),
    ]

    out_default = [0] * len(boundary_inputs)
    out_core = [0] * len(boundary_inputs)

    err_default = fpq16_exp_array_checked_default(boundary_inputs, out_default, len(boundary_inputs))
    err_core, out_core = fpq16_exp_array_checked(boundary_inputs, out_core, len(boundary_inputs))

    assert err_default == err_core == FP_Q16_OK
    assert out_default == out_core

    rng = random.Random(20260419_484)
    for _ in range(2000):
        count = rng.randint(0, 96)
        if count == 0:
            inputs = []
            out_default = [0x33, 0x44]
            out_core = out_default.copy()
        else:
            inputs = [rng.randint(-(1 << 62), (1 << 62)) for _ in range(count)]
            out_default = [0x55] * count
            out_core = out_default.copy()

        err_default = fpq16_exp_array_checked_default(inputs, out_default, count)
        err_core, out_core = fpq16_exp_array_checked(inputs, out_core, count)

        assert err_default == err_core
        assert out_default == out_core


def run() -> None:
    test_source_contains_default_wrapper_shape()
    test_null_pointer_and_negative_count_surfaces()
    test_zero_count_no_write()
    test_pointer_span_overflow_parity()
    test_boundary_and_randomized_wrapper_equivalence()
    print("intexp_exp_array_checked_default=ok")


if __name__ == "__main__":
    run()
