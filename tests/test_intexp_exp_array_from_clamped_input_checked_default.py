#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayFromClampedInputCheckedDefault wrapper."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_intexp_exp_array_from_clamped_input_checked import (
    EXP_Q16_MAX_INPUT,
    EXP_Q16_MIN_INPUT,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
    fpq16_exp_array_from_clamped_input_checked,
)


def fpq16_exp_array_from_clamped_input_checked_default(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    length: int,
    *,
    synthetic_input_addr: int = 0x1A00,
    synthetic_output_addr: int = 0x2B00,
) -> int:
    return fpq16_exp_array_from_clamped_input_checked(
        input_q16,
        output_q16,
        length,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=synthetic_output_addr,
    )[0]


def test_source_contains_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayFromClampedInputCheckedDefault" in source
    assert "return FPQ16ExpArrayFromClampedInputChecked(input_q16," in source


def test_null_pointer_and_negative_length_surfaces() -> None:
    assert fpq16_exp_array_from_clamped_input_checked_default(None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_from_clamped_input_checked_default([0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_from_clamped_input_checked_default([0], [0], -1) == FP_Q16_ERR_BAD_PARAM


def test_zero_length_no_write() -> None:
    out = [101, 202, 303]
    err = fpq16_exp_array_from_clamped_input_checked_default([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert out == [101, 202, 303]


def test_pointer_span_overflow_parity() -> None:
    inp = [0, 1]
    out_a = [77, 88]
    out_b = out_a.copy()

    huge_length = (I64_MAX_VALUE >> 3) + 2
    err_a = fpq16_exp_array_from_clamped_input_checked_default(inp, out_a, huge_length)
    err_b, _ = fpq16_exp_array_from_clamped_input_checked(inp, out_b, huge_length)
    assert err_a == err_b == FP_Q16_ERR_OVERFLOW
    assert out_a == out_b == [77, 88]

    near_end = U64_MAX_VALUE - (((len(inp) - 1) << 3)) + 1
    err_a = fpq16_exp_array_from_clamped_input_checked_default(
        inp,
        out_a,
        len(inp),
        synthetic_input_addr=near_end,
    )
    err_b, _ = fpq16_exp_array_from_clamped_input_checked(
        inp,
        out_b,
        len(inp),
        synthetic_input_addr=near_end,
    )
    assert err_a == err_b == FP_Q16_ERR_OVERFLOW


def test_domain_boundary_vectors() -> None:
    inputs = [
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -123456,
        0,
        123456,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
    ]
    out_default = [0] * len(inputs)
    out_core = [0] * len(inputs)

    err_default = fpq16_exp_array_from_clamped_input_checked_default(inputs, out_default, len(inputs))
    err_core, out_core = fpq16_exp_array_from_clamped_input_checked(inputs, out_core, len(inputs))

    assert err_default == err_core == FP_Q16_OK
    assert out_default == out_core


def test_malformed_unclamped_rejects_without_partial_parity() -> None:
    inputs = [0, EXP_Q16_MAX_INPUT + 1, 2]
    out_default = [0xAA, 0xAA, 0xAA]
    out_core = out_default.copy()

    err_default = fpq16_exp_array_from_clamped_input_checked_default(inputs, out_default, len(inputs))
    err_core, out_core = fpq16_exp_array_from_clamped_input_checked(inputs, out_core, len(inputs))

    assert err_default == err_core == FP_Q16_ERR_BAD_PARAM
    assert out_default == out_core == [0xAA, 0xAA, 0xAA]


def test_randomized_wrapper_equivalence() -> None:
    rng = random.Random(20260419_478)

    for _ in range(2000):
        length = rng.randint(0, 96)
        if length == 0:
            inputs = []
            out_default = [0x33, 0x44]
            out_core = out_default.copy()
        else:
            mode = rng.randint(0, 3)
            if mode == 0:
                inputs = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(length)]
            elif mode == 1:
                inputs = [rng.randint(EXP_Q16_MIN_INPUT - 4000, EXP_Q16_MAX_INPUT + 4000) for _ in range(length)]
            else:
                inputs = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(length)]
            out_default = [0x55] * length
            out_core = out_default.copy()

        err_default = fpq16_exp_array_from_clamped_input_checked_default(inputs, out_default, length)
        err_core, out_core = fpq16_exp_array_from_clamped_input_checked(inputs, out_core, length)

        assert err_default == err_core
        assert out_default == out_core


def run() -> None:
    test_source_contains_default_wrapper_shape()
    test_null_pointer_and_negative_length_surfaces()
    test_zero_length_no_write()
    test_pointer_span_overflow_parity()
    test_domain_boundary_vectors()
    test_malformed_unclamped_rejects_without_partial_parity()
    test_randomized_wrapper_equivalence()
    print("intexp_exp_array_from_clamped_input_checked_default=ok")


if __name__ == "__main__":
    run()
