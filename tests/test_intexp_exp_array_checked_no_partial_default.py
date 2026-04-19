#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayCheckedNoPartialDefault wrapper."""

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
)
from test_intexp_exp_array_checked_no_partial import (
    explicit_staged_composition,
    fpq16_exp_array_checked_no_partial,
)


def fpq16_exp_array_checked_no_partial_default(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1200,
    synthetic_output_addr: int = 0x2300,
) -> int:
    return fpq16_exp_array_checked_no_partial(
        input_q16,
        output_q16,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=synthetic_output_addr,
    )


def test_source_contains_no_partial_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayCheckedNoPartialDefault" in source
    assert "status = FPQ16ExpArrayCheckedNoPartial(input_q16," in source
    assert "return status;" in source


def test_null_bad_count_and_zero_count_surfaces() -> None:
    assert fpq16_exp_array_checked_no_partial_default(None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_no_partial_default([0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_no_partial_default([0], [0], -1) == FP_Q16_ERR_BAD_PARAM

    out = [111, 222, 333]
    err = fpq16_exp_array_checked_no_partial_default([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert out == [111, 222, 333]


def test_malformed_length_overflow_passthrough_no_partial_write() -> None:
    inp = [0, 1]
    out = [77, 88]

    huge_count = (I64_MAX_VALUE >> 3) + 2
    err = fpq16_exp_array_checked_no_partial_default(inp, out, huge_count)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]

    near_end = U64_MAX_VALUE - ((len(inp) - 1) << 3) + 1
    err = fpq16_exp_array_checked_no_partial_default(
        inp,
        out,
        len(inp),
        synthetic_input_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]

    err = fpq16_exp_array_checked_no_partial_default(
        inp,
        out,
        len(inp),
        synthetic_output_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]


def test_clamp_boundary_and_randomized_parity_vs_staged_core() -> None:
    boundary_inputs = [
        -(1 << 62),
        -700000,
        -655360,
        -655359,
        -1,
        0,
        1,
        655359,
        655360,
        700000,
        (1 << 62),
    ]

    out_default = [0x44] * len(boundary_inputs)
    out_core = out_default.copy()
    err_default = fpq16_exp_array_checked_no_partial_default(
        boundary_inputs,
        out_default,
        len(boundary_inputs),
    )
    err_core = explicit_staged_composition(boundary_inputs, out_core, len(boundary_inputs))
    assert err_default == err_core == FP_Q16_OK
    assert out_default == out_core

    rng = random.Random(20260419_485)
    for _ in range(1500):
        count = rng.randint(1, 96)
        inputs = [rng.randint(-(1 << 62), (1 << 62)) for _ in range(count)]

        out_default = [0x2A] * count
        out_core = out_default.copy()

        err_default = fpq16_exp_array_checked_no_partial_default(inputs, out_default, count)
        err_core = explicit_staged_composition(inputs, out_core, count)

        assert err_default == err_core
        if err_core == FP_Q16_OK:
            assert out_default == out_core
        else:
            assert out_default == [0x2A] * count


def run() -> None:
    test_source_contains_no_partial_default_wrapper_shape()
    test_null_bad_count_and_zero_count_surfaces()
    test_malformed_length_overflow_passthrough_no_partial_write()
    test_clamp_boundary_and_randomized_parity_vs_staged_core()
    print("intexp_exp_array_checked_no_partial_default=ok")


if __name__ == "__main__":
    run()
