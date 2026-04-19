#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayCheckedDefaultNoPartial wrapper."""

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
from test_intexp_exp_array_checked_default import fpq16_exp_array_checked_default


def fpq16_exp_array_checked_default_no_partial(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1600,
    synthetic_output_addr: int = 0x2700,
) -> int:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    staging = [0] * count
    err = fpq16_exp_array_checked_default(
        input_q16,
        staging,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=0x5A00,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        output_q16[i] = staging[i]

    return FP_Q16_OK


def explicit_staged_composition(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1600,
    synthetic_output_addr: int = 0x2700,
) -> int:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    staging = [0] * count
    err = fpq16_exp_array_checked_default(
        input_q16,
        staging,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=0x5B00,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        output_q16[i] = staging[i]

    return FP_Q16_OK


def test_source_contains_no_partial_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayCheckedDefaultNoPartial" in source
    assert "status = FPQ16ExpArrayCheckedDefault(input_q16," in source
    assert "staged_output_q16 = MAlloc(staging_bytes);" in source


def test_null_bad_count_and_zero_count_surfaces() -> None:
    assert fpq16_exp_array_checked_default_no_partial(None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_default_no_partial([0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_default_no_partial([0], [0], -1) == FP_Q16_ERR_BAD_PARAM

    out = [701, 702, 703]
    err = fpq16_exp_array_checked_default_no_partial([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert out == [701, 702, 703]


def test_malformed_count_and_pointer_span_overflow_no_partial_write() -> None:
    inp = [0, 1]
    out = [41, 42]

    huge_count = (I64_MAX_VALUE >> 3) + 2
    err = fpq16_exp_array_checked_default_no_partial(inp, out, huge_count)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [41, 42]

    near_end = U64_MAX_VALUE - ((len(inp) - 1) << 3) + 1
    err = fpq16_exp_array_checked_default_no_partial(
        inp,
        out,
        len(inp),
        synthetic_input_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [41, 42]

    err = fpq16_exp_array_checked_default_no_partial(
        inp,
        out,
        len(inp),
        synthetic_output_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [41, 42]


def test_boundary_and_randomized_parity_vs_explicit_staged_composition() -> None:
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

    out_a = [0x5E] * len(boundary_inputs)
    out_b = out_a.copy()

    err_a = fpq16_exp_array_checked_default_no_partial(
        boundary_inputs,
        out_a,
        len(boundary_inputs),
    )
    err_b = explicit_staged_composition(
        boundary_inputs,
        out_b,
        len(boundary_inputs),
    )

    assert err_a == err_b == FP_Q16_OK
    assert out_a == out_b

    rng = random.Random(20260419_499)
    for _ in range(1700):
        count = rng.randint(1, 96)
        mode = rng.randint(0, 2)
        if mode == 0:
            inputs = [rng.randint(-700000, 700000) for _ in range(count)]
        elif mode == 1:
            inputs = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(count)]
        else:
            inputs = [rng.randint(-(1 << 62), (1 << 62)) for _ in range(count)]

        out_wrap = [0x7A] * count
        out_ref = out_wrap.copy()

        err_wrap = fpq16_exp_array_checked_default_no_partial(inputs, out_wrap, count)
        err_ref = explicit_staged_composition(inputs, out_ref, count)

        assert err_wrap == err_ref
        if err_ref == FP_Q16_OK:
            assert out_wrap == out_ref
        else:
            assert out_wrap == [0x7A] * count


def run() -> None:
    test_source_contains_no_partial_default_wrapper_shape()
    test_null_bad_count_and_zero_count_surfaces()
    test_malformed_count_and_pointer_span_overflow_no_partial_write()
    test_boundary_and_randomized_parity_vs_explicit_staged_composition()
    print("intexp_exp_array_checked_default_no_partial=ok")


if __name__ == "__main__":
    run()
