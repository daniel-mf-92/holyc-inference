#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayFromClampedInputCheckedDefaultNoPartial wrapper."""

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
)
from test_intexp_exp_array_from_clamped_input_checked_default import (
    fpq16_exp_array_from_clamped_input_checked_default,
)


def fpq16_exp_array_from_clamped_input_checked_default_no_partial(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    length: int,
    *,
    synthetic_input_addr: int = 0x3A00,
    synthetic_output_addr: int = 0x4B00,
) -> int:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if length < 0:
        return FP_Q16_ERR_BAD_PARAM
    if length == 0:
        return FP_Q16_OK

    last_index = length - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if length > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    staging = [0] * length
    err = fpq16_exp_array_from_clamped_input_checked_default(
        input_q16,
        staging,
        length,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=0x5C00,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(length):
        output_q16[i] = staging[i]

    return FP_Q16_OK


def explicit_staged_composition(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    length: int,
    *,
    synthetic_input_addr: int = 0x3A00,
    synthetic_output_addr: int = 0x4B00,
) -> int:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if length < 0:
        return FP_Q16_ERR_BAD_PARAM
    if length == 0:
        return FP_Q16_OK

    last_index = length - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if length > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    staging = [0] * length
    err = fpq16_exp_array_from_clamped_input_checked_default(
        input_q16,
        staging,
        length,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=0x5D00,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(length):
        output_q16[i] = staging[i]

    return FP_Q16_OK


def test_source_contains_no_partial_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayFromClampedInputCheckedDefaultNoPartial" in source
    assert "status = FPQ16ExpArrayFromClampedInputCheckedDefault(input_q16," in source
    assert "staging_output_q16 = MAlloc(staging_bytes);" in source


def test_null_bad_length_and_zero_length_surfaces() -> None:
    assert fpq16_exp_array_from_clamped_input_checked_default_no_partial(None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_from_clamped_input_checked_default_no_partial([0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_from_clamped_input_checked_default_no_partial([0], [0], -1) == FP_Q16_ERR_BAD_PARAM

    out = [111, 222, 333]
    err = fpq16_exp_array_from_clamped_input_checked_default_no_partial([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert out == [111, 222, 333]


def test_malformed_length_overflow_and_pointer_span_no_partial_write() -> None:
    inp = [0, 1]
    out = [77, 88]

    huge_length = (I64_MAX_VALUE >> 3) + 2
    err = fpq16_exp_array_from_clamped_input_checked_default_no_partial(inp, out, huge_length)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]

    near_end = U64_MAX_VALUE - ((len(inp) - 1) << 3) + 1
    err = fpq16_exp_array_from_clamped_input_checked_default_no_partial(
        inp,
        out,
        len(inp),
        synthetic_input_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]

    err = fpq16_exp_array_from_clamped_input_checked_default_no_partial(
        inp,
        out,
        len(inp),
        synthetic_output_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [77, 88]


def test_domain_reject_has_no_partial_writes() -> None:
    inputs = [0, EXP_Q16_MIN_INPUT - 1, 10]
    out = [301, 302, 303]

    err = fpq16_exp_array_from_clamped_input_checked_default_no_partial(
        inputs,
        out,
        len(inputs),
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [301, 302, 303]


def test_boundary_and_randomized_parity_vs_explicit_staged_composition() -> None:
    boundary_inputs = [
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -123456,
        -1,
        0,
        1,
        123456,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
    ]

    out_a = [0xAA] * len(boundary_inputs)
    out_b = out_a.copy()
    err_a = fpq16_exp_array_from_clamped_input_checked_default_no_partial(
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

    rng = random.Random(20260419_493)
    for _ in range(1500):
        length = rng.randint(1, 96)
        mode = rng.randint(0, 3)
        if mode == 0:
            inputs = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(length)]
        elif mode == 1:
            inputs = [rng.randint(EXP_Q16_MIN_INPUT - 4000, EXP_Q16_MAX_INPUT + 4000) for _ in range(length)]
        else:
            inputs = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(length)]

        out_wrap = [0x31] * length
        out_ref = out_wrap.copy()

        err_wrap = fpq16_exp_array_from_clamped_input_checked_default_no_partial(
            inputs,
            out_wrap,
            length,
        )
        err_ref = explicit_staged_composition(
            inputs,
            out_ref,
            length,
        )

        assert err_wrap == err_ref
        if err_ref == FP_Q16_OK:
            assert out_wrap == out_ref
        else:
            assert out_wrap == [0x31] * length


def run() -> None:
    test_source_contains_no_partial_default_wrapper_shape()
    test_null_bad_length_and_zero_length_surfaces()
    test_malformed_length_overflow_and_pointer_span_no_partial_write()
    test_domain_reject_has_no_partial_writes()
    test_boundary_and_randomized_parity_vs_explicit_staged_composition()
    print("intexp_exp_array_from_clamped_input_checked_default_no_partial=ok")


if __name__ == "__main__":
    run()
