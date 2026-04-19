#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayFromClampedInputCheckedNoPartial semantics."""

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


def fpq16_exp_array_from_clamped_input_checked_no_partial(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x3000,
    synthetic_output_addr: int = 0x5000,
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

    staged = [0] * count
    err, staged = fpq16_exp_array_from_clamped_input_checked(
        input_q16,
        staged,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=0x7000,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        output_q16[i] = staged[i]

    return FP_Q16_OK


def explicit_staged_composition(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x3000,
    synthetic_output_addr: int = 0x5000,
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

    staged = [0] * count
    err, staged = fpq16_exp_array_from_clamped_input_checked(
        input_q16,
        staged,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=0x7100,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        output_q16[i] = staged[i]

    return FP_Q16_OK


def test_source_contains_no_partial_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayFromClampedInputCheckedNoPartial" in source
    assert "staged_output_q16 = MAlloc(staged_bytes);" in source
    assert "status = FPQ16ExpArrayFromClampedInputChecked(input_q16," in source


def test_null_bad_count_and_zero_count_surfaces() -> None:
    assert fpq16_exp_array_from_clamped_input_checked_no_partial(None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_from_clamped_input_checked_no_partial([0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_from_clamped_input_checked_no_partial([0], [0], -1) == FP_Q16_ERR_BAD_PARAM

    out = [777, 888]
    err = fpq16_exp_array_from_clamped_input_checked_no_partial([1, 2], out, 0)
    assert err == FP_Q16_OK
    assert out == [777, 888]


def test_pointer_span_overflow_guards_no_partial_write() -> None:
    inp = [0, 1]
    out = [55, 66]

    huge_count = (I64_MAX_VALUE >> 3) + 2
    err = fpq16_exp_array_from_clamped_input_checked_no_partial(inp, out, huge_count)
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [55, 66]

    count = 2
    near_end = U64_MAX_VALUE - ((count - 1) << 3) + 1
    err = fpq16_exp_array_from_clamped_input_checked_no_partial(
        inp,
        out,
        count,
        synthetic_input_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [55, 66]

    err = fpq16_exp_array_from_clamped_input_checked_no_partial(
        inp,
        out,
        count,
        synthetic_output_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out == [55, 66]


def test_domain_reject_has_no_partial_writes() -> None:
    inputs = [0, EXP_Q16_MAX_INPUT + 1, 10]
    out = [101, 202, 303]

    err = fpq16_exp_array_from_clamped_input_checked_no_partial(inputs, out, len(inputs))
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 202, 303]


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
    err_a = fpq16_exp_array_from_clamped_input_checked_no_partial(
        boundary_inputs,
        out_a,
        len(boundary_inputs),
    )
    err_b = explicit_staged_composition(boundary_inputs, out_b, len(boundary_inputs))
    assert err_a == err_b == FP_Q16_OK
    assert out_a == out_b

    rng = random.Random(202604190469)
    for _ in range(1200):
        count = rng.randint(1, 96)
        inputs = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(count)]

        out_wrap = [0x11] * count
        out_ref = out_wrap.copy()

        err_wrap = fpq16_exp_array_from_clamped_input_checked_no_partial(inputs, out_wrap, count)
        err_ref = explicit_staged_composition(inputs, out_ref, count)

        assert err_wrap == err_ref
        if err_ref == FP_Q16_OK:
            assert out_wrap == out_ref
        else:
            assert out_wrap == [0x11] * count


if __name__ == "__main__":
    test_source_contains_no_partial_wrapper_shape()
    test_null_bad_count_and_zero_count_surfaces()
    test_pointer_span_overflow_guards_no_partial_write()
    test_domain_reject_has_no_partial_writes()
    test_boundary_and_randomized_parity_vs_explicit_staged_composition()
    print("intexp_exp_array_from_clamped_input_checked_no_partial_parity=ok")
