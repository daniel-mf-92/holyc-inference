#!/usr/bin/env python3
"""Parity checks for FPQ16ExpRangeReduceLn2ArrayCheckedDefault wrapper."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_intexp_q16_range_reduce_array_checked import (  # noqa: E402
    EXP_Q16_MAX_INPUT,
    EXP_Q16_MIN_INPUT,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX,
    U64_MAX,
    _u64,
    fpq16_exp_range_reduce_ln2_array_checked,
)


def fpq16_exp_range_reduce_ln2_array_checked_default(
    input_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    length: int,
    *,
    in_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
) -> int:
    if input_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if length < 0:
        return FP_Q16_ERR_BAD_PARAM
    if length == 0:
        return FP_Q16_OK

    last_index = length - 1
    if last_index > (I64_MAX >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if _u64(in_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW
    if _u64(out_k_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW
    if _u64(out_r_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW

    return fpq16_exp_range_reduce_ln2_array_checked(
        input_q16,
        out_k,
        out_r_q16,
        length,
        in_addr,
        out_k_addr,
        out_r_addr,
    )


def test_source_contains_default_wrapper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpRangeReduceLn2ArrayCheckedDefault" in source
    assert "return FPQ16ExpRangeReduceLn2ArrayChecked(input_q16," in source


def test_null_pointer_and_negative_length_surfaces() -> None:
    in_data = [0]
    out_k = [11]
    out_r = [22]

    assert fpq16_exp_range_reduce_ln2_array_checked_default(None, out_k, out_r, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked_default(in_data, None, out_r, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked_default(in_data, out_k, None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked_default(in_data, out_k, out_r, -1) == FP_Q16_ERR_BAD_PARAM


def test_zero_length_is_noop() -> None:
    in_data = [123, 456]
    out_k = [77, 88]
    out_r = [99, 111]

    err = fpq16_exp_range_reduce_ln2_array_checked_default(in_data, out_k, out_r, 0)
    assert err == FP_Q16_OK
    assert out_k == [77, 88]
    assert out_r == [99, 111]


def test_pointer_span_overflow_parity() -> None:
    in_data = [1, 2]
    out_k_default = [3, 4]
    out_r_default = [5, 6]
    out_k_core = out_k_default.copy()
    out_r_core = out_r_default.copy()

    huge_length = (I64_MAX >> 3) + 2
    err_default = fpq16_exp_range_reduce_ln2_array_checked_default(
        in_data,
        out_k_default,
        out_r_default,
        huge_length,
    )
    err_core = fpq16_exp_range_reduce_ln2_array_checked(
        in_data,
        out_k_core,
        out_r_core,
        huge_length,
    )
    assert err_default == err_core == FP_Q16_ERR_OVERFLOW

    near_end = U64_MAX - 7 + 1
    err_default = fpq16_exp_range_reduce_ln2_array_checked_default(
        in_data,
        out_k_default,
        out_r_default,
        2,
        in_addr=near_end,
    )
    err_core = fpq16_exp_range_reduce_ln2_array_checked(
        in_data,
        out_k_core,
        out_r_core,
        2,
        in_addr=near_end,
    )
    assert err_default == err_core == FP_Q16_ERR_OVERFLOW


def test_clamp_boundary_vector_parity() -> None:
    inputs = [
        EXP_Q16_MIN_INPUT - 1,
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -1,
        0,
        1,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
        EXP_Q16_MAX_INPUT + 1,
    ]

    out_k_default = [0x55] * len(inputs)
    out_r_default = [0x66] * len(inputs)
    out_k_core = out_k_default.copy()
    out_r_core = out_r_default.copy()

    err_default = fpq16_exp_range_reduce_ln2_array_checked_default(
        inputs,
        out_k_default,
        out_r_default,
        len(inputs),
    )
    err_core = fpq16_exp_range_reduce_ln2_array_checked(
        inputs,
        out_k_core,
        out_r_core,
        len(inputs),
    )

    assert err_default == err_core == FP_Q16_OK
    assert out_k_default == out_k_core
    assert out_r_default == out_r_core


def test_randomized_wrapper_equivalence() -> None:
    rng = random.Random(20260419_481)

    for _ in range(1500):
        length = rng.randint(0, 128)
        if length == 0:
            inputs = []
            out_k_default = [777, 888]
            out_r_default = [999, 111]
            out_k_core = out_k_default.copy()
            out_r_core = out_r_default.copy()
        else:
            inputs = [rng.randint(EXP_Q16_MIN_INPUT - 400000, EXP_Q16_MAX_INPUT + 400000) for _ in range(length)]
            out_k_default = [0x11] * length
            out_r_default = [0x22] * length
            out_k_core = out_k_default.copy()
            out_r_core = out_r_default.copy()

        err_default = fpq16_exp_range_reduce_ln2_array_checked_default(
            inputs,
            out_k_default,
            out_r_default,
            length,
        )
        err_core = fpq16_exp_range_reduce_ln2_array_checked(
            inputs,
            out_k_core,
            out_r_core,
            length,
        )

        assert err_default == err_core
        assert out_k_default == out_k_core
        assert out_r_default == out_r_core


def run() -> None:
    test_source_contains_default_wrapper_shape()
    test_null_pointer_and_negative_length_surfaces()
    test_zero_length_is_noop()
    test_pointer_span_overflow_parity()
    test_clamp_boundary_vector_parity()
    test_randomized_wrapper_equivalence()
    print("intexp_q16_range_reduce_array_checked_default=ok")


if __name__ == "__main__":
    run()
