#!/usr/bin/env python3
"""Parity checks for FPQ16ExpRangeReduceLn2ArrayCheckedNoPartial semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_intexp_q16_range_reduce_array_checked import (
    EXP_Q16_LN2,
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX,
    U64_MAX,
    fpq16_exp_range_reduce_ln2_array_checked,
)


def _u64(v: int) -> int:
    return v & U64_MAX


def fpq16_exp_range_reduce_ln2_array_checked_no_partial(
    input_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    in_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
) -> int:
    if input_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if _u64(in_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW
    if _u64(out_k_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW
    if _u64(out_r_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW

    if count > (I64_MAX >> 3):
        return FP_Q16_ERR_OVERFLOW

    staged_k = [0] * count
    staged_r = [0] * count

    err = fpq16_exp_range_reduce_ln2_array_checked(
        input_q16,
        staged_k,
        staged_r,
        count,
        in_addr=in_addr,
        out_k_addr=0x4000,
        out_r_addr=0x5000,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        out_k[i] = staged_k[i]
        out_r_q16[i] = staged_r[i]

    return FP_Q16_OK


def explicit_staged_composition(
    input_q16: list[int],
    out_k: list[int],
    out_r_q16: list[int],
    count: int,
    in_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
) -> int:
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count == 0:
        return FP_Q16_OK

    last_index = count - 1
    if last_index > (I64_MAX >> 3):
        return FP_Q16_ERR_OVERFLOW

    last_byte_offset = last_index << 3
    if _u64(in_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW
    if _u64(out_k_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW
    if _u64(out_r_addr) > (U64_MAX - _u64(last_byte_offset)):
        return FP_Q16_ERR_OVERFLOW

    if count > (I64_MAX >> 3):
        return FP_Q16_ERR_OVERFLOW

    staged_k = [0] * count
    staged_r = [0] * count
    err = fpq16_exp_range_reduce_ln2_array_checked(
        input_q16,
        staged_k,
        staged_r,
        count,
        in_addr=in_addr,
        out_k_addr=0x6000,
        out_r_addr=0x7000,
    )
    if err != FP_Q16_OK:
        return err

    for i in range(count):
        out_k[i] = staged_k[i]
        out_r_q16[i] = staged_r[i]
    return FP_Q16_OK


def test_source_contains_no_partial_helper_shape() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpRangeReduceLn2ArrayCheckedNoPartial" in source
    assert "staged_k = MAlloc(staged_bytes);" in source
    assert "staged_r_q16 = MAlloc(staged_bytes);" in source


def test_null_and_bad_count_surfaces() -> None:
    assert fpq16_exp_range_reduce_ln2_array_checked_no_partial(None, [0], [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked_no_partial([0], None, [0], 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked_no_partial([0], [0], None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked_no_partial([0], [0], [0], -1) == FP_Q16_ERR_BAD_PARAM


def test_pointer_span_overflow_surfaces() -> None:
    in_data = [1, 2]
    out_k = [3, 4]
    out_r = [5, 6]

    huge_count = (I64_MAX >> 3) + 2
    assert (
        fpq16_exp_range_reduce_ln2_array_checked_no_partial(in_data, out_k, out_r, huge_count)
        == FP_Q16_ERR_OVERFLOW
    )

    assert (
        fpq16_exp_range_reduce_ln2_array_checked_no_partial(
            in_data,
            out_k,
            out_r,
            2,
            in_addr=U64_MAX - 7 + 1,
        )
        == FP_Q16_ERR_OVERFLOW
    )
    assert (
        fpq16_exp_range_reduce_ln2_array_checked_no_partial(
            in_data,
            out_k,
            out_r,
            2,
            out_k_addr=U64_MAX - 7 + 1,
        )
        == FP_Q16_ERR_OVERFLOW
    )
    assert (
        fpq16_exp_range_reduce_ln2_array_checked_no_partial(
            in_data,
            out_k,
            out_r,
            2,
            out_r_addr=U64_MAX - 7 + 1,
        )
        == FP_Q16_ERR_OVERFLOW
    )


def test_boundary_and_multilane_parity_vs_explicit_staged_composition() -> None:
    inputs = [
        -2_000_000,
        -655_361,
        -655_360,
        -655_359,
        -1,
        0,
        1,
        655_359,
        655_360,
        655_361,
        2_000_000,
    ]

    out_k_ref = [0x4444] * len(inputs)
    out_r_ref = [0x5555] * len(inputs)
    out_k_wrapped = out_k_ref.copy()
    out_r_wrapped = out_r_ref.copy()

    err_ref = explicit_staged_composition(inputs, out_k_ref, out_r_ref, len(inputs))
    err_wrapped = fpq16_exp_range_reduce_ln2_array_checked_no_partial(
        inputs,
        out_k_wrapped,
        out_r_wrapped,
        len(inputs),
    )

    assert err_wrapped == err_ref == FP_Q16_OK
    assert out_k_wrapped == out_k_ref
    assert out_r_wrapped == out_r_ref
    for rem in out_r_wrapped:
        assert 0 <= rem < EXP_Q16_LN2


def test_no_partial_write_on_core_failure() -> None:
    import test_intexp_q16_range_reduce_array_checked as base_mod

    inputs = [123, 456, 789]
    out_k = [11, 22, 33]
    out_r = [44, 55, 66]

    saved_ln2 = base_mod.EXP_Q16_LN2
    try:
        base_mod.EXP_Q16_LN2 = 0
        err = fpq16_exp_range_reduce_ln2_array_checked_no_partial(inputs, out_k, out_r, len(inputs))
        assert err == FP_Q16_ERR_BAD_PARAM
        assert out_k == [11, 22, 33]
        assert out_r == [44, 55, 66]
    finally:
        base_mod.EXP_Q16_LN2 = saved_ln2


def test_randomized_parity_against_explicit_staged_composition() -> None:
    rng = random.Random(20260419_466)

    for _ in range(1000):
        count = rng.randint(1, 96)
        inputs = [rng.randint(-2_000_000, 2_000_000) for _ in range(count)]

        out_k_ref = [0xA1] * count
        out_r_ref = [0xB2] * count
        out_k_wrapped = out_k_ref.copy()
        out_r_wrapped = out_r_ref.copy()

        err_ref = explicit_staged_composition(inputs, out_k_ref, out_r_ref, count)
        err_wrapped = fpq16_exp_range_reduce_ln2_array_checked_no_partial(
            inputs,
            out_k_wrapped,
            out_r_wrapped,
            count,
        )

        assert err_wrapped == err_ref
        assert out_k_wrapped == out_k_ref
        assert out_r_wrapped == out_r_ref


def run() -> None:
    test_source_contains_no_partial_helper_shape()
    test_null_and_bad_count_surfaces()
    test_pointer_span_overflow_surfaces()
    test_boundary_and_multilane_parity_vs_explicit_staged_composition()
    test_no_partial_write_on_core_failure()
    test_randomized_parity_against_explicit_staged_composition()
    print("intexp_q16_range_reduce_array_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
