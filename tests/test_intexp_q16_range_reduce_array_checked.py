#!/usr/bin/env python3
"""Parity checks for FPQ16ExpRangeReduceLn2ArrayChecked semantics."""

from __future__ import annotations

import pathlib
import random

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


def _u64(v: int) -> int:
    return v & U64_MAX


def _holyc_floor_div_k(x_q16: int) -> int:
    k = int(x_q16 / EXP_Q16_LN2)
    if x_q16 < 0 and (x_q16 % EXP_Q16_LN2):
        k -= 1
    return k


def fpq16_exp_clamp_to_input_domain_checked(input_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0
    if EXP_Q16_MIN_INPUT > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0
    if input_q16 < EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, EXP_Q16_MIN_INPUT
    if input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, EXP_Q16_MAX_INPUT
    return FP_Q16_OK, input_q16


def fpq16_exp_range_reduce_ln2_checked(input_q16: int, out_k_present: bool = True, out_r_present: bool = True) -> tuple[int, int, int]:
    if not out_k_present or not out_r_present:
        return FP_Q16_ERR_NULL_PTR, 0, 0

    err, clamped_q16 = fpq16_exp_clamp_to_input_domain_checked(input_q16)
    if err != FP_Q16_OK:
        return err, 0, 0

    if clamped_q16 < EXP_Q16_MIN_INPUT or clamped_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if EXP_Q16_LN2 <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    k = _holyc_floor_div_k(clamped_q16)
    r_q16 = clamped_q16 - (k * EXP_Q16_LN2)
    if r_q16 < 0 or r_q16 >= EXP_Q16_LN2:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    return FP_Q16_OK, k, r_q16


def fpq16_exp_range_reduce_ln2_array_checked(
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

    # Pass 1: no writes.
    for i in range(count):
        err, _, _ = fpq16_exp_range_reduce_ln2_checked(input_q16[i])
        if err != FP_Q16_OK:
            return err

    # Pass 2: commit.
    for i in range(count):
        err, got_k, got_r = fpq16_exp_range_reduce_ln2_checked(input_q16[i])
        if err != FP_Q16_OK:
            return err
        out_k[i] = got_k
        out_r_q16[i] = got_r

    return FP_Q16_OK


def test_source_contains_helper_and_two_pass_write_pattern() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpRangeReduceLn2ArrayChecked" in source
    assert "// Pass 1: full-lane validation without caller-buffer writes." in source
    assert "// Pass 2: deterministic commit after successful preflight." in source


def test_null_and_bad_count_surfaces() -> None:
    in_data = [0]
    out_k = [11]
    out_r = [22]

    assert fpq16_exp_range_reduce_ln2_array_checked(None, out_k, out_r, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked(in_data, None, out_r, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked(in_data, out_k, None, 1) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_range_reduce_ln2_array_checked(in_data, out_k, out_r, -1) == FP_Q16_ERR_BAD_PARAM


def test_pointer_span_overflow_surfaces() -> None:
    in_data = [1, 2]
    out_k = [3, 4]
    out_r = [5, 6]

    huge_count = (I64_MAX >> 3) + 2
    assert fpq16_exp_range_reduce_ln2_array_checked(in_data, out_k, out_r, huge_count) == FP_Q16_ERR_OVERFLOW

    assert (
        fpq16_exp_range_reduce_ln2_array_checked(
            in_data,
            out_k,
            out_r,
            2,
            in_addr=U64_MAX - 7 + 1,
        )
        == FP_Q16_ERR_OVERFLOW
    )
    assert (
        fpq16_exp_range_reduce_ln2_array_checked(
            in_data,
            out_k,
            out_r,
            2,
            out_k_addr=U64_MAX - 7 + 1,
        )
        == FP_Q16_ERR_OVERFLOW
    )
    assert (
        fpq16_exp_range_reduce_ln2_array_checked(
            in_data,
            out_k,
            out_r,
            2,
            out_r_addr=U64_MAX - 7 + 1,
        )
        == FP_Q16_ERR_OVERFLOW
    )


def test_scalar_composition_parity_and_reconstruction() -> None:
    inputs = [
        EXP_Q16_MIN_INPUT - 1,
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -500000,
        -1,
        0,
        1,
        500000,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
        EXP_Q16_MAX_INPUT + 1,
    ]

    out_k = [999] * len(inputs)
    out_r = [888] * len(inputs)
    err = fpq16_exp_range_reduce_ln2_array_checked(inputs, out_k, out_r, len(inputs))
    assert err == FP_Q16_OK

    for i, x_q16 in enumerate(inputs):
        serr, sk, sr = fpq16_exp_range_reduce_ln2_checked(x_q16)
        assert serr == FP_Q16_OK
        assert out_k[i] == sk
        assert out_r[i] == sr

        _, clamped = fpq16_exp_clamp_to_input_domain_checked(x_q16)
        assert clamped == (out_k[i] * EXP_Q16_LN2) + out_r[i]
        assert 0 <= out_r[i] < EXP_Q16_LN2


def test_no_partial_write_on_preflight_failure() -> None:
    global EXP_Q16_LN2

    inputs = [123, 456, 789]
    out_k = [111, 222, 333]
    out_r = [444, 555, 666]

    saved_ln2 = EXP_Q16_LN2
    try:
        EXP_Q16_LN2 = 0
        err = fpq16_exp_range_reduce_ln2_array_checked(inputs, out_k, out_r, len(inputs))
        assert err == FP_Q16_ERR_BAD_PARAM
        assert out_k == [111, 222, 333]
        assert out_r == [444, 555, 666]
    finally:
        EXP_Q16_LN2 = saved_ln2


def test_randomized_dense_parity() -> None:
    rng = random.Random(20260419_457)

    for _ in range(500):
        count = rng.randint(1, 128)
        inputs = [rng.randint(EXP_Q16_MIN_INPUT - 400000, EXP_Q16_MAX_INPUT + 400000) for _ in range(count)]
        out_k = [-1] * count
        out_r = [-1] * count

        err = fpq16_exp_range_reduce_ln2_array_checked(inputs, out_k, out_r, count)
        assert err == FP_Q16_OK

        for i, x_q16 in enumerate(inputs):
            serr, sk, sr = fpq16_exp_range_reduce_ln2_checked(x_q16)
            assert serr == FP_Q16_OK
            assert out_k[i] == sk
            assert out_r[i] == sr


def run() -> None:
    test_source_contains_helper_and_two_pass_write_pattern()
    test_null_and_bad_count_surfaces()
    test_pointer_span_overflow_surfaces()
    test_scalar_composition_parity_and_reconstruction()
    test_no_partial_write_on_preflight_failure()
    test_randomized_dense_parity()
    print("intexp_q16_range_reduce_array_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
