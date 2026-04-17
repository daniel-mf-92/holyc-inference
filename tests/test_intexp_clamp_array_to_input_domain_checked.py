#!/usr/bin/env python3
"""Reference checks for FPQ16ExpClampArrayToInputDomainChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4


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


def fpq16_exp_clamp_array_to_input_domain_checked(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1000,
    synthetic_output_addr: int = 0x2000,
) -> tuple[int, list[int] | None]:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, output_q16
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, output_q16
    if count == 0:
        return FP_Q16_OK, output_q16

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, output_q16

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW, output_q16
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW, output_q16

    if len(input_q16) < count or len(output_q16) < count:
        raise ValueError("test harness requires count <= provided buffer lengths")

    for i in range(count):
        err, clamped = fpq16_exp_clamp_to_input_domain_checked(input_q16[i])
        if err != FP_Q16_OK:
            return err, output_q16
        output_q16[i] = clamped

    return FP_Q16_OK, output_q16


def test_null_pointer_surfaces() -> None:
    err, _ = fpq16_exp_clamp_array_to_input_domain_checked(None, [0], 1)
    assert err == FP_Q16_ERR_NULL_PTR

    err, _ = fpq16_exp_clamp_array_to_input_domain_checked([0], None, 1)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_negative_count() -> None:
    err, out = fpq16_exp_clamp_array_to_input_domain_checked([1, 2], [0, 0], -1)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [0, 0]


def test_zero_count_no_write() -> None:
    out = [111, 222, 333]
    err, got = fpq16_exp_clamp_array_to_input_domain_checked([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert got == [111, 222, 333]


def test_overflow_pointer_span_guards() -> None:
    huge_count = (I64_MAX_VALUE >> 3) + 2
    err, _ = fpq16_exp_clamp_array_to_input_domain_checked([0], [0], huge_count)
    assert err == FP_Q16_ERR_OVERFLOW

    count = 4
    end_minus_two = U64_MAX_VALUE - ((count - 1) << 3) + 1
    err, _ = fpq16_exp_clamp_array_to_input_domain_checked(
        [0] * count,
        [0] * count,
        count,
        synthetic_input_addr=end_minus_two,
    )
    assert err == FP_Q16_ERR_OVERFLOW

    err, _ = fpq16_exp_clamp_array_to_input_domain_checked(
        [0] * count,
        [0] * count,
        count,
        synthetic_output_addr=end_minus_two,
    )
    assert err == FP_Q16_ERR_OVERFLOW


def test_lanewise_scalar_parity_on_boundaries() -> None:
    data = [
        EXP_Q16_MIN_INPUT - 99,
        EXP_Q16_MIN_INPUT,
        EXP_Q16_MIN_INPUT + 1,
        -1,
        0,
        1,
        EXP_Q16_MAX_INPUT - 1,
        EXP_Q16_MAX_INPUT,
        EXP_Q16_MAX_INPUT + 99,
    ]
    out = [0] * len(data)

    err, got = fpq16_exp_clamp_array_to_input_domain_checked(data, out, len(data))
    assert err == FP_Q16_OK
    assert got is not None

    for i, value in enumerate(data):
        scalar_err, scalar_out = fpq16_exp_clamp_to_input_domain_checked(value)
        assert scalar_err == FP_Q16_OK
        assert got[i] == scalar_out


def test_randomized_scalar_composition_and_idempotence() -> None:
    rng = random.Random(20260417_163)

    for _ in range(2000):
        count = rng.randint(1, 128)
        data = [rng.randint(-(1 << 62), (1 << 62)) for _ in range(count)]
        out = [0] * count

        err, first = fpq16_exp_clamp_array_to_input_domain_checked(data, out, count)
        assert err == FP_Q16_OK
        assert first is not None

        for i in range(count):
            scalar_err, scalar_out = fpq16_exp_clamp_to_input_domain_checked(data[i])
            assert scalar_err == FP_Q16_OK
            assert first[i] == scalar_out

        second_buf = [0] * count
        err2, second = fpq16_exp_clamp_array_to_input_domain_checked(first.copy(), second_buf, count)
        assert err2 == FP_Q16_OK
        assert second == first


def test_inplace_alias_path_same_results() -> None:
    values = [
        EXP_Q16_MIN_INPUT - 1,
        -123,
        0,
        456,
        EXP_Q16_MAX_INPUT + 1,
    ]

    err, got = fpq16_exp_clamp_array_to_input_domain_checked(values, values, len(values))
    assert err == FP_Q16_OK
    assert got == [
        EXP_Q16_MIN_INPUT,
        -123,
        0,
        456,
        EXP_Q16_MAX_INPUT,
    ]


def run() -> None:
    test_null_pointer_surfaces()
    test_bad_param_negative_count()
    test_zero_count_no_write()
    test_overflow_pointer_span_guards()
    test_lanewise_scalar_parity_on_boundaries()
    test_randomized_scalar_composition_and_idempotence()
    test_inplace_alias_path_same_results()
    print("intexp_clamp_array_to_input_domain_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
