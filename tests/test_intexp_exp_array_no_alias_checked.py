#!/usr/bin/env python3
"""Reference checks for FPQ16ExpArrayCheckedNoAliasChecked semantics."""

from __future__ import annotations

import random

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

EXP_Q16_LN2 = 45426
EXP_Q16_MAX_INPUT = 655360
EXP_Q16_MIN_INPUT = -655360

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4


def fpq16_mul(a: int, b: int) -> int:
    prod = a * b
    if prod >= 0:
        return (prod + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT
    return -(((-prod) + (1 << (FP_Q16_SHIFT - 1))) >> FP_Q16_SHIFT)


def fpq16_exp_from_clamped_input_checked(clamped_input_q16: int, out_present: bool = True) -> tuple[int, int]:
    if not out_present:
        return FP_Q16_ERR_NULL_PTR, 0

    if clamped_input_q16 < EXP_Q16_MIN_INPUT or clamped_input_q16 > EXP_Q16_MAX_INPUT:
        return FP_Q16_ERR_BAD_PARAM, 0

    if clamped_input_q16 >= EXP_Q16_MAX_INPUT:
        return FP_Q16_OK, I64_MAX_VALUE
    if clamped_input_q16 <= EXP_Q16_MIN_INPUT:
        return FP_Q16_OK, 0

    k = clamped_input_q16 // EXP_Q16_LN2
    if clamped_input_q16 < 0 and (clamped_input_q16 % EXP_Q16_LN2):
        k -= 1

    r = clamped_input_q16 - (k * EXP_Q16_LN2)

    r2 = fpq16_mul(r, r)
    r3 = fpq16_mul(r2, r)
    r4 = fpq16_mul(r3, r)

    poly = FP_Q16_ONE + r + (r2 // 2) + (r3 // 6) + (r4 // 24)

    if k >= 0:
        if k >= 30:
            return FP_Q16_OK, I64_MAX_VALUE
        if poly > (I64_MAX_VALUE >> k):
            return FP_Q16_OK, I64_MAX_VALUE
        return FP_Q16_OK, poly << k

    if k <= -63:
        return FP_Q16_OK, 0

    return FP_Q16_OK, poly >> (-k)


def fpq16_exp_array_from_clamped_input_checked(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x3000,
    synthetic_output_addr: int = 0x5000,
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
        if input_q16[i] < EXP_Q16_MIN_INPUT or input_q16[i] > EXP_Q16_MAX_INPUT:
            return FP_Q16_ERR_BAD_PARAM, output_q16

    for i in range(count):
        err, lane = fpq16_exp_from_clamped_input_checked(input_q16[i])
        if err != FP_Q16_OK:
            return err, output_q16
        output_q16[i] = lane

    return FP_Q16_OK, output_q16


def fpq16_exp_array_checked_no_alias_checked(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    aliased: bool = False,
    synthetic_input_addr: int = 0x3000,
    synthetic_output_addr: int = 0x5000,
) -> tuple[int, list[int] | None]:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, output_q16
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, output_q16
    if aliased:
        return FP_Q16_ERR_BAD_PARAM, output_q16

    return fpq16_exp_array_from_clamped_input_checked(
        input_q16,
        output_q16,
        count,
        synthetic_input_addr=synthetic_input_addr,
        synthetic_output_addr=synthetic_output_addr,
    )


def test_null_pointer_surfaces() -> None:
    err, _ = fpq16_exp_array_checked_no_alias_checked(None, [0], 1)
    assert err == FP_Q16_ERR_NULL_PTR

    err, _ = fpq16_exp_array_checked_no_alias_checked([0], None, 1)
    assert err == FP_Q16_ERR_NULL_PTR


def test_bad_param_negative_count() -> None:
    out = [111, 222]
    err, got = fpq16_exp_array_checked_no_alias_checked([1, 2], out, -1)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert got == [111, 222]


def test_bad_param_rejects_alias_even_zero_count() -> None:
    buf = [7, 8, 9]

    err, got = fpq16_exp_array_checked_no_alias_checked(buf, buf, 0, aliased=True)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert got == [7, 8, 9]

    err, got = fpq16_exp_array_checked_no_alias_checked(buf, buf, len(buf), aliased=True)
    assert err == FP_Q16_ERR_BAD_PARAM
    assert got == [7, 8, 9]


def test_zero_count_non_alias_ok_no_write() -> None:
    out = [111, 222, 333]
    err, got = fpq16_exp_array_checked_no_alias_checked([1, 2, 3], out, 0)
    assert err == FP_Q16_OK
    assert got == [111, 222, 333]


def test_overflow_pointer_span_guards() -> None:
    huge_count = (I64_MAX_VALUE >> 3) + 2
    err, _ = fpq16_exp_array_checked_no_alias_checked([0], [0], huge_count)
    assert err == FP_Q16_ERR_OVERFLOW

    count = 4
    near_end = U64_MAX_VALUE - ((count - 1) << 3) + 1
    err, _ = fpq16_exp_array_checked_no_alias_checked(
        [0] * count,
        [0] * count,
        count,
        synthetic_input_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW

    err, _ = fpq16_exp_array_checked_no_alias_checked(
        [0] * count,
        [0] * count,
        count,
        synthetic_output_addr=near_end,
    )
    assert err == FP_Q16_ERR_OVERFLOW


def test_domain_prescan_no_partial_writes() -> None:
    inputs = [0, EXP_Q16_MAX_INPUT + 1, 42]
    out = [999, 999, 999]

    err, got = fpq16_exp_array_checked_no_alias_checked(inputs, out, len(inputs))
    assert err == FP_Q16_ERR_BAD_PARAM
    assert got == [999, 999, 999]


def test_parity_with_preclamped_array_helper_for_non_alias() -> None:
    rng = random.Random(20260417_168)

    for _ in range(1500):
        count = rng.randint(1, 96)
        inputs = [rng.randint(EXP_Q16_MIN_INPUT, EXP_Q16_MAX_INPUT) for _ in range(count)]

        out_expected = [0] * count
        err_expected, expected = fpq16_exp_array_from_clamped_input_checked(inputs.copy(), out_expected, count)
        assert err_expected == FP_Q16_OK
        assert expected is not None

        out_got = [0] * count
        err_got, got = fpq16_exp_array_checked_no_alias_checked(inputs, out_got, count)
        assert err_got == FP_Q16_OK
        assert got == expected


def run() -> None:
    test_null_pointer_surfaces()
    test_bad_param_negative_count()
    test_bad_param_rejects_alias_even_zero_count()
    test_zero_count_non_alias_ok_no_write()
    test_overflow_pointer_span_guards()
    test_domain_prescan_no_partial_writes()
    test_parity_with_preclamped_array_helper_for_non_alias()
    print("intexp_exp_array_no_alias_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
