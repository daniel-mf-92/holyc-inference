#!/usr/bin/env python3
"""Reference checks for FPQ16RMSNormRowsChecked (IQ-1135)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_DOMAIN = 3
FP_Q16_ERR_OVERFLOW = 4


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_try_apply_sign_from_u64_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0

    if is_negative:
        if mag >= (1 << 63):
            return FP_Q16_OK, I64_MIN_VALUE
        return FP_Q16_OK, -mag

    return FP_Q16_OK, mag


def fpq16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    if a_q16 == 0 or b_q16 == 0:
        return FP_Q16_OK, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)
    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    return fp_try_apply_sign_from_u64_checked(rounded_mag, is_negative)


def fpq16_from_int(x: int) -> int:
    max_int = I64_MAX_VALUE >> FP_Q16_SHIFT
    min_int = -(1 << (63 - FP_Q16_SHIFT))
    if x > max_int:
        return I64_MAX_VALUE
    if x < min_int:
        return I64_MIN_VALUE
    return x << FP_Q16_SHIFT


def fpq16_div(num: int, den: int) -> int:
    if den == 0:
        return 0

    abs_num = fp_abs_to_u64(num)
    abs_den = fp_abs_to_u64(den)
    is_negative = (num < 0) ^ (den < 0)

    limit = I64_MAX_VALUE
    if is_negative:
        limit = 1 << 63

    int_part = abs_num // abs_den
    if int_part > (limit >> FP_Q16_SHIFT):
        return I64_MIN_VALUE if is_negative else I64_MAX_VALUE

    result_mag = int_part << FP_Q16_SHIFT
    rem = abs_num % abs_den

    for bit in range(FP_Q16_SHIFT - 1, -1, -1):
        rem <<= 1
        if rem >= abs_den:
            rem -= abs_den
            add = 1 << bit
            if result_mag <= limit - add:
                result_mag |= add
            else:
                result_mag = limit

    if rem >= ((abs_den + 1) >> 1):
        if result_mag < limit:
            result_mag += 1

    _, signed = fp_try_apply_sign_from_u64_checked(result_mag, is_negative)
    return signed


def int_sqrt_u64(x: int) -> int:
    res = 0
    bit = 1 << 62

    while bit > x:
        bit >>= 2

    while bit:
        if x >= res + bit:
            x -= res + bit
            res = (res >> 1) + bit
        else:
            res >>= 1
        bit >>= 2

    return res


def fpq16_sqrt(x_q16: int) -> int:
    if x_q16 <= 0:
        return 0

    if x_q16 > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        shifted = I64_MAX_VALUE
    else:
        shifted = x_q16 << FP_Q16_SHIFT

    return int_sqrt_u64(shifted)


def fpq16_rmsnorm_compute_inv_denom_checked(
    input_row_q16: list[int] | None,
    lane_count: int,
    eps_q16: int,
) -> tuple[int, int, int]:
    if input_row_q16 is None:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if lane_count <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if eps_q16 < 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if lane_count > (I64_MAX_VALUE >> FP_Q16_SHIFT):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    sum_sq_q16 = 0
    for i in range(lane_count):
        status, sq_q16 = fpq16_mul_checked(input_row_q16[i], input_row_q16[i])
        if status != FP_Q16_OK:
            return status, 0, 0
        if sum_sq_q16 > I64_MAX_VALUE - sq_q16:
            return FP_Q16_ERR_OVERFLOW, 0, 0
        sum_sq_q16 += sq_q16

    count_q16 = fpq16_from_int(lane_count)
    if count_q16 <= 0:
        return FP_Q16_ERR_OVERFLOW, 0, 0

    mean_sq_q16 = fpq16_div(sum_sq_q16, count_q16)
    if mean_sq_q16 < 0:
        return FP_Q16_ERR_DOMAIN, 0, 0

    if mean_sq_q16 > I64_MAX_VALUE - eps_q16:
        return FP_Q16_ERR_OVERFLOW, 0, 0

    denom_arg_q16 = mean_sq_q16 + eps_q16
    denom_q16 = fpq16_sqrt(denom_arg_q16)
    if denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0, 0

    inv_denom_q16 = fpq16_div(FP_Q16_ONE, denom_q16)
    if inv_denom_q16 <= 0:
        return FP_Q16_ERR_DOMAIN, 0, 0

    return FP_Q16_OK, inv_denom_q16, denom_q16


def fpq16_try_add_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if rhs > 0:
        if lhs > I64_MAX_VALUE - rhs:
            return FP_Q16_ERR_OVERFLOW, 0
    elif rhs < 0:
        if lhs < I64_MIN_VALUE - rhs:
            return FP_Q16_ERR_OVERFLOW, 0
    return FP_Q16_OK, lhs + rhs


def fpq16_try_mul_i64_checked(lhs: int, rhs: int) -> tuple[int, int]:
    if lhs == 0 or rhs == 0:
        return FP_Q16_OK, 0

    if lhs == I64_MIN_VALUE and rhs != 1:
        return FP_Q16_ERR_OVERFLOW, 0
    if rhs == I64_MIN_VALUE and lhs != 1:
        return FP_Q16_ERR_OVERFLOW, 0

    prod = lhs * rhs
    if prod < I64_MIN_VALUE or prod > I64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, 0

    return FP_Q16_OK, prod


def fpq16_rmsnorm_rows_checked_reference(
    input_q16: list[int] | None,
    input_capacity: int,
    input_row_stride: int,
    gamma_q16: list[int] | None,
    gamma_capacity: int,
    gamma_row_stride: int,
    output_q16: list[int] | None,
    output_capacity: int,
    output_row_stride: int,
    row_count: int,
    lane_count: int,
    eps_q16: int,
    *,
    input_addr: int = 0x1000,
    gamma_addr: int = 0x2000,
    output_addr: int = 0x3000,
) -> tuple[int, list[int]]:
    if input_q16 is None or gamma_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, []

    if input_capacity < 0 or gamma_capacity < 0 or output_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if input_row_stride < 0 or gamma_row_stride < 0 or output_row_stride < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if row_count < 0 or lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, []
    if eps_q16 < 0:
        return FP_Q16_ERR_BAD_PARAM, []

    if row_count == 0 or lane_count == 0:
        return FP_Q16_OK, list(output_q16)

    if (
        input_row_stride < lane_count
        or gamma_row_stride < lane_count
        or output_row_stride < lane_count
    ):
        return FP_Q16_ERR_BAD_PARAM, []

    if input_q16 is output_q16 and input_row_stride != output_row_stride:
        return FP_Q16_ERR_BAD_PARAM, []
    if gamma_q16 is output_q16:
        return FP_Q16_ERR_BAD_PARAM, []

    last_row = row_count - 1

    status, required_input_cells = fpq16_try_mul_i64_checked(last_row, input_row_stride)
    if status != FP_Q16_OK:
        return status, []
    status, required_input_cells = fpq16_try_add_i64_checked(required_input_cells, lane_count)
    if status != FP_Q16_OK:
        return status, []

    status, required_gamma_cells = fpq16_try_mul_i64_checked(last_row, gamma_row_stride)
    if status != FP_Q16_OK:
        return status, []
    status, required_gamma_cells = fpq16_try_add_i64_checked(required_gamma_cells, lane_count)
    if status != FP_Q16_OK:
        return status, []

    status, required_output_cells = fpq16_try_mul_i64_checked(last_row, output_row_stride)
    if status != FP_Q16_OK:
        return status, []
    status, required_output_cells = fpq16_try_add_i64_checked(required_output_cells, lane_count)
    if status != FP_Q16_OK:
        return status, []

    if required_input_cells > input_capacity:
        return FP_Q16_ERR_BAD_PARAM, []
    if required_gamma_cells > gamma_capacity:
        return FP_Q16_ERR_BAD_PARAM, []
    if required_output_cells > output_capacity:
        return FP_Q16_ERR_BAD_PARAM, []

    for required_cells, addr in (
        (required_input_cells, input_addr),
        (required_gamma_cells, gamma_addr),
        (required_output_cells, output_addr),
    ):
        last_index = required_cells - 1
        if last_index > (I64_MAX_VALUE >> 3):
            return FP_Q16_ERR_OVERFLOW, []
        last_byte_offset = last_index << 3
        if addr > (U64_MAX_VALUE - last_byte_offset):
            return FP_Q16_ERR_OVERFLOW, []

    in_base = 0
    gamma_base = 0
    out_base = 0

    for _ in range(row_count):
        in_row = input_q16[in_base : in_base + lane_count]
        status, inv_denom_q16, _ = fpq16_rmsnorm_compute_inv_denom_checked(
            in_row,
            lane_count,
            eps_q16,
        )
        if status != FP_Q16_OK:
            return status, []

        for lane in range(lane_count):
            status, norm_lane_q16 = fpq16_mul_checked(input_q16[in_base + lane], inv_denom_q16)
            if status != FP_Q16_OK:
                return status, []
            status, _ = fpq16_mul_checked(norm_lane_q16, gamma_q16[gamma_base + lane])
            if status != FP_Q16_OK:
                return status, []

        status, in_base = fpq16_try_add_i64_checked(in_base, input_row_stride)
        if status != FP_Q16_OK:
            return status, []
        status, gamma_base = fpq16_try_add_i64_checked(gamma_base, gamma_row_stride)
        if status != FP_Q16_OK:
            return status, []
        status, out_base = fpq16_try_add_i64_checked(out_base, output_row_stride)
        if status != FP_Q16_OK:
            return status, []

    out = list(output_q16)
    in_base = 0
    gamma_base = 0
    out_base = 0

    for _ in range(row_count):
        in_row = input_q16[in_base : in_base + lane_count]
        status, inv_denom_q16, _ = fpq16_rmsnorm_compute_inv_denom_checked(
            in_row,
            lane_count,
            eps_q16,
        )
        if status != FP_Q16_OK:
            return status, []

        for lane in range(lane_count):
            status, norm_lane_q16 = fpq16_mul_checked(input_q16[in_base + lane], inv_denom_q16)
            if status != FP_Q16_OK:
                return status, []
            status, weighted_lane_q16 = fpq16_mul_checked(norm_lane_q16, gamma_q16[gamma_base + lane])
            if status != FP_Q16_OK:
                return status, []
            out[out_base + lane] = weighted_lane_q16

        status, in_base = fpq16_try_add_i64_checked(in_base, input_row_stride)
        if status != FP_Q16_OK:
            return status, []
        status, gamma_base = fpq16_try_add_i64_checked(gamma_base, gamma_row_stride)
        if status != FP_Q16_OK:
            return status, []
        status, out_base = fpq16_try_add_i64_checked(out_base, output_row_stride)
        if status != FP_Q16_OK:
            return status, []

    return FP_Q16_OK, out


def test_source_contains_iq1135_rows_function_and_contract_guards() -> None:
    source = Path("src/math/rmsnorm.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16RMSNormRowsChecked(I64 *input_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("U0 FPQ16RMSNorm(", 1)[0]

    assert "if (input_row_stride < 0 || gamma_row_stride < 0 || output_row_stride < 0)" in body
    assert "if (input_q16 == output_q16 && input_row_stride != output_row_stride)" in body
    assert "if (required_input_cells > input_capacity)" in body
    assert "if ((U64)output_q16 > (U64_MAX_VALUE - (U64)last_byte_offset))" in body
    assert "// Global preflight: prove all rows can normalize without overflow/domain" in body


def test_zero_row_or_lane_short_circuit_no_writes() -> None:
    out = [123, 456, 789]

    status, out_after = fpq16_rmsnorm_rows_checked_reference(
        [1, 2, 3],
        3,
        3,
        [1, 1, 1],
        3,
        3,
        out,
        3,
        3,
        0,
        3,
        0,
    )
    assert status == FP_Q16_OK
    assert out_after == out

    status, out_after = fpq16_rmsnorm_rows_checked_reference(
        [1, 2, 3],
        3,
        3,
        [1, 1, 1],
        3,
        3,
        out,
        3,
        3,
        3,
        0,
        0,
    )
    assert status == FP_Q16_OK
    assert out_after == out


def test_stride_capacity_and_alias_guards() -> None:
    input_buf = [1000, 2000, 3000, 4000, 5000, 6000]
    gamma_buf = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]
    out_buf = [0] * 6

    status, _ = fpq16_rmsnorm_rows_checked_reference(
        input_buf,
        6,
        1,
        gamma_buf,
        4,
        2,
        out_buf,
        6,
        2,
        2,
        2,
        64,
    )
    assert status == FP_Q16_ERR_BAD_PARAM

    status, _ = fpq16_rmsnorm_rows_checked_reference(
        input_buf,
        3,
        3,
        gamma_buf,
        4,
        2,
        out_buf,
        6,
        3,
        2,
        2,
        64,
    )
    assert status == FP_Q16_ERR_BAD_PARAM

    alias_out = input_buf
    status, _ = fpq16_rmsnorm_rows_checked_reference(
        input_buf,
        6,
        3,
        gamma_buf,
        4,
        2,
        alias_out,
        6,
        2,
        2,
        2,
        64,
    )
    assert status == FP_Q16_ERR_BAD_PARAM

    status, _ = fpq16_rmsnorm_rows_checked_reference(
        input_buf,
        6,
        3,
        out_buf,
        6,
        3,
        out_buf,
        6,
        3,
        2,
        2,
        64,
    )
    assert status == FP_Q16_ERR_BAD_PARAM


def test_pointer_span_overflow_guard() -> None:
    status, _ = fpq16_rmsnorm_rows_checked_reference(
        [1000, 2000, 3000, 4000],
        4,
        2,
        [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE],
        4,
        2,
        [0, 0, 0, 0],
        4,
        2,
        2,
        2,
        64,
        output_addr=U64_MAX_VALUE,
    )
    assert status == FP_Q16_ERR_OVERFLOW


def test_nopartial_write_when_preflight_fails() -> None:
    input_buf = [
        5 * FP_Q16_ONE,
        -7 * FP_Q16_ONE,
        3 * FP_Q16_ONE,
        4 * FP_Q16_ONE,
    ]
    gamma_buf = [FP_Q16_ONE, FP_Q16_ONE, I64_MAX_VALUE, I64_MAX_VALUE]
    out_buf = [111, 222, 333, 444]

    status, out_after = fpq16_rmsnorm_rows_checked_reference(
        input_buf,
        4,
        2,
        gamma_buf,
        4,
        2,
        out_buf,
        4,
        2,
        2,
        2,
        64,
    )
    assert status in (FP_Q16_ERR_OVERFLOW, FP_Q16_ERR_DOMAIN)
    assert out_after == []
    assert out_buf == [111, 222, 333, 444]


def test_deterministic_numeric_case_two_rows() -> None:
    eps_q16 = int(0.0001 * FP_Q16_ONE)

    input_buf = [
        int(0.5 * FP_Q16_ONE),
        int(-0.25 * FP_Q16_ONE),
        int(1.0 * FP_Q16_ONE),
        int(-1.0 * FP_Q16_ONE),
        int(0.125 * FP_Q16_ONE),
        int(0.75 * FP_Q16_ONE),
    ]

    gamma_buf = [
        int(1.0 * FP_Q16_ONE),
        int(0.5 * FP_Q16_ONE),
        int(2.0 * FP_Q16_ONE),
        int(0.75 * FP_Q16_ONE),
        int(1.25 * FP_Q16_ONE),
        int(0.25 * FP_Q16_ONE),
    ]

    out_buf = [0] * 6

    status, out = fpq16_rmsnorm_rows_checked_reference(
        input_buf,
        6,
        3,
        gamma_buf,
        6,
        3,
        out_buf,
        6,
        3,
        2,
        3,
        eps_q16,
    )
    assert status == FP_Q16_OK

    assert len(out) == 6
    assert out[0] > 0
    assert out[1] < 0
    assert out[2] > out[0]
    assert out[3] < 0
    assert out[4] > 0


def test_randomized_valid_reference_invariants() -> None:
    rng = random.Random(20260422_1135)

    for _ in range(2000):
        row_count = rng.randint(1, 5)
        lane_count = rng.randint(1, 16)
        input_row_stride = rng.randint(lane_count, lane_count + 4)
        gamma_row_stride = rng.randint(lane_count, lane_count + 4)
        output_row_stride = rng.randint(lane_count, lane_count + 4)

        input_capacity = (row_count - 1) * input_row_stride + lane_count
        gamma_capacity = (row_count - 1) * gamma_row_stride + lane_count
        output_capacity = (row_count - 1) * output_row_stride + lane_count

        input_buf = [0] * input_capacity
        gamma_buf = [0] * gamma_capacity
        out_buf = [rng.randint(-999, 999) for _ in range(output_capacity)]

        for row in range(row_count):
            in_base = row * input_row_stride
            gamma_base = row * gamma_row_stride
            for lane in range(lane_count):
                input_buf[in_base + lane] = rng.randint(-3 * FP_Q16_ONE, 3 * FP_Q16_ONE)
                gamma_buf[gamma_base + lane] = rng.randint(-2 * FP_Q16_ONE, 2 * FP_Q16_ONE)

        status, out = fpq16_rmsnorm_rows_checked_reference(
            input_buf,
            input_capacity,
            input_row_stride,
            gamma_buf,
            gamma_capacity,
            gamma_row_stride,
            out_buf,
            output_capacity,
            output_row_stride,
            row_count,
            lane_count,
            rng.randint(0, 1024),
        )

        assert status in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW, FP_Q16_ERR_DOMAIN)
        if status == FP_Q16_OK:
            assert len(out) == output_capacity

            for row in range(row_count):
                out_base = row * output_row_stride
                for lane in range(lane_count):
                    val = out[out_base + lane]
                    assert I64_MIN_VALUE <= val <= I64_MAX_VALUE


def run() -> None:
    test_source_contains_iq1135_rows_function_and_contract_guards()
    test_zero_row_or_lane_short_circuit_no_writes()
    test_stride_capacity_and_alias_guards()
    test_pointer_span_overflow_guard()
    test_nopartial_write_when_preflight_fails()
    test_deterministic_numeric_case_two_rows()
    test_randomized_valid_reference_invariants()
    print("rmsnorm_q16_rows_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
