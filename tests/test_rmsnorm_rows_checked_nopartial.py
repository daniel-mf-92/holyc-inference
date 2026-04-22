#!/usr/bin/env python3
"""Reference checks for FPQ16RMSNormRowsCheckedNoPartial (IQ-1163)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rmsnorm_q16_rows_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_DOMAIN,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    FP_Q16_ONE,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
    fpq16_mul_checked,
    fpq16_rmsnorm_compute_inv_denom_checked,
    fpq16_try_add_i64_checked,
    fpq16_try_mul_i64_checked,
)


def fpq16_rmsnorm_rows_checked_nopartial_reference(
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

    if input_row_stride < lane_count or gamma_row_stride < lane_count or output_row_stride < lane_count:
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

    input_span_start = input_addr
    gamma_span_start = gamma_addr
    output_span_start = output_addr

    input_span_end = input_span_start + (required_input_cells << 3)
    gamma_span_end = gamma_span_start + (required_gamma_cells << 3)
    output_span_end = output_span_start + (required_output_cells << 3)

    if input_q16 is not output_q16:
        if not (input_span_end <= output_span_start or output_span_end <= input_span_start):
            return FP_Q16_ERR_BAD_PARAM, []

    if not (gamma_span_end <= output_span_start or output_span_end <= gamma_span_start):
        return FP_Q16_ERR_BAD_PARAM, []

    if row_count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, []
    if required_output_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, []

    inv_rows = [0] * row_count
    staged_out = [0] * required_output_cells

    in_base = 0
    gamma_base = 0
    out_base = 0

    for row in range(row_count):
        in_row = input_q16[in_base : in_base + lane_count]
        status, inv_denom_q16, _ = fpq16_rmsnorm_compute_inv_denom_checked(
            in_row,
            lane_count,
            eps_q16,
        )
        if status != FP_Q16_OK:
            return status, []

        inv_rows[row] = inv_denom_q16

        for lane in range(lane_count):
            status, norm_lane_q16 = fpq16_mul_checked(input_q16[in_base + lane], inv_denom_q16)
            if status != FP_Q16_OK:
                return status, []
            status, weighted_lane_q16 = fpq16_mul_checked(norm_lane_q16, gamma_q16[gamma_base + lane])
            if status != FP_Q16_OK:
                return status, []
            staged_out[out_base + lane] = weighted_lane_q16

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
    out_base = 0

    for row in range(row_count):
        if inv_rows[row] <= 0:
            return FP_Q16_ERR_DOMAIN, []

        for lane in range(lane_count):
            out[out_base + lane] = staged_out[out_base + lane]

        status, out_base = fpq16_try_add_i64_checked(out_base, output_row_stride)
        if status != FP_Q16_OK:
            return status, []

    return FP_Q16_OK, out


def test_source_contains_iq1163_rows_nopartial_kernel() -> None:
    source = Path("src/math/rmsnorm.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16RMSNormRowsCheckedNoPartial(I64 *input_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("U0 FPQ16RMSNorm(", 1)[0]

    assert "status = FPQ16RMSNormRowsChecked(input_q16," in body
    assert "staged_output_q16 = MAlloc(staged_output_bytes);" in body
    assert "// Phase 1 (preflight): run full checked kernel into private staging so" in body
    assert "// Phase 2 (commit): one publish pass into caller output." in body
    assert "snapshot_row_count" in body


def test_zero_row_or_lane_short_circuit_no_writes() -> None:
    out = [333, 444, 555]

    status, out_after = fpq16_rmsnorm_rows_checked_nopartial_reference(
        [100, 200, 300],
        3,
        3,
        [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE],
        3,
        3,
        out,
        3,
        3,
        0,
        3,
        64,
    )
    assert status == FP_Q16_OK
    assert out_after == out

    status, out_after = fpq16_rmsnorm_rows_checked_nopartial_reference(
        [100, 200, 300],
        3,
        3,
        [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE],
        3,
        3,
        out,
        3,
        3,
        2,
        0,
        64,
    )
    assert status == FP_Q16_OK
    assert out_after == out


def test_stride_capacity_and_alias_guards() -> None:
    input_buf = [1000, 2000, 3000, 4000, 5000, 6000]
    gamma_buf = [FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE, FP_Q16_ONE]
    out_buf = [0] * 6

    status, _ = fpq16_rmsnorm_rows_checked_nopartial_reference(
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

    status, _ = fpq16_rmsnorm_rows_checked_nopartial_reference(
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
    status, _ = fpq16_rmsnorm_rows_checked_nopartial_reference(
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

    status, _ = fpq16_rmsnorm_rows_checked_nopartial_reference(
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
    status, _ = fpq16_rmsnorm_rows_checked_nopartial_reference(
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


def test_nopartial_write_when_stage_fails() -> None:
    input_buf = [
        5 * FP_Q16_ONE,
        -7 * FP_Q16_ONE,
        3 * FP_Q16_ONE,
        4 * FP_Q16_ONE,
    ]
    gamma_buf = [FP_Q16_ONE, FP_Q16_ONE, I64_MAX_VALUE, I64_MAX_VALUE]
    out_buf = [111, 222, 333, 444]

    status, _ = fpq16_rmsnorm_rows_checked_nopartial_reference(
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
    assert status == FP_Q16_ERR_OVERFLOW
    assert out_buf == [111, 222, 333, 444]


def test_success_matches_expected_rows_and_strides() -> None:
    input_buf = [
        2 * FP_Q16_ONE,
        4 * FP_Q16_ONE,
        7,
        3 * FP_Q16_ONE,
        6 * FP_Q16_ONE,
        11,
    ]
    gamma_buf = [
        FP_Q16_ONE,
        2 * FP_Q16_ONE,
        9,
        (FP_Q16_ONE // 2),
        FP_Q16_ONE,
        13,
    ]
    out_buf = [999, 888, 777, 666, 555, 444]

    status, out_after = fpq16_rmsnorm_rows_checked_nopartial_reference(
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
        2,
        64,
    )
    assert status == FP_Q16_OK
    assert out_after[2] == 777
    assert out_after[5] == 444

    row0 = out_after[0:2]
    row1 = out_after[3:5]

    assert row0[0] > 0
    assert row0[1] > row0[0]
    assert row1[0] > 0
    assert row1[1] > row1[0]


def test_randomized_rows_reference_invariants() -> None:
    random.seed(1163)

    for _ in range(160):
        row_count = random.randint(1, 5)
        lane_count = random.randint(1, 6)

        input_row_stride = lane_count + random.randint(0, 3)
        gamma_row_stride = lane_count + random.randint(0, 3)
        output_row_stride = lane_count + random.randint(0, 3)

        required_input = (row_count - 1) * input_row_stride + lane_count
        required_gamma = (row_count - 1) * gamma_row_stride + lane_count
        required_output = (row_count - 1) * output_row_stride + lane_count

        input_capacity = required_input + random.randint(0, 2)
        gamma_capacity = required_gamma + random.randint(0, 2)
        output_capacity = required_output + random.randint(0, 2)

        input_buf = [random.randint(-(4 * FP_Q16_ONE), 4 * FP_Q16_ONE) for _ in range(input_capacity)]
        gamma_buf = [random.randint(-(2 * FP_Q16_ONE), 2 * FP_Q16_ONE) for _ in range(gamma_capacity)]
        out_buf = [random.randint(-999, 999) for _ in range(output_capacity)]

        eps_q16 = random.randint(1, 512)

        status, out_after = fpq16_rmsnorm_rows_checked_nopartial_reference(
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
            eps_q16,
        )

        if status == FP_Q16_OK:
            assert len(out_after) == len(out_buf)
            row_base = 0
            for _ in range(row_count):
                for lane in range(lane_count):
                    assert isinstance(out_after[row_base + lane], int)
                row_base += output_row_stride
        else:
            assert status in {
                FP_Q16_ERR_BAD_PARAM,
                FP_Q16_ERR_DOMAIN,
                FP_Q16_ERR_OVERFLOW,
            }


if __name__ == "__main__":
    test_source_contains_iq1163_rows_nopartial_kernel()
    test_zero_row_or_lane_short_circuit_no_writes()
    test_stride_capacity_and_alias_guards()
    test_pointer_span_overflow_guard()
    test_nopartial_write_when_stage_fails()
    test_success_matches_expected_rows_and_strides()
    test_randomized_rows_reference_invariants()
    print("ok")
