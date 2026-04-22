#!/usr/bin/env python3
"""Parity harness for IQ-1151 FPQ16RMSNormRowsCheckedNoPartial wrapper semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_rmsnorm_q16_rows_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    fpq16_rmsnorm_rows_checked_reference,
)


def fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
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

    snapshot = (
        row_count,
        lane_count,
        input_capacity,
        gamma_capacity,
        output_capacity,
        input_row_stride,
        gamma_row_stride,
        output_row_stride,
        id(input_q16),
        id(gamma_q16),
        id(output_q16),
    )

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
    required_output_cells = last_row * output_row_stride + lane_count
    if required_output_cells < 0:
        return FP_Q16_ERR_OVERFLOW, []
    if required_output_cells > output_capacity:
        return FP_Q16_ERR_BAD_PARAM, []
    if required_output_cells > (1 << 60):
        return FP_Q16_ERR_OVERFLOW, []

    staged_out = [0] * required_output_cells

    status, staged_full = fpq16_rmsnorm_rows_checked_reference(
        input_q16,
        input_capacity,
        input_row_stride,
        gamma_q16,
        gamma_capacity,
        gamma_row_stride,
        staged_out,
        required_output_cells,
        output_row_stride,
        row_count,
        lane_count,
        eps_q16,
        input_addr=input_addr,
        gamma_addr=gamma_addr,
        output_addr=0xA000,
    )
    if status != FP_Q16_OK:
        return status, []

    current = (
        row_count,
        lane_count,
        input_capacity,
        gamma_capacity,
        output_capacity,
        input_row_stride,
        gamma_row_stride,
        output_row_stride,
        id(input_q16),
        id(gamma_q16),
        id(output_q16),
    )
    if current != snapshot:
        return FP_Q16_ERR_BAD_PARAM, []

    # Commit walks row bases with monotonic stride progression.
    out = list(output_q16)
    out_base = 0
    for _ in range(row_count):
        next_base = out_base + output_row_stride
        if next_base < out_base:
            return FP_Q16_ERR_OVERFLOW, []
        out_base = next_base

    out_base = 0
    for _ in range(row_count):
        for lane in range(lane_count):
            out[out_base + lane] = staged_full[out_base + lane]
        out_base += output_row_stride

    return FP_Q16_OK, out


def test_source_contains_iq1151_wrapper_pattern() -> None:
    source = Path("src/math/rmsnorm.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16RMSNormRowsCheckedNoPartial("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = FPQ16RMSNormRowsChecked(input_q16," in body
    assert "staged_output_q16" in body
    assert "snapshot_row_count" in body
    assert "snapshot_lane_count" in body
    assert "snapshot_input_capacity" in body
    assert "snapshot_gamma_capacity" in body
    assert "snapshot_output_capacity" in body
    assert "snapshot_input_row_stride" in body
    assert "snapshot_gamma_row_stride" in body
    assert "snapshot_output_row_stride" in body


def test_alias_stride_guard_rejected() -> None:
    data = [11, 22, 33, 44, 55, 66]
    gamma = [1 << 16] * 6

    err, out = fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
        data,
        len(data),
        3,
        gamma,
        len(gamma),
        3,
        data,
        len(data),
        4,
        2,
        2,
        1,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == []


def test_preflight_error_preserves_output() -> None:
    rng = random.Random(115101)

    row_count = 3
    lane_count = 4
    in_stride = 6
    gamma_stride = 6
    out_stride = 7

    input_capacity = row_count * in_stride
    gamma_capacity = row_count * gamma_stride

    input_q16 = [rng.randint(-8 << 16, 8 << 16) for _ in range(input_capacity)]
    gamma_q16 = [rng.randint(1, 3 << 16) for _ in range(gamma_capacity)]
    output_q16 = [777777, -333333]  # intentionally undersized
    before = output_q16[:]

    err, out = fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
        input_q16,
        input_capacity,
        in_stride,
        gamma_q16,
        gamma_capacity,
        gamma_stride,
        output_q16,
        len(output_q16),
        out_stride,
        row_count,
        lane_count,
        1,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == []
    assert output_q16 == before


def test_overflow_required_output_cells_rejected() -> None:
    tiny = [0]
    gamma = [1 << 16]
    out = [5]

    # required_output_cells = last_row*out_stride + lane_count -> overflow domain
    err, out_ref = fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
        tiny,
        1,
        1,
        gamma,
        1,
        1,
        out,
        1,
        (1 << 62),
        2,
        2,
        1,
    )
    assert err in (FP_Q16_ERR_BAD_PARAM, FP_Q16_ERR_OVERFLOW)
    assert out_ref == []


def test_random_parity_vs_rows_checked_reference() -> None:
    rng = random.Random(115102)

    for _ in range(220):
        row_count = rng.randint(1, 6)
        lane_count = rng.randint(1, 8)
        in_stride = lane_count + rng.randint(0, 3)
        gamma_stride = lane_count + rng.randint(0, 3)
        out_stride = lane_count + rng.randint(0, 3)

        input_capacity = row_count * in_stride
        gamma_capacity = row_count * gamma_stride
        output_capacity = row_count * out_stride

        input_q16 = [rng.randint(-6 << 16, 6 << 16) for _ in range(input_capacity)]
        gamma_q16 = [rng.randint(1, 4 << 16) for _ in range(gamma_capacity)]
        output_q16 = [rng.randint(-(1 << 20), 1 << 20) for _ in range(output_capacity)]

        status_base, expected = fpq16_rmsnorm_rows_checked_reference(
            input_q16,
            input_capacity,
            in_stride,
            gamma_q16,
            gamma_capacity,
            gamma_stride,
            output_q16,
            output_capacity,
            out_stride,
            row_count,
            lane_count,
            1,
        )

        status_wrap, got = fpq16_rmsnorm_rows_checked_nopartial_wrapper_reference(
            input_q16,
            input_capacity,
            in_stride,
            gamma_q16,
            gamma_capacity,
            gamma_stride,
            output_q16,
            output_capacity,
            out_stride,
            row_count,
            lane_count,
            1,
        )

        assert status_wrap == status_base
        if status_wrap == FP_Q16_OK:
            assert got == expected


if __name__ == "__main__":
    test_source_contains_iq1151_wrapper_pattern()
    test_alias_stride_guard_rejected()
    test_preflight_error_preserves_output()
    test_overflow_required_output_cells_rejected()
    test_random_parity_vs_rows_checked_reference()
    print("rmsnorm_q16_rows_checked_nopartial=ok")
