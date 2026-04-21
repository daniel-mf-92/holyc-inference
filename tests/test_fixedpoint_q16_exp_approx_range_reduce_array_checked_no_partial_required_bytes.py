#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes (IQ-904)."""

from __future__ import annotations

import random
from pathlib import Path

from test_fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    fpq16_exp_approx_range_reduce_checked_no_partial_array,
)


def fpq16_exp_approx_range_reduce_checked_no_partial_array_required_bytes(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_required_output_bytes_present: bool,
    x_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
    out_required_output_bytes_addr: int = 0x4000,
) -> tuple[int, int]:
    if not out_required_output_bytes_present:
        return FP_Q16_ERR_NULL_PTR, -1

    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, -1

    # Wrapper-level alias rejection for scalar diagnostics output.
    if out_required_output_bytes_addr in (x_addr, out_k_addr, out_r_addr):
        return FP_Q16_ERR_BAD_PARAM, -1

    # Span overlap checks between diagnostics scalar and lane arrays.
    if count:
        scalar_start = out_required_output_bytes_addr
        scalar_end = scalar_start + 8

        x_end = x_addr + (count << 3)
        k_end = out_k_addr + (count << 3)
        r_end = out_r_addr + (count << 3)

        if not (scalar_end <= x_addr or x_end <= scalar_start):
            return FP_Q16_ERR_BAD_PARAM, -1
        if not (scalar_end <= out_k_addr or k_end <= scalar_start):
            return FP_Q16_ERR_BAD_PARAM, -1
        if not (scalar_end <= out_r_addr or r_end <= scalar_start):
            return FP_Q16_ERR_BAD_PARAM, -1

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array(
        x_q16,
        out_k,
        out_r_q16,
        count,
        x_addr=x_addr,
        out_k_addr=out_k_addr,
        out_r_addr=out_r_addr,
    )
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status, -1

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, -1

    required_output_bytes = count << 3
    return status, required_output_bytes


def test_source_contains_iq904_required_bytes_helper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16ExpApproxChecked", 1)[0]
    assert "if (!x_q16 || !out_k || !out_r_q16 || !out_required_output_bytes)" in body
    assert "if (out_required_output_bytes == x_q16 ||" in body
    assert "status = FPArrayI64SpanChecked(x_q16, count, &x_base, &x_end);" in body
    assert "if (FPAddressRangesOverlap(required_base, required_end, x_base, x_end) ||" in body
    assert "status = FPQ16ExpApproxRangeReduceCheckedNoPartialArray(x_q16," in body
    assert "staged_required_output_bytes = count << 3;" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_null_required_bytes_and_hard_fail_no_write() -> None:
    x = [0, 1, -1]
    out_k = [0x11, 0x22, 0x33]
    out_r = [0x44, 0x55, 0x66]

    status, required = fpq16_exp_approx_range_reduce_checked_no_partial_array_required_bytes(
        x,
        out_k,
        out_r,
        3,
        out_required_output_bytes_present=False,
    )
    assert status == FP_Q16_ERR_NULL_PTR
    assert required == -1

    before_k = out_k.copy()
    before_r = out_r.copy()
    status, required = fpq16_exp_approx_range_reduce_checked_no_partial_array_required_bytes(
        x,
        out_k,
        out_r,
        -1,
        out_required_output_bytes_present=True,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert required == -1
    assert out_k == before_k
    assert out_r == before_r


def test_known_vector_required_bytes_and_lane_results() -> None:
    x = [-(1 << 20), -65_536, -1, 0, 1, 65_536, 1 << 20]
    out_k = [0xAAAA] * len(x)
    out_r = [0xBBBB] * len(x)

    status, required = fpq16_exp_approx_range_reduce_checked_no_partial_array_required_bytes(
        x,
        out_k,
        out_r,
        len(x),
        out_required_output_bytes_present=True,
    )
    assert status == FP_Q16_OK
    assert required == len(x) * 8

    for idx, x_lane in enumerate(x):
        status_lane = fpq16_exp_approx_range_reduce_checked_no_partial_array(
            [x_lane],
            [0],
            [0],
            1,
            x_addr=0x5000,
            out_k_addr=0x6000,
            out_r_addr=0x7000,
        )
        assert status_lane == FP_Q16_OK
        # Ensure helper wrote outputs (not sentinel left-behind).
        assert out_k[idx] != 0xAAAA
        assert out_r[idx] != 0xBBBB


def test_scalar_output_overlap_rejected() -> None:
    x = [0, 1, 2, 3]
    out_k = [0] * 4
    out_r = [0] * 4

    status, required = fpq16_exp_approx_range_reduce_checked_no_partial_array_required_bytes(
        x,
        out_k,
        out_r,
        4,
        out_required_output_bytes_present=True,
        x_addr=0x1000,
        out_k_addr=0x2000,
        out_r_addr=0x3000,
        out_required_output_bytes_addr=0x1008,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert required == -1


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260421_904)

    for _ in range(4000):
        count = rng.randint(0, 64)
        x = [rng.randint(-(1 << 40), 1 << 40) for _ in range(max(1, count))]

        if count == 0:
            x = []

        out_k_a = [0x55AA] * max(1, count)
        out_r_a = [0xAA55] * max(1, count)
        out_k_b = out_k_a.copy()
        out_r_b = out_r_a.copy()

        x_addr = 0x1000
        out_k_addr = 0x2000
        out_r_addr = 0x3000
        out_required_addr = 0x4000

        if rng.random() < 0.15 and count > 0:
            # Induce scalar/output overlap failure path.
            out_required_addr = out_k_addr + rng.randint(0, max(0, (count << 3) - 1))

        got_status, got_required = fpq16_exp_approx_range_reduce_checked_no_partial_array_required_bytes(
            x if count > 0 else [],
            out_k_a,
            out_r_a,
            count,
            out_required_output_bytes_present=True,
            x_addr=x_addr,
            out_k_addr=out_k_addr,
            out_r_addr=out_r_addr,
            out_required_output_bytes_addr=out_required_addr,
        )

        exp_status = fpq16_exp_approx_range_reduce_checked_no_partial_array(
            x if count > 0 else [],
            out_k_b,
            out_r_b,
            count,
            x_addr=x_addr,
            out_k_addr=out_k_addr,
            out_r_addr=out_r_addr,
        )

        if out_required_addr in (x_addr, out_k_addr, out_r_addr):
            exp_status = FP_Q16_ERR_BAD_PARAM

        if count > 0:
            scalar_start = out_required_addr
            scalar_end = scalar_start + 8
            x_end = x_addr + (count << 3)
            k_end = out_k_addr + (count << 3)
            r_end = out_r_addr + (count << 3)
            if not (scalar_end <= x_addr or x_end <= scalar_start):
                exp_status = FP_Q16_ERR_BAD_PARAM
            if not (scalar_end <= out_k_addr or k_end <= scalar_start):
                exp_status = FP_Q16_ERR_BAD_PARAM
            if not (scalar_end <= out_r_addr or r_end <= scalar_start):
                exp_status = FP_Q16_ERR_BAD_PARAM

        assert got_status == exp_status

        if got_status in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
            assert got_required == (count << 3)
        else:
            assert got_required == -1


def run() -> None:
    test_source_contains_iq904_required_bytes_helper()
    test_null_required_bytes_and_hard_fail_no_write()
    test_known_vector_required_bytes_and_lane_results()
    test_scalar_output_overlap_rejected()
    test_randomized_parity_against_explicit_composition()
    print("fixedpoint_q16_exp_approx_range_reduce_array_checked_no_partial_required_bytes=ok")


if __name__ == "__main__":
    run()
