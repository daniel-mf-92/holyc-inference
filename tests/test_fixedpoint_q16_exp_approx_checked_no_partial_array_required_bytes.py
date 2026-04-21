#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes (IQ-909)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_fixedpoint_q16_exp_approx_checked_no_partial_array import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    FP_Q16_ONE,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
    fpq16_exp_approx_checked_no_partial_array,
)


def fp_address_ranges_overlap(a_base: int, a_end_exclusive: int, b_base: int, b_end_exclusive: int) -> int:
    if a_end_exclusive <= a_base:
        return 0
    if b_end_exclusive <= b_base:
        return 0
    if a_end_exclusive <= b_base:
        return 0
    if b_end_exclusive <= a_base:
        return 0
    return 1


def fp_array_i64_span_checked(base_addr: int | None, count: int) -> tuple[int, int, int]:
    if base_addr is None:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if count > (U64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    byte_count = (count << 3) & U64_MAX_VALUE
    base = base_addr & U64_MAX_VALUE

    if base > (U64_MAX_VALUE - byte_count):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    return FP_Q16_OK, base, (base + byte_count) & U64_MAX_VALUE


def fpq16_exp_approx_checked_no_partial_array_required_bytes(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_slot: list[int] | None,
    x_addr: int = 0x1000,
    out_addr: int = 0x3000,
    out_required_addr: int = 0x5000,
) -> int:
    if x_q16 is None or out_q16 is None or out_required_output_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if out_required_output_bytes_slot is x_q16 or out_required_output_bytes_slot is out_q16:
        return FP_Q16_ERR_BAD_PARAM

    if count:
        status, x_base, x_end = fp_array_i64_span_checked(x_addr, count)
        if status != FP_Q16_OK:
            return status
        status, out_base, out_end = fp_array_i64_span_checked(out_addr, count)
        if status != FP_Q16_OK:
            return status

        if out_required_addr > (U64_MAX_VALUE - 7):
            return FP_Q16_ERR_OVERFLOW
        required_end = out_required_addr + 8

        if (
            fp_address_ranges_overlap(out_required_addr, required_end, x_base, x_end)
            or fp_address_ranges_overlap(out_required_addr, required_end, out_base, out_end)
        ):
            return FP_Q16_ERR_BAD_PARAM

    status = fpq16_exp_approx_checked_no_partial_array(x_q16, out_q16, count, x_addr=x_addr, out_addr=out_addr)
    if status != FP_Q16_OK:
        return status

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    out_required_output_bytes_slot[0] = count << 3
    return FP_Q16_OK


def test_source_contains_iq909_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "status = FPQ16ExpApproxCheckedNoPartialArray(x_q16," in body
    assert "if (count > (I64_MAX_VALUE >> 3))" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_null_bad_count_alias_and_overlap_guards() -> None:
    x = [0, FP_Q16_ONE]
    out = [0x1111, 0x2222]
    required = [0x9999]

    assert fpq16_exp_approx_checked_no_partial_array_required_bytes(None, out, 1, required) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes(x, None, 1, required) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, 1, None) == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, -1, required) == FP_Q16_ERR_BAD_PARAM
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, 1, x) == FP_Q16_ERR_BAD_PARAM
    assert fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, 1, out) == FP_Q16_ERR_BAD_PARAM

    out_before = out.copy()
    req_before = required[0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
        x,
        out,
        2,
        required,
        x_addr=0x1000,
        out_addr=0x2000,
        out_required_addr=0x1000,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == out_before
    assert required[0] == req_before


def test_no_partial_on_delegate_overflow() -> None:
    x = [0, I64_MAX_VALUE, 1]
    out = [0xAAAA, 0xBBBB, 0xCCCC]
    required = [0xDDDD]

    out_before = out.copy()
    req_before = required[0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, len(x), required)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == out_before
    assert required[0] == req_before


def test_zero_count_writes_zero_required_bytes() -> None:
    x: list[int] = []
    out: list[int] = []
    required = [0x7777]

    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, 0, required)
    assert status == FP_Q16_OK
    assert required[0] == 0


def test_known_vectors_and_required_bytes() -> None:
    x = [
        -(8 * FP_Q16_ONE),
        -(4 * FP_Q16_ONE),
        -FP_Q16_ONE,
        -1,
        0,
        1,
        FP_Q16_ONE,
        4 * FP_Q16_ONE,
        8 * FP_Q16_ONE,
    ]
    out = [0x1212] * len(x)
    out_ref = out.copy()
    required = [0xFEED]

    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(x, out, len(x), required)
    assert status == FP_Q16_OK
    assert required[0] == len(x) * 8

    ref_status = fpq16_exp_approx_checked_no_partial_array(x, out_ref, len(x))
    assert ref_status == FP_Q16_OK
    assert out == out_ref


def test_randomized_parity() -> None:
    rng = random.Random(20260421_909)

    for _ in range(3000):
        count = rng.randint(0, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(count)]

        out_expected = [0x4141] * count
        out_got = [0x4141] * count
        required_expected = [0x5151]
        required_got = [0x5151]

        expected_status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
            x,
            out_expected,
            count,
            required_expected,
            x_addr=0x1000,
            out_addr=0x4000,
            out_required_addr=0x9000,
        )
        got_status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
            x,
            out_got,
            count,
            required_got,
            x_addr=0x1000,
            out_addr=0x4000,
            out_required_addr=0x9000,
        )

        assert got_status == expected_status
        assert out_got == out_expected
        assert required_got == required_expected


def run() -> None:
    test_source_contains_iq909_function()
    test_null_bad_count_alias_and_overlap_guards()
    test_no_partial_on_delegate_overflow()
    test_zero_count_writes_zero_required_bytes()
    test_known_vectors_and_required_bytes()
    test_randomized_parity()
    print("fixedpoint_q16_exp_approx_checked_no_partial_array_required_bytes=ok")


if __name__ == "__main__":
    run()
