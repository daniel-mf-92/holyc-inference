#!/usr/bin/env python3
"""Parity harness for FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnly (IQ-1191)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_shift_round_checked_nopartial_array import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
    fpq16_mul_shift_round_checked_nopartial_array_reference,
)


def fp_address_ranges_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_end > b_start and b_end > a_start


def fp_array_i64_span_checked(base_addr: int, count: int) -> tuple[int, int, int]:
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if count == 0:
        return FP_Q16_OK, base_addr, base_addr
    if base_addr < 0 or base_addr > U64_MAX_VALUE:
        return FP_Q16_ERR_OVERFLOW, 0, 0
    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    span_bytes = count << 3
    if base_addr > U64_MAX_VALUE - span_bytes:
        return FP_Q16_ERR_OVERFLOW, 0, 0

    return FP_Q16_OK, base_addr, base_addr + span_bytes


def fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
    lhs_values: list[int] | None,
    lhs_capacity: int,
    rhs_values: list[int] | None,
    rhs_capacity: int,
    out_values: list[int] | None,
    out_capacity: int,
    count: int,
    shift: int,
    out_required_lhs: list[int] | None,
    out_required_rhs: list[int] | None,
    out_required_out: list[int] | None,
    *,
    lhs_addr: int = 0x1000,
    rhs_addr: int = 0x4000,
    out_addr: int = 0x7000,
    req_lhs_addr: int = 0xA000,
    req_rhs_addr: int = 0xB000,
    req_out_addr: int = 0xC000,
    mutate_snapshot: bool = False,
) -> int:
    if (
        lhs_values is None
        or rhs_values is None
        or out_values is None
        or out_required_lhs is None
        or out_required_rhs is None
        or out_required_out is None
    ):
        return FP_Q16_ERR_NULL_PTR

    if lhs_capacity < 0 or rhs_capacity < 0 or out_capacity < 0 or count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if out_required_lhs is out_required_rhs or out_required_lhs is out_required_out or out_required_rhs is out_required_out:
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_lhs is lhs_values
        or out_required_lhs is rhs_values
        or out_required_lhs is out_values
        or out_required_rhs is lhs_values
        or out_required_rhs is rhs_values
        or out_required_rhs is out_values
        or out_required_out is lhs_values
        or out_required_out is rhs_values
        or out_required_out is out_values
    ):
        return FP_Q16_ERR_BAD_PARAM

    if count > lhs_capacity or count > rhs_capacity or count > out_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if count:
        status, lhs_base, lhs_end = fp_array_i64_span_checked(lhs_addr, count)
        if status != FP_Q16_OK:
            return status
        status, rhs_base, rhs_end = fp_array_i64_span_checked(rhs_addr, count)
        if status != FP_Q16_OK:
            return status
        status, out_base, out_end = fp_array_i64_span_checked(out_addr, count)
        if status != FP_Q16_OK:
            return status

        if req_lhs_addr > U64_MAX_VALUE - 7 or req_rhs_addr > U64_MAX_VALUE - 7 or req_out_addr > U64_MAX_VALUE - 7:
            return FP_Q16_ERR_OVERFLOW

        req_lhs_end = req_lhs_addr + 8
        req_rhs_end = req_rhs_addr + 8
        req_out_end = req_out_addr + 8

        if (
            fp_address_ranges_overlap(req_lhs_addr, req_lhs_end, lhs_base, lhs_end)
            or fp_address_ranges_overlap(req_lhs_addr, req_lhs_end, rhs_base, rhs_end)
            or fp_address_ranges_overlap(req_lhs_addr, req_lhs_end, out_base, out_end)
            or fp_address_ranges_overlap(req_rhs_addr, req_rhs_end, lhs_base, lhs_end)
            or fp_address_ranges_overlap(req_rhs_addr, req_rhs_end, rhs_base, rhs_end)
            or fp_address_ranges_overlap(req_rhs_addr, req_rhs_end, out_base, out_end)
            or fp_address_ranges_overlap(req_out_addr, req_out_end, lhs_base, lhs_end)
            or fp_address_ranges_overlap(req_out_addr, req_out_end, rhs_base, rhs_end)
            or fp_address_ranges_overlap(req_out_addr, req_out_end, out_base, out_end)
            or fp_address_ranges_overlap(req_lhs_addr, req_lhs_end, req_rhs_addr, req_rhs_end)
            or fp_address_ranges_overlap(req_lhs_addr, req_lhs_end, req_out_addr, req_out_end)
            or fp_address_ranges_overlap(req_rhs_addr, req_rhs_end, req_out_addr, req_out_end)
        ):
            return FP_Q16_ERR_BAD_PARAM

    snapshot_lhs = lhs_values
    snapshot_rhs = rhs_values
    snapshot_out = out_values
    snapshot_lhs_capacity = lhs_capacity
    snapshot_rhs_capacity = rhs_capacity
    snapshot_out_capacity = out_capacity
    snapshot_count = count
    snapshot_shift = shift

    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        lhs_values,
        lhs_capacity,
        rhs_values,
        rhs_capacity,
        out_values,
        out_capacity,
        count,
        shift,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        shift += 1

    if (
        snapshot_lhs is not lhs_values
        or snapshot_rhs is not rhs_values
        or snapshot_out is not out_values
        or snapshot_lhs_capacity != lhs_capacity
        or snapshot_rhs_capacity != rhs_capacity
        or snapshot_out_capacity != out_capacity
        or snapshot_count != count
        or snapshot_shift != shift
    ):
        return FP_Q16_ERR_BAD_PARAM

    staged_required_lhs = count
    staged_required_rhs = count
    staged_required_out = count

    recomputed_required_lhs = snapshot_count
    recomputed_required_rhs = snapshot_count
    recomputed_required_out = snapshot_count

    if (
        staged_required_lhs != recomputed_required_lhs
        or staged_required_rhs != recomputed_required_rhs
        or staged_required_out != recomputed_required_out
    ):
        return FP_Q16_ERR_BAD_PARAM

    out_required_lhs[0] = staged_required_lhs
    out_required_rhs[0] = staged_required_rhs
    out_required_out[0] = staged_required_out
    return FP_Q16_OK


def explicit_expected(
    lhs_values: list[int],
    rhs_values: list[int],
    out_values: list[int],
    count: int,
    shift: int,
    out_required_lhs: list[int],
    out_required_rhs: list[int],
    out_required_out: list[int],
) -> int:
    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        lhs_values,
        count,
        rhs_values,
        count,
        out_values,
        count,
        count,
        shift,
    )
    if status != FP_Q16_OK:
        return status
    out_required_lhs[0] = count
    out_required_rhs[0] = count
    out_required_out[0] = count
    return FP_Q16_OK


def test_source_contains_iq1191_function_and_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnly(I64 *lhs_values,"
    assert sig in source
    body = source.split(sig, 1)[1].split("// Geometry diagnostics helper for FPQ16MulCheckedNoPartialArray.", 1)[0]
    assert "snapshot_shift = shift;" in body
    assert "status = FPQ16MulShiftRoundCheckedNoPartialArray(lhs_values," in body
    assert "if (count > lhs_capacity || count > rhs_capacity || count > out_capacity)" in body
    assert "*out_required_lhs = staged_required_lhs;" in body
    assert "*out_required_rhs = staged_required_rhs;" in body
    assert "*out_required_out = staged_required_out;" in body


def test_alias_capacity_shift_and_overflow_guards_preserve_outputs() -> None:
    lhs = [1, 2, 3]
    rhs = [4, 5, 6]
    out = [101, 102, 103]
    req_lhs = [701]
    req_rhs = [702]
    req_out = [703]

    status = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
        lhs,
        3,
        rhs,
        3,
        out,
        3,
        4,
        1,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 102, 103]
    assert req_lhs == [701] and req_rhs == [702] and req_out == [703]

    status = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
        lhs,
        3,
        rhs,
        3,
        out,
        3,
        3,
        -1,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 102, 103]
    assert req_lhs == [701] and req_rhs == [702] and req_out == [703]

    status = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
        [I64_MAX_VALUE],
        1,
        [I64_MAX_VALUE],
        1,
        [909],
        1,
        1,
        0,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert status == FP_Q16_ERR_OVERFLOW
    assert req_lhs == [701] and req_rhs == [702] and req_out == [703]


def test_success_publish_and_snapshot_mutation_guard() -> None:
    lhs = [5, -5, 9, -9]
    rhs = [3, 3, -3, -3]
    out = [0, 0, 0, 0]
    req_lhs = [0]
    req_rhs = [0]
    req_out = [0]

    status = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
        lhs,
        4,
        rhs,
        4,
        out,
        4,
        4,
        1,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert status == FP_Q16_OK
    assert req_lhs == [4] and req_rhs == [4] and req_out == [4]

    req_lhs_fail = [911]
    req_rhs_fail = [922]
    req_out_fail = [933]
    out_fail = [44, 45, 46, 47]
    status = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
        lhs,
        4,
        rhs,
        4,
        out_fail,
        4,
        4,
        1,
        req_lhs_fail,
        req_rhs_fail,
        req_out_fail,
        mutate_snapshot=True,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert req_lhs_fail == [911] and req_rhs_fail == [922] and req_out_fail == [933]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260423_1191)
    for _ in range(12000):
        count = rng.randint(0, 48)
        shift = rng.randint(0, 62)

        lhs = [rng.randint(-(1 << 63), (1 << 63) - 1) for _ in range(count)]
        rhs = [rng.randint(-(1 << 63), (1 << 63) - 1) for _ in range(count)]

        out_ref = [0xA55A] * count
        out_exp = [0xA55A] * count

        req_lhs_ref = [0x11]
        req_rhs_ref = [0x22]
        req_out_ref = [0x33]

        req_lhs_exp = [0x11]
        req_rhs_exp = [0x22]
        req_out_exp = [0x33]

        status_ref = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
            lhs,
            count,
            rhs,
            count,
            out_ref,
            count,
            count,
            shift,
            req_lhs_ref,
            req_rhs_ref,
            req_out_ref,
        )
        status_exp = explicit_expected(
            lhs,
            rhs,
            out_exp,
            count,
            shift,
            req_lhs_exp,
            req_rhs_exp,
            req_out_exp,
        )

        assert status_ref == status_exp
        assert out_ref == out_exp
        assert req_lhs_ref == req_lhs_exp
        assert req_rhs_ref == req_rhs_exp
        assert req_out_ref == req_out_exp


def run() -> None:
    test_source_contains_iq1191_function_and_contract()
    test_alias_capacity_shift_and_overflow_guards_preserve_outputs()
    test_success_publish_and_snapshot_mutation_guard()
    test_randomized_parity_against_explicit_composition()
    print("fixedpoint_q16_mul_shift_round_checked_nopartial_array_commit_only=ok")


if __name__ == "__main__":
    run()
