#!/usr/bin/env python3
"""Parity harness for FPQ16MulCheckedNoPartialArrayCommitOnly (IQ-968)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_checked_no_partial_array import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    fpq16_mul_checked_no_partial_array,
)

U64_MAX_VALUE = (1 << 64) - 1


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


def fpq16_mul_checked_no_partial_array_commit_only(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_capacity: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    *,
    lhs_addr: int = 0x1000,
    rhs_addr: int = 0x4000,
    out_addr: int = 0x7000,
    out_required_cells_addr: int = 0xA000,
    out_required_bytes_addr: int = 0xB000,
    mutate_snapshot: bool = False,
) -> int:
    if (
        lhs_q16 is None
        or rhs_q16 is None
        or out_q16 is None
        or out_required_cells is None
        or out_required_bytes is None
    ):
        return FP_Q16_ERR_NULL_PTR
    if count < 0 or out_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM

    if out_required_cells is out_required_bytes:
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_cells is lhs_q16
        or out_required_cells is rhs_q16
        or out_required_cells is out_q16
        or out_required_bytes is lhs_q16
        or out_required_bytes is rhs_q16
        or out_required_bytes is out_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if count > out_capacity:
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

        if out_required_cells_addr > U64_MAX_VALUE - 7 or out_required_bytes_addr > U64_MAX_VALUE - 7:
            return FP_Q16_ERR_OVERFLOW

        req_cells_end = out_required_cells_addr + 8
        req_bytes_end = out_required_bytes_addr + 8

        if (
            fp_address_ranges_overlap(out_required_cells_addr, req_cells_end, lhs_base, lhs_end)
            or fp_address_ranges_overlap(out_required_cells_addr, req_cells_end, rhs_base, rhs_end)
            or fp_address_ranges_overlap(out_required_cells_addr, req_cells_end, out_base, out_end)
            or fp_address_ranges_overlap(out_required_bytes_addr, req_bytes_end, lhs_base, lhs_end)
            or fp_address_ranges_overlap(out_required_bytes_addr, req_bytes_end, rhs_base, rhs_end)
            or fp_address_ranges_overlap(out_required_bytes_addr, req_bytes_end, out_base, out_end)
            or fp_address_ranges_overlap(out_required_cells_addr, req_cells_end, out_required_bytes_addr, req_bytes_end)
        ):
            return FP_Q16_ERR_BAD_PARAM

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    snapshot_lhs = lhs_q16
    snapshot_rhs = rhs_q16
    snapshot_out = out_q16
    snapshot_count = count
    snapshot_out_capacity = out_capacity

    status = fpq16_mul_checked_no_partial_array(lhs_q16, rhs_q16, out_q16, count)
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        out_capacity += 1

    if (
        snapshot_lhs is not lhs_q16
        or snapshot_rhs is not rhs_q16
        or snapshot_out is not out_q16
        or snapshot_count != count
        or snapshot_out_capacity != out_capacity
    ):
        return FP_Q16_ERR_BAD_PARAM

    staged_required_cells = count
    staged_required_bytes = count << 3
    recomputed_required_cells = snapshot_count
    recomputed_required_bytes = snapshot_count << 3

    if (
        staged_required_cells != recomputed_required_cells
        or staged_required_bytes != recomputed_required_bytes
    ):
        return FP_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_required_cells
    out_required_bytes[0] = staged_required_bytes
    return FP_Q16_OK


def explicit_commit_only_composition(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_capacity: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
) -> int:
    if out_required_cells is None or out_required_bytes is None:
        return FP_Q16_ERR_NULL_PTR
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0 or out_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count > out_capacity:
        return FP_Q16_ERR_BAD_PARAM

    status = fpq16_mul_checked_no_partial_array(lhs_q16, rhs_q16, out_q16, count)
    if status != FP_Q16_OK:
        return status

    out_required_cells[0] = count
    out_required_bytes[0] = count << 3
    return FP_Q16_OK


def test_source_contains_iq968_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayCommitOnly(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatCheckedNoPartialArray", 1)[0]
    assert "snapshot_out_capacity = out_capacity;" in body
    assert "if (count > out_capacity)" in body
    assert "status = FPQ16MulCheckedNoPartialArray(lhs_q16," in body
    assert "if (snapshot_lhs_q16 != lhs_q16 ||" in body
    assert "*out_required_cells = staged_required_cells;" in body


def test_null_bad_and_no_publish() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0, 0]
    out_cells = [101]
    out_bytes = [202]

    err = fpq16_mul_checked_no_partial_array_commit_only(
        None,
        rhs,
        out,
        2,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cells == [101]
    assert out_bytes == [202]

    err = fpq16_mul_checked_no_partial_array_commit_only(
        lhs,
        rhs,
        out,
        -1,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [101]
    assert out_bytes == [202]

    err = fpq16_mul_checked_no_partial_array_commit_only(
        lhs,
        rhs,
        out,
        2,
        1,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [101]
    assert out_bytes == [202]


def test_success_publish_and_mutation_guard() -> None:
    lhs = [5 << 16] * 12
    rhs = [7 << 16] * 12
    out = [0] * 12

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_commit_only(
        lhs,
        rhs,
        out,
        12,
        12,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK
    assert out_cells == [12]
    assert out_bytes == [12 << 3]

    out_cells_fail = [901]
    out_bytes_fail = [902]
    err = fpq16_mul_checked_no_partial_array_commit_only(
        lhs,
        rhs,
        out,
        12,
        12,
        out_cells_fail,
        out_bytes_fail,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells_fail == [901]
    assert out_bytes_fail == [902]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260421_968)

    for _ in range(2500):
        count = rng.randint(0, 64)
        capacity = count + rng.randint(0, 6)
        lhs = [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(max(count, 1))]
        rhs = [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(max(count, 1))]

        got_out = [0xABCD] * max(count, 1)
        exp_out = got_out.copy()

        got_cells = [0xAAAA]
        got_bytes = [0xBBBB]
        exp_cells = [0xCCCC]
        exp_bytes = [0xDDDD]

        got = fpq16_mul_checked_no_partial_array_commit_only(
            lhs,
            rhs,
            got_out,
            count,
            capacity,
            got_cells,
            got_bytes,
        )
        exp = explicit_commit_only_composition(
            lhs,
            rhs,
            exp_out,
            count,
            capacity,
            exp_cells,
            exp_bytes,
        )

        assert got == exp
        if got == FP_Q16_OK:
            assert got_out == exp_out
            assert got_cells == exp_cells
            assert got_bytes == exp_bytes
        else:
            assert got_cells == [0xAAAA]
            assert got_bytes == [0xBBBB]


def run() -> None:
    test_source_contains_iq968_function()
    test_null_bad_and_no_publish()
    test_success_publish_and_mutation_guard()
    test_randomized_parity_against_explicit_composition()
    print("fixedpoint_q16_mul_checked_no_partial_array_commit_only=ok")


if __name__ == "__main__":
    run()
