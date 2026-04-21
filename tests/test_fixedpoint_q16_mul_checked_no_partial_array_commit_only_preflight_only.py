#!/usr/bin/env python3
"""Harness for FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly (IQ-973)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_checked_no_partial_array import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
)
from test_fixedpoint_q16_mul_checked_no_partial_array_required_bytes_commit_only import (
    fpq16_mul_checked_no_partial_array_required_bytes_commit_only,
)


def fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_capacity: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
    *,
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
    if count > out_capacity:
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

    snapshot_lhs_q16 = lhs_q16
    snapshot_rhs_q16 = rhs_q16
    snapshot_out_q16 = out_q16
    snapshot_count = count
    snapshot_out_capacity = out_capacity

    staged_required_cells = [0]
    staged_required_bytes = [0]

    status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        staged_required_cells,
        staged_required_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        out_capacity += 1

    if (
        snapshot_lhs_q16 is not lhs_q16
        or snapshot_rhs_q16 is not rhs_q16
        or snapshot_out_q16 is not out_q16
        or snapshot_count != count
        or snapshot_out_capacity != out_capacity
    ):
        return FP_Q16_ERR_BAD_PARAM

    if snapshot_count > snapshot_out_capacity:
        return FP_Q16_ERR_BAD_PARAM

    recomputed_required_cells = snapshot_count
    if snapshot_count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    recomputed_required_bytes = snapshot_count << 3

    if (
        staged_required_cells[0] != recomputed_required_cells
        or staged_required_bytes[0] != recomputed_required_bytes
    ):
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

    out_required_cells[0] = staged_required_cells[0]
    out_required_bytes[0] = staged_required_bytes[0]
    return FP_Q16_OK


def explicit_preflight_only_composition(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_capacity: int,
    out_required_cells: list[int] | None,
    out_required_bytes: list[int] | None,
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
    if count > out_capacity:
        return FP_Q16_ERR_BAD_PARAM

    staged_cells = [0]
    staged_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    if staged_cells[0] != count or staged_bytes[0] != (count << 3):
        return FP_Q16_ERR_BAD_PARAM

    out_required_cells[0] = staged_cells[0]
    out_required_bytes[0] = staged_bytes[0]
    return FP_Q16_OK


def test_source_contains_iq973_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatCheckedNoPartialArray", 1)[0]

    assert "if (count < 0 || out_capacity < 0)" in body
    assert "if (count > out_capacity)" in body
    assert "snapshot_out_capacity = out_capacity;" in body
    assert "status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly(" in body
    assert "if (snapshot_out_capacity != out_capacity)" in body
    assert "if (snapshot_count > snapshot_out_capacity)" in body
    assert "*out_required_cells = staged_required_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_null_bad_param_and_no_publish_on_fail() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0x11, 0x22]

    out_cells = [701]
    out_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        None,
        rhs,
        out,
        2,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cells == [701]
    assert out_bytes == [702]

    out_cells = [701]
    out_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        -1,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [701]
    assert out_bytes == [702]

    out_cells = [701]
    out_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        2,
        1,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [701]
    assert out_bytes == [702]


def test_publish_and_mutation_guard() -> None:
    lhs = [5 << 16] * 10
    rhs = [7 << 16] * 10
    out = [0x44] * 10

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        10,
        10,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK
    assert out_cells == [10]
    assert out_bytes == [80]
    assert out == [0x44] * 10

    out_cells_fail = [801]
    out_bytes_fail = [802]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        10,
        10,
        out_cells_fail,
        out_bytes_fail,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells_fail == [801]
    assert out_bytes_fail == [802]


def test_randomized_matches_explicit_composition() -> None:
    rng = random.Random(20260421_973)

    for _ in range(4000):
        count = rng.randint(0, 48)
        out_capacity = count + rng.randint(0, 4)

        lhs = [rng.randint(-(1 << 31), (1 << 31) - 1) << 16 for _ in range(count)]
        rhs = [rng.randint(-(1 << 31), (1 << 31) - 1) << 16 for _ in range(count)]

        out_a = [0x55AA55AA] * count
        out_b = out_a.copy()

        got_cells = [13]
        got_bytes = [17]
        exp_cells = [13]
        exp_bytes = [17]

        err_got = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
            lhs.copy(),
            rhs.copy(),
            out_a,
            count,
            out_capacity,
            got_cells,
            got_bytes,
        )
        err_exp = explicit_preflight_only_composition(
            lhs.copy(),
            rhs.copy(),
            out_b,
            count,
            out_capacity,
            exp_cells,
            exp_bytes,
        )

        assert err_got == err_exp
        if err_got == FP_Q16_OK:
            assert got_cells == exp_cells
            assert got_bytes == exp_bytes
            assert out_a == out_b


def run() -> None:
    test_source_contains_iq973_function()
    test_null_bad_param_and_no_publish_on_fail()
    test_publish_and_mutation_guard()
    test_randomized_matches_explicit_composition()
    print("fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
