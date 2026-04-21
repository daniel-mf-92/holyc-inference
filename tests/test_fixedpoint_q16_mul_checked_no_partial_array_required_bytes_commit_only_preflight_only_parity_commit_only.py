#!/usr/bin/env python3
"""Harness for FPQ16...RequiredBytesCommitOnlyPreflightOnlyParityCommitOnly (IQ-977)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_checked_no_partial_array_required_bytes import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_OK,
)
from test_fixedpoint_q16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity import (
    fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity,
)


def fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
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

    if count < 0:
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

    staged_required_cells = [0]
    staged_required_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity(
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
        count += 1

    if (
        snapshot_lhs_q16 is not lhs_q16
        or snapshot_rhs_q16 is not rhs_q16
        or snapshot_out_q16 is not out_q16
        or snapshot_count != count
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


def explicit_composition(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
) -> tuple[int, tuple[int, int]]:
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, (0, 0)

    staged_cells = [0]
    staged_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    return FP_Q16_OK, (staged_cells[0], staged_bytes[0])


def test_source_contains_iq977_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatCheckedNoPartialArray", 1)[0]

    assert "status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParity(" in body
    assert "if (snapshot_lhs_q16 != lhs_q16)" in body
    assert "if (snapshot_rhs_q16 != rhs_q16)" in body
    assert "if (snapshot_out_q16 != out_q16)" in body
    assert "if (snapshot_count != count)" in body
    assert "*out_required_cells = staged_required_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_null_alias_and_no_publish_on_fail() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0, 0]

    out_cells = [111]
    out_bytes = [222]

    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
        None,
        rhs,
        out,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        -1,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        2,
        out_cells,
        out_cells,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]


def test_publish_and_snapshot_mutation_guard() -> None:
    lhs = [5 << 16] * 12
    rhs = [7 << 16] * 12
    out = [0x44] * 12

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        12,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK
    assert out_cells == [12]
    assert out_bytes == [96]

    fail_cells = [701]
    fail_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        12,
        fail_cells,
        fail_bytes,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert fail_cells == [701]
    assert fail_bytes == [702]


def test_randomized_vs_explicit_composition() -> None:
    rng = random.Random(20260421_977)

    for _ in range(4000):
        count = rng.randint(0, 64)
        lhs = [rng.randint(-(1 << 31), (1 << 31) - 1) << 16 for _ in range(count)]
        rhs = [rng.randint(-(1 << 31), (1 << 31) - 1) << 16 for _ in range(count)]
        out_a = [0xA5A5] * count
        out_b = out_a.copy()

        got_cells = [13]
        got_bytes = [17]

        err_got = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
            lhs.copy(),
            rhs.copy(),
            out_a,
            count,
            got_cells,
            got_bytes,
        )
        err_exp, exp_tuple = explicit_composition(
            lhs.copy(),
            rhs.copy(),
            out_b,
            count,
        )

        assert err_got == err_exp
        if err_got == FP_Q16_OK:
            assert (got_cells[0], got_bytes[0]) == exp_tuple
        else:
            assert got_cells == [13]
            assert got_bytes == [17]


if __name__ == "__main__":
    test_source_contains_iq977_function()
    test_null_alias_and_no_publish_on_fail()
    test_publish_and_snapshot_mutation_guard()
    test_randomized_vs_explicit_composition()
    print("fixedpoint_q16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only=ok")
