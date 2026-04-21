#!/usr/bin/env python3
"""Harness for FPQ16...RequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-979)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_checked_no_partial_array_required_bytes import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_OK,
    fpq16_mul_checked_no_partial_array_required_bytes,
)
from test_fixedpoint_q16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only import (
    fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only,
)


def fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    snapshot = (lhs_q16, rhs_q16, out_q16, count)

    staged_required_cells = [0]
    staged_required_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
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

    if snapshot != (lhs_q16, rhs_q16, out_q16, count):
        return FP_Q16_ERR_BAD_PARAM

    status, canonical_required_cells, canonical_required_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        True,
        True,
    )
    if status != FP_Q16_OK:
        return status

    if staged_required_cells[0] != canonical_required_cells:
        return FP_Q16_ERR_BAD_PARAM
    if staged_required_bytes[0] != canonical_required_bytes:
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
    status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    status, canonical_cells, canonical_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        True,
        True,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    if staged_cells[0] != canonical_cells or staged_bytes[0] != canonical_bytes:
        return FP_Q16_ERR_BAD_PARAM, (0, 0)

    return FP_Q16_OK, (staged_cells[0], staged_bytes[0])


def test_source_contains_iq979_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatCheckedNoPartialArray", 1)[0]

    assert "status = FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "status = FPTryMulI64Checked(snapshot_count," in body
    assert "if (recomputed_required_bytes_checked != recomputed_required_bytes)" in body
    assert "if (staged_required_cells != recomputed_required_cells ||" in body
    assert "if (snapshot_lhs_q16 != lhs_q16 ||" in body
    assert "*out_required_cells = staged_required_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_null_bad_param_and_no_publish_on_fail() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0, 0]

    out_cells = [111]
    out_bytes = [222]

    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    lhs = [5 << 16] * 9
    rhs = [7 << 16] * 9
    out = [0x44] * 9

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        9,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK
    assert out_cells == [9]
    assert out_bytes == [72]

    fail_cells = [701]
    fail_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        9,
        fail_cells,
        fail_bytes,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert fail_cells == [701]
    assert fail_bytes == [702]


def test_randomized_parity_with_explicit_composition() -> None:
    rng = random.Random(20260421_979)

    for _ in range(2000):
        count = rng.randint(0, 64)
        lhs = [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(count)]
        rhs = [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(count)]
        out = [0x55AA55AA55AA55AA] * count

        exp_status, exp_tuple = explicit_composition(lhs.copy(), rhs.copy(), out.copy(), count)

        out_cells = [0x1111]
        out_bytes = [0x2222]
        got_status = fpq16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only(
            lhs,
            rhs,
            out,
            count,
            out_cells,
            out_bytes,
        )

        assert got_status == exp_status
        if got_status == FP_Q16_OK:
            assert out_cells[0] == exp_tuple[0]
            assert out_bytes[0] == exp_tuple[1]
        else:
            assert out_cells == [0x1111]
            assert out_bytes == [0x2222]


def run() -> None:
    test_source_contains_iq979_function()
    test_null_bad_param_and_no_publish_on_fail()
    test_publish_and_snapshot_mutation_guard()
    test_randomized_parity_with_explicit_composition()
    print("fixedpoint_q16_mul_checked_no_partial_array_required_bytes_commit_only_preflight_only_parity_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
