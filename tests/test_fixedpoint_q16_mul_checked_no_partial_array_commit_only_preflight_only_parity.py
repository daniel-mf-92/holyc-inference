#!/usr/bin/env python3
"""Parity harness for FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParity (IQ-978)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_checked_no_partial_array import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_OK,
)
from test_fixedpoint_q16_mul_checked_no_partial_array_commit_only import (  # noqa: E402
    fpq16_mul_checked_no_partial_array_commit_only,
)
from test_fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only import (  # noqa: E402
    fpq16_mul_checked_no_partial_array_commit_only_preflight_only,
)


def fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
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

    snapshot = (lhs_q16, rhs_q16, out_q16, count, out_capacity)

    staged_preflight_cells = [0]
    staged_preflight_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        staged_preflight_cells,
        staged_preflight_bytes,
    )
    if status != FP_Q16_OK:
        return status

    staged_commit_cells = [0]
    staged_commit_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        staged_commit_cells,
        staged_commit_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        out_capacity += 1

    if snapshot != (lhs_q16, rhs_q16, out_q16, count, out_capacity):
        return FP_Q16_ERR_BAD_PARAM

    if staged_preflight_cells[0] != staged_commit_cells[0]:
        return FP_Q16_ERR_BAD_PARAM
    if staged_preflight_bytes[0] != staged_commit_bytes[0]:
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

    out_required_cells[0] = staged_preflight_cells[0]
    out_required_bytes[0] = staged_preflight_bytes[0]
    return FP_Q16_OK


def explicit_parity_composition(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_capacity: int,
) -> tuple[int, tuple[int, int]]:
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, (0, 0)

    preflight_cells = [0]
    preflight_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        preflight_cells,
        preflight_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    commit_cells = [0]
    commit_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        commit_cells,
        commit_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    if preflight_cells[0] != commit_cells[0] or preflight_bytes[0] != commit_bytes[0]:
        return FP_Q16_ERR_BAD_PARAM, (0, 0)

    return FP_Q16_OK, (preflight_cells[0], preflight_bytes[0])


def test_source_contains_iq978_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParity(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnly", 1)[0]

    assert "status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16MulCheckedNoPartialArrayCommitOnly(" in body
    assert "if (count < 0 || out_capacity < 0)" in body
    assert "if (count > out_capacity)" in body
    assert "snapshot_out_capacity = out_capacity;" in body
    assert "if (staged_from_preflight_cells != staged_from_commit_cells ||" in body
    assert "if (out_required_cells == out_required_bytes)" in body
    assert "*out_required_cells = staged_from_preflight_cells;" in body
    assert "*out_required_bytes = staged_from_preflight_bytes;" in body


def test_null_bad_param_and_no_publish() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0x11, 0x22]

    out_cells = [701]
    out_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
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

    out_cells = [703]
    out_bytes = [704]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        -1,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [703]
    assert out_bytes == [704]

    out_cells = [705]
    out_bytes = [706]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        3,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [705]
    assert out_bytes == [706]


def test_alias_rejection_and_snapshot_guard() -> None:
    lhs = [5 << 16] * 8
    rhs = [7 << 16] * 8
    out = [0x33] * 8

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        8,
        8,
        lhs,
        [0],
    )
    assert err == FP_Q16_ERR_BAD_PARAM

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        8,
        8,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK
    assert out_cells == [8]
    assert out_bytes == [64]

    fail_cells = [991]
    fail_bytes = [992]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        8,
        8,
        fail_cells,
        fail_bytes,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert fail_cells == [991]
    assert fail_bytes == [992]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(978)

    for _ in range(300):
        count = rng.randint(0, 128)
        out_capacity = count + rng.randint(0, 24)

        lhs = [rng.randint(-(1 << 20), 1 << 20) for _ in range(count)]
        rhs = [rng.randint(-(1 << 20), 1 << 20) for _ in range(count)]
        out = [0x5A5A for _ in range(out_capacity)]

        got_cells = [0]
        got_bytes = [0]
        got = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
            lhs,
            rhs,
            out,
            count,
            out_capacity,
            got_cells,
            got_bytes,
        )

        exp_status, (exp_cells, exp_bytes) = explicit_parity_composition(
            lhs,
            rhs,
            out,
            count,
            out_capacity,
        )

        assert got == exp_status
        if got == FP_Q16_OK:
            assert got_cells == [exp_cells]
            assert got_bytes == [exp_bytes]


def test_zero_count_writes_zero_tuple() -> None:
    lhs = [123 << 16]
    rhs = [456 << 16]
    out = [0x22]
    out_cells = [111]
    out_bytes = [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        0,
        0,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK
    assert out_cells == [0]
    assert out_bytes == [0]
