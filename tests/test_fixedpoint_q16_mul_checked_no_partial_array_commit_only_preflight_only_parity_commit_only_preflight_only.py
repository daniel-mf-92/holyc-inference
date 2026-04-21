#!/usr/bin/env python3
"""Parity harness for FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-978)."""

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


def fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
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
        return status

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
        return status

    if mutate_snapshot:
        out_capacity += 1

    if snapshot != (lhs_q16, rhs_q16, out_q16, count, out_capacity):
        return FP_Q16_ERR_BAD_PARAM

    if preflight_cells[0] != commit_cells[0] or preflight_bytes[0] != commit_bytes[0]:
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

    out_required_cells[0] = preflight_cells[0]
    out_required_bytes[0] = preflight_bytes[0]
    return FP_Q16_OK


def explicit_parity_composition(
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
        return status

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
        return status

    if preflight_cells[0] != commit_cells[0] or preflight_bytes[0] != commit_bytes[0]:
        return FP_Q16_ERR_BAD_PARAM

    out_required_cells[0] = preflight_cells[0]
    out_required_bytes[0] = preflight_bytes[0]
    return FP_Q16_OK


def test_source_contains_iq978_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly", 1
    )[0]

    assert "status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16MulCheckedNoPartialArrayCommitOnly(" in body
    assert "if (snapshot_lhs_q16 != lhs_q16 ||" in body
    assert "snapshot_out_capacity != out_capacity" in body
    assert "if (staged_from_preflight_cells != staged_from_commit_cells ||" in body
    assert "if (count < 0 || out_capacity < 0)" in body
    assert "if (count > out_capacity)" in body
    assert "FPAddressRangesOverlap(req_cells_base, req_cells_end, lhs_base, lhs_end)" in body
    assert "*out_required_cells = staged_from_preflight_cells;" in body
    assert "*out_required_bytes = staged_from_preflight_bytes;" in body


def test_null_bad_param_and_no_publish_on_fail() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0, 0]

    out_cells = [777]
    out_bytes = [888]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        None,
        rhs,
        out,
        2,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cells == [777]
    assert out_bytes == [888]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        -1,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [777]
    assert out_bytes == [888]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        2,
        1,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [777]
    assert out_bytes == [888]


def test_publish_and_snapshot_mutation_guard() -> None:
    lhs = [5 << 16] * 12
    rhs = [7 << 16] * 12
    out = [0] * 12

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    assert out_bytes == [96]

    fail_cells = [111]
    fail_bytes = [222]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        12,
        12,
        fail_cells,
        fail_bytes,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert fail_cells == [111]
    assert fail_bytes == [222]


def test_output_alias_rejected_without_publish() -> None:
    lhs = [1 << 16, 2 << 16, 3 << 16]
    rhs = [4 << 16, 5 << 16, 6 << 16]
    out = [0, 0, 0]

    out_bytes = [123]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs,
        rhs,
        out,
        3,
        3,
        out_bytes,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_bytes == [123]


def test_randomized_vs_explicit_composition() -> None:
    rng = random.Random(20260421_974)

    for _ in range(4000):
        count = rng.randint(0, 64)
        out_capacity = count + rng.randint(0, 6)
        lhs = [rng.randint(-(1 << 23), (1 << 23)) for _ in range(max(count, 1))]
        rhs = [rng.randint(-(1 << 23), (1 << 23)) for _ in range(max(count, 1))]
        out = [0] * max(count, 1)

        out_cells_actual = [0x55AA]
        out_bytes_actual = [0xAA55]
        status_actual = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
            lhs,
            rhs,
            out,
            count,
            out_capacity,
            out_cells_actual,
            out_bytes_actual,
        )

        out_copy = list(out)
        out_cells_expected = [0x55AA]
        out_bytes_expected = [0xAA55]
        status_expected = explicit_parity_composition(
            lhs,
            rhs,
            out_copy,
            count,
            out_capacity,
            out_cells_expected,
            out_bytes_expected,
        )

        assert status_actual == status_expected
        assert out == out_copy
        assert out_cells_actual == out_cells_expected
        assert out_bytes_actual == out_bytes_expected
