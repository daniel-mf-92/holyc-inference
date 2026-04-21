#!/usr/bin/env python3
"""Parity harness for FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity (IQ-982)."""

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
from test_fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only import (
    fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only,
)
from test_fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only import (
    fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only,
)


def fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

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

    from_preflight_cells = [0]
    from_preflight_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        from_preflight_cells,
        from_preflight_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if snapshot != (lhs_q16, rhs_q16, out_q16, count, out_capacity):
        return FP_Q16_ERR_BAD_PARAM

    from_commit_cells = [0]
    from_commit_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        from_commit_cells,
        from_commit_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if mutate_snapshot:
        out_capacity += 1

    if snapshot != (lhs_q16, rhs_q16, out_q16, count, out_capacity):
        return FP_Q16_ERR_BAD_PARAM

    if (
        from_preflight_cells[0] != from_commit_cells[0]
        or from_preflight_bytes[0] != from_commit_bytes[0]
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

    out_required_cells[0] = from_preflight_cells[0]
    out_required_bytes[0] = from_preflight_bytes[0]
    return FP_Q16_OK


def explicit_composition(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_capacity: int,
) -> tuple[int, tuple[int, int]]:
    if lhs_q16 is None or rhs_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, (0, 0)
    if count < 0 or out_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM, (0, 0)
    if count > out_capacity:
        return FP_Q16_ERR_BAD_PARAM, (0, 0)
    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, (0, 0)

    preflight_cells = [0]
    preflight_bytes = [0]
    status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
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


def test_source_contains_iq982_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly", 1
    )[0]

    assert "status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (snapshot_lhs_q16 != lhs_q16 ||" in body
    assert "snapshot_out_capacity != out_capacity" in body
    assert "if (count > (I64_MAX_VALUE >> 3))" in body
    assert "if (staged_from_preflight_cells != staged_from_commit_only_cells ||" in body
    assert "*out_required_cells = staged_from_preflight_cells;" in body
    assert "*out_required_bytes = staged_from_preflight_bytes;" in body


def test_null_bad_param_overflow_and_no_publish_on_fail() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0, 0]

    out_cells = [111]
    out_bytes = [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        None, rhs, out, 2, 2, out_cells, out_bytes
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs, rhs, out, -1, 2, out_cells, out_bytes
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs, rhs, out, 3, 2, out_cells, out_bytes
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        (I64_MAX_VALUE >> 3) + 1,
        (I64_MAX_VALUE >> 3) + 1,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs, rhs, out, 2, 2, out_cells, out_cells
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]


def test_publish_and_snapshot_mutation_guard() -> None:
    lhs = [5 << 16] * 12
    rhs = [7 << 16] * 12
    out = [0x44] * 12

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs, rhs, out, 12, 12, out_cells, out_bytes
    )
    assert err == FP_Q16_OK
    assert out_cells == [12]
    assert out_bytes == [96]

    fail_cells = [701]
    fail_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
    assert fail_cells == [701]
    assert fail_bytes == [702]


def test_adversarial_alias_and_output_immutability() -> None:
    lhs = [1 << 16, 2 << 16, 3 << 16, 4 << 16]
    rhs = [5 << 16, 6 << 16, 7 << 16, 8 << 16]
    out = [0x777] * 4

    out_cells = [901]
    out_bytes = [902]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        4,
        4,
        lhs,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [901]
    assert out_bytes == [902]
    assert out == [0x777] * 4

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        4,
        4,
        out_cells,
        out,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [901]
    assert out_bytes == [902]
    assert out == [0x777] * 4


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_982)

    for _ in range(3200):
        count = rng.randint(0, 64)
        out_capacity = count + rng.randint(0, 8)
        lhs = [rng.randint(-(1 << 23), (1 << 23) - 1) for _ in range(count)]
        rhs = [rng.randint(-(1 << 23), (1 << 23) - 1) for _ in range(count)]
        out = [0x55AA55AA55AA55AA] * count

        exp_status, exp_tuple = explicit_composition(
            lhs.copy(), rhs.copy(), out.copy(), count, out_capacity
        )

        out_cells = [0x1111]
        out_bytes = [0x2222]
        got_status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            lhs, rhs, out, count, out_capacity, out_cells, out_bytes
        )

        assert got_status == exp_status
        if got_status == FP_Q16_OK:
            assert out_cells[0] == exp_tuple[0]
            assert out_bytes[0] == exp_tuple[1]
        else:
            assert out_cells == [0x1111]
            assert out_bytes == [0x2222]


def test_boundary_count_max_shifted_overflow_guard() -> None:
    lhs = [1 << 16]
    rhs = [1 << 16]
    out = [0]
    out_cells = [1234]
    out_bytes = [5678]

    max_ok = I64_MAX_VALUE >> 3
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        max_ok,
        max_ok,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [1234]
    assert out_bytes == [5678]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        lhs,
        rhs,
        out,
        max_ok + 1,
        max_ok + 1,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_cells == [1234]
    assert out_bytes == [5678]


if __name__ == "__main__":
    test_source_contains_iq982_function()
    test_null_bad_param_overflow_and_no_publish_on_fail()
    test_publish_and_snapshot_mutation_guard()
    test_adversarial_alias_and_output_immutability()
    test_randomized_parity_vs_explicit_composition()
    test_boundary_count_max_shifted_overflow_guard()
    print("fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity=ok")
