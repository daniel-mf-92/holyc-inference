#!/usr/bin/env python3
"""Harness for IQ-983 commit-only wrapper over IQ-982 parity preflight."""

from __future__ import annotations

import importlib.util
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

_PREV_PATH = Path(
    "tests/test_fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity.py"
)

_spec_prev = importlib.util.spec_from_file_location("fpq16_iq982_prev", _PREV_PATH)
assert _spec_prev and _spec_prev.loader
_prev = importlib.util.module_from_spec(_spec_prev)
sys.modules[_spec_prev.name] = _prev
_spec_prev.loader.exec_module(_prev)

preflight_only_parity = (
    _prev.fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity
)


def try_mul_i64_checked(a: int, b: int) -> tuple[bool, int]:
    product = a * b
    if product < -(1 << 63) or product > ((1 << 63) - 1):
        return False, 0
    return True, product


def fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    if count:
        if lhs_q16 is rhs_q16 or lhs_q16 is out_q16 or rhs_q16 is out_q16:
            return FP_Q16_ERR_BAD_PARAM

    snapshot_lhs_q16 = lhs_q16
    snapshot_rhs_q16 = rhs_q16
    snapshot_out_q16 = out_q16
    snapshot_count = count
    snapshot_out_capacity = out_capacity

    staged_required_cells = [0]
    staged_required_bytes = [0]
    status = preflight_only_parity(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
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

    canonical_required_cells = snapshot_count
    ok, canonical_required_bytes_checked = try_mul_i64_checked(snapshot_count, 8)
    if not ok:
        return FP_Q16_ERR_OVERFLOW

    if snapshot_count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    canonical_required_bytes = snapshot_count << 3

    if canonical_required_bytes != canonical_required_bytes_checked:
        return FP_Q16_ERR_BAD_PARAM

    if (
        staged_required_cells[0] != canonical_required_cells
        or staged_required_bytes[0] != canonical_required_bytes
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

    staged_required_cells = [0]
    staged_required_bytes = [0]
    status = preflight_only_parity(
        lhs_q16,
        rhs_q16,
        out_q16,
        count,
        out_capacity,
        staged_required_cells,
        staged_required_bytes,
    )
    if status != FP_Q16_OK:
        return status, (0, 0)

    ok, canonical_required_bytes_checked = try_mul_i64_checked(count, 8)
    if not ok:
        return FP_Q16_ERR_OVERFLOW, (0, 0)

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, (0, 0)
    canonical_required_bytes = count << 3

    if canonical_required_bytes != canonical_required_bytes_checked:
        return FP_Q16_ERR_BAD_PARAM, (0, 0)

    if (
        staged_required_cells[0] != count
        or staged_required_bytes[0] != canonical_required_bytes
    ):
        return FP_Q16_ERR_BAD_PARAM, (0, 0)

    return FP_Q16_OK, (staged_required_cells[0], staged_required_bytes[0])


def test_source_contains_iq983_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 FPQ16MulCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly", 1
    )[0]

    assert (
        "status = FPQ16MulCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
        in body
    )
    assert "if (snapshot_lhs_q16 != lhs_q16 ||" in body
    assert "snapshot_out_capacity != out_capacity" in body
    assert "status = FPTryMulI64Checked(snapshot_count," in body
    assert "if (FPAddressRangesOverlap(req_cells_base, req_cells_end, req_bytes_base, req_bytes_end))" in body
    assert "if (staged_required_cells != canonical_required_cells ||" in body
    assert "*out_required_cells = staged_required_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_null_bad_param_overflow_and_no_publish_on_fail() -> None:
    lhs = [1 << 16, 2 << 16]
    rhs = [3 << 16, 4 << 16]
    out = [0, 0]

    out_cells = [111]
    out_bytes = [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        None,
        rhs,
        out,
        2,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        -1,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        3,
        2,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]
    assert out_bytes == [222]

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        2,
        2,
        out_cells,
        out_cells,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [111]


def test_publish_and_snapshot_mutation_guard() -> None:
    lhs = [5 << 16] * 10
    rhs = [7 << 16] * 10
    out = [0x44] * 10

    out_cells = [0]
    out_bytes = [0]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
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

    fail_cells = [701]
    fail_bytes = [702]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        10,
        10,
        fail_cells,
        fail_bytes,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert fail_cells == [701]
    assert fail_bytes == [702]


def test_alias_guards_and_randomized_parity() -> None:
    lhs = [1 << 16, 2 << 16, 3 << 16]
    rhs = [4 << 16, 5 << 16, 6 << 16]
    out = [0, 0, 0]

    out_cells = [88]
    out_bytes = [99]
    err = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        lhs,
        rhs,
        out,
        3,
        3,
        lhs,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_cells == [88]
    assert out_bytes == [99]

    rng = random.Random(20260422_983)
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
        got_status = fpq16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
            lhs,
            rhs,
            out,
            count,
            out_capacity,
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


if __name__ == "__main__":
    test_source_contains_iq983_function()
    test_null_bad_param_overflow_and_no_publish_on_fail()
    test_publish_and_snapshot_mutation_guard()
    test_alias_guards_and_randomized_parity()
    print(
        "fixedpoint_q16_mul_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only=ok"
    )
