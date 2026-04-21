#!/usr/bin/env python3
"""Parity harness for FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly (IQ-985)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_sat_checked_no_partial_array import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
)
from test_fixedpoint_q16_mac_sat_checked_no_partial_array_required_bytes_commit_only import (
    fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only,
)

I64_MAX_VALUE = (1 << 63) - 1


def try_mul_i64_checked(a: int, b: int) -> tuple[bool, int]:
    product = a * b
    if product < -(1 << 63) or product > ((1 << 63) - 1):
        return False, 0
    return True, product


def fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes: list[int] | None,
    *,
    mutate_snapshot: bool = False,
) -> int:
    if (
        acc_q16 is None
        or a_q16 is None
        or b_q16 is None
        or out_q16 is None
        or out_required_output_bytes is None
    ):
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM
    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    if (
        out_required_output_bytes is acc_q16
        or out_required_output_bytes is a_q16
        or out_required_output_bytes is b_q16
        or out_required_output_bytes is out_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    snapshot_acc_q16 = acc_q16
    snapshot_a_q16 = a_q16
    snapshot_b_q16 = b_q16
    snapshot_out_q16 = out_q16
    snapshot_count = count
    snapshot_out_cells = out_q16.copy()

    scratch_out_q16 = [0] * max(count, 1)

    staged_required_output_bytes = [0]
    status = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
        acc_q16,
        a_q16,
        b_q16,
        scratch_out_q16,
        count,
        staged_required_output_bytes,
    )
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status

    if mutate_snapshot:
        count += 1

    if (
        snapshot_acc_q16 is not acc_q16
        or snapshot_a_q16 is not a_q16
        or snapshot_b_q16 is not b_q16
        or snapshot_out_q16 is not out_q16
        or snapshot_count != count
    ):
        return FP_Q16_ERR_BAD_PARAM

    if out_q16 != snapshot_out_cells:
        return FP_Q16_ERR_BAD_PARAM

    ok, recomputed_required_output_bytes_checked = try_mul_i64_checked(snapshot_count, 8)
    if not ok:
        return FP_Q16_ERR_OVERFLOW

    if snapshot_count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    recomputed_required_output_bytes = snapshot_count << 3

    if recomputed_required_output_bytes != recomputed_required_output_bytes_checked:
        return FP_Q16_ERR_BAD_PARAM

    if staged_required_output_bytes[0] != recomputed_required_output_bytes:
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_output_bytes is acc_q16
        or out_required_output_bytes is a_q16
        or out_required_output_bytes is b_q16
        or out_required_output_bytes is out_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    out_required_output_bytes[0] = staged_required_output_bytes[0]

    if status == FP_Q16_ERR_OVERFLOW:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def explicit_composition(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
) -> tuple[int, int]:
    if out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, 0

    scratch_out_q16 = [0] * max(count, 1)
    staged_required_output_bytes = [0]
    status = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
        acc_q16,
        a_q16,
        b_q16,
        scratch_out_q16,
        count,
        staged_required_output_bytes,
    )
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status, 0

    ok, recomputed_required_output_bytes_checked = try_mul_i64_checked(count, 8)
    if not ok:
        return FP_Q16_ERR_OVERFLOW, 0

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0
    recomputed_required_output_bytes = count << 3

    if recomputed_required_output_bytes != recomputed_required_output_bytes_checked:
        return FP_Q16_ERR_BAD_PARAM, 0

    if staged_required_output_bytes[0] != recomputed_required_output_bytes:
        return FP_Q16_ERR_BAD_PARAM, 0

    if status == FP_Q16_ERR_OVERFLOW:
        return FP_Q16_ERR_OVERFLOW, staged_required_output_bytes[0]
    return FP_Q16_OK, staged_required_output_bytes[0]


def test_source_contains_required_bytes_commit_only_preflight_only_wrapper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly(I64 *acc_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "// Checked elementwise Q16 divide helper with no-partial-write contract.", 1
    )[0]

    assert "scratch_out_q16 = MAlloc(scratch_bytes);" in body
    assert "status = FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnly(acc_q16," in body
    assert "scratch_out_q16," in body
    assert "Free(scratch_out_q16);" in body
    assert "snapshot_acc_q16 = acc_q16;" in body
    assert "snapshot_a_q16 = a_q16;" in body
    assert "snapshot_b_q16 = b_q16;" in body
    assert "snapshot_out_q16 = out_q16;" in body
    assert "snapshot_count = count;" in body
    assert "status = FPTryMulI64Checked(staged_required_cells," in body
    assert "if (staged_required_output_bytes != recomputed_required_output_bytes)" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_null_bad_param_overflow_and_no_publish_on_fail() -> None:
    acc = [1 << 16, 2 << 16]
    a = [3 << 16, 4 << 16]
    b = [5 << 16, 6 << 16]
    out = [0xAAAA, 0xBBBB]
    out_required_output_bytes = [0x7777]

    err = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
        None, a, b, out, 2, out_required_output_bytes
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert out_required_output_bytes == [0x7777]

    err = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
        acc, a, b, out, -1, out_required_output_bytes
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_required_output_bytes == [0x7777]

    err = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
        acc, a, b, out, 2, acc
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_required_output_bytes == [0x7777]

    err = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
        acc,
        a,
        b,
        out,
        (I64_MAX_VALUE >> 3) + 1,
        out_required_output_bytes,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert out_required_output_bytes == [0x7777]


def test_publish_and_snapshot_mutation_guard() -> None:
    count = 9
    acc = [1 << 16] * count
    a = [2 << 16] * count
    b = [3 << 16] * count
    out = [0x44] * count

    out_required_output_bytes = [0]
    err = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
        acc,
        a,
        b,
        out,
        count,
        out_required_output_bytes,
    )
    assert err == FP_Q16_OK
    assert out_required_output_bytes == [count << 3]
    assert out == [0x44] * count

    fail_out_required_output_bytes = [0x2222]
    err = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
        acc,
        a,
        b,
        out,
        count,
        fail_out_required_output_bytes,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert fail_out_required_output_bytes == [0x2222]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260422_985)

    for _ in range(2500):
        count = rng.randint(0, 32)
        acc = [rng.randint(-(1 << 62), (1 << 62) - 1) for _ in range(count)]
        a = [rng.randint(-(1 << 62), (1 << 62) - 1) for _ in range(count)]
        b = [rng.randint(-(1 << 62), (1 << 62) - 1) for _ in range(count)]
        out_wrapped = [0x55AA55AA55AA55AA] * count
        out_explicit = [0x55AA55AA55AA55AA] * count
        out_wrapped_before = out_wrapped.copy()
        out_explicit_before = out_explicit.copy()

        exp_status, exp_required_output_bytes = explicit_composition(
            acc.copy(), a.copy(), b.copy(), out_explicit, count
        )

        got_required_output_bytes = [0x9999]
        got_status = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only(
            acc,
            a,
            b,
            out_wrapped,
            count,
            got_required_output_bytes,
        )

        assert got_status == exp_status
        if got_status in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
            assert got_required_output_bytes[0] == exp_required_output_bytes
        else:
            assert got_required_output_bytes == [0x9999]

        assert out_wrapped == out_wrapped_before
        assert out_explicit == out_explicit_before


if __name__ == "__main__":
    test_source_contains_required_bytes_commit_only_preflight_only_wrapper()
    test_null_bad_param_overflow_and_no_publish_on_fail()
    test_publish_and_snapshot_mutation_guard()
    test_randomized_parity_against_explicit_composition()
    print("fixedpoint_q16_mac_sat_checked_no_partial_array_required_bytes_commit_only_preflight_only=ok")
