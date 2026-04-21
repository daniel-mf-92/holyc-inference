#!/usr/bin/env python3
"""Parity harness for FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnly (IQ-896)."""

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
    fpq16_mul_sat_checked,
)


def fpq16_mac_sat_checked_no_partial_array(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
) -> int:
    if acc_q16 is None or a_q16 is None or b_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if (
        acc_q16 is a_q16
        or acc_q16 is b_q16
        or acc_q16 is out_q16
        or a_q16 is b_q16
        or a_q16 is out_q16
        or b_q16 is out_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK

    # Preflight all lanes first.
    for i in range(count):
        err, _ = fpq16_mul_sat_checked(a_q16[i], b_q16[i], out_present=True)
        if err not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
            return err

    overflow_seen = False
    for i in range(count):
        err, mul_q16 = fpq16_mul_sat_checked(a_q16[i], b_q16[i], out_present=True)
        lane = acc_q16[i] + mul_q16

        if mul_q16 > 0 and acc_q16[i] > ((1 << 63) - 1) - mul_q16:
            lane = (1 << 63) - 1
            overflow_seen = True
        elif mul_q16 < 0 and acc_q16[i] < (-(1 << 63)) - mul_q16:
            lane = -(1 << 63)
            overflow_seen = True
        elif err == FP_Q16_ERR_OVERFLOW:
            overflow_seen = True

        out_q16[i] = lane

    if overflow_seen:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def fpq16_mac_sat_checked_no_partial_array_required_bytes(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes: list[int] | None,
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

    if (
        out_required_output_bytes is acc_q16
        or out_required_output_bytes is a_q16
        or out_required_output_bytes is b_q16
        or out_required_output_bytes is out_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    status = fpq16_mac_sat_checked_no_partial_array(acc_q16, a_q16, b_q16, out_q16, count)
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status

    out_required_output_bytes[0] = count << 3
    if status == FP_Q16_ERR_OVERFLOW:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes: list[int] | None,
) -> int:
    if out_required_output_bytes is None:
        return FP_Q16_ERR_NULL_PTR

    snapshot_count = count
    snapshot_acc_q16 = acc_q16
    snapshot_a_q16 = a_q16
    snapshot_b_q16 = b_q16
    snapshot_out_q16 = out_q16

    staged_required_output_bytes = [0]
    status = fpq16_mac_sat_checked_no_partial_array_required_bytes(
        acc_q16,
        a_q16,
        b_q16,
        out_q16,
        count,
        staged_required_output_bytes,
    )
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status

    if snapshot_count != count:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_acc_q16 is not acc_q16:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_a_q16 is not a_q16:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_b_q16 is not b_q16:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_out_q16 is not out_q16:
        return FP_Q16_ERR_BAD_PARAM

    out_required_output_bytes[0] = staged_required_output_bytes[0]
    if status == FP_Q16_ERR_OVERFLOW:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def explicit_commit_only_composition(
    acc_q16: list[int] | None,
    a_q16: list[int] | None,
    b_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes: list[int] | None,
) -> int:
    if out_required_output_bytes is None:
        return FP_Q16_ERR_NULL_PTR

    staged_required_output_bytes = [0]
    status = fpq16_mac_sat_checked_no_partial_array_required_bytes(
        acc_q16,
        a_q16,
        b_q16,
        out_q16,
        count,
        staged_required_output_bytes,
    )
    if status not in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW):
        return status

    out_required_output_bytes[0] = staged_required_output_bytes[0]
    if status == FP_Q16_ERR_OVERFLOW:
        return FP_Q16_ERR_OVERFLOW
    return FP_Q16_OK


def test_source_contains_required_bytes_commit_only_wrapper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MacSatCheckedNoPartialArrayRequiredBytesCommitOnly(I64 *acc_q16,"
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "FPQ16MacSatCheckedNoPartialArrayRequiredBytes(acc_q16," in body
    assert "snapshot_count = count;" in body
    assert "snapshot_acc_q16 = acc_q16;" in body
    assert "snapshot_a_q16 = a_q16;" in body
    assert "snapshot_b_q16 = b_q16;" in body
    assert "snapshot_out_q16 = out_q16;" in body
    assert "if (snapshot_count != count)" in body
    assert "if (snapshot_acc_q16 != acc_q16)" in body
    assert "if (snapshot_a_q16 != a_q16)" in body
    assert "if (snapshot_b_q16 != b_q16)" in body
    assert "if (snapshot_out_q16 != out_q16)" in body
    assert "*out_required_output_bytes = staged_required_output_bytes;" in body


def test_known_vector_commit_only_parity() -> None:
    acc = [1 << 16, -(2 << 16), 3 << 16, -(4 << 16)]
    a = [2 << 16, 3 << 16, -(5 << 16), 6 << 16]
    b = [4 << 16, -(1 << 16), 7 << 16, 8 << 16]

    out_wrapped = [0x55AA] * len(acc)
    out_explicit = [0x55AA] * len(acc)
    bytes_wrapped = [0]
    bytes_explicit = [0]

    wrapped_status = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
        acc.copy(), a.copy(), b.copy(), out_wrapped, len(acc), bytes_wrapped
    )
    explicit_status = explicit_commit_only_composition(
        acc.copy(), a.copy(), b.copy(), out_explicit, len(acc), bytes_explicit
    )

    assert wrapped_status == explicit_status
    assert out_wrapped == out_explicit
    assert bytes_wrapped == bytes_explicit


def test_null_and_alias_contracts_no_partial() -> None:
    acc = [1 << 16, 2 << 16]
    a = [3 << 16, 4 << 16]
    b = [5 << 16, 6 << 16]
    out = [0xDEAD, 0xBEEF]
    out_before = out.copy()

    assert (
        fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
            acc,
            a,
            b,
            out,
            2,
            None,
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert out == out_before

    # Commit-only wrapper does not reject out_required_output_bytes aliasing input
    # arrays because it stages required bytes internally before publish.
    shared = [1 << 16]
    out_shared = [0xAAAA]
    wrapped_status = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
        shared,
        a,
        b,
        out_shared,
        1,
        shared,
    )
    explicit_status = explicit_commit_only_composition(
        shared,
        a,
        b,
        [0xAAAA],
        1,
        [0],
    )
    assert wrapped_status == explicit_status


def test_randomized_explicit_composition_parity() -> None:
    rng = random.Random(20260421_896)

    for _ in range(4000):
        count = rng.randint(0, 32)
        acc = [rng.randint(-(1 << 62), (1 << 62) - 1) for _ in range(max(1, count))]
        a = [rng.randint(-(1 << 62), (1 << 62) - 1) for _ in range(max(1, count))]
        b = [rng.randint(-(1 << 62), (1 << 62) - 1) for _ in range(max(1, count))]

        if count == 0:
            acc = []
            a = []
            b = []

        out_wrapped = [0x1111] * max(1, count)
        out_explicit = out_wrapped.copy()
        bytes_wrapped = [0x2222]
        bytes_explicit = [0x3333]

        wrapped_status = fpq16_mac_sat_checked_no_partial_array_required_bytes_commit_only(
            acc,
            a,
            b,
            out_wrapped,
            count,
            bytes_wrapped,
        )
        explicit_status = explicit_commit_only_composition(
            acc,
            a,
            b,
            out_explicit,
            count,
            bytes_explicit,
        )

        assert wrapped_status == explicit_status
        assert out_wrapped == out_explicit
        assert bytes_wrapped == bytes_explicit


def run() -> None:
    test_source_contains_required_bytes_commit_only_wrapper()
    test_known_vector_commit_only_parity()
    test_null_and_alias_contracts_no_partial()
    test_randomized_explicit_composition_parity()
    print("fixedpoint_q16_mac_sat_checked_no_partial_array_required_bytes_commit_only=ok")


if __name__ == "__main__":
    run()
