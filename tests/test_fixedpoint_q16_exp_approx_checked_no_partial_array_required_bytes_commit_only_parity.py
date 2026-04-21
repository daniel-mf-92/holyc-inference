#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParity (IQ-912)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_exp_approx_checked_no_partial_array_required_bytes import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_OK,
    fpq16_exp_approx_checked_no_partial_array_required_bytes,
)
from test_fixedpoint_q16_exp_approx_checked_no_partial_array_required_bytes_commit_only import (
    fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only,
)


def fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_output_bytes_slot: list[int] | None,
) -> int:
    if x_q16 is None or out_q16 is None or out_required_output_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_q16:
        return FP_Q16_ERR_BAD_PARAM

    if out_required_output_bytes_slot is x_q16 or out_required_output_bytes_slot is out_q16:
        return FP_Q16_ERR_BAD_PARAM

    snapshot = (x_q16, out_q16, count)

    staged_commit_required_output_bytes = [0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(
        x_q16,
        out_q16,
        count,
        staged_commit_required_output_bytes,
    )
    if status != FP_Q16_OK:
        return status

    staged_canonical_required_output_bytes = [0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
        x_q16,
        out_q16,
        count,
        staged_canonical_required_output_bytes,
    )
    if status != FP_Q16_OK:
        return status

    if snapshot != (x_q16, out_q16, count):
        return FP_Q16_ERR_BAD_PARAM

    if staged_commit_required_output_bytes[0] != staged_canonical_required_output_bytes[0]:
        return FP_Q16_ERR_BAD_PARAM

    out_required_output_bytes_slot[0] = staged_commit_required_output_bytes[0]
    return FP_Q16_OK


def explicit_composition(
    x_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
) -> tuple[int, int]:
    if x_q16 is None or out_q16 is None:
        return FP_Q16_ERR_NULL_PTR, 0
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, 0
    if x_q16 is out_q16:
        return FP_Q16_ERR_BAD_PARAM, 0

    commit_required = [0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only(
        x_q16,
        out_q16,
        count,
        commit_required,
    )
    if status != FP_Q16_OK:
        return status, 0

    canonical_required = [0]
    status = fpq16_exp_approx_checked_no_partial_array_required_bytes(
        x_q16,
        out_q16,
        count,
        canonical_required,
    )
    if status != FP_Q16_OK:
        return status, 0

    if commit_required[0] != canonical_required[0]:
        return FP_Q16_ERR_BAD_PARAM, 0

    return FP_Q16_OK, commit_required[0]


def test_source_contains_iq912_function() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyParity(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnlyPreflightOnly", 1
    )[0]
    assert "FPQ16ExpApproxCheckedNoPartialArrayRequiredBytesCommitOnly(" in body
    assert "FPQ16ExpApproxCheckedNoPartialArrayRequiredBytes(" in body
    assert "if (snapshot_x_q16 != x_q16)" in body
    assert "if (staged_commit_required_output_bytes != staged_canonical_required_output_bytes)" in body
    assert "FPAddressRangesOverlap(required_base, required_end, x_base, x_end)" in body
    assert "FPAddressRangesOverlap(required_base, required_end, out_base, out_end)" in body
    assert "*out_required_output_bytes = staged_commit_required_output_bytes;" in body


def test_null_bad_param_guards() -> None:
    x = [0, 1]
    out = [0x1111, 0x2222]
    required = [0x3333]

    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            None, out, 1, required
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x, None, 1, required
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x, out, 1, None
        )
        == FP_Q16_ERR_NULL_PTR
    )
    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x, out, -1, required
        )
        == FP_Q16_ERR_BAD_PARAM
    )
    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x, x, 1, required
        )
        == FP_Q16_ERR_BAD_PARAM
    )
    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x, out, 1, x
        )
        == FP_Q16_ERR_BAD_PARAM
    )
    assert (
        fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x, out, 1, out
        )
        == FP_Q16_ERR_BAD_PARAM
    )


def test_no_publish_on_failure() -> None:
    x = [0, (1 << 63) - 1, 1]
    out = [0xAAAA, 0xBBBB, 0xCCCC]
    required = [0xDEAD]

    out_before = out.copy()
    req_before = required[0]

    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
        x,
        out,
        len(x),
        required,
    )
    assert status != FP_Q16_OK
    assert out == out_before
    assert required[0] == req_before


def test_zero_count_parity() -> None:
    x: list[int] = []
    out: list[int] = []
    required = [0x5555]

    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
        x,
        out,
        0,
        required,
    )
    assert status == FP_Q16_OK
    assert required[0] == 0


def test_required_output_slot_alias_rejected_no_partial_write() -> None:
    x = [0, 1, -1, 3]
    out = [0x1111, 0x2222, 0x3333, 0x4444]
    out_before = out.copy()

    status = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
        x,
        out,
        len(x),
        out,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == out_before


def test_known_vectors_parity() -> None:
    q16_one = 1 << 16
    x = [-(8 * q16_one), -(4 * q16_one), -q16_one, -1, 0, 1, q16_one, 4 * q16_one, 8 * q16_one]

    out_impl = [0x1111] * len(x)
    out_ref = [0x1111] * len(x)
    required_impl = [0x7777]

    status_impl = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
        x,
        out_impl,
        len(x),
        required_impl,
    )
    status_ref, required_ref = explicit_composition(x, out_ref, len(x))

    assert status_impl == status_ref
    assert out_impl == out_ref
    if status_impl == FP_Q16_OK:
        assert required_impl[0] == required_ref


def test_randomized_parity() -> None:
    rng = random.Random(20260421_912)

    for _ in range(3000):
        count = rng.randint(0, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(count)]

        out_impl = [rng.randint(-10000, 10000) for _ in range(count)]
        out_ref = out_impl.copy()
        required_impl = [rng.randint(-10000, 10000)]

        status_impl = fpq16_exp_approx_checked_no_partial_array_required_bytes_commit_only_parity(
            x,
            out_impl,
            count,
            required_impl,
        )
        status_ref, required_ref = explicit_composition(x, out_ref, count)

        assert status_impl == status_ref
        assert out_impl == out_ref
        if status_impl == FP_Q16_OK:
            assert required_impl[0] == required_ref


if __name__ == "__main__":
    test_source_contains_iq912_function()
    test_null_bad_param_guards()
    test_no_publish_on_failure()
    test_zero_count_parity()
    test_required_output_slot_alias_rejected_no_partial_write()
    test_known_vectors_parity()
    test_randomized_parity()
    print("ok")
