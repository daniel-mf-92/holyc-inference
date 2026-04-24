#!/usr/bin/env python3
"""Parity harness for FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnly (IQ-1206)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_fixedpoint_q16_mul_shift_round_checked_nopartial_array import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    fpq16_mul_shift_round_checked_nopartial_array_reference,
)
from test_fixedpoint_q16_mul_shift_round_checked_nopartial_array_commit_only import (  # noqa: E402
    fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference,
)


def fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
    lhs_values: list[int] | None,
    lhs_capacity: int,
    rhs_values: list[int] | None,
    rhs_capacity: int,
    out_values: list[int] | None,
    out_capacity: int,
    count: int,
    shift: int,
    out_required_lhs: list[int] | None,
    out_required_rhs: list[int] | None,
    out_required_out: list[int] | None,
    *,
    mutate_snapshot: bool = False,
    mutate_required: bool = False,
) -> int:
    if (
        lhs_values is None
        or rhs_values is None
        or out_values is None
        or out_required_lhs is None
        or out_required_rhs is None
        or out_required_out is None
    ):
        return FP_Q16_ERR_NULL_PTR

    if lhs_capacity < 0 or rhs_capacity < 0 or out_capacity < 0 or count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_lhs is out_required_rhs
        or out_required_lhs is out_required_out
        or out_required_rhs is out_required_out
    ):
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_lhs is lhs_values
        or out_required_lhs is rhs_values
        or out_required_lhs is out_values
        or out_required_rhs is lhs_values
        or out_required_rhs is rhs_values
        or out_required_rhs is out_values
        or out_required_out is lhs_values
        or out_required_out is rhs_values
        or out_required_out is out_values
    ):
        return FP_Q16_ERR_BAD_PARAM

    if count > lhs_capacity or count > rhs_capacity or count > out_capacity:
        return FP_Q16_ERR_BAD_PARAM

    snapshot_lhs_capacity = lhs_capacity
    snapshot_rhs_capacity = rhs_capacity
    snapshot_out_capacity = out_capacity
    snapshot_count = count
    snapshot_shift = shift

    staged_commit_out = list(out_values)
    staged_canonical_out = list(out_values)

    staged_commit_required_lhs = [0x11]
    staged_commit_required_rhs = [0x22]
    staged_commit_required_out = [0x33]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_reference(
        lhs_values,
        lhs_capacity,
        rhs_values,
        rhs_capacity,
        staged_commit_out,
        out_capacity,
        count,
        shift,
        staged_commit_required_lhs,
        staged_commit_required_rhs,
        staged_commit_required_out,
    )
    if err != FP_Q16_OK:
        return err

    err = fpq16_mul_shift_round_checked_nopartial_array_reference(
        lhs_values,
        lhs_capacity,
        rhs_values,
        rhs_capacity,
        staged_canonical_out,
        out_capacity,
        count,
        shift,
    )
    if err != FP_Q16_OK:
        return err

    if mutate_snapshot:
        snapshot_shift += 1

    if (
        snapshot_lhs_capacity != lhs_capacity
        or snapshot_rhs_capacity != rhs_capacity
        or snapshot_out_capacity != out_capacity
        or snapshot_count != count
        or snapshot_shift != shift
    ):
        return FP_Q16_ERR_BAD_PARAM

    staged_canonical_required_lhs = snapshot_count
    staged_canonical_required_rhs = snapshot_count
    staged_canonical_required_out = snapshot_count

    if mutate_required:
        staged_canonical_required_out += 1

    if (
        staged_commit_required_lhs[0] != staged_canonical_required_lhs
        or staged_commit_required_rhs[0] != staged_canonical_required_rhs
        or staged_commit_required_out[0] != staged_canonical_required_out
    ):
        return FP_Q16_ERR_BAD_PARAM

    out_required_lhs[0] = staged_commit_required_lhs[0]
    out_required_rhs[0] = staged_commit_required_rhs[0]
    out_required_out[0] = staged_commit_required_out[0]
    return FP_Q16_OK


def explicit_preflight_composition(
    lhs_values: list[int],
    rhs_values: list[int],
    out_values: list[int],
    count: int,
    shift: int,
    out_required_lhs: list[int],
    out_required_rhs: list[int],
    out_required_out: list[int],
) -> int:
    staged_out = list(out_values)
    status = fpq16_mul_shift_round_checked_nopartial_array_reference(
        lhs_values,
        count,
        rhs_values,
        count,
        staged_out,
        count,
        count,
        shift,
    )
    if status != FP_Q16_OK:
        return status

    out_required_lhs[0] = count
    out_required_rhs[0] = count
    out_required_out[0] = count
    return FP_Q16_OK


def test_source_contains_iq1206_function_and_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnly(I64 *lhs_values,"
    assert sig in source
    body = source.split(sig, 1)[1].split("// Geometry diagnostics helper for FPQ16MulCheckedNoPartialArray.", 1)[0]

    assert "staged_commit_out_values = MAlloc(stage_bytes);" in body
    assert "staged_canonical_out_values = MAlloc(stage_bytes);" in body
    assert "status = FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnly(lhs_values," in body
    assert "status = FPQ16MulShiftRoundCheckedNoPartialArray(lhs_values," in body
    assert "staged_canonical_required_lhs = snapshot_count;" in body
    assert "*out_required_lhs = staged_commit_required_lhs;" in body
    assert "*out_required_rhs = staged_commit_required_rhs;" in body
    assert "*out_required_out = staged_commit_required_out;" in body


def test_alias_capacity_shift_overflow_and_no_partial_publish() -> None:
    lhs = [3, 4, 5]
    rhs = [6, 7, 8]
    out = [101, 102, 103]

    req_lhs = [701]
    req_rhs = [702]
    req_out = [703]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
        lhs,
        3,
        rhs,
        3,
        out,
        3,
        4,
        1,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 102, 103]
    assert req_lhs == [701] and req_rhs == [702] and req_out == [703]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
        lhs,
        3,
        rhs,
        3,
        out,
        3,
        3,
        -1,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert out == [101, 102, 103]
    assert req_lhs == [701] and req_rhs == [702] and req_out == [703]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
        [I64_MAX_VALUE],
        1,
        [I64_MAX_VALUE],
        1,
        [909],
        1,
        1,
        0,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert err == FP_Q16_ERR_OVERFLOW
    assert req_lhs == [701] and req_rhs == [702] and req_out == [703]


def test_success_publish_with_zero_write_output_behavior() -> None:
    lhs = [5, -5, 9, -9]
    rhs = [3, 3, -3, -3]
    out = [41, 42, 43, 44]
    out_before = list(out)

    req_lhs = [0]
    req_rhs = [0]
    req_out = [0]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
        lhs,
        4,
        rhs,
        4,
        out,
        4,
        4,
        1,
        req_lhs,
        req_rhs,
        req_out,
    )
    assert err == FP_Q16_OK
    assert out == out_before
    assert req_lhs == [4] and req_rhs == [4] and req_out == [4]


def test_snapshot_and_required_tuple_parity_guards() -> None:
    lhs = [12, -13]
    rhs = [7, 8]
    out = [91, 92]

    req_lhs = [501]
    req_rhs = [502]
    req_out = [503]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
        lhs,
        2,
        rhs,
        2,
        out,
        2,
        2,
        2,
        req_lhs,
        req_rhs,
        req_out,
        mutate_snapshot=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert req_lhs == [501] and req_rhs == [502] and req_out == [503]

    err = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
        lhs,
        2,
        rhs,
        2,
        out,
        2,
        2,
        2,
        req_lhs,
        req_rhs,
        req_out,
        mutate_required=True,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert req_lhs == [501] and req_rhs == [502] and req_out == [503]


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260424_1206)
    for _ in range(12000):
        count = rng.randint(0, 48)
        shift = rng.randint(0, 62)

        lhs = [rng.randint(-(1 << 63), (1 << 63) - 1) for _ in range(count)]
        rhs = [rng.randint(-(1 << 63), (1 << 63) - 1) for _ in range(count)]

        out_ref = [0xA55A] * count
        out_exp = [0xA55A] * count

        req_lhs_ref = [0x11]
        req_rhs_ref = [0x22]
        req_out_ref = [0x33]

        req_lhs_exp = [0x11]
        req_rhs_exp = [0x22]
        req_out_exp = [0x33]

        status_ref = fpq16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only_reference(
            lhs,
            count,
            rhs,
            count,
            out_ref,
            count,
            count,
            shift,
            req_lhs_ref,
            req_rhs_ref,
            req_out_ref,
        )
        status_exp = explicit_preflight_composition(
            lhs,
            rhs,
            out_exp,
            count,
            shift,
            req_lhs_exp,
            req_rhs_exp,
            req_out_exp,
        )

        assert status_ref == status_exp
        assert out_ref == out_exp
        assert req_lhs_ref == req_lhs_exp
        assert req_rhs_ref == req_rhs_exp
        assert req_out_ref == req_out_exp


def run() -> None:
    test_source_contains_iq1206_function_and_contract()
    test_alias_capacity_shift_overflow_and_no_partial_publish()
    test_success_publish_with_zero_write_output_behavior()
    test_snapshot_and_required_tuple_parity_guards()
    test_randomized_parity_against_explicit_composition()
    print("fixedpoint_q16_mul_shift_round_checked_nopartial_array_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
