#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnly (IQ-906)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
U64_MAX_VALUE = (1 << 64) - 1

EXP_LN2_Q16 = 45_426
EXP_HALF_LN2_Q16 = 22_713


def _i64_wrap(value: int) -> int:
    value &= U64_MAX_VALUE
    if value >= (1 << 63):
        value -= (1 << 64)
    return value


def _u64(value: int) -> int:
    return value & U64_MAX_VALUE


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_address_ranges_overlap(a_base: int, a_end_exclusive: int, b_base: int, b_end_exclusive: int) -> int:
    if a_end_exclusive <= a_base:
        return 0
    if b_end_exclusive <= b_base:
        return 0
    if a_end_exclusive <= b_base:
        return 0
    if b_end_exclusive <= a_base:
        return 0
    return 1


def fp_array_i64_span_checked(base_addr: int | None, count: int) -> tuple[int, int, int]:
    if base_addr is None:
        return FP_Q16_ERR_NULL_PTR, 0, 0
    if count <= 0:
        return FP_Q16_ERR_BAD_PARAM, 0, 0
    if count > (U64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    byte_count = _u64(count << 3)
    if _u64(base_addr) > (U64_MAX_VALUE - byte_count):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    base = _u64(base_addr)
    return FP_Q16_OK, base, _u64(base + byte_count)


def fpq16_exp_approx_range_reduce_checked(x_q16: int) -> tuple[int, int, int]:
    x_q16 = _i64_wrap(x_q16)

    if x_q16 >= 0:
        k = (x_q16 + EXP_HALF_LN2_Q16) // EXP_LN2_Q16
    else:
        k = -(((-x_q16) + EXP_HALF_LN2_Q16) // EXP_LN2_Q16)

    abs_k_u64 = fp_abs_to_u64(k)
    if abs_k_u64 > (U64_MAX_VALUE // EXP_LN2_Q16):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    if k >= 0:
        k_ln2_q16 = _i64_wrap(abs_k_u64 * EXP_LN2_Q16)
    else:
        k_ln2_q16 = _i64_wrap(-(abs_k_u64 * EXP_LN2_Q16))

    if (k_ln2_q16 > 0 and x_q16 < _i64_wrap(-(1 << 63) + k_ln2_q16)) or (
        k_ln2_q16 < 0 and x_q16 > _i64_wrap(I64_MAX_VALUE + k_ln2_q16)
    ):
        return FP_Q16_ERR_OVERFLOW, 0, 0

    residual_q16 = _i64_wrap(x_q16 - k_ln2_q16)

    if residual_q16 < -EXP_HALF_LN2_Q16 or residual_q16 > EXP_HALF_LN2_Q16:
        return FP_Q16_ERR_BAD_PARAM, 0, 0

    return FP_Q16_OK, _i64_wrap(k), residual_q16


def fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_count_slot: list[int] | None,
    out_required_cells_slot: list[int] | None,
    out_required_bytes_slot: list[int] | None,
) -> int:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if out_count_slot is None or out_required_cells_slot is None or out_required_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_k or x_q16 is out_r_q16 or out_k is out_r_q16:
        return FP_Q16_ERR_BAD_PARAM

    if out_count_slot is out_required_cells_slot or out_count_slot is out_required_bytes_slot or out_required_cells_slot is out_required_bytes_slot:
        return FP_Q16_ERR_BAD_PARAM

    for i in range(count):
        status, lane_k, lane_r = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status
        out_k[i] = lane_k
        out_r_q16[i] = lane_r

    if count > (I64_MAX_VALUE >> 1):
        return FP_Q16_ERR_OVERFLOW

    required_cells = count << 1
    if required_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    out_count_slot[0] = count
    out_required_cells_slot[0] = required_cells
    out_required_bytes_slot[0] = required_cells << 3
    return FP_Q16_OK


def fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_count_slot: list[int] | None,
    out_required_cells_slot: list[int] | None,
    out_required_bytes_slot: list[int] | None,
) -> int:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if out_count_slot is None or out_required_cells_slot is None or out_required_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_k or x_q16 is out_r_q16 or out_k is out_r_q16:
        return FP_Q16_ERR_BAD_PARAM

    if out_count_slot is out_required_cells_slot or out_count_slot is out_required_bytes_slot or out_required_cells_slot is out_required_bytes_slot:
        return FP_Q16_ERR_BAD_PARAM

    snapshot_out_k = out_k.copy()
    snapshot_out_r = out_r_q16.copy()

    for i in range(count):
        status, _, _ = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status

    if out_k != snapshot_out_k or out_r_q16 != snapshot_out_r:
        return FP_Q16_ERR_BAD_PARAM

    if count > (I64_MAX_VALUE >> 1):
        return FP_Q16_ERR_OVERFLOW

    required_cells = count << 1
    if required_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    out_count_slot[0] = count
    out_required_cells_slot[0] = required_cells
    out_required_bytes_slot[0] = required_cells << 3
    return FP_Q16_OK


def explicit_checked_composition(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_count_slot: list[int] | None,
    out_required_cells_slot: list[int] | None,
    out_required_bytes_slot: list[int] | None,
) -> int:
    if out_count_slot is None or out_required_cells_slot is None or out_required_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR

    staged_k = list(out_k) if out_k is not None else None
    staged_r = list(out_r_q16) if out_r_q16 is not None else None

    staged_count = [0]
    staged_cells = [0]
    staged_bytes = [0]
    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
        x_q16,
        staged_k,
        staged_r,
        count,
        staged_count,
        staged_cells,
        staged_bytes,
    )
    if status != FP_Q16_OK:
        return status

    # Explicit preflight-only composition: validate all lanes again but do not
    # publish lane-array mutations.
    if x_q16 is None or staged_k is None or staged_r is None:
        return FP_Q16_ERR_NULL_PTR

    for i in range(count):
        status, _, _ = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status

    out_count_slot[0] = staged_count[0]
    out_required_cells_slot[0] = staged_cells[0]
    out_required_bytes_slot[0] = staged_bytes[0]
    return FP_Q16_OK


def test_source_contains_iq906_preflight_only_wrapper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnlyPreflightOnly(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayRequiredBytes", 1)[0]
    assert "status = FPQ16ExpApproxRangeReduceChecked(x_q16[i]," in body
    assert "if (snapshot_out_k != out_k)" in body
    assert "if (snapshot_out_r_q16 != out_r_q16)" in body
    assert "*out_count = staged_count;" in body
    assert "*out_required_cells = staged_required_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_known_vector_preflight_only_no_array_writes() -> None:
    x = [-(1 << 20), -65_536, -1, 0, 1, 65_536, 1 << 20]
    out_k = [0xAAAA] * len(x)
    out_r = [0xBBBB] * len(x)
    before_k = out_k.copy()
    before_r = out_r.copy()

    out_count = [0x11]
    out_cells = [0x22]
    out_bytes = [0x33]

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
        x,
        out_k,
        out_r,
        len(x),
        out_count,
        out_cells,
        out_bytes,
    )
    assert status == FP_Q16_OK
    assert out_k == before_k
    assert out_r == before_r
    assert out_count[0] == len(x)
    assert out_cells[0] == len(x) * 2
    assert out_bytes[0] == len(x) * 16


def test_null_and_bad_param_no_partial_publish() -> None:
    x = [0, 1]
    out_k = [2, 3]
    out_r = [4, 5]

    out_count = [0xCAFE]
    out_cells = [0xBEEF]
    out_bytes = [0x1234]

    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
            None,
            out_k,
            out_r,
            2,
            out_count,
            out_cells,
            out_bytes,
        )
        == FP_Q16_ERR_NULL_PTR
    )

    before = (out_count[0], out_cells[0], out_bytes[0])
    err = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
        x,
        out_k,
        out_r,
        -1,
        out_count,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert (out_count[0], out_cells[0], out_bytes[0]) == before


def test_randomized_parity_against_explicit_checked_composition() -> None:
    rng = random.Random(20260421_906)

    for _ in range(3000):
        count = rng.randint(0, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(max(count, 1))]

        got_k = [0xCAFE] * max(count, 1)
        got_r = [0xBEEF] * max(count, 1)
        got_k_before = got_k.copy()
        got_r_before = got_r.copy()

        exp_k = got_k.copy()
        exp_r = got_r.copy()

        got_count = [0x55]
        got_cells = [0x66]
        got_bytes = [0x77]

        exp_count = [0x55]
        exp_cells = [0x66]
        exp_bytes = [0x77]

        expected_status = explicit_checked_composition(
            x,
            exp_k,
            exp_r,
            count,
            exp_count,
            exp_cells,
            exp_bytes,
        )
        got_status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only_preflight_only(
            x,
            got_k,
            got_r,
            count,
            got_count,
            got_cells,
            got_bytes,
        )

        assert got_status == expected_status
        if got_status == FP_Q16_OK:
            assert got_count == exp_count
            assert got_cells == exp_cells
            assert got_bytes == exp_bytes
            assert got_k == got_k_before
            assert got_r == got_r_before


if __name__ == "__main__":
    test_source_contains_iq906_preflight_only_wrapper()
    test_known_vector_preflight_only_no_array_writes()
    test_null_and_bad_param_no_partial_publish()
    test_randomized_parity_against_explicit_checked_composition()
    print("ok")
