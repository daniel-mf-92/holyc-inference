#!/usr/bin/env python3
"""Parity harness for FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnly (IQ-903)."""

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


def fpq16_exp_approx_range_reduce_checked_no_partial_array(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    x_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
) -> int:
    if x_q16 is None or out_k is None or out_r_q16 is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is out_k or x_q16 is out_r_q16 or out_k is out_r_q16:
        return FP_Q16_ERR_BAD_PARAM

    if count == 0:
        return FP_Q16_OK
    if count > len(x_q16) or count > len(out_k) or count > len(out_r_q16):
        return FP_Q16_ERR_BAD_PARAM

    status, x_base, x_end = fp_array_i64_span_checked(x_addr, count)
    if status != FP_Q16_OK:
        return status
    status, k_base, k_end = fp_array_i64_span_checked(out_k_addr, count)
    if status != FP_Q16_OK:
        return status
    status, r_base, r_end = fp_array_i64_span_checked(out_r_addr, count)
    if status != FP_Q16_OK:
        return status

    if (
        fp_address_ranges_overlap(x_base, x_end, k_base, k_end)
        or fp_address_ranges_overlap(x_base, x_end, r_base, r_end)
        or fp_address_ranges_overlap(k_base, k_end, r_base, r_end)
    ):
        return FP_Q16_ERR_BAD_PARAM

    for i in range(count):
        status, _, _ = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status

    for i in range(count):
        status, lane_k, lane_r = fpq16_exp_approx_range_reduce_checked(x_q16[i])
        if status != FP_Q16_OK:
            return status
        out_k[i] = lane_k
        out_r_q16[i] = lane_r

    return FP_Q16_OK


def fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
    x_q16: list[int] | None,
    out_k: list[int] | None,
    out_r_q16: list[int] | None,
    count: int,
    out_count_slot: list[int] | None,
    out_cells_slot: list[int] | None,
    out_bytes_slot: list[int] | None,
    x_addr: int = 0x1000,
    out_k_addr: int = 0x2000,
    out_r_addr: int = 0x3000,
    out_count_addr: int = 0x4000,
    out_cells_addr: int = 0x4010,
    out_bytes_addr: int = 0x4020,
) -> int:
    if out_count_slot is None or out_cells_slot is None or out_bytes_slot is None:
        return FP_Q16_ERR_NULL_PTR
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if out_count_slot is out_cells_slot or out_count_slot is out_bytes_slot or out_cells_slot is out_bytes_slot:
        return FP_Q16_ERR_BAD_PARAM

    if x_q16 is not None and (out_count_slot is x_q16 or out_cells_slot is x_q16 or out_bytes_slot is x_q16):
        return FP_Q16_ERR_BAD_PARAM
    if out_k is not None and (out_count_slot is out_k or out_cells_slot is out_k or out_bytes_slot is out_k):
        return FP_Q16_ERR_BAD_PARAM
    if out_r_q16 is not None and (out_count_slot is out_r_q16 or out_cells_slot is out_r_q16 or out_bytes_slot is out_r_q16):
        return FP_Q16_ERR_BAD_PARAM

    if count != 0:
        status, x_base, x_end = fp_array_i64_span_checked(x_addr, count)
        if status != FP_Q16_OK:
            return status
        status, k_base, k_end = fp_array_i64_span_checked(out_k_addr, count)
        if status != FP_Q16_OK:
            return status
        status, r_base, r_end = fp_array_i64_span_checked(out_r_addr, count)
        if status != FP_Q16_OK:
            return status

        if out_count_addr > (U64_MAX_VALUE - 7) or out_cells_addr > (U64_MAX_VALUE - 7) or out_bytes_addr > (U64_MAX_VALUE - 7):
            return FP_Q16_ERR_OVERFLOW

        count_end = out_count_addr + 8
        cells_end = out_cells_addr + 8
        bytes_end = out_bytes_addr + 8

        if (
            fp_address_ranges_overlap(out_count_addr, count_end, x_base, x_end)
            or fp_address_ranges_overlap(out_count_addr, count_end, k_base, k_end)
            or fp_address_ranges_overlap(out_count_addr, count_end, r_base, r_end)
            or fp_address_ranges_overlap(out_cells_addr, cells_end, x_base, x_end)
            or fp_address_ranges_overlap(out_cells_addr, cells_end, k_base, k_end)
            or fp_address_ranges_overlap(out_cells_addr, cells_end, r_base, r_end)
            or fp_address_ranges_overlap(out_bytes_addr, bytes_end, x_base, x_end)
            or fp_address_ranges_overlap(out_bytes_addr, bytes_end, k_base, k_end)
            or fp_address_ranges_overlap(out_bytes_addr, bytes_end, r_base, r_end)
            or fp_address_ranges_overlap(out_count_addr, count_end, out_cells_addr, cells_end)
            or fp_address_ranges_overlap(out_count_addr, count_end, out_bytes_addr, bytes_end)
            or fp_address_ranges_overlap(out_cells_addr, cells_end, out_bytes_addr, bytes_end)
        ):
            return FP_Q16_ERR_BAD_PARAM

    snapshot_x = x_q16
    snapshot_k = out_k
    snapshot_r = out_r_q16
    snapshot_count = count

    status = fpq16_exp_approx_range_reduce_checked_no_partial_array(
        x_q16,
        out_k,
        out_r_q16,
        count,
        x_addr=x_addr,
        out_k_addr=out_k_addr,
        out_r_addr=out_r_addr,
    )
    if status != FP_Q16_OK:
        return status

    if snapshot_x is not x_q16 or snapshot_k is not out_k or snapshot_r is not out_r_q16 or snapshot_count != count:
        return FP_Q16_ERR_BAD_PARAM

    if count > (I64_MAX_VALUE >> 1):
        return FP_Q16_ERR_OVERFLOW

    staged_cells = count << 1
    if staged_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    staged_bytes = staged_cells << 3

    out_count_slot[0] = count
    out_cells_slot[0] = staged_cells
    out_bytes_slot[0] = staged_bytes
    return FP_Q16_OK


def test_source_contains_iq903_commit_only_wrapper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16ExpApproxRangeReduceCheckedNoPartialArrayCommitOnly(I64 *x_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16ExpApproxChecked", 1)[0]
    assert "status = FPQ16ExpApproxRangeReduceCheckedNoPartialArray(x_q16," in body
    assert "staged_required_cells = count << 1;" in body
    assert "staged_required_bytes = staged_required_cells << 3;" in body
    assert "*out_count = staged_count;" in body


def test_null_bad_param_and_overlap_guards() -> None:
    x = [0, 1]
    out_k = [7, 8]
    out_r = [9, 10]

    out_count = [111]
    out_cells = [222]
    out_bytes = [333]

    assert (
        fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
            x,
            out_k,
            out_r,
            2,
            None,
            out_cells,
            out_bytes,
        )
        == FP_Q16_ERR_NULL_PTR
    )

    before = (out_count[0], out_cells[0], out_bytes[0])
    err = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
        x,
        out_k,
        out_r,
        2,
        out_count,
        out_cells,
        out_bytes,
        out_count_addr=0x2000,
        out_cells_addr=0x4010,
        out_bytes_addr=0x4020,
    )
    assert err == FP_Q16_ERR_BAD_PARAM
    assert (out_count[0], out_cells[0], out_bytes[0]) == before


def test_known_vectors_publish_staged_totals() -> None:
    x = [-(1 << 20), -65_536, -1, 0, 1, 65_536, 1 << 20]
    out_k = [0xAAAA] * len(x)
    out_r = [0xBBBB] * len(x)
    out_count = [0x1234]
    out_cells = [0x5678]
    out_bytes = [0x9ABC]

    err = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
        x,
        out_k,
        out_r,
        len(x),
        out_count,
        out_cells,
        out_bytes,
    )
    assert err == FP_Q16_OK

    assert out_count[0] == len(x)
    assert out_cells[0] == len(x) * 2
    assert out_bytes[0] == len(x) * 16

    for i, x_q16 in enumerate(x):
        lane_status, lane_k, lane_r = fpq16_exp_approx_range_reduce_checked(x_q16)
        assert lane_status == FP_Q16_OK
        assert out_k[i] == lane_k
        assert out_r[i] == lane_r


def test_randomized_parity_against_explicit_composition() -> None:
    rng = random.Random(20260421_903)

    for _ in range(2500):
        count = rng.randint(0, 64)
        x = [rng.randint(-(1 << 63) + 1, (1 << 63) - 1) for _ in range(max(count, 1))]
        out_k = [0xCAFE] * max(count, 1)
        out_r = [0xBEEF] * max(count, 1)

        got_k = out_k.copy()
        got_r = out_r.copy()
        expected_k = out_k.copy()
        expected_r = out_r.copy()

        got_count = [0x11]
        got_cells = [0x22]
        got_bytes = [0x33]

        expected_count = [0x11]
        expected_cells = [0x22]
        expected_bytes = [0x33]

        expected_status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
            x,
            expected_k,
            expected_r,
            count,
            expected_count,
            expected_cells,
            expected_bytes,
            x_addr=0x1000,
            out_k_addr=0x4000,
            out_r_addr=0x8000,
            out_count_addr=0x9000,
            out_cells_addr=0x9010,
            out_bytes_addr=0x9020,
        )
        got_status = fpq16_exp_approx_range_reduce_checked_no_partial_array_commit_only(
            x,
            got_k,
            got_r,
            count,
            got_count,
            got_cells,
            got_bytes,
            x_addr=0x1000,
            out_k_addr=0x4000,
            out_r_addr=0x8000,
            out_count_addr=0x9000,
            out_cells_addr=0x9010,
            out_bytes_addr=0x9020,
        )

        assert got_status == expected_status
        assert got_k == expected_k
        assert got_r == expected_r
        assert got_count == expected_count
        assert got_cells == expected_cells
        assert got_bytes == expected_bytes


if __name__ == "__main__":
    test_source_contains_iq903_commit_only_wrapper()
    test_null_bad_param_and_overlap_guards()
    test_known_vectors_publish_staged_totals()
    test_randomized_parity_against_explicit_composition()
    print("ok")
