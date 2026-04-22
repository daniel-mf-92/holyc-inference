#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxPrefixSumCheckedNoPartial (IQ-1149)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_ONE = 1 << FP_Q16_SHIFT

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1


def fpq16_softmax_prefix_sum_checked_reference(
    probs_q16: list[int],
    lane_count: int,
    *,
    probs_present: bool = True,
    prefix_present: bool = True,
    out_total_present: bool = True,
) -> tuple[int, list[int], int]:
    if not probs_present or not prefix_present or not out_total_present:
        return FP_Q16_ERR_NULL_PTR, [], 0
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, [], 0
    if lane_count == 0:
        return FP_Q16_OK, [], 0
    if lane_count != len(probs_q16):
        return FP_Q16_ERR_BAD_PARAM, [], 0

    running_sum_q16 = 0
    for lane_q16 in probs_q16:
        if lane_q16 < 0:
            return FP_Q16_ERR_BAD_PARAM, [], 0
        if lane_q16 > I64_MAX_VALUE - running_sum_q16:
            return FP_Q16_ERR_OVERFLOW, [], 0
        running_sum_q16 += lane_q16

    corrected = list(probs_q16)
    diff_q16 = FP_Q16_ONE - running_sum_q16

    if diff_q16 > 0:
        if corrected[-1] > I64_MAX_VALUE - diff_q16:
            return FP_Q16_ERR_OVERFLOW, [], 0
        corrected[-1] += diff_q16
    elif diff_q16 < 0:
        borrow_q16 = -diff_q16
        for idx in range(lane_count - 1, -1, -1):
            if borrow_q16 <= 0:
                break
            take_q16 = min(corrected[idx], borrow_q16)
            corrected[idx] -= take_q16
            borrow_q16 -= take_q16
        if borrow_q16 != 0:
            return FP_Q16_ERR_BAD_PARAM, [], 0

    prefix: list[int] = []
    running = 0
    for lane_q16 in corrected:
        if lane_q16 < 0:
            return FP_Q16_ERR_BAD_PARAM, [], 0
        if lane_q16 > I64_MAX_VALUE - running:
            return FP_Q16_ERR_OVERFLOW, [], 0
        running += lane_q16
        prefix.append(running)

    if running != FP_Q16_ONE:
        return FP_Q16_ERR_OVERFLOW, [], 0

    return FP_Q16_OK, prefix, running


def fpq16_softmax_prefix_sum_checked_nopartial_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    probs_capacity: int,
    prefix_sums_q16: list[int] | None,
    prefix_capacity: int,
    out_total_q16: list[int] | None,
) -> int:
    if probs_q16 is None or prefix_sums_q16 is None or out_total_q16 is None:
        return FP_Q16_ERR_NULL_PTR

    if lane_count < 0 or probs_capacity < 0 or prefix_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM

    snapshot_lane_count = lane_count
    snapshot_probs_capacity = probs_capacity
    snapshot_prefix_capacity = prefix_capacity

    required_probs = snapshot_lane_count
    required_prefix = snapshot_lane_count

    if required_probs > snapshot_probs_capacity:
        return FP_Q16_ERR_BAD_PARAM
    if required_prefix > snapshot_prefix_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if snapshot_lane_count > len(probs_q16) or snapshot_lane_count > len(prefix_sums_q16):
        return FP_Q16_ERR_BAD_PARAM

    status, staged_prefix, staged_total = fpq16_softmax_prefix_sum_checked_reference(
        probs_q16[:snapshot_lane_count],
        snapshot_lane_count,
    )
    if status != FP_Q16_OK:
        return status

    if (
        snapshot_lane_count != lane_count
        or snapshot_probs_capacity != probs_capacity
        or snapshot_prefix_capacity != prefix_capacity
    ):
        return FP_Q16_ERR_BAD_PARAM

    for i, lane in enumerate(staged_prefix):
        prefix_sums_q16[i] = lane

    out_total_q16[0] = staged_total
    return FP_Q16_OK


def test_source_contains_iq1149_function_and_snapshot_contract() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SoftmaxPrefixSumCheckedNoPartial(I64 *probs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("// Checked composed helper for alias-permitted split-phase softmax.", 1)[0]
    assert "snapshot_lane_count = lane_count;" in body
    assert "snapshot_probs_capacity = probs_capacity;" in body
    assert "snapshot_prefix_capacity = prefix_capacity;" in body
    assert "status = FPQ16SoftmaxPrefixSumChecked(probs_q16," in body
    assert "*out_total_q16 = staged_total_q16;" in body


def test_null_and_capacity_guards_preserve_total() -> None:
    probs = [5000, 7000, 9000]
    prefix = [111, 222, 333]
    out_total = [444]

    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        None,
        3,
        3,
        prefix,
        3,
        out_total,
    )
    assert status == FP_Q16_ERR_NULL_PTR
    assert out_total == [444]

    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        probs,
        3,
        2,
        prefix,
        3,
        out_total,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out_total == [444]

    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        probs,
        3,
        3,
        prefix,
        2,
        out_total,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out_total == [444]


def test_negative_lane_and_overflow_preserve_total() -> None:
    out_total = [0xA5A5]
    prefix = [9, 9]

    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        [100, -1],
        2,
        2,
        prefix,
        2,
        out_total,
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out_total == [0xA5A5]

    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        [I64_MAX_VALUE, 1],
        2,
        2,
        prefix,
        2,
        out_total,
    )
    assert status == FP_Q16_ERR_OVERFLOW
    assert out_total == [0xA5A5]


def test_success_vectors_and_randomized_invariants() -> None:
    probs = [10_000, 12_000, 13_000]
    prefix = [0, 0, 0]
    out_total = [0]

    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        probs,
        len(probs),
        len(probs),
        prefix,
        len(prefix),
        out_total,
    )
    assert status == FP_Q16_OK
    assert prefix == [10_000, 22_000, FP_Q16_ONE]
    assert out_total[0] == FP_Q16_ONE

    rng = random.Random(20260422_1149)
    for _ in range(6000):
        lanes = rng.randint(0, 64)
        probs = [rng.randint(0, FP_Q16_ONE * 2) for _ in range(lanes)]
        prefix = [0x7F7F7F7F] * lanes
        out_total = [0x3C3C3C3C]

        expected_status, expected_prefix, expected_total = fpq16_softmax_prefix_sum_checked_reference(
            probs,
            lanes,
        )

        status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
            probs,
            lanes,
            lanes,
            prefix,
            lanes,
            out_total,
        )
        assert status == expected_status

        if status == FP_Q16_OK:
            assert prefix == expected_prefix
            assert out_total[0] == expected_total
        else:
            assert prefix == [0x7F7F7F7F] * lanes
            assert out_total[0] == 0x3C3C3C3C


def run() -> None:
    test_source_contains_iq1149_function_and_snapshot_contract()
    test_null_and_capacity_guards_preserve_total()
    test_negative_lane_and_overflow_preserve_total()
    test_success_vectors_and_randomized_invariants()
    print("softmax_q16_prefix_sum_checked_nopartial=ok")


if __name__ == "__main__":
    run()
