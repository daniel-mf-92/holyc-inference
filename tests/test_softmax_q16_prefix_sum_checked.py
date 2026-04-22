#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxPrefixSumChecked (IQ-1134)."""

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


def q16(frac: float) -> int:
    return int(round(frac * FP_Q16_ONE))


def fpq16_softmax_prefix_sum_checked_reference(
    probs_q16: list[int],
    lane_count: int,
    *,
    probs_present: bool = True,
    prefix_present: bool = True,
    out_total_present: bool = True,
) -> tuple[int, list[int], int | None]:
    if not probs_present or not prefix_present or not out_total_present:
        return FP_Q16_ERR_NULL_PTR, [0] * max(lane_count, 0), None
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM, [0] * max(lane_count, 0), 0

    if lane_count == 0:
        return FP_Q16_OK, [], 0

    probs = probs_q16[:lane_count]

    running_sum_q16 = 0
    for lane in probs:
        if lane < 0:
            return FP_Q16_ERR_BAD_PARAM, [0] * lane_count, 0
        if lane > (I64_MAX_VALUE - running_sum_q16):
            return FP_Q16_ERR_OVERFLOW, [0] * lane_count, 0
        running_sum_q16 += lane

    diff_q16 = FP_Q16_ONE - running_sum_q16
    corrected_total_q16 = running_sum_q16

    if diff_q16 > 0:
        if corrected_total_q16 > (I64_MAX_VALUE - diff_q16):
            return FP_Q16_ERR_OVERFLOW, [0] * lane_count, 0
        corrected_total_q16 += diff_q16
    elif diff_q16 < 0:
        borrow_q16 = -diff_q16
        for i in range(lane_count - 1, -1, -1):
            if borrow_q16 <= 0:
                break
            lane = probs[i]
            if lane >= borrow_q16:
                borrow_q16 = 0
            else:
                borrow_q16 -= lane
        if borrow_q16:
            return FP_Q16_ERR_BAD_PARAM, [0] * lane_count, 0
        corrected_total_q16 += diff_q16

    if corrected_total_q16 != FP_Q16_ONE:
        return FP_Q16_ERR_OVERFLOW, [0] * lane_count, 0

    corrected = probs[:]
    if diff_q16 >= 0:
        corrected[-1] += diff_q16
    else:
        borrow_q16 = -diff_q16
        for i in range(lane_count - 1, -1, -1):
            if borrow_q16 <= 0:
                break
            take_q16 = min(corrected[i], borrow_q16)
            corrected[i] -= take_q16
            borrow_q16 -= take_q16
        if borrow_q16:
            return FP_Q16_ERR_BAD_PARAM, [0] * lane_count, 0

    prefix = [0] * lane_count
    running_sum_q16 = 0
    for i, lane in enumerate(corrected):
        if lane < 0:
            return FP_Q16_ERR_BAD_PARAM, [0] * lane_count, 0
        if lane > (I64_MAX_VALUE - running_sum_q16):
            return FP_Q16_ERR_OVERFLOW, [0] * lane_count, 0
        running_sum_q16 += lane
        prefix[i] = running_sum_q16

    if running_sum_q16 != FP_Q16_ONE:
        return FP_Q16_ERR_OVERFLOW, [0] * lane_count, 0

    return FP_Q16_OK, prefix, running_sum_q16


def test_source_contains_iq1134_function_and_correction_logic() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SoftmaxPrefixSumChecked(I64 *probs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16SoftmaxFromPreclampedChecked", 1)[0]
    assert "diff_q16 = FP_Q16_ONE - running_sum_q16;" in body
    assert "if (diff_q16 >= 0)" in body
    assert "for (i = lane_count - 1; i >= 0; i--)" in body
    assert "prefix_sums_q16[i] = running_sum_q16;" in body


def test_nullptr_and_domain_failures() -> None:
    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([q16(0.5)], 1, probs_present=False)
    assert status == FP_Q16_ERR_NULL_PTR

    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([q16(0.5)], -1)
    assert status == FP_Q16_ERR_BAD_PARAM

    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([q16(0.2), -1], 2)
    assert status == FP_Q16_ERR_BAD_PARAM


def test_empty_input_contract() -> None:
    status, prefix, total = fpq16_softmax_prefix_sum_checked_reference([], 0)
    assert status == FP_Q16_OK
    assert prefix == []
    assert total == 0


def test_positive_diff_adds_to_last_lane() -> None:
    status, prefix, total = fpq16_softmax_prefix_sum_checked_reference([q16(0.2), q16(0.3)], 2)
    assert status == FP_Q16_OK
    assert prefix == [q16(0.2), FP_Q16_ONE]
    assert total == FP_Q16_ONE


def test_negative_diff_borrow_tail_first() -> None:
    status, prefix, total = fpq16_softmax_prefix_sum_checked_reference([q16(0.8), q16(0.25), q16(0.1)], 3)
    assert status == FP_Q16_OK
    assert prefix[0] == q16(0.8)
    assert prefix[1] == FP_Q16_ONE
    assert prefix[2] == FP_Q16_ONE
    assert total == FP_Q16_ONE


def test_overflow_guard_on_accumulation() -> None:
    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([I64_MAX_VALUE, 1], 2)
    assert status == FP_Q16_ERR_OVERFLOW


def test_randomized_invariants() -> None:
    rng = random.Random(20260422_1134)

    for _ in range(5000):
        lane_count = rng.randint(1, 16)
        probs = [rng.randint(0, FP_Q16_ONE) for _ in range(lane_count)]

        status, prefix, total = fpq16_softmax_prefix_sum_checked_reference(probs, lane_count)
        assert status in (FP_Q16_OK, FP_Q16_ERR_OVERFLOW, FP_Q16_ERR_BAD_PARAM)

        if status == FP_Q16_OK:
            assert len(prefix) == lane_count
            assert prefix[-1] == FP_Q16_ONE
            assert total == FP_Q16_ONE
            for i in range(1, lane_count):
                assert prefix[i] >= prefix[i - 1]


def run() -> None:
    test_source_contains_iq1134_function_and_correction_logic()
    test_nullptr_and_domain_failures()
    test_empty_input_contract()
    test_positive_diff_adds_to_last_lane()
    test_negative_diff_borrow_tail_first()
    test_overflow_guard_on_accumulation()
    test_randomized_invariants()
    print("softmax_q16_prefix_sum_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
