#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxPrefixSumChecked (IQ-1134)."""

from __future__ import annotations

from pathlib import Path
import random

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

    corrected_lanes = list(probs_q16)
    diff_q16 = FP_Q16_ONE - running_sum_q16

    if diff_q16 > 0:
        if corrected_lanes[-1] > I64_MAX_VALUE - diff_q16:
            return FP_Q16_ERR_OVERFLOW, [], 0
        corrected_lanes[-1] += diff_q16
    elif diff_q16 < 0:
        borrow_q16 = -diff_q16
        for idx in range(lane_count - 1, -1, -1):
            if borrow_q16 <= 0:
                break
            take_q16 = min(corrected_lanes[idx], borrow_q16)
            corrected_lanes[idx] -= take_q16
            borrow_q16 -= take_q16
        if borrow_q16 != 0:
            return FP_Q16_ERR_BAD_PARAM, [], 0

    corrected_total = sum(corrected_lanes)
    if corrected_total != FP_Q16_ONE:
        return FP_Q16_ERR_OVERFLOW, [], 0

    prefix_sums_q16: list[int] = []
    prefix_running = 0
    for lane_q16 in corrected_lanes:
        if lane_q16 < 0:
            return FP_Q16_ERR_BAD_PARAM, [], 0
        if lane_q16 > I64_MAX_VALUE - prefix_running:
            return FP_Q16_ERR_OVERFLOW, [], 0
        prefix_running += lane_q16
        prefix_sums_q16.append(prefix_running)

    if prefix_running != FP_Q16_ONE:
        return FP_Q16_ERR_OVERFLOW, [], 0

    return FP_Q16_OK, prefix_sums_q16, prefix_running


def test_source_contains_iq1134_function_and_correction_paths() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SoftmaxPrefixSumChecked(I64 *probs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16SoftmaxFromPreclampedChecked", 1)[0]
    assert "diff_q16 = FP_Q16_ONE - running_sum_q16;" in body
    assert "if (diff_q16 > 0)" in body
    assert "else if (diff_q16 < 0)" in body
    assert "for (i = lane_count - 1; i >= 0; i--)" in body
    assert "// Preflight source lanes and total before any output write." in body
    assert "if (running_sum_q16 != FP_Q16_ONE)" in body


def test_null_and_badparam_guards() -> None:
    status, _, _ = fpq16_softmax_prefix_sum_checked_reference(
        [FP_Q16_ONE], 1, probs_present=False
    )
    assert status == FP_Q16_ERR_NULL_PTR

    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([FP_Q16_ONE], -1)
    assert status == FP_Q16_ERR_BAD_PARAM

    status, out, total = fpq16_softmax_prefix_sum_checked_reference([], 0)
    assert status == FP_Q16_OK
    assert out == []
    assert total == 0


def test_exact_one_passthrough() -> None:
    probs = [FP_Q16_ONE // 4, FP_Q16_ONE // 2, FP_Q16_ONE // 4]
    status, out, total = fpq16_softmax_prefix_sum_checked_reference(probs, len(probs))
    assert status == FP_Q16_OK
    assert out == [FP_Q16_ONE // 4, (FP_Q16_ONE * 3) // 4, FP_Q16_ONE]
    assert total == FP_Q16_ONE


def test_positive_diff_tail_injection() -> None:
    probs = [10_000, 20_000, 30_000]
    status, out, total = fpq16_softmax_prefix_sum_checked_reference(probs, len(probs))
    assert status == FP_Q16_OK
    assert out[-1] == FP_Q16_ONE
    assert total == FP_Q16_ONE


def test_negative_diff_reverse_borrow() -> None:
    probs = [40_000, 40_000, 20_000]
    status, out, total = fpq16_softmax_prefix_sum_checked_reference(probs, len(probs))
    assert status == FP_Q16_OK
    assert out[-1] == FP_Q16_ONE
    assert total == FP_Q16_ONE


def test_negative_diff_reverse_borrow_spills_across_multiple_tail_lanes() -> None:
    probs = [50_000, 30_000, 1_000]
    status, out, total = fpq16_softmax_prefix_sum_checked_reference(probs, len(probs))
    assert status == FP_Q16_OK
    assert out == [50_000, FP_Q16_ONE, FP_Q16_ONE]
    assert total == FP_Q16_ONE


def test_positive_diff_injects_entire_remainder_into_tail_lane_only() -> None:
    probs = [10_000, 12_000, 13_000]
    status, out, total = fpq16_softmax_prefix_sum_checked_reference(probs, len(probs))
    assert status == FP_Q16_OK
    assert out == [10_000, 22_000, FP_Q16_ONE]
    assert total == FP_Q16_ONE


def test_negative_input_and_sum_overflow_rejected() -> None:
    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([1, -1], 2)
    assert status == FP_Q16_ERR_BAD_PARAM

    status, _, _ = fpq16_softmax_prefix_sum_checked_reference([I64_MAX_VALUE, 1], 2)
    assert status == FP_Q16_ERR_OVERFLOW


def test_randomized_invariants() -> None:
    rng = random.Random(20260422_1134)

    for _ in range(5000):
        lanes = rng.randint(1, 64)
        probs = [rng.randint(0, FP_Q16_ONE * 2) for _ in range(lanes)]

        status, out, total = fpq16_softmax_prefix_sum_checked_reference(probs, lanes)
        assert status in (FP_Q16_OK, FP_Q16_ERR_BAD_PARAM, FP_Q16_ERR_OVERFLOW)

        if status == FP_Q16_OK:
            assert len(out) == lanes
            assert out[-1] == FP_Q16_ONE
            assert total == FP_Q16_ONE
            assert all(out[i] >= out[i - 1] for i in range(1, len(out)))


def run() -> None:
    test_source_contains_iq1134_function_and_correction_paths()
    test_null_and_badparam_guards()
    test_exact_one_passthrough()
    test_positive_diff_tail_injection()
    test_negative_diff_reverse_borrow()
    test_negative_diff_reverse_borrow_spills_across_multiple_tail_lanes()
    test_positive_diff_injects_entire_remainder_into_tail_lane_only()
    test_negative_input_and_sum_overflow_rejected()
    test_randomized_invariants()
    print("softmax_q16_prefix_sum_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
