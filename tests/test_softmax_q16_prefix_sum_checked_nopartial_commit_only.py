#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxPrefixSumCheckedNoPartialCommitOnly (IQ-1192)."""

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
    probs_q16: list[int], lane_count: int
) -> tuple[int, list[int], int]:
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

    if snapshot_lane_count > snapshot_probs_capacity:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_lane_count > snapshot_prefix_capacity:
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_lane_count > len(probs_q16):
        return FP_Q16_ERR_BAD_PARAM
    if snapshot_lane_count > len(prefix_sums_q16):
        return FP_Q16_ERR_BAD_PARAM

    status, staged_prefix, staged_total = fpq16_softmax_prefix_sum_checked_reference(
        probs_q16[:snapshot_lane_count], snapshot_lane_count
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


def fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    probs_capacity: int,
    prefix_sums_q16: list[int] | None,
    prefix_capacity: int,
    out_total_q16: list[int] | None,
    out_last_prefix_q16: list[int] | None,
) -> int:
    if (
        probs_q16 is None
        or prefix_sums_q16 is None
        or out_total_q16 is None
        or out_last_prefix_q16 is None
    ):
        return FP_Q16_ERR_NULL_PTR
    if lane_count < 0 or probs_capacity < 0 or prefix_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM

    snapshot_lane_count = lane_count
    snapshot_probs_capacity = probs_capacity
    snapshot_prefix_capacity = prefix_capacity

    staged_total = [0]
    status = fpq16_softmax_prefix_sum_checked_nopartial_reference(
        probs_q16,
        snapshot_lane_count,
        snapshot_probs_capacity,
        prefix_sums_q16,
        snapshot_prefix_capacity,
        staged_total,
    )
    if status != FP_Q16_OK:
        return status

    staged_last = prefix_sums_q16[snapshot_lane_count - 1] if snapshot_lane_count else 0

    status, recomputed_prefix, recomputed_total = fpq16_softmax_prefix_sum_checked_reference(
        probs_q16[:snapshot_lane_count], snapshot_lane_count
    )
    if status != FP_Q16_OK:
        return status
    recomputed_last = recomputed_prefix[-1] if snapshot_lane_count else 0

    if (
        snapshot_lane_count != lane_count
        or snapshot_probs_capacity != probs_capacity
        or snapshot_prefix_capacity != prefix_capacity
    ):
        return FP_Q16_ERR_BAD_PARAM

    if staged_total[0] != recomputed_total or staged_last != recomputed_last:
        return FP_Q16_ERR_OVERFLOW

    out_total_q16[0] = staged_total[0]
    out_last_prefix_q16[0] = staged_last
    return FP_Q16_OK


def test_source_contains_iq1192_signature_and_parity_contract() -> None:
    src = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SoftmaxPrefixSumCheckedNoPartialCommitOnly(I64 *probs_q16,"
    assert sig in src
    body = src.split(sig, 1)[1].split(
        "// Commit-only wrapper over FPQ16SoftmaxPrefixSumChecked.", 1
    )[0]
    assert "snapshot_lane_count = lane_count;" in body
    assert "snapshot_probs_capacity = probs_capacity;" in body
    assert "snapshot_prefix_capacity = prefix_capacity;" in body
    assert "status = FPQ16SoftmaxPrefixSumCheckedNoPartial(snapshot_probs_q16," in body
    assert "status = FPQ16SoftmaxPrefixSumChecked(snapshot_probs_q16," in body
    assert "if (staged_total_q16 != recomputed_total_q16 ||" in body
    assert "staged_last_prefix_q16 != recomputed_last_prefix_q16)" in body
    assert "*out_total_q16 = staged_total_q16;" in body
    assert "*out_last_prefix_q16 = staged_last_prefix_q16;" in body


def test_null_and_bad_param_guards_preserve_outputs() -> None:
    probs = [1000, 2000]
    prefix = [0x1111, 0x2222]
    out_total = [0x3333]
    out_last = [0x4444]

    st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
        None, 2, 2, prefix, 2, out_total, out_last
    )
    assert st == FP_Q16_ERR_NULL_PTR
    assert out_total == [0x3333]
    assert out_last == [0x4444]

    st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
        probs, -1, 2, prefix, 2, out_total, out_last
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert out_total == [0x3333]
    assert out_last == [0x4444]


def test_capacity_and_negative_lane_errors_preserve_outputs() -> None:
    out_total = [0xA5A5]
    out_last = [0x5A5A]

    st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
        [100, 200, 300], 3, 2, [9, 9, 9], 3, out_total, out_last
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert out_total == [0xA5A5]
    assert out_last == [0x5A5A]

    st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
        [100, -1, 300], 3, 3, [9, 9, 9], 3, out_total, out_last
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert out_total == [0xA5A5]
    assert out_last == [0x5A5A]


def test_zero_lane_success() -> None:
    prefix = [0xABCD, 0x1234]
    out_total = [0xBBBB]
    out_last = [0xCCCC]

    st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
        [], 0, 0, prefix, 0, out_total, out_last
    )
    assert st == FP_Q16_OK
    assert out_total[0] == 0
    assert out_last[0] == 0
    assert prefix == [0xABCD, 0x1234]


def test_success_vectors_and_randomized_parity() -> None:
    probs = [10_000, 12_000, 13_000]
    prefix = [0, 0, 0]
    out_total = [0]
    out_last = [0]

    st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
        probs,
        len(probs),
        len(probs),
        prefix,
        len(prefix),
        out_total,
        out_last,
    )
    assert st == FP_Q16_OK

    exp_status, exp_prefix, exp_total = fpq16_softmax_prefix_sum_checked_reference(
        probs, len(probs)
    )
    assert exp_status == FP_Q16_OK
    assert prefix == exp_prefix
    assert out_total[0] == exp_total
    assert out_last[0] == exp_prefix[-1]

    rng = random.Random(20260423_1192)
    for _ in range(7000):
        lanes = rng.randint(0, 64)
        probs = [rng.randint(0, FP_Q16_ONE * 2) for _ in range(lanes)]
        prefix = [0x7F7F7F7F] * max(lanes, 1)
        out_total = [0x41414141]
        out_last = [0x52525252]

        expected_status, expected_prefix, expected_total = fpq16_softmax_prefix_sum_checked_reference(
            probs, lanes
        )

        st = fpq16_softmax_prefix_sum_checked_nopartial_commit_only_reference(
            probs,
            lanes,
            lanes,
            prefix,
            lanes,
            out_total,
            out_last,
        )
        assert st == expected_status

        if st == FP_Q16_OK:
            assert prefix[:lanes] == expected_prefix
            assert out_total[0] == expected_total
            assert out_last[0] == (expected_prefix[-1] if lanes else 0)
        else:
            assert out_total[0] == 0x41414141
            assert out_last[0] == 0x52525252


def run() -> None:
    test_source_contains_iq1192_signature_and_parity_contract()
    test_null_and_bad_param_guards_preserve_outputs()
    test_capacity_and_negative_lane_errors_preserve_outputs()
    test_zero_lane_success()
    test_success_vectors_and_randomized_parity()
    print("softmax_q16_prefix_sum_checked_nopartial_commit_only=ok")


if __name__ == "__main__":
    run()
