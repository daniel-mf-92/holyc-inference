#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnly (IQ-1150)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_softmax_q16_prefix_sum_checked import (  # noqa: E402
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    fpq16_softmax_prefix_sum_checked_reference,
)


def fpq16_softmax_prefix_sum_checked_commit_only_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    prefix_sums_q16: list[int] | None,
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
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM

    if lane_count == 0:
        out_total_q16[0] = 0
        out_last_prefix_q16[0] = 0
        return FP_Q16_OK

    if lane_count > len(probs_q16) or lane_count > len(prefix_sums_q16):
        return FP_Q16_ERR_BAD_PARAM

    status, staged_prefix, staged_total = fpq16_softmax_prefix_sum_checked_reference(
        probs_q16[:lane_count], lane_count
    )
    if status != FP_Q16_OK:
        return status

    for i in range(lane_count):
        prefix_sums_q16[i] = staged_prefix[i]

    out_total_q16[0] = staged_total
    out_last_prefix_q16[0] = staged_prefix[-1]
    return FP_Q16_OK


def fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    prefix_sums_q16: list[int] | None,
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
    if lane_count < 0:
        return FP_Q16_ERR_BAD_PARAM

    out_total_q16[0] = 0
    out_last_prefix_q16[0] = 0

    if lane_count == 0:
        return FP_Q16_OK

    if lane_count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW

    commit_prefix = [0] * lane_count
    commit_total = [0]
    commit_last = [0]

    status = fpq16_softmax_prefix_sum_checked_commit_only_reference(
        probs_q16,
        lane_count,
        commit_prefix,
        commit_total,
        commit_last,
    )
    if status != FP_Q16_OK:
        return status

    status, checked_prefix, checked_total = fpq16_softmax_prefix_sum_checked_reference(
        probs_q16[:lane_count], lane_count
    )
    if status != FP_Q16_OK:
        return status

    checked_last = checked_prefix[-1]
    if commit_total[0] != checked_total or commit_last[0] != checked_last:
        return FP_Q16_ERR_OVERFLOW

    out_total_q16[0] = commit_total[0]
    out_last_prefix_q16[0] = commit_last[0]
    return FP_Q16_OK


def test_source_contains_iq1150_signature_and_zero_write_contract() -> None:
    src = Path("src/math/softmax.HC").read_text(encoding="utf-8")

    commit_sig = "I32 FPQ16SoftmaxPrefixSumCheckedCommitOnly(I64 *probs_q16,"
    assert commit_sig in src
    commit_body = src.split(commit_sig, 1)[1].split(
        "I32 FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnly(I64 *probs_q16,", 1
    )[0]
    assert "status = FPQ16SoftmaxPrefixSumChecked(snapshot_probs_q16," in commit_body
    assert "staged_last_prefix_q16 = staged_prefix_q16[snapshot_lane_count - 1];" in commit_body
    assert "prefix_sums_q16[i] = staged_prefix_q16[i];" in commit_body

    preflight_sig = "I32 FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnly(I64 *probs_q16,"
    assert preflight_sig in src
    preflight_body = src.split(preflight_sig, 1)[1].split(
        "// Checked composed helper for alias-permitted split-phase softmax.", 1
    )[0]
    assert "*out_total_q16 = 0;" in preflight_body
    assert "*out_last_prefix_q16 = 0;" in preflight_body
    assert "status = FPQ16SoftmaxPrefixSumCheckedCommitOnly(probs_q16," in preflight_body
    assert "status = FPQ16SoftmaxPrefixSumChecked(probs_q16," in preflight_body
    assert "if (staged_total_q16 != canonical_total_q16 ||" in preflight_body
    assert "staged_last_prefix_q16 != canonical_last_prefix_q16)" in preflight_body
    assert "prefix_sums_q16[i] =" not in preflight_body


def test_null_and_bad_param_guards() -> None:
    sentinel_prefix = [0xAA, 0xBB]
    out_total = [0x11]
    out_last = [0x22]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
        None, 2, sentinel_prefix, out_total, out_last
    )
    assert st == FP_Q16_ERR_NULL_PTR
    assert sentinel_prefix == [0xAA, 0xBB]
    assert out_total == [0x11]
    assert out_last == [0x22]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
        [100, 200], -1, sentinel_prefix, out_total, out_last
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert sentinel_prefix == [0xAA, 0xBB]
    assert out_total == [0x11]
    assert out_last == [0x22]


def test_zero_lane_success_and_no_prefix_mutation() -> None:
    prefix = [0x7777, 0x8888]
    out_total = [0x9999]
    out_last = [0xAAAA]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
        [], 0, prefix, out_total, out_last
    )
    assert st == FP_Q16_OK
    assert prefix == [0x7777, 0x8888]
    assert out_total == [0]
    assert out_last == [0]


def test_error_paths_zero_outputs_and_preserve_prefix() -> None:
    prefix = [0x1357, 0x2468, 0x369A]
    out_total = [0xDEAD]
    out_last = [0xBEEF]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
        [10_000, -1, 30_000], 3, prefix, out_total, out_last
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert prefix == [0x1357, 0x2468, 0x369A]
    assert out_total == [0]
    assert out_last == [0]


def test_success_vectors_match_canonical_summary_and_preserve_prefix() -> None:
    probs = [10_000, 12_000, 13_000]
    prefix = [0x4444, 0x5555, 0x6666]
    out_total = [0x7777]
    out_last = [0x8888]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
        probs, len(probs), prefix, out_total, out_last
    )
    assert st == FP_Q16_OK
    assert prefix == [0x4444, 0x5555, 0x6666]

    exp_status, exp_prefix, exp_total = fpq16_softmax_prefix_sum_checked_reference(
        probs, len(probs)
    )
    assert exp_status == FP_Q16_OK
    assert out_total[0] == exp_total
    assert out_last[0] == exp_prefix[-1]


def test_randomized_summary_parity_and_no_prefix_writes() -> None:
    rng = random.Random(20260422_1150)
    for _ in range(6000):
        lanes = rng.randint(0, 64)
        probs = [rng.randint(0, (1 << 16) * 2) for _ in range(lanes)]
        prefix = [0x7A7A7A7A] * lanes
        out_total = [0x5B5B5B5B]
        out_last = [0x6C6C6C6C]

        expected_status, expected_prefix, expected_total = (
            fpq16_softmax_prefix_sum_checked_reference(probs, lanes)
        )

        st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
            probs, lanes, prefix, out_total, out_last
        )
        assert st == expected_status
        assert prefix == [0x7A7A7A7A] * lanes

        if st == FP_Q16_OK:
            assert out_total[0] == expected_total
            assert out_last[0] == (expected_prefix[-1] if lanes else 0)
        else:
            assert out_total[0] == 0
            assert out_last[0] == 0


def run() -> None:
    test_source_contains_iq1150_signature_and_zero_write_contract()
    test_null_and_bad_param_guards()
    test_zero_lane_success_and_no_prefix_mutation()
    test_error_paths_zero_outputs_and_preserve_prefix()
    test_success_vectors_match_canonical_summary_and_preserve_prefix()
    test_randomized_summary_parity_and_no_prefix_writes()
    print("softmax_q16_prefix_sum_checked_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
