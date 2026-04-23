#!/usr/bin/env python3
"""Reference checks for FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnlyParity (IQ-1193)."""

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
    fpq16_softmax_prefix_sum_checked_reference,
)
from test_softmax_q16_prefix_sum_checked_commit_only_preflight_only import (  # noqa: E402
    fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference,
    fpq16_softmax_prefix_sum_checked_commit_only_reference,
)


def fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_parity_reference(
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

    snapshot_lane_count = lane_count
    snapshot_out_total = out_total_q16
    snapshot_out_last = out_last_prefix_q16

    staged_total = [0]
    staged_last = [0]
    status = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_reference(
        probs_q16,
        snapshot_lane_count,
        prefix_sums_q16,
        staged_total,
        staged_last,
    )
    if status != FP_Q16_OK:
        return status

    canonical_total = [0]
    canonical_last = [0]
    status = fpq16_softmax_prefix_sum_checked_commit_only_reference(
        probs_q16,
        snapshot_lane_count,
        prefix_sums_q16,
        canonical_total,
        canonical_last,
    )
    if status != FP_Q16_OK:
        return status

    if (
        snapshot_lane_count != lane_count
        or snapshot_out_total is not out_total_q16
        or snapshot_out_last is not out_last_prefix_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if staged_total[0] != canonical_total[0] or staged_last[0] != canonical_last[0]:
        return FP_Q16_ERR_OVERFLOW

    out_total_q16[0] = staged_total[0]
    out_last_prefix_q16[0] = staged_last[0]
    return FP_Q16_OK


def test_source_contains_iq1193_signature_and_parity_path() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnlyParity(I64 *probs_q16,"
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split(
        "// Checked composed helper for alias-permitted split-phase softmax.",
        1,
    )[0]
    assert "status = FPQ16SoftmaxPrefixSumCheckedCommitOnlyPreflightOnly(" in body
    assert "status = FPQ16SoftmaxPrefixSumCheckedCommitOnly(probs_q16," in body
    assert "snapshot_lane_count = lane_count;" in body
    assert "if (snapshot_lane_count != lane_count ||" in body
    assert "if (staged_total_q16 != canonical_total_q16 ||" in body
    assert "*out_total_q16 = staged_total_q16;" in body
    assert "*out_last_prefix_q16 = staged_last_prefix_q16;" in body


def test_null_and_badparam_guards() -> None:
    prefix = [0xAAAA, 0xBBBB]
    out_total = [0xCCCC]
    out_last = [0xDDDD]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_parity_reference(
        None, 2, prefix, out_total, out_last
    )
    assert st == FP_Q16_ERR_NULL_PTR
    assert prefix == [0xAAAA, 0xBBBB]
    assert out_total == [0xCCCC]
    assert out_last == [0xDDDD]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_parity_reference(
        [100, 200], -1, prefix, out_total, out_last
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert prefix == [0xAAAA, 0xBBBB]
    assert out_total == [0xCCCC]
    assert out_last == [0xDDDD]


def test_success_known_vector_and_tuple_parity() -> None:
    probs = [10_000, 12_000, 13_000]
    prefix = [0x1234, 0x5678, 0x9ABC]
    out_total = [0x1357]
    out_last = [0x2468]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_parity_reference(
        probs,
        len(probs),
        prefix,
        out_total,
        out_last,
    )
    assert st == FP_Q16_OK

    base_status, base_prefix, base_total = fpq16_softmax_prefix_sum_checked_reference(
        probs, len(probs)
    )
    assert base_status == FP_Q16_OK
    assert out_total[0] == base_total
    assert out_last[0] == base_prefix[-1]


def test_error_propagation_and_no_partial_publish() -> None:
    probs = [10_000, -5, 30_000]
    prefix = [0xDEAD, 0xBEEF, 0xFACE]
    out_total = [0x1111]
    out_last = [0x2222]

    st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_parity_reference(
        probs,
        len(probs),
        prefix,
        out_total,
        out_last,
    )
    assert st == FP_Q16_ERR_BAD_PARAM
    assert prefix == [0xDEAD, 0xBEEF, 0xFACE]
    assert out_total == [0x1111]
    assert out_last == [0x2222]


def test_randomized_parity_against_canonical() -> None:
    rng = random.Random(20260423_1193)

    for _ in range(5000):
        lanes = rng.randint(0, 64)
        probs = [rng.randint(0, (1 << 16) * 2) for _ in range(lanes)]
        if lanes and rng.random() < 0.08:
            probs[rng.randrange(lanes)] = -rng.randint(1, 100)

        prefix = [0x7A7A7A7A] * lanes
        out_total = [0x5B5B5B5B]
        out_last = [0x6C6C6C6C]

        expected_status, expected_prefix, expected_total = (
            fpq16_softmax_prefix_sum_checked_reference(probs, lanes)
        )

        st = fpq16_softmax_prefix_sum_checked_commit_only_preflight_only_parity_reference(
            probs,
            lanes,
            prefix,
            out_total,
            out_last,
        )
        assert st == expected_status

        if st == FP_Q16_OK:
            assert out_total[0] == expected_total
            assert out_last[0] == (expected_prefix[-1] if lanes else 0)
        else:
            assert out_total[0] == 0x5B5B5B5B
            assert out_last[0] == 0x6C6C6C6C


def run() -> None:
    test_source_contains_iq1193_signature_and_parity_path()
    test_null_and_badparam_guards()
    test_success_known_vector_and_tuple_parity()
    test_error_propagation_and_no_partial_publish()
    test_randomized_parity_against_canonical()
    print("softmax_q16_prefix_sum_checked_commit_only_preflight_only_parity_reference_checks=ok")


if __name__ == "__main__":
    run()
