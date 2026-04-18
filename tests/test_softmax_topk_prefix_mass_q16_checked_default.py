#!/usr/bin/env python3
"""Parity checks for FPQ16TopKPrefixMassQ16CheckedDefault semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_softmax_topk_prefix_mass_q16_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_OK,
    fpq16_topk_prefix_mass_q16_checked_reference,
)


def fpq16_topk_prefix_mass_q16_checked_default_reference(
    probs_q16: list[int] | None,
    lane_count: int,
    topk_indices: list[int] | None,
    k: int,
    out_prefix_mass_q16: list[int] | None,
    probs_addr: int = 0x1000,
    topk_indices_addr: int = 0x3000,
    out_prefix_mass_addr: int = 0x5000,
) -> int:
    return fpq16_topk_prefix_mass_q16_checked_reference(
        probs_q16=probs_q16,
        lane_count=lane_count,
        topk_indices=topk_indices,
        k=k,
        out_prefix_mass_q16=out_prefix_mass_q16,
        out_mass_capacity=k,
        probs_addr=probs_addr,
        topk_indices_addr=topk_indices_addr,
        out_prefix_mass_addr=out_prefix_mass_addr,
    )


def test_source_contains_default_wrapper_and_core_delegation() -> None:
    source = pathlib.Path("src/math/softmax.HC").read_text(encoding="utf-8")
    assert "FPQ16TopKPrefixMassQ16CheckedDefault" in source
    assert "FPQ16TopKPrefixMassQ16Checked(probs_q16," in source
    assert "out_prefix_mass_q16,\n                                         k);" in source


def test_default_matches_explicit_capacity_success_vector() -> None:
    probs = [18000, 16000, 12000, 11000, 8536]
    indices = [2, 0, 3]

    out_default = [777] * 6
    out_explicit = [777] * 6

    default_err = fpq16_topk_prefix_mass_q16_checked_default_reference(
        probs_q16=probs,
        lane_count=len(probs),
        topk_indices=indices,
        k=3,
        out_prefix_mass_q16=out_default,
    )
    explicit_err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs_q16=probs,
        lane_count=len(probs),
        topk_indices=indices,
        k=3,
        out_prefix_mass_q16=out_explicit,
        out_mass_capacity=3,
    )

    assert default_err == FP_Q16_OK
    assert explicit_err == FP_Q16_OK
    assert out_default == out_explicit
    assert out_default[:3] == [12000, 30000, 41000]
    assert out_default[3:] == [777, 777, 777]


def test_default_matches_explicit_capacity_error_and_no_partial() -> None:
    probs = [30000, 20000, 15536]
    indices = [0, 1, 2]

    out_default = [41, 42, 43]
    out_explicit = [41, 42, 43]

    default_err = fpq16_topk_prefix_mass_q16_checked_default_reference(
        probs_q16=probs,
        lane_count=3,
        topk_indices=indices,
        k=4,
        out_prefix_mass_q16=out_default,
    )
    explicit_err = fpq16_topk_prefix_mass_q16_checked_reference(
        probs_q16=probs,
        lane_count=3,
        topk_indices=indices,
        k=4,
        out_prefix_mass_q16=out_explicit,
        out_mass_capacity=4,
    )

    assert default_err == FP_Q16_ERR_BAD_PARAM
    assert explicit_err == FP_Q16_ERR_BAD_PARAM
    assert out_default == out_explicit == [41, 42, 43]


def test_default_vs_explicit_randomized_parity() -> None:
    rng = random.Random(356)

    for _ in range(2000):
        lane_count = rng.randint(0, 40)
        k = rng.randint(0, 48)

        if lane_count > 0:
            cuts = sorted(rng.sample(range(0, 65536 + 1), lane_count - 1))
            points = [0] + cuts + [65536]
            probs = [points[i + 1] - points[i] for i in range(lane_count)]
        else:
            probs = []

        indices = list(range(lane_count))
        rng.shuffle(indices)
        indices = indices[:k] if k <= lane_count else indices

        out_len = max(1, lane_count + 5)
        out_default = [903] * out_len
        out_explicit = [903] * out_len

        default_err = fpq16_topk_prefix_mass_q16_checked_default_reference(
            probs_q16=probs,
            lane_count=lane_count,
            topk_indices=indices,
            k=k,
            out_prefix_mass_q16=out_default,
        )
        explicit_err = fpq16_topk_prefix_mass_q16_checked_reference(
            probs_q16=probs,
            lane_count=lane_count,
            topk_indices=indices,
            k=k,
            out_prefix_mass_q16=out_explicit,
            out_mass_capacity=k,
        )

        assert default_err == explicit_err
        assert out_default == out_explicit


if __name__ == "__main__":
    test_source_contains_default_wrapper_and_core_delegation()
    test_default_matches_explicit_capacity_success_vector()
    test_default_matches_explicit_capacity_error_and_no_partial()
    test_default_vs_explicit_randomized_parity()
    print("ok")
