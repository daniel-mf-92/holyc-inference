#!/usr/bin/env python3
"""Parity checks for FPQ16TopKSelectLogitsCheckedNoAliasDefault semantics."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from test_softmax_topk_select_logits_checked_no_alias import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_OK,
    fpq16_topk_select_logits_checked_no_alias_reference,
    stable_topk_pairs_reference,
)


def fpq16_topk_select_logits_checked_no_alias_default_reference(
    logits_q16: list[int] | None,
    lane_count: int,
    k: int,
    out_topk_logits_q16: list[int] | None,
    out_topk_indices: list[int] | None,
    logits_addr: int = 0x1000,
    out_logits_addr: int = 0x3000,
    out_indices_addr: int = 0x5000,
) -> int:
    return fpq16_topk_select_logits_checked_no_alias_reference(
        logits_q16=logits_q16,
        lane_count=lane_count,
        k=k,
        out_topk_logits_q16=out_topk_logits_q16,
        out_topk_indices=out_topk_indices,
        out_lane_capacity=k,
        logits_addr=logits_addr,
        out_logits_addr=out_logits_addr,
        out_indices_addr=out_indices_addr,
    )


def test_source_contains_default_wrapper_and_core_delegation() -> None:
    source = pathlib.Path("src/math/softmax.HC").read_text(encoding="utf-8")
    assert "FPQ16TopKSelectLogitsCheckedNoAliasDefault" in source
    assert "FPQ16TopKSelectLogitsCheckedNoAlias(logits_q16," in source
    assert "out_topk_indices,\n                                               k);" in source


def test_default_matches_explicit_capacity_core_on_success_vectors() -> None:
    logits = [1400, 4200, 4200, -200, 1000, 17]
    out_logits_default = [-11] * 6
    out_indices_default = [-22] * 6
    out_logits_explicit = [-11] * 6
    out_indices_explicit = [-22] * 6

    default_err = fpq16_topk_select_logits_checked_no_alias_default_reference(
        logits_q16=logits,
        lane_count=len(logits),
        k=4,
        out_topk_logits_q16=out_logits_default,
        out_topk_indices=out_indices_default,
    )
    explicit_err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits_q16=logits,
        lane_count=len(logits),
        k=4,
        out_topk_logits_q16=out_logits_explicit,
        out_topk_indices=out_indices_explicit,
        out_lane_capacity=4,
    )

    assert default_err == FP_Q16_OK
    assert explicit_err == FP_Q16_OK
    assert out_logits_default == out_logits_explicit
    assert out_indices_default == out_indices_explicit

    exp_idx, exp_logits = stable_topk_pairs_reference(logits, 4)
    assert out_indices_default[:4] == exp_idx
    assert out_logits_default[:4] == exp_logits
    assert out_indices_default[4:] == [-22, -22]
    assert out_logits_default[4:] == [-11, -11]


def test_default_matches_explicit_capacity_error_and_no_partial_write() -> None:
    logits = [9, 8, 7]
    out_logits_default = [301, 302, 303]
    out_indices_default = [401, 402, 403]
    out_logits_explicit = [301, 302, 303]
    out_indices_explicit = [401, 402, 403]

    default_err = fpq16_topk_select_logits_checked_no_alias_default_reference(
        logits_q16=logits,
        lane_count=3,
        k=4,
        out_topk_logits_q16=out_logits_default,
        out_topk_indices=out_indices_default,
    )
    explicit_err = fpq16_topk_select_logits_checked_no_alias_reference(
        logits_q16=logits,
        lane_count=3,
        k=4,
        out_topk_logits_q16=out_logits_explicit,
        out_topk_indices=out_indices_explicit,
        out_lane_capacity=4,
    )

    assert default_err == FP_Q16_ERR_BAD_PARAM
    assert explicit_err == FP_Q16_ERR_BAD_PARAM
    assert out_logits_default == out_logits_explicit == [301, 302, 303]
    assert out_indices_default == out_indices_explicit == [401, 402, 403]


def test_default_vs_explicit_randomized_parity() -> None:
    rng = random.Random(350)

    for _ in range(250):
        lane_count = rng.randint(0, 16)
        logits = [rng.randint(-8000, 8000) for _ in range(lane_count)]
        k = rng.randint(0, 18)

        out_len = max(1, lane_count + 3)
        out_logits_default = [777] * out_len
        out_indices_default = [888] * out_len
        out_logits_explicit = [777] * out_len
        out_indices_explicit = [888] * out_len

        default_err = fpq16_topk_select_logits_checked_no_alias_default_reference(
            logits_q16=logits,
            lane_count=lane_count,
            k=k,
            out_topk_logits_q16=out_logits_default,
            out_topk_indices=out_indices_default,
            logits_addr=0x1000,
            out_logits_addr=0x3000,
            out_indices_addr=0x5000,
        )
        explicit_err = fpq16_topk_select_logits_checked_no_alias_reference(
            logits_q16=logits,
            lane_count=lane_count,
            k=k,
            out_topk_logits_q16=out_logits_explicit,
            out_topk_indices=out_indices_explicit,
            out_lane_capacity=k,
            logits_addr=0x1000,
            out_logits_addr=0x3000,
            out_indices_addr=0x5000,
        )

        assert default_err == explicit_err
        assert out_logits_default == out_logits_explicit
        assert out_indices_default == out_indices_explicit


if __name__ == "__main__":
    test_source_contains_default_wrapper_and_core_delegation()
    test_default_matches_explicit_capacity_core_on_success_vectors()
    test_default_matches_explicit_capacity_error_and_no_partial_write()
    test_default_vs_explicit_randomized_parity()
    print("ok")
