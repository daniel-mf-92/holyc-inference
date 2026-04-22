#!/usr/bin/env python3
"""Harness for SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly (IQ-1107)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_OK,
)
from test_softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only import (
    softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only,
)
from test_softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
)


def softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
    logits_q16: list[int],
    logits_count: int,
    top_k: int,
    workspace_capacity: int,
    out_required_workspace_cells: list[int],
    out_selected_count: list[int],
    out_max_logit_q16: list[int],
    staged_fn=softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
    canonical_fn=softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only,
) -> int:
    staged_required = [0]
    staged_selected = [0]
    staged_max = [0]

    err = staged_fn(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        staged_required,
        staged_selected,
        staged_max,
    )
    if err != FP_Q16_OK:
        return err

    canonical_required = [0]
    canonical_selected = [0]
    canonical_max = [0]

    err = canonical_fn(
        logits_q16,
        logits_count,
        top_k,
        workspace_capacity,
        canonical_required,
        canonical_selected,
        canonical_max,
    )
    if err != FP_Q16_OK:
        return err

    if staged_selected[0] != top_k or canonical_selected[0] != top_k:
        return FP_Q16_ERR_BAD_PARAM

    if staged_required[0] < logits_count or canonical_required[0] < logits_count:
        return FP_Q16_ERR_BAD_PARAM

    if staged_required[0] > workspace_capacity or canonical_required[0] > workspace_capacity:
        return FP_Q16_ERR_BAD_PARAM

    if (
        staged_required[0] != canonical_required[0]
        or staged_selected[0] != canonical_selected[0]
        or staged_max[0] != canonical_max[0]
    ):
        return FP_Q16_ERR_BAD_PARAM

    out_required_workspace_cells[0] = staged_required[0]
    out_selected_count[0] = staged_selected[0]
    out_max_logit_q16[0] = staged_max[0]
    return FP_Q16_OK


def test_source_contains_signature_and_chain() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split("I32 FPQ16TopKSelectLogitsCheckedNoAlias(", 1)[0]
    assert "SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (staged_selected_count != snapshot_top_k ||" in body


def test_known_vector() -> None:
    logits = [13, -4, 8, 13]
    out_required = [0xAAAA]
    out_selected = [0xBBBB]
    out_max = [0xCCCC]

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
        logits,
        logits_count=4,
        top_k=3,
        workspace_capacity=7,
        out_required_workspace_cells=out_required,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
    )

    assert err == FP_Q16_OK
    assert out_required == [7]
    assert out_selected == [3]
    assert out_max == [13]


def test_fuzz() -> None:
    rng = random.Random(20260422_1107)
    for _ in range(256):
        logits_count = rng.randint(0, 64)
        logits = [rng.randint(-200000, 200000) for _ in range(logits_count)]
        top_k = 0 if logits_count == 0 else rng.randint(0, logits_count)
        req = logits_count + top_k
        workspace_capacity = max(0, req + rng.randint(-2, 3))

        out_required = [111]
        out_selected = [222]
        out_max = [333]
        err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only(
            logits,
            logits_count,
            top_k,
            workspace_capacity,
            out_required,
            out_selected,
            out_max,
        )

        if logits_count == 0:
            assert err == FP_Q16_OK
            assert out_required == [0]
            assert out_selected == [0]
            assert out_max == [0]
            continue

        if req > workspace_capacity:
            assert err == FP_Q16_ERR_BAD_PARAM
            continue

        assert err == FP_Q16_OK
        assert out_required == [req]
        assert out_selected == [top_k]
        assert out_max == [max(logits)]


if __name__ == "__main__":
    test_source_contains_signature_and_chain()
    test_known_vector()
    test_fuzz()
    print("ok")
