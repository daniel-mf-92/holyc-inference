#!/usr/bin/env python3
"""Harness for SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly (IQ-1102)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_OK,
    softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only,
)
from test_softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity import (
    softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity,
)


def softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
    logits_q16: list[int] | None,
    logits_count: int,
    top_k: int,
    workspace_capacity: int,
    out_required_workspace_cells: list[int] | None,
    out_selected_count: list[int] | None,
    out_max_logit_q16: list[int] | None,
    staged_fn=softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity,
    canonical_fn=softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only,
) -> int:
    if (
        logits_q16 is None
        or out_required_workspace_cells is None
        or out_selected_count is None
        or out_max_logit_q16 is None
    ):
        return FP_Q16_ERR_NULL_PTR

    if (
        out_required_workspace_cells is logits_q16
        or out_selected_count is logits_q16
        or out_max_logit_q16 is logits_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if (
        out_required_workspace_cells is out_selected_count
        or out_required_workspace_cells is out_max_logit_q16
        or out_selected_count is out_max_logit_q16
    ):
        return FP_Q16_ERR_BAD_PARAM

    if logits_count < 0 or top_k < 0 or workspace_capacity < 0:
        return FP_Q16_ERR_BAD_PARAM
    if top_k > logits_count:
        return FP_Q16_ERR_BAD_PARAM

    snapshot_logits_ref = logits_q16
    snapshot_logits_count = logits_count
    snapshot_top_k = top_k
    snapshot_workspace_capacity = workspace_capacity

    snapshot_out_required = out_required_workspace_cells
    snapshot_out_selected = out_selected_count
    snapshot_out_max = out_max_logit_q16

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

    if (
        snapshot_logits_ref is not logits_q16
        or snapshot_logits_count != logits_count
        or snapshot_top_k != top_k
        or snapshot_workspace_capacity != workspace_capacity
    ):
        return FP_Q16_ERR_BAD_PARAM

    if (
        snapshot_out_required is not out_required_workspace_cells
        or snapshot_out_selected is not out_selected_count
        or snapshot_out_max is not out_max_logit_q16
    ):
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


def test_source_contains_signature_and_composition_chain() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split(
        "I32 FPQ16TopKSelectLogitsCheckedNoAlias(",
        1,
    )[0]
    assert "SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (staged_required_workspace_cells < 0 ||" in body
    assert "if (staged_selected_count > staged_required_workspace_cells ||" in body
    assert "if (staged_required_workspace_cells > snapshot_workspace_capacity ||" in body
    assert "if (staged_required_workspace_cells != canonical_required_workspace_cells ||" in body


def test_known_vector() -> None:
    logits = [10, -20, 5, 10]
    out_required = [0xAAAA]
    out_selected = [0xBBBB]
    out_max = [0xCCCC]

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        logits,
        logits_count=4,
        top_k=2,
        workspace_capacity=6,
        out_required_workspace_cells=out_required,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
    )

    assert err == FP_Q16_OK
    assert out_required == [6]
    assert out_selected == [2]
    assert out_max == [10]


def test_error_no_partial_write() -> None:
    logits = [3, 1, 2]
    out_required = [0x1111]
    out_selected = [0x2222]
    out_max = [0x3333]

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        logits,
        logits_count=3,
        top_k=2,
        workspace_capacity=4,
        out_required_workspace_cells=out_required,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_required == [0x1111]
    assert out_selected == [0x2222]
    assert out_max == [0x3333]


def test_logits_output_alias_rejected() -> None:
    logits = [1, 2, 3]
    out_selected = [111]
    out_max = [222]

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        logits,
        logits_count=3,
        top_k=1,
        workspace_capacity=4,
        out_required_workspace_cells=logits,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_selected == [111]
    assert out_max == [222]


def test_fuzz_parity_vs_components() -> None:
    rng = random.Random(20260422_1102)

    for _ in range(512):
        logits_count = rng.randint(0, 64)
        logits = [rng.randint(-200_000, 200_000) for _ in range(logits_count)]
        top_k = 0 if logits_count == 0 else rng.randint(0, logits_count)

        required = logits_count + top_k
        workspace_capacity = required + rng.randint(-2, 3)
        if workspace_capacity < 0:
            workspace_capacity = 0

        out_required = [0xABCD]
        out_selected = [0xBCDE]
        out_max = [0xCDEF]

        err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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

        if required > workspace_capacity:
            assert err == FP_Q16_ERR_BAD_PARAM
            assert out_required == [0xABCD]
            assert out_selected == [0xBCDE]
            assert out_max == [0xCDEF]
            continue

        assert err == FP_Q16_OK
        assert out_required == [required]
        assert out_selected == [top_k]
        assert out_max == [max(logits)]


def test_rejects_negative_cardinality_from_staged_path() -> None:
    logits = [9, -2, 4]
    out_required = [0x1111]
    out_selected = [0x2222]
    out_max = [0x3333]

    def bad_staged(*_args, **_kwargs):
        _kwargs["out_required_workspace_cells"][0] = -1
        _kwargs["out_selected_count"][0] = 1
        _kwargs["out_max_logit_q16"][0] = 9
        return FP_Q16_OK

    def canonical(*_args, **_kwargs):
        _kwargs["out_required_workspace_cells"][0] = 4
        _kwargs["out_selected_count"][0] = 1
        _kwargs["out_max_logit_q16"][0] = 9
        return FP_Q16_OK

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        logits,
        logits_count=3,
        top_k=1,
        workspace_capacity=8,
        out_required_workspace_cells=out_required,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
        staged_fn=lambda l, lc, tk, wc, r, s, m: bad_staged(
            logits_q16=l,
            logits_count=lc,
            top_k=tk,
            workspace_capacity=wc,
            out_required_workspace_cells=r,
            out_selected_count=s,
            out_max_logit_q16=m,
        ),
        canonical_fn=lambda l, lc, tk, wc, r, s, m: canonical(
            logits_q16=l,
            logits_count=lc,
            top_k=tk,
            workspace_capacity=wc,
            out_required_workspace_cells=r,
            out_selected_count=s,
            out_max_logit_q16=m,
        ),
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_required == [0x1111]
    assert out_selected == [0x2222]
    assert out_max == [0x3333]


def test_rejects_workspace_oversubscription_before_publish() -> None:
    logits = [7, 6, 5, 4]
    out_required = [0xAAAA]
    out_selected = [0xBBBB]
    out_max = [0xCCCC]

    def staged_ok(*_args, **_kwargs):
        _kwargs["out_required_workspace_cells"][0] = 10
        _kwargs["out_selected_count"][0] = 2
        _kwargs["out_max_logit_q16"][0] = 7
        return FP_Q16_OK

    def canonical_ok(*_args, **_kwargs):
        _kwargs["out_required_workspace_cells"][0] = 10
        _kwargs["out_selected_count"][0] = 2
        _kwargs["out_max_logit_q16"][0] = 7
        return FP_Q16_OK

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        logits,
        logits_count=4,
        top_k=2,
        workspace_capacity=9,
        out_required_workspace_cells=out_required,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
        staged_fn=lambda l, lc, tk, wc, r, s, m: staged_ok(
            logits_q16=l,
            logits_count=lc,
            top_k=tk,
            workspace_capacity=wc,
            out_required_workspace_cells=r,
            out_selected_count=s,
            out_max_logit_q16=m,
        ),
        canonical_fn=lambda l, lc, tk, wc, r, s, m: canonical_ok(
            logits_q16=l,
            logits_count=lc,
            top_k=tk,
            workspace_capacity=wc,
            out_required_workspace_cells=r,
            out_selected_count=s,
            out_max_logit_q16=m,
        ),
    )

    assert err == FP_Q16_ERR_BAD_PARAM
    assert out_required == [0xAAAA]
    assert out_selected == [0xBBBB]
    assert out_max == [0xCCCC]


if __name__ == "__main__":
    test_source_contains_signature_and_composition_chain()
    test_known_vector()
    test_error_no_partial_write()
    test_logits_output_alias_rejected()
    test_fuzz_parity_vs_components()
    test_rejects_negative_cardinality_from_staged_path()
    test_rejects_workspace_oversubscription_before_publish()
    print("ok")
