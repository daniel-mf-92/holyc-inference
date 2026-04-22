#!/usr/bin/env python3
"""Harness for SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly (IQ-1083)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only(
    logits_q16: list[int] | None,
    logits_count: int,
    top_k: int,
    workspace_capacity: int,
    out_required_workspace_cells: list[int] | None,
    out_selected_count: list[int] | None,
    out_max_logit_q16: list[int] | None,
    logits_addr: int = 0,
) -> int:
    if (
        logits_q16 is None
        or out_required_workspace_cells is None
        or out_selected_count is None
        or out_max_logit_q16 is None
    ):
        return FP_Q16_ERR_NULL_PTR

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

    if logits_count == 0:
        out_required_workspace_cells[0] = 0
        out_selected_count[0] = 0
        out_max_logit_q16[0] = 0
        return FP_Q16_OK

    last_index = logits_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW

    if len(logits_q16) < logits_count:
        return FP_Q16_ERR_BAD_PARAM

    if logits_count > (I64_MAX_VALUE - top_k):
        return FP_Q16_ERR_OVERFLOW
    required_workspace = logits_count + top_k
    if required_workspace > workspace_capacity:
        return FP_Q16_ERR_BAD_PARAM

    staged_max = logits_q16[0]
    for i in range(1, logits_count):
        if logits_q16[i] > staged_max:
            staged_max = logits_q16[i]

    for i in range(logits_count):
        lane = logits_q16[i]
        if lane < 0 and staged_max > (I64_MAX_VALUE + lane):
            return FP_Q16_ERR_OVERFLOW
        delta = staged_max - lane
        if delta < 0:
            return FP_Q16_ERR_BAD_PARAM

    out_required_workspace_cells[0] = required_workspace
    out_selected_count[0] = top_k
    out_max_logit_q16[0] = staged_max
    return FP_Q16_OK


def test_source_contains_iq_1083_symbol() -> None:
    source = Path("src/math/softmax.HC").read_text(encoding="utf-8")
    sig = "I32 SoftmaxQ16StableTopKCheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 FPQ16TopKSelectLogitsCheckedNoAlias(",
        1,
    )[0]
    assert "staged_required_workspace_cells = logits_count + top_k;" in body
    assert "staged_max_logit_q16 = logits_q16[0];" in body


def test_known_vector() -> None:
    logits = [10, -20, 5, 10]
    out_required = [0xAAAA]
    out_selected = [0xBBBB]
    out_max = [0xCCCC]

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only(
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

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only(
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


def test_fuzz_geometry_and_max_parity() -> None:
    rng = random.Random(20260422_1083)

    for _ in range(512):
        logits_count = rng.randint(0, 64)
        logits = [rng.randint(-200_000, 200_000) for _ in range(logits_count)]

        if logits_count == 0:
            top_k = 0
        else:
            top_k = rng.randint(0, logits_count)

        required = logits_count + top_k
        workspace_capacity = required + rng.randint(-2, 3)
        if workspace_capacity < 0:
            workspace_capacity = 0

        out_required = [0xABCD]
        out_selected = [0xBCDE]
        out_max = [0xCDEF]

        err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only(
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


def test_pointer_span_overflow_guard() -> None:
    logits = [1, 2, 3]
    out_required = [7]
    out_selected = [8]
    out_max = [9]

    err = softmax_q16_stable_topk_checked_nopartial_commit_only_preflight_only(
        logits,
        logits_count=3,
        top_k=1,
        workspace_capacity=4,
        out_required_workspace_cells=out_required,
        out_selected_count=out_selected,
        out_max_logit_q16=out_max,
        logits_addr=U64_MAX_VALUE - 15,
    )

    assert err == FP_Q16_ERR_OVERFLOW
    assert out_required == [7]
    assert out_selected == [8]
    assert out_max == [9]


if __name__ == "__main__":
    test_source_contains_iq_1083_symbol()
    test_known_vector()
    test_error_no_partial_write()
    test_fuzz_geometry_and_max_parity()
    test_pointer_span_overflow_guard()
    print("ok")
