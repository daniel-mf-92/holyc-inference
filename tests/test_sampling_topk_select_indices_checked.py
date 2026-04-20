#!/usr/bin/env python3
"""Reference checks for SamplingTopKSelectIndicesChecked semantics (IQ-746)."""

from __future__ import annotations

import random
from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2
SAMPLING_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = 0x7FFFFFFFFFFFFFFF
U64_MAX_VALUE = 0xFFFFFFFFFFFFFFFF


def fpq16_topk_select_indices_checked_reference(
    logits_q16: list[int] | None,
    lane_count: int,
    k: int,
    out_indices: list[int] | None,
    out_index_capacity: int,
    logits_addr: int = 0,
    out_addr: int = 0,
) -> int:
    if logits_q16 is None or out_indices is None:
        return SAMPLING_Q16_ERR_NULL_PTR
    if lane_count < 0 or k < 0 or out_index_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if k > lane_count or k > out_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    if k == 0:
        return SAMPLING_Q16_OK

    last_index = lane_count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if logits_addr > (U64_MAX_VALUE - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    last_index = k - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return SAMPLING_Q16_ERR_OVERFLOW
    last_byte_offset = last_index << 3
    if out_addr > (U64_MAX_VALUE - last_byte_offset):
        return SAMPLING_Q16_ERR_OVERFLOW

    if len(logits_q16) < lane_count or len(out_indices) < out_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for rank in range(k):
        best_index = -1
        best_logit_q16 = 0
        for i in range(lane_count):
            if i in out_indices[:rank]:
                continue

            candidate = logits_q16[i]
            if best_index < 0:
                best_index = i
                best_logit_q16 = candidate
            elif candidate > best_logit_q16:
                best_index = i
                best_logit_q16 = candidate
            elif candidate == best_logit_q16 and i < best_index:
                best_index = i

        if best_index < 0:
            return SAMPLING_Q16_ERR_OVERFLOW
        out_indices[rank] = best_index

    return SAMPLING_Q16_OK


def sampling_topk_select_indices_checked_reference(
    logits_q16: list[int] | None,
    logits_capacity: int,
    vocab_size: int,
    k: int,
    out_indices: list[int] | None,
    out_index_capacity: int,
    logits_addr: int = 0,
    out_addr: int = 0,
) -> int:
    if logits_q16 is None or out_indices is None:
        return SAMPLING_Q16_ERR_NULL_PTR

    if logits_capacity < 0 or vocab_size < 0 or k < 0 or out_index_capacity < 0:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if vocab_size > logits_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if k > vocab_size or k > out_index_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM

    return fpq16_topk_select_indices_checked_reference(
        logits_q16,
        vocab_size,
        k,
        out_indices,
        out_index_capacity,
        logits_addr=logits_addr,
        out_addr=out_addr,
    )


def stable_topk_reference(logits_q16: list[int], k: int) -> list[int]:
    return sorted(range(len(logits_q16)), key=lambda i: (-logits_q16[i], i))[:k]


def test_source_contains_wrapper_signature_and_delegate() -> None:
    source = Path("src/model/sampling.HC").read_text(encoding="utf-8")
    assert "I32 SamplingTopKSelectIndicesChecked(" in source
    assert "vocab_size > logits_capacity" in source
    assert "return FPQ16TopKSelectIndicesChecked(logits_q16," in source


def test_null_and_bad_param_contracts() -> None:
    out = [99, 98, 97]

    err = sampling_topk_select_indices_checked_reference(None, 3, 3, 2, out, 3)
    assert err == SAMPLING_Q16_ERR_NULL_PTR
    assert out == [99, 98, 97]

    err = sampling_topk_select_indices_checked_reference([1, 2, 3], 3, 3, 2, None, 3)
    assert err == SAMPLING_Q16_ERR_NULL_PTR

    err = sampling_topk_select_indices_checked_reference([1, 2, 3], -1, 3, 2, out, 3)
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [99, 98, 97]

    err = sampling_topk_select_indices_checked_reference([1, 2, 3], 3, 4, 2, out, 3)
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [99, 98, 97]

    err = sampling_topk_select_indices_checked_reference([1, 2, 3], 3, 3, 4, out, 3)
    assert err == SAMPLING_Q16_ERR_BAD_PARAM
    assert out == [99, 98, 97]


def test_tie_break_and_vocab_window_behavior() -> None:
    logits = [50, 50, 50, 9999, 9998]
    out = [77, 77, 77]

    err = sampling_topk_select_indices_checked_reference(
        logits_q16=logits,
        logits_capacity=len(logits),
        vocab_size=3,
        k=3,
        out_indices=out,
        out_index_capacity=len(out),
    )
    assert err == SAMPLING_Q16_OK
    assert out == [0, 1, 2]


def test_randomized_parity_vs_sorted_reference() -> None:
    rng = random.Random(20260420_746)

    for _ in range(2500):
        logits_capacity = rng.randint(1, 96)
        vocab_size = rng.randint(1, logits_capacity)
        k = rng.randint(0, vocab_size)
        out_capacity = k + rng.randint(0, 16)

        logits = [rng.randint(-500_000, 500_000) for _ in range(logits_capacity)]
        out = [-1] * out_capacity

        err = sampling_topk_select_indices_checked_reference(
            logits,
            logits_capacity,
            vocab_size,
            k,
            out,
            out_capacity,
        )
        assert err == SAMPLING_Q16_OK
        assert out[:k] == stable_topk_reference(logits[:vocab_size], k)


def test_pointer_overflow_and_no_partial_write() -> None:
    logits = [4, 3, 2, 1]
    out = [11, 22, 33]

    err = sampling_topk_select_indices_checked_reference(
        logits,
        logits_capacity=4,
        vocab_size=4,
        k=3,
        out_indices=out,
        out_index_capacity=3,
        logits_addr=U64_MAX_VALUE - 15,
        out_addr=0,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW
    assert out == [11, 22, 33]

    err = sampling_topk_select_indices_checked_reference(
        logits,
        logits_capacity=4,
        vocab_size=4,
        k=3,
        out_indices=out,
        out_index_capacity=3,
        logits_addr=0,
        out_addr=U64_MAX_VALUE - 15,
    )
    assert err == SAMPLING_Q16_ERR_OVERFLOW
    assert out == [11, 22, 33]


if __name__ == "__main__":
    test_source_contains_wrapper_signature_and_delegate()
    test_null_and_bad_param_contracts()
    test_tie_break_and_vocab_window_behavior()
    test_randomized_parity_vs_sorted_reference()
    test_pointer_overflow_and_no_partial_write()
    print("sampling_topk_select_indices_checked_reference_checks=ok")
