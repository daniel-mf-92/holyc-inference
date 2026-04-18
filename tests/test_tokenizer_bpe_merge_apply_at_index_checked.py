#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeApplyAtIndexChecked semantics."""

from __future__ import annotations

import random

TOKENIZER_BPE_OK = 0
TOKENIZER_BPE_ERR_NULL_PTR = 101
TOKENIZER_BPE_ERR_BAD_PARAM = 102
TOKENIZER_BPE_ERR_OVERFLOW = 103

I64_MAX = (1 << 63) - 1


def tokenizer_bpe_merge_apply_at_index_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    left_index: int,
    merged_token: int,
    out_token_count: list[int] | None,
) -> int:
    if token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if token_count > token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if token_count < 2:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if left_index >= token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    right_index = left_index + 1
    if right_index <= left_index:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if right_index >= token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    token_ids[left_index] = merged_token

    src_index = right_index + 1
    while src_index < token_count:
        dst_index = src_index - 1
        if dst_index >= token_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

        token_ids[dst_index] = token_ids[src_index]
        src_index += 1

    out_token_count[0] = token_count - 1
    return TOKENIZER_BPE_OK


def reference_merge_apply_at_index(
    token_ids: list[int],
    token_count: int,
    left_index: int,
    merged_token: int,
) -> tuple[list[int], int]:
    out = token_ids[:token_count]
    out[left_index] = merged_token
    del out[left_index + 1]
    return out, token_count - 1


def test_known_merge_vectors_middle_start_end() -> None:
    tokens = [11, 22, 33, 44, 55, -1, -1]
    out_count = [999]
    err = tokenizer_bpe_merge_apply_at_index_checked(tokens, 5, 7, 2, 777, out_count)
    assert err == TOKENIZER_BPE_OK
    assert out_count[0] == 4
    assert tokens[:4] == [11, 22, 777, 55]

    tokens2 = [9, 8, 7, 6, -1]
    out_count2 = [888]
    err = tokenizer_bpe_merge_apply_at_index_checked(tokens2, 4, 5, 0, 42, out_count2)
    assert err == TOKENIZER_BPE_OK
    assert out_count2[0] == 3
    assert tokens2[:3] == [42, 7, 6]

    tokens3 = [1, 2, 3, 4, 5, -1]
    out_count3 = [777]
    err = tokenizer_bpe_merge_apply_at_index_checked(tokens3, 5, 6, 3, 99, out_count3)
    assert err == TOKENIZER_BPE_OK
    assert out_count3[0] == 4
    assert tokens3[:4] == [1, 2, 3, 99]


def test_parameter_contracts_and_no_partial_on_error() -> None:
    base = [10, 20, 30, 40, 50]

    out_count = [111]
    assert (
        tokenizer_bpe_merge_apply_at_index_checked(None, 5, 5, 1, 99, out_count)
        == TOKENIZER_BPE_ERR_NULL_PTR
    )
    assert out_count[0] == 111

    out_count2 = [112]
    toks2 = base.copy()
    assert (
        tokenizer_bpe_merge_apply_at_index_checked(toks2, 5, I64_MAX + 1, 1, 99, out_count2)
        == TOKENIZER_BPE_ERR_OVERFLOW
    )
    assert out_count2[0] == 112
    assert toks2 == base

    out_count3 = [113]
    toks3 = base.copy()
    assert (
        tokenizer_bpe_merge_apply_at_index_checked(toks3, 6, 5, 1, 99, out_count3)
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )
    assert out_count3[0] == 113
    assert toks3 == base

    out_count4 = [114]
    toks4 = [7]
    assert (
        tokenizer_bpe_merge_apply_at_index_checked(toks4, 1, 1, 0, 99, out_count4)
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )
    assert out_count4[0] == 114
    assert toks4 == [7]

    out_count5 = [115]
    toks5 = base.copy()
    assert (
        tokenizer_bpe_merge_apply_at_index_checked(toks5, 5, 5, 5, 99, out_count5)
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )
    assert out_count5[0] == 115
    assert toks5 == base

    out_count6 = [116]
    toks6 = base.copy()
    assert (
        tokenizer_bpe_merge_apply_at_index_checked(toks6, 5, 5, 4, 99, out_count6)
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )
    assert out_count6[0] == 116
    assert toks6 == base


def test_randomized_reference_parity() -> None:
    rng = random.Random(20260418_355)

    for _ in range(5000):
        token_count = rng.randint(2, 128)
        slack = rng.randint(0, 8)
        token_capacity = token_count + slack
        left_index = rng.randint(0, token_count - 2)
        merged_token = rng.randint(-1000, 1000)

        data = [rng.randint(-500, 500) for _ in range(token_capacity)]
        original = data[:]

        out_count = [4242]
        err = tokenizer_bpe_merge_apply_at_index_checked(
            data,
            token_count,
            token_capacity,
            left_index,
            merged_token,
            out_count,
        )
        assert err == TOKENIZER_BPE_OK

        ref_tokens, ref_count = reference_merge_apply_at_index(
            original,
            token_count,
            left_index,
            merged_token,
        )

        assert out_count[0] == ref_count
        assert data[:ref_count] == ref_tokens
        assert ref_count == token_count - 1


def test_tail_compaction_preserves_post_count_garbage_lane_irrelevance() -> None:
    # Contract validates [0, out_count) content only; trailing lanes are scratch.
    tokens = [5, 6, 7, 8, 9, 111, 222, 333]
    out_count = [0]
    err = tokenizer_bpe_merge_apply_at_index_checked(tokens, 5, 8, 1, 42, out_count)
    assert err == TOKENIZER_BPE_OK
    assert out_count[0] == 4
    assert tokens[:4] == [5, 42, 8, 9]

