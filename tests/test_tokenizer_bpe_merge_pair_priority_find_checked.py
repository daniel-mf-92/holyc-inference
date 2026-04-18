#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairPriorityFindChecked semantics."""

from __future__ import annotations

import random

TOKENIZER_BPE_OK = 0
TOKENIZER_BPE_ERR_NULL_PTR = 101
TOKENIZER_BPE_ERR_BAD_PARAM = 102
TOKENIZER_BPE_ERR_OVERFLOW = 103

I64_MAX = (1 << 63) - 1


def tokenizer_bpe_merge_pair_priority_find_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_left_index: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if token_ids is None or out_left_index is None or out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_capacity > I64_MAX or rank_table_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if token_count > token_capacity or rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None or rank_right_tokens is None or rank_values is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count < 2:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    found = False
    best_left_index = 0
    best_rank = 0

    for i in range(token_count - 1):
        left = token_ids[i]
        right = token_ids[i + 1]

        pair_rank_found = False
        pair_rank = 0
        for j in range(rank_table_count):
            if rank_left_tokens[j] != left:
                continue
            if rank_right_tokens[j] != right:
                continue
            if not pair_rank_found or rank_values[j] < pair_rank:
                pair_rank = rank_values[j]
                pair_rank_found = True

        if not pair_rank_found:
            continue

        if not found or pair_rank < best_rank:
            best_rank = pair_rank
            best_left_index = i
            found = True

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_left_index[0] = best_left_index
    out_rank[0] = best_rank
    out_found[0] = True
    return TOKENIZER_BPE_OK


def reference_find(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
) -> tuple[bool, int, int]:
    found = False
    best_index = -1
    best_rank = 0

    for i in range(len(token_ids) - 1):
        left = token_ids[i]
        right = token_ids[i + 1]
        pair_ranks = [
            rank_values[j]
            for j in range(len(rank_values))
            if rank_left_tokens[j] == left and rank_right_tokens[j] == right
        ]
        if not pair_ranks:
            continue
        pair_best = min(pair_ranks)
        if not found or pair_best < best_rank:
            found = True
            best_rank = pair_best
            best_index = i

    return found, best_index, best_rank


def run_case(
    token_ids: list[int],
    rank_left: list[int],
    rank_right: list[int],
    rank_values: list[int],
) -> None:
    out_idx = [777]
    out_rank = [888]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        len(token_ids),
        len(token_ids),
        rank_left,
        rank_right,
        rank_values,
        len(rank_values),
        len(rank_values),
        out_idx,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK

    ref_found, ref_idx, ref_rank = reference_find(token_ids, rank_left, rank_right, rank_values)
    assert out_found[0] == ref_found
    if ref_found:
        assert out_idx[0] == ref_idx
        assert out_rank[0] == ref_rank
    else:
        assert out_idx[0] == 777
        assert out_rank[0] == 888


def test_known_priority_and_tie_break_vectors() -> None:
    run_case(
        [10, 20, 30, 20],
        [10, 20, 30],
        [20, 30, 20],
        [7, 4, 9],
    )

    # Equal rank appears in two positions; leftmost index must win.
    run_case(
        [1, 2, 3, 2],
        [1, 2],
        [2, 3],
        [5, 5],
    )

    # Duplicate table entries for same pair should pick lowest rank.
    run_case(
        [4, 5, 6],
        [4, 4, 5],
        [5, 5, 6],
        [12, 3, 9],
    )


def test_no_match_and_small_span_behavior() -> None:
    out_idx = [123]
    out_rank = [456]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_priority_find_checked(
        [7],
        1,
        1,
        [7],
        [8],
        [1],
        1,
        1,
        out_idx,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is False
    assert out_idx[0] == 123
    assert out_rank[0] == 456

    out_idx2 = [111]
    out_rank2 = [222]
    out_found2 = [True]
    err = tokenizer_bpe_merge_pair_priority_find_checked(
        [10, 11, 12],
        3,
        3,
        [1, 2],
        [3, 4],
        [5, 6],
        2,
        2,
        out_idx2,
        out_rank2,
        out_found2,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found2[0] is False
    assert out_idx2[0] == 111
    assert out_rank2[0] == 222


def test_parameter_contracts_and_no_partial_on_error() -> None:
    idx = [90]
    rank = [91]
    found = [True]

    assert (
        tokenizer_bpe_merge_pair_priority_find_checked(
            None,
            0,
            0,
            [],
            [],
            [],
            0,
            0,
            idx,
            rank,
            found,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_merge_pair_priority_find_checked(
            [1, 2],
            2,
            I64_MAX + 1,
            [1],
            [2],
            [3],
            1,
            1,
            idx,
            rank,
            found,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )

    idx2 = [70]
    rank2 = [71]
    found2 = [True]
    assert (
        tokenizer_bpe_merge_pair_priority_find_checked(
            [1, 2],
            3,
            2,
            [1],
            [2],
            [3],
            1,
            1,
            idx2,
            rank2,
            found2,
        )
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )
    assert idx2[0] == 70 and rank2[0] == 71 and found2[0] is True

    idx3 = [80]
    rank3 = [81]
    found3 = [True]
    assert (
        tokenizer_bpe_merge_pair_priority_find_checked(
            [1, 2],
            2,
            2,
            None,
            [2],
            [3],
            1,
            1,
            idx3,
            rank3,
            found3,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )
    assert idx3[0] == 80 and rank3[0] == 81 and found3[0] is True


def test_randomized_ranked_reference_corpora() -> None:
    rng = random.Random(20260418_351)
    for _ in range(3000):
        token_count = rng.randint(0, 40)
        token_ids = [rng.randint(0, 60) for _ in range(token_count)]

        rank_count = rng.randint(0, 200)
        rank_left = [rng.randint(0, 60) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 60) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 2000) for _ in range(rank_count)]

        out_idx = [999]
        out_rank = [888]
        out_found = [False]
        err = tokenizer_bpe_merge_pair_priority_find_checked(
            token_ids,
            len(token_ids),
            len(token_ids),
            rank_left,
            rank_right,
            rank_values,
            rank_count,
            rank_count,
            out_idx,
            out_rank,
            out_found,
        )
        assert err == TOKENIZER_BPE_OK

        ref_found, ref_idx, ref_rank = reference_find(
            token_ids,
            rank_left,
            rank_right,
            rank_values,
        )
        assert out_found[0] == ref_found
        if ref_found:
            assert out_idx[0] == ref_idx
            assert out_rank[0] == ref_rank
        else:
            assert out_idx[0] == 999
            assert out_rank[0] == 888


if __name__ == "__main__":
    test_known_priority_and_tie_break_vectors()
    test_no_match_and_small_span_behavior()
    test_parameter_contracts_and_no_partial_on_error()
    test_randomized_ranked_reference_corpora()
    print("tokenizer_bpe_merge_pair_priority_find_checked_reference_checks=ok")
