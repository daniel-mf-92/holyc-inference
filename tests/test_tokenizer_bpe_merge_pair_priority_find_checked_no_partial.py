#!/usr/bin/env python3
"""Parity checks for TokenizerBPEMergePairPriorityFindCheckedNoPartial semantics."""

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


def tokenizer_bpe_merge_pair_priority_find_checked_no_partial(
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

    staged_left_index = [out_left_index[0]]
    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]

    err = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        token_count,
        token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_capacity,
        staged_left_index,
        staged_rank,
        staged_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_left_index[0] = staged_left_index[0]
    out_rank[0] = staged_rank[0]
    out_found[0] = staged_found[0]
    return TOKENIZER_BPE_OK


def run_once(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    seed_idx: int,
) -> None:
    init_idx = 1000 + seed_idx
    init_rank = 2000 + seed_idx
    init_found = (seed_idx % 2) == 0

    idx_core = [init_idx]
    rank_core = [init_rank]
    found_core = [init_found]

    idx_np = [init_idx]
    rank_np = [init_rank]
    found_np = [init_found]

    err_core = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        len(token_ids),
        len(token_ids),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        len(rank_values),
        len(rank_values),
        idx_core,
        rank_core,
        found_core,
    )
    err_np = tokenizer_bpe_merge_pair_priority_find_checked_no_partial(
        token_ids,
        len(token_ids),
        len(token_ids),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        len(rank_values),
        len(rank_values),
        idx_np,
        rank_np,
        found_np,
    )

    assert err_core == err_np == TOKENIZER_BPE_OK
    assert idx_np[0] == idx_core[0]
    assert rank_np[0] == rank_core[0]
    assert found_np[0] == found_core[0]


def test_success_and_no_match_parity_against_checked_core() -> None:
    run_once([10, 20, 30, 20], [10, 20, 30], [20, 30, 20], [7, 4, 9], 1)
    run_once([4, 5, 6], [4, 4, 5], [5, 5, 6], [12, 3, 9], 2)
    run_once([1, 2, 3], [9, 8], [7, 6], [5, 4], 3)
    run_once([7], [7], [8], [1], 4)


def test_no_partial_write_on_error_paths() -> None:
    idx = [111]
    rank = [222]
    found = [True]

    err = tokenizer_bpe_merge_pair_priority_find_checked_no_partial(
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
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert idx[0] == 111 and rank[0] == 222 and found[0] is True

    idx = [333]
    rank = [444]
    found = [False]
    err = tokenizer_bpe_merge_pair_priority_find_checked_no_partial(
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
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert idx[0] == 333 and rank[0] == 444 and found[0] is False

    idx = [555]
    rank = [666]
    found = [True]
    err = tokenizer_bpe_merge_pair_priority_find_checked_no_partial(
        [1, 2],
        3,
        2,
        [1],
        [2],
        [3],
        1,
        1,
        idx,
        rank,
        found,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert idx[0] == 555 and rank[0] == 666 and found[0] is True


def test_randomized_parity_against_checked_core() -> None:
    rng = random.Random(20260418_366)

    for i in range(3000):
        token_count = rng.randint(0, 40)
        token_ids = [rng.randint(0, 50) for _ in range(token_count)]

        rank_count = rng.randint(0, 180)
        rank_left = [rng.randint(0, 50) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 50) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 4000) for _ in range(rank_count)]

        run_once(token_ids, rank_left, rank_right, rank_values, i + 10)


if __name__ == "__main__":
    test_success_and_no_match_parity_against_checked_core()
    test_no_partial_write_on_error_paths()
    test_randomized_parity_against_checked_core()
    print("tokenizer_bpe_merge_pair_priority_find_checked_no_partial_reference_checks=ok")
