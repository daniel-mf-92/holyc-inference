#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairPriorityFindCheckedDefaultNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_pair_priority_find_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_merge_pair_priority_find_checked,
)


def tokenizer_bpe_merge_pair_priority_find_checked_default_no_partial(
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    out_left_index: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if token_ids is None or out_left_index is None or out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or rank_table_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_token_capacity = token_count
    derived_rank_table_capacity = rank_table_count

    staged_left_index = [out_left_index[0]]
    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]

    err = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        token_count,
        derived_token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_rank_table_capacity,
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


def run_case(
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    seed: int,
) -> None:
    init_idx = 300000 + seed
    init_rank = 400000 + seed
    init_found = (seed % 2) == 0

    idx_expected = [init_idx]
    rank_expected = [init_rank]
    found_expected = [init_found]

    idx_default_np = [init_idx]
    rank_default_np = [init_rank]
    found_default_np = [init_found]

    # Explicit staged composition baseline:
    # 1) derive default capacities, 2) run checked core on staged outputs,
    # 3) commit only on success.
    staged_idx = [idx_expected[0]]
    staged_rank = [rank_expected[0]]
    staged_found = [found_expected[0]]
    err_expected = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        token_count,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_count,
        staged_idx,
        staged_rank,
        staged_found,
    )
    if err_expected == TOKENIZER_BPE_OK:
        idx_expected[0] = staged_idx[0]
        rank_expected[0] = staged_rank[0]
        found_expected[0] = staged_found[0]

    err_default_np = tokenizer_bpe_merge_pair_priority_find_checked_default_no_partial(
        token_ids,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        idx_default_np,
        rank_default_np,
        found_default_np,
    )

    assert err_default_np == err_expected
    assert idx_default_np[0] == idx_expected[0]
    assert rank_default_np[0] == rank_expected[0]
    assert found_default_np[0] == found_expected[0]


def test_success_and_no_match_parity_vs_explicit_staged_composition() -> None:
    run_case([10, 20, 30, 20], 4, [10, 20, 30], [20, 30, 20], [7, 4, 9], 3, 1)
    run_case([4, 5, 6], 3, [4, 4, 5], [5, 5, 6], [12, 3, 9], 3, 2)
    run_case([1, 2, 3], 3, [9, 8], [7, 6], [5, 4], 2, 3)
    run_case([7], 1, [7], [8], [1], 1, 4)


def test_exact_no_partial_write_parity_on_error_paths() -> None:
    idx = [111]
    rank = [222]
    found = [True]

    err = tokenizer_bpe_merge_pair_priority_find_checked_default_no_partial(
        None,
        0,
        [],
        [],
        [],
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
    err = tokenizer_bpe_merge_pair_priority_find_checked_default_no_partial(
        [1, 2],
        I64_MAX + 1,
        [1],
        [2],
        [3],
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
    err = tokenizer_bpe_merge_pair_priority_find_checked_default_no_partial(
        [1, 2],
        2,
        None,
        [2],
        [3],
        1,
        idx,
        rank,
        found,
    )
    assert err != TOKENIZER_BPE_OK
    assert idx[0] == 555 and rank[0] == 666 and found[0] is True


def test_randomized_parity_against_explicit_staged_composition() -> None:
    rng = random.Random(20260418_381)

    for i in range(6000):
        token_count = rng.randint(0, 48)
        token_ids = [rng.randint(0, 150) for _ in range(token_count)]

        rank_count = rng.randint(0, 220)
        rank_left = [rng.randint(0, 150) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 150) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 8000) for _ in range(rank_count)]

        run_case(
            token_ids,
            len(token_ids),
            rank_left,
            rank_right,
            rank_values,
            len(rank_values),
            i + 10,
        )


if __name__ == "__main__":
    test_success_and_no_match_parity_vs_explicit_staged_composition()
    test_exact_no_partial_write_parity_on_error_paths()
    test_randomized_parity_against_explicit_staged_composition()
    print("tokenizer_bpe_merge_pair_priority_find_checked_default_no_partial_reference_checks=ok")
