#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairPriorityFindCheckedDefault."""

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


def tokenizer_bpe_merge_pair_priority_find_checked_default(
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

    return tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        token_count,
        derived_token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_rank_table_capacity,
        out_left_index,
        out_rank,
        out_found,
    )


def run_case(
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    seed: int,
) -> None:
    init_idx = 100000 + seed
    init_rank = 200000 + seed
    init_found = (seed % 2) == 0

    idx_core = [init_idx]
    rank_core = [init_rank]
    found_core = [init_found]

    idx_default = [init_idx]
    rank_default = [init_rank]
    found_default = [init_found]

    err_core = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        token_count,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_count,
        idx_core,
        rank_core,
        found_core,
    )
    err_default = tokenizer_bpe_merge_pair_priority_find_checked_default(
        token_ids,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        idx_default,
        rank_default,
        found_default,
    )

    assert err_default == err_core
    assert idx_default[0] == idx_core[0]
    assert rank_default[0] == rank_core[0]
    assert found_default[0] == found_core[0]


def test_success_and_no_match_parity_vs_explicit_capacity_core() -> None:
    run_case([10, 20, 30, 20], 4, [10, 20, 30], [20, 30, 20], [7, 4, 9], 3, 1)
    run_case([4, 5, 6], 3, [4, 4, 5], [5, 5, 6], [12, 3, 9], 3, 2)
    run_case([1, 2, 3], 3, [9, 8], [7, 6], [5, 4], 2, 3)
    run_case([7], 1, [7], [8], [1], 1, 4)


def test_exact_no_partial_write_parity_on_error_paths() -> None:
    idx_default = [111]
    rank_default = [222]
    found_default = [True]

    err = tokenizer_bpe_merge_pair_priority_find_checked_default(
        None,
        0,
        [],
        [],
        [],
        0,
        idx_default,
        rank_default,
        found_default,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert idx_default[0] == 111 and rank_default[0] == 222 and found_default[0] is True

    idx_default = [333]
    rank_default = [444]
    found_default = [False]
    err = tokenizer_bpe_merge_pair_priority_find_checked_default(
        [1, 2],
        I64_MAX + 1,
        [1],
        [2],
        [3],
        1,
        idx_default,
        rank_default,
        found_default,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert idx_default[0] == 333 and rank_default[0] == 444 and found_default[0] is False

    idx_default = [555]
    rank_default = [666]
    found_default = [True]
    err = tokenizer_bpe_merge_pair_priority_find_checked_default(
        [1, 2],
        2,
        [1],
        [2],
        [3],
        I64_MAX + 1,
        idx_default,
        rank_default,
        found_default,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert idx_default[0] == 555 and rank_default[0] == 666 and found_default[0] is True


def test_randomized_parity_against_explicit_capacity_core() -> None:
    rng = random.Random(20260418_380)

    for i in range(5000):
        token_count = rng.randint(0, 42)
        token_ids = [rng.randint(0, 120) for _ in range(token_count)]

        rank_count = rng.randint(0, 200)
        rank_left = [rng.randint(0, 120) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 120) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 5000) for _ in range(rank_count)]

        run_case(
            token_ids,
            len(token_ids),
            rank_left,
            rank_right,
            rank_values,
            len(rank_values),
            i + 10,
        )


def test_malformed_capacity_adversarial_equivalence_boundary() -> None:
    idx_core = [777]
    rank_core = [888]
    found_core = [True]
    idx_default = [777]
    rank_default = [888]
    found_default = [True]

    # Core explicit path can be malformed when capacity < count.
    err_core = tokenizer_bpe_merge_pair_priority_find_checked(
        [1, 2],
        2,
        1,
        [1],
        [2],
        [3],
        1,
        1,
        idx_core,
        rank_core,
        found_core,
    )
    assert err_core != TOKENIZER_BPE_OK
    assert idx_core[0] == 777 and rank_core[0] == 888 and found_core[0] is True

    # Default wrapper constructs the well-formed equivalent by definition.
    err_default = tokenizer_bpe_merge_pair_priority_find_checked_default(
        [1, 2],
        2,
        [1],
        [2],
        [3],
        1,
        idx_default,
        rank_default,
        found_default,
    )
    assert err_default == TOKENIZER_BPE_OK
    assert idx_default[0] == 0
    assert rank_default[0] == 3
    assert found_default[0] is True


if __name__ == "__main__":
    test_success_and_no_match_parity_vs_explicit_capacity_core()
    test_exact_no_partial_write_parity_on_error_paths()
    test_randomized_parity_against_explicit_capacity_core()
    test_malformed_capacity_adversarial_equivalence_boundary()
    print("tokenizer_bpe_merge_pair_priority_find_checked_default_reference_checks=ok")
