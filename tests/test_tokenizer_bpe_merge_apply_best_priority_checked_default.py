#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeApplyBestPriorityCheckedDefault semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_apply_best_priority_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_merge_apply_best_priority_checked,
)


def tokenizer_bpe_merge_apply_best_priority_checked_default(
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    out_merged_token: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
    out_token_count: list[int] | None,
) -> int:
    if (
        token_ids is None
        or out_merged_token is None
        or out_rank is None
        or out_found is None
        or out_token_count is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or rank_table_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_token_capacity = token_count
    derived_rank_table_capacity = rank_table_count

    return tokenizer_bpe_merge_apply_best_priority_checked(
        token_ids,
        token_count,
        derived_token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        derived_rank_table_capacity,
        out_merged_token,
        out_rank,
        out_found,
        out_token_count,
    )


def run_case(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    rank_merged_tokens: list[int],
) -> None:
    tokens_default = token_ids.copy()
    tokens_core = token_ids.copy()

    out_merged_default = [123]
    out_rank_default = [456]
    out_found_default = [False]
    out_count_default = [789]

    out_merged_core = [123]
    out_rank_core = [456]
    out_found_core = [False]
    out_count_core = [789]

    err_default = tokenizer_bpe_merge_apply_best_priority_checked_default(
        tokens_default,
        len(tokens_default),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        len(rank_values),
        out_merged_default,
        out_rank_default,
        out_found_default,
        out_count_default,
    )

    err_core = tokenizer_bpe_merge_apply_best_priority_checked(
        tokens_core,
        len(tokens_core),
        len(tokens_core),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        len(rank_values),
        len(rank_values),
        out_merged_core,
        out_rank_core,
        out_found_core,
        out_count_core,
    )

    assert err_default == err_core
    assert tokens_default == tokens_core
    assert out_merged_default[0] == out_merged_core[0]
    assert out_rank_default[0] == out_rank_core[0]
    assert out_found_default[0] == out_found_core[0]
    assert out_count_default[0] == out_count_core[0]


def test_known_vectors_parity() -> None:
    run_case(
        [10, 20, 30, 20],
        [10, 20, 30],
        [20, 30, 20],
        [7, 4, 9],
        [120, 230, 320],
    )

    run_case(
        [1, 2, 3, 2],
        [1, 2],
        [2, 3],
        [5, 5],
        [12, 23],
    )

    run_case(
        [4, 5, 6],
        [4, 4, 5],
        [5, 5, 6],
        [12, 3, 9],
        [450, 403, 560],
    )

    run_case(
        [9, 8, 7],
        [1, 2],
        [3, 4],
        [10, 20],
        [100, 200],
    )


def test_null_and_overflow_contracts() -> None:
    out_merged = [111]
    out_rank = [222]
    out_found = [True]
    out_count = [333]

    err = tokenizer_bpe_merge_apply_best_priority_checked_default(
        None,
        0,
        [],
        [],
        [],
        [],
        0,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_merge_apply_best_priority_checked_default(
        [],
        I64_MAX + 1,
        [],
        [],
        [],
        [],
        0,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    err = tokenizer_bpe_merge_apply_best_priority_checked_default(
        [],
        0,
        [],
        [],
        [],
        [],
        I64_MAX + 1,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW


def test_no_partial_output_parity_on_error() -> None:
    tokens_default = [1, 2, 3]
    baseline_default = tokens_default.copy()
    tokens_core = [1, 2, 3]

    out_merged_default = [901]
    out_rank_default = [902]
    out_found_default = [True]
    out_count_default = [903]

    out_merged_core = [901]
    out_rank_core = [902]
    out_found_core = [True]
    out_count_core = [903]

    err_default = tokenizer_bpe_merge_apply_best_priority_checked_default(
        tokens_default,
        len(tokens_default),
        [1, 2],
        [2, 3],
        [10, 11],
        None,
        2,
        out_merged_default,
        out_rank_default,
        out_found_default,
        out_count_default,
    )
    assert err_default == TOKENIZER_BPE_ERR_NULL_PTR
    assert tokens_default == baseline_default
    assert out_merged_default[0] == 901
    assert out_rank_default[0] == 902
    assert out_found_default[0] is True
    assert out_count_default[0] == 903

    err_core = tokenizer_bpe_merge_apply_best_priority_checked(
        tokens_core,
        len(tokens_core),
        len(tokens_core),
        [1, 2],
        [2, 3],
        [10, 11],
        None,
        2,
        2,
        out_merged_core,
        out_rank_core,
        out_found_core,
        out_count_core,
    )
    assert err_core == err_default
    assert out_merged_core[0] == out_merged_default[0]
    assert out_rank_core[0] == out_rank_default[0]
    assert out_found_core[0] == out_found_default[0]
    assert out_count_core[0] == out_count_default[0]


def test_randomized_parity_vs_explicit_capacity_core() -> None:
    rng = random.Random(20260418_382)

    for _ in range(6000):
        token_count = rng.randint(0, 40)
        tokens = [rng.randint(0, 40) for _ in range(token_count)]

        rank_count = rng.randint(0, 220)
        raw = [
            (rng.randint(0, 40), rng.randint(0, 40), rng.randint(0, 4096), rng.randint(1000, 6000))
            for _ in range(rank_count)
        ]
        raw.sort(key=lambda row: (row[0], row[1]))

        rank_left = [r[0] for r in raw]
        rank_right = [r[1] for r in raw]
        rank_values = [r[2] for r in raw]
        rank_merged = [r[3] for r in raw]

        run_case(tokens, rank_left, rank_right, rank_values, rank_merged)


def test_malformed_capacity_adversarial_case_is_unreachable_in_default() -> None:
    out_merged_core = [555]
    out_rank_core = [666]
    out_found_core = [True]
    out_count_core = [777]

    err_core = tokenizer_bpe_merge_apply_best_priority_checked(
        [1, 2, 3],
        3,
        2,
        [1, 2],
        [2, 3],
        [9, 10],
        [101, 202],
        2,
        2,
        out_merged_core,
        out_rank_core,
        out_found_core,
        out_count_core,
    )
    assert err_core == TOKENIZER_BPE_ERR_BAD_PARAM

    out_merged_default = [555]
    out_rank_default = [666]
    out_found_default = [True]
    out_count_default = [777]

    err_default = tokenizer_bpe_merge_apply_best_priority_checked_default(
        [1, 2, 3],
        3,
        [1, 2],
        [2, 3],
        [9, 10],
        [101, 202],
        2,
        out_merged_default,
        out_rank_default,
        out_found_default,
        out_count_default,
    )
    assert err_default == TOKENIZER_BPE_OK


if __name__ == "__main__":
    test_known_vectors_parity()
    test_null_and_overflow_contracts()
    test_no_partial_output_parity_on_error()
    test_randomized_parity_vs_explicit_capacity_core()
    test_malformed_capacity_adversarial_case_is_unreachable_in_default()
    print("tokenizer_bpe_merge_apply_best_priority_checked_default_reference_checks=ok")
