#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairTokenLookupCheckedDefault semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_apply_best_priority_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_merge_pair_token_lookup_checked,
)


def tokenizer_bpe_merge_pair_token_lookup_checked_default(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    out_merged_token: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if out_merged_token is None or out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if rank_table_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = rank_table_count

    return tokenizer_bpe_merge_pair_token_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        derived_capacity,
        out_merged_token,
        out_rank,
        out_found,
    )


def run_case(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
) -> None:
    out_merged_core = [0x3131]
    out_rank_core = [0x4141]
    out_found_core = [False]

    out_merged_default = [0x3131]
    out_rank_default = [0x4141]
    out_found_default = [False]

    err_core = tokenizer_bpe_merge_pair_token_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_count,
        out_merged_core,
        out_rank_core,
        out_found_core,
    )
    err_default = tokenizer_bpe_merge_pair_token_lookup_checked_default(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        out_merged_default,
        out_rank_default,
        out_found_default,
    )

    assert err_default == err_core
    assert out_merged_default[0] == out_merged_core[0]
    assert out_rank_default[0] == out_rank_core[0]
    assert out_found_default[0] == out_found_core[0]


def test_known_vectors_duplicate_rank_and_missing_pair() -> None:
    rank_rows = sorted(
        [
            (4, 7, 30, 1004),
            (4, 7, 2, 1001),
            (4, 7, 2, 1002),
            (4, 8, 9, 1008),
            (9, 9, 1, 1099),
        ],
        key=lambda x: (x[0], x[1]),
    )

    rank_left = [row[0] for row in rank_rows]
    rank_right = [row[1] for row in rank_rows]
    rank_values = [row[2] for row in rank_rows]
    rank_merged = [row[3] for row in rank_rows]

    run_case(4, 7, rank_left, rank_right, rank_values, rank_merged, len(rank_rows))
    run_case(9, 9, rank_left, rank_right, rank_values, rank_merged, len(rank_rows))
    run_case(5, 5, rank_left, rank_right, rank_values, rank_merged, len(rank_rows))


def test_null_and_overflow_contracts() -> None:
    out_merged = [0x7777]
    out_rank = [0x6666]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default(
        1,
        2,
        [],
        [],
        [],
        [],
        I64_MAX + 1,
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_merged[0] == 0x7777
    assert out_rank[0] == 0x6666
    assert out_found[0] is True

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default(
        1,
        2,
        [],
        [],
        [],
        [],
        0,
        None,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_parity_vs_explicit_capacity_core() -> None:
    rng = random.Random(20260418_396)

    for _ in range(6000):
        n = rng.randint(0, 320)
        rows = [
            (
                rng.randint(0, 127),
                rng.randint(0, 127),
                rng.randint(0, 10000),
                rng.randint(0, 65535),
            )
            for _ in range(n)
        ]
        rows.sort(key=lambda row: (row[0], row[1]))

        rank_left = [row[0] for row in rows]
        rank_right = [row[1] for row in rows]
        rank_values = [row[2] for row in rows]
        rank_merged = [row[3] for row in rows]

        query_left = rng.randint(0, 127)
        query_right = rng.randint(0, 127)

        run_case(
            query_left,
            query_right,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            len(rows),
        )


def test_malformed_capacity_adversarial_parity() -> None:
    out_merged_core = [333]
    out_rank_core = [444]
    out_found_core = [True]

    out_merged_default = [333]
    out_rank_default = [444]
    out_found_default = [True]

    # Explicit malformed capacity in core should fail and preserve outputs.
    err_core = tokenizer_bpe_merge_pair_token_lookup_checked(
        3,
        5,
        [3],
        [5],
        [1],
        [999],
        1,
        0,
        out_merged_core,
        out_rank_core,
        out_found_core,
    )
    assert err_core != TOKENIZER_BPE_OK
    assert out_merged_core[0] == 333
    assert out_rank_core[0] == 444
    assert out_found_core[0] is True

    # Default wrapper always derives well-formed capacity == count.
    err_default = tokenizer_bpe_merge_pair_token_lookup_checked_default(
        3,
        5,
        [3],
        [5],
        [1],
        [999],
        1,
        out_merged_default,
        out_rank_default,
        out_found_default,
    )
    assert err_default == TOKENIZER_BPE_OK
    assert out_merged_default[0] == 999
    assert out_rank_default[0] == 1
    assert out_found_default[0] is True


if __name__ == "__main__":
    test_known_vectors_duplicate_rank_and_missing_pair()
    test_null_and_overflow_contracts()
    test_randomized_parity_vs_explicit_capacity_core()
    test_malformed_capacity_adversarial_parity()
    print("tokenizer_bpe_merge_pair_token_lookup_checked_default_reference_checks=ok")
