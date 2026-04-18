#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairTokenLookupCheckedDefaultValidateRanks."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_apply_best_priority_checked import TOKENIZER_BPE_ERR_BAD_PARAM
from test_tokenizer_bpe_merge_pair_token_lookup_checked_default import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    tokenizer_bpe_merge_pair_token_lookup_checked_default,
)

TOKENIZER_BPE_OK = 0


def tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks(
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

    if rank_table_count > 0 and (
        rank_left_tokens is None
        or rank_right_tokens is None
        or rank_values is None
        or rank_merged_tokens is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    for i in range(1, rank_table_count):
        if rank_left_tokens[i - 1] > rank_left_tokens[i]:  # type: ignore[index]
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if (
            rank_left_tokens[i - 1] == rank_left_tokens[i]  # type: ignore[index]
            and rank_right_tokens[i - 1] > rank_right_tokens[i]  # type: ignore[index]
        ):
            return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_merge_pair_token_lookup_checked_default(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        out_merged_token,
        out_rank,
        out_found,
    )


def _run_parity_case(
    left_token: int,
    right_token: int,
    rank_left: list[int],
    rank_right: list[int],
    rank_values: list[int],
    rank_merged: list[int],
) -> None:
    out_merged_a = [0x1111]
    out_rank_a = [0x2222]
    out_found_a = [False]

    out_merged_b = [0x1111]
    out_rank_b = [0x2222]
    out_found_b = [False]

    err_a = tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks(
        left_token,
        right_token,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_values),
        out_merged_a,
        out_rank_a,
        out_found_a,
    )
    err_b = tokenizer_bpe_merge_pair_token_lookup_checked_default(
        left_token,
        right_token,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_values),
        out_merged_b,
        out_rank_b,
        out_found_b,
    )

    assert err_a == err_b
    assert out_merged_a[0] == out_merged_b[0]
    assert out_rank_a[0] == out_rank_b[0]
    assert out_found_a[0] == out_found_b[0]


def test_unsorted_rank_table_rejected_without_writes() -> None:
    rank_left = [10, 9, 10]
    rank_right = [20, 30, 20]
    rank_values = [5, 1, 3]
    rank_merged = [100, 200, 300]

    out_merged = [0x7777]
    out_rank = [0x6666]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks(
        10,
        20,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_values),
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_merged[0] == 0x7777
    assert out_rank[0] == 0x6666
    assert out_found[0] is True


def test_sorted_table_success_parity_and_duplicate_min_rank() -> None:
    rows = sorted(
        [
            (4, 7, 30, 1004),
            (4, 7, 2, 1001),
            (4, 7, 2, 1002),
            (4, 8, 9, 1008),
            (9, 9, 1, 1099),
        ],
        key=lambda x: (x[0], x[1]),
    )

    rank_left = [row[0] for row in rows]
    rank_right = [row[1] for row in rows]
    rank_values = [row[2] for row in rows]
    rank_merged = [row[3] for row in rows]

    _run_parity_case(4, 7, rank_left, rank_right, rank_values, rank_merged)

    out_merged = [0x1111]
    out_rank = [0x2222]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks(
        4,
        7,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rows),
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is True
    assert out_rank[0] == 2
    assert out_merged[0] == 1001


def test_null_and_overflow_contracts() -> None:
    out_merged = [123]
    out_rank = [456]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks(
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
    assert out_merged[0] == 123
    assert out_rank[0] == 456
    assert out_found[0] is True

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks(
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


def test_randomized_sorted_parity_vs_default_wrapper() -> None:
    rng = random.Random(20260418_411)

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

        _run_parity_case(
            query_left,
            query_right,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
        )


if __name__ == "__main__":
    test_unsorted_rank_table_rejected_without_writes()
    test_sorted_table_success_parity_and_duplicate_min_rank()
    test_null_and_overflow_contracts()
    test_randomized_sorted_parity_vs_default_wrapper()
    print(
        "tokenizer_bpe_merge_pair_token_lookup_checked_default_validate_ranks_reference_checks=ok"
    )
