#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairRankLookupCheckedDefaultValidateRanks."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_pair_rank_lookup_checked_default import (
    tokenizer_bpe_merge_pair_rank_lookup_checked_default,
)
from test_tokenizer_bpe_token_pair_rank_lookup_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if rank_table_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > 0 and (
        rank_left_tokens is None or rank_right_tokens is None or rank_values is None
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

    run_start = 0
    while run_start < rank_table_count:
        key_left = rank_left_tokens[run_start]  # type: ignore[index]
        key_right = rank_right_tokens[run_start]  # type: ignore[index]

        min_rank_set = False
        run_scan = run_start
        while run_scan < rank_table_count:
            if rank_left_tokens[run_scan] != key_left:  # type: ignore[index]
                break
            if rank_right_tokens[run_scan] != key_right:  # type: ignore[index]
                break

            if not min_rank_set:
                min_rank_set = True

            run_scan += 1

        if not min_rank_set:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        run_start = run_scan

    return tokenizer_bpe_merge_pair_rank_lookup_checked_default(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        out_rank,
        out_found,
    )


def _run_parity_case(
    left_token: int,
    right_token: int,
    rank_left: list[int],
    rank_right: list[int],
    rank_values: list[int],
) -> None:
    out_rank_a = [0x3131]
    out_found_a = [True]

    out_rank_b = [0x3131]
    out_found_b = [True]

    err_a = tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks(
        left_token,
        right_token,
        rank_left,
        rank_right,
        rank_values,
        len(rank_values),
        out_rank_a,
        out_found_a,
    )
    err_b = tokenizer_bpe_merge_pair_rank_lookup_checked_default(
        left_token,
        right_token,
        rank_left,
        rank_right,
        rank_values,
        len(rank_values),
        out_rank_b,
        out_found_b,
    )

    assert err_a == err_b
    assert out_rank_a[0] == out_rank_b[0]
    assert out_found_a[0] == out_found_b[0]


def test_unsorted_rank_table_rejected_without_writes() -> None:
    rank_left = [8, 7, 8]
    rank_right = [10, 11, 10]
    rank_values = [4, 1, 2]

    out_rank = [0x7777]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks(
        8,
        10,
        rank_left,
        rank_right,
        rank_values,
        len(rank_values),
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_rank[0] == 0x7777
    assert out_found[0] is True


def test_sorted_table_success_parity_and_duplicate_min_rank() -> None:
    rows = sorted(
        [
            (4, 7, 30),
            (4, 7, 2),
            (4, 7, 2),
            (4, 8, 9),
            (9, 9, 1),
        ],
        key=lambda x: (x[0], x[1]),
    )

    rank_left = [row[0] for row in rows]
    rank_right = [row[1] for row in rows]
    rank_values = [row[2] for row in rows]

    _run_parity_case(4, 7, rank_left, rank_right, rank_values)

    out_rank = [0x1111]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks(
        4,
        7,
        rank_left,
        rank_right,
        rank_values,
        len(rows),
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is True
    assert out_rank[0] == 2


def test_null_and_overflow_contracts() -> None:
    out_rank = [123]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks(
        1,
        2,
        [],
        [],
        [],
        I64_MAX + 1,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_rank[0] == 123
    assert out_found[0] is True

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks(
        1,
        2,
        [],
        [],
        [],
        0,
        None,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_sorted_parity_vs_default_wrapper() -> None:
    rng = random.Random(20260418_413)

    for _ in range(7000):
        n = rng.randint(0, 320)
        rows = [
            (
                rng.randint(0, 127),
                rng.randint(0, 127),
                rng.randint(0, 10000),
            )
            for _ in range(n)
        ]
        rows.sort(key=lambda row: (row[0], row[1]))

        rank_left = [row[0] for row in rows]
        rank_right = [row[1] for row in rows]
        rank_values = [row[2] for row in rows]

        query_left = rng.randint(0, 127)
        query_right = rng.randint(0, 127)

        _run_parity_case(
            query_left,
            query_right,
            rank_left,
            rank_right,
            rank_values,
        )


if __name__ == "__main__":
    test_unsorted_rank_table_rejected_without_writes()
    test_sorted_table_success_parity_and_duplicate_min_rank()
    test_null_and_overflow_contracts()
    test_randomized_sorted_parity_vs_default_wrapper()
    print(
        "tokenizer_bpe_merge_pair_rank_lookup_checked_default_validate_ranks_reference_checks=ok"
    )
