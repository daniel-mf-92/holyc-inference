#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairTokenLookupCheckedDefaultNoPartialValidateRanks."""

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
    tokenizer_bpe_merge_pair_token_lookup_checked,
)


def tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
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

    run_start = 0
    while run_start < rank_table_count:
        key_left = rank_left_tokens[run_start]  # type: ignore[index]
        key_right = rank_right_tokens[run_start]  # type: ignore[index]

        min_rank_set = False
        min_rank = 0
        run_scan = run_start
        while run_scan < rank_table_count:
            if rank_left_tokens[run_scan] != key_left:  # type: ignore[index]
                break
            if rank_right_tokens[run_scan] != key_right:  # type: ignore[index]
                break

            if (not min_rank_set) or rank_values[run_scan] < min_rank:  # type: ignore[index]
                min_rank = rank_values[run_scan]  # type: ignore[index]
                min_rank_set = True

            run_scan += 1

        if not min_rank_set:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        run_start = run_scan

    staged_merged = [out_merged_token[0]]
    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]

    err = tokenizer_bpe_merge_pair_token_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_count,
        staged_merged,
        staged_rank,
        staged_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_merged_token[0] = staged_merged[0]
    out_rank[0] = staged_rank[0]
    out_found[0] = staged_found[0]
    return TOKENIZER_BPE_OK


def explicit_staged_composition(
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

    run_start = 0
    while run_start < rank_table_count:
        key_left = rank_left_tokens[run_start]  # type: ignore[index]
        key_right = rank_right_tokens[run_start]  # type: ignore[index]

        min_rank_set = False
        min_rank = 0
        run_scan = run_start
        while run_scan < rank_table_count:
            if rank_left_tokens[run_scan] != key_left:  # type: ignore[index]
                break
            if rank_right_tokens[run_scan] != key_right:  # type: ignore[index]
                break

            if (not min_rank_set) or rank_values[run_scan] < min_rank:  # type: ignore[index]
                min_rank = rank_values[run_scan]  # type: ignore[index]
                min_rank_set = True

            run_scan += 1

        if not min_rank_set:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        run_start = run_scan

    staged_merged = [out_merged_token[0]]
    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]

    err = tokenizer_bpe_merge_pair_token_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_count,
        staged_merged,
        staged_rank,
        staged_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_merged_token[0] = staged_merged[0]
    out_rank[0] = staged_rank[0]
    out_found[0] = staged_found[0]
    return TOKENIZER_BPE_OK


def run_case(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
) -> None:
    out_merged_a = [0x4A4A]
    out_rank_a = [0x5B5B]
    out_found_a = [True]

    out_merged_b = [0x4A4A]
    out_rank_b = [0x5B5B]
    out_found_b = [True]

    err_a = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        out_merged_a,
        out_rank_a,
        out_found_a,
    )
    err_b = explicit_staged_composition(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        out_merged_b,
        out_rank_b,
        out_found_b,
    )

    assert err_a == err_b
    assert out_merged_a[0] == out_merged_b[0]
    assert out_rank_a[0] == out_rank_b[0]
    assert out_found_a[0] == out_found_b[0]


def test_unsorted_rank_table_rejected_without_partial_commit() -> None:
    out_merged = [0x1111]
    out_rank = [0x2222]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
        5,
        7,
        [5, 4],
        [7, 7],
        [3, 1],
        [100, 101],
        2,
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_merged[0] == 0x1111
    assert out_rank[0] == 0x2222
    assert out_found[0] is True


def test_sorted_duplicate_min_rank_success() -> None:
    rows = sorted(
        [
            (9, 11, 30, 3011),
            (9, 11, 4, 3004),
            (9, 11, 4, 3005),
            (9, 12, 7, 3012),
            (12, 12, 1, 3121),
        ],
        key=lambda x: (x[0], x[1]),
    )

    left = [row[0] for row in rows]
    right = [row[1] for row in rows]
    rank = [row[2] for row in rows]
    merged = [row[3] for row in rows]

    out_merged = [0xA1A1]
    out_rank = [0xB2B2]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
        9,
        11,
        left,
        right,
        rank,
        merged,
        len(rows),
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is True
    assert out_rank[0] == 4
    assert out_merged[0] == 3004


def test_null_output_and_overflow_contracts() -> None:
    out_merged = [0xAAAA]
    out_rank = [0xBBBB]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
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
    assert out_merged[0] == 0xAAAA
    assert out_rank[0] == 0xBBBB
    assert out_found[0] is True

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
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


def test_malformed_rank_columns_preserve_outputs() -> None:
    out_merged = [0x7777]
    out_rank = [0x8888]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks(
        4,
        7,
        [4],
        None,
        [2],
        [1002],
        1,
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert out_merged[0] == 0x7777
    assert out_rank[0] == 0x8888
    assert out_found[0] is False


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_412)

    for _ in range(8000):
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


if __name__ == "__main__":
    test_unsorted_rank_table_rejected_without_partial_commit()
    test_sorted_duplicate_min_rank_success()
    test_null_output_and_overflow_contracts()
    test_malformed_rank_columns_preserve_outputs()
    test_randomized_parity_vs_explicit_staged_composition()
    print(
        "tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_validate_ranks_reference_checks=ok"
    )
