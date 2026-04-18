#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeCandidatesBuildCheckedDefaultValidateRanks."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_candidates_build_checked import TOKENIZER_BPE_ERR_BAD_PARAM
from test_tokenizer_bpe_merge_candidates_build_checked_default import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    tokenizer_bpe_merge_candidates_build_checked_default,
)


def tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks(
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    out_left_tokens: list[int] | None,
    out_right_tokens: list[int] | None,
    out_left_indices: list[int] | None,
    out_ranks: list[int] | None,
    out_candidate_count: list[int] | None,
) -> int:
    if (
        token_ids is None
        or out_left_tokens is None
        or out_right_tokens is None
        or out_left_indices is None
        or out_ranks is None
        or out_candidate_count is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or rank_table_count > I64_MAX:
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

    return tokenizer_bpe_merge_candidates_build_checked_default(
        token_ids,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        out_candidate_count,
    )


def _run_parity_case(
    token_ids: list[int],
    rank_left: list[int],
    rank_right: list[int],
    rank_values: list[int],
    seed: int,
) -> None:
    token_count = len(token_ids)
    rank_count = len(rank_values)
    cap = max(1, token_count + 8)

    out_left_a = [100000 + seed + i for i in range(cap)]
    out_right_a = [200000 + seed + i for i in range(cap)]
    out_indices_a = [300000 + seed + i for i in range(cap)]
    out_ranks_a = [400000 + seed + i for i in range(cap)]
    out_count_a = [500000 + seed]

    out_left_b = out_left_a.copy()
    out_right_b = out_right_a.copy()
    out_indices_b = out_indices_a.copy()
    out_ranks_b = out_ranks_a.copy()
    out_count_b = out_count_a.copy()

    err_a = tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks(
        token_ids,
        token_count,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        out_left_a,
        out_right_a,
        out_indices_a,
        out_ranks_a,
        out_count_a,
    )
    err_b = tokenizer_bpe_merge_candidates_build_checked_default(
        token_ids,
        token_count,
        rank_left,
        rank_right,
        rank_values,
        rank_count,
        out_left_b,
        out_right_b,
        out_indices_b,
        out_ranks_b,
        out_count_b,
    )

    assert err_a == err_b
    assert out_left_a == out_left_b
    assert out_right_a == out_right_b
    assert out_indices_a == out_indices_b
    assert out_ranks_a == out_ranks_b
    assert out_count_a == out_count_b


def test_unsorted_rank_table_rejected_without_writes() -> None:
    tokens = [10, 20, 30, 20]
    rank_left = [10, 9, 20]
    rank_right = [20, 30, 20]
    rank_values = [5, 3, 7]

    out_left = [111, 112, 113, 114]
    out_right = [211, 212, 213, 214]
    out_indices = [311, 312, 313, 314]
    out_ranks = [411, 412, 413, 414]
    out_count = [511]

    err = tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks(
        tokens,
        len(tokens),
        rank_left,
        rank_right,
        rank_values,
        len(rank_values),
        out_left,
        out_right,
        out_indices,
        out_ranks,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_left == [111, 112, 113, 114]
    assert out_right == [211, 212, 213, 214]
    assert out_indices == [311, 312, 313, 314]
    assert out_ranks == [411, 412, 413, 414]
    assert out_count == [511]


def test_duplicate_pair_ties_sorted_success_parity() -> None:
    rows = sorted(
        [
            (10, 20, 9),
            (10, 20, 3),
            (10, 20, 3),
            (20, 30, 2),
            (30, 20, 8),
        ],
        key=lambda row: (row[0], row[1]),
    )
    rank_left = [row[0] for row in rows]
    rank_right = [row[1] for row in rows]
    rank_values = [row[2] for row in rows]

    _run_parity_case([10, 20, 30, 20], rank_left, rank_right, rank_values, 100)


def test_null_and_overflow_contracts() -> None:
    out_left = [1]
    out_right = [2]
    out_indices = [3]
    out_ranks = [4]
    out_count = [5]

    err = tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks(
        None,
        0,
        [],
        [],
        [],
        0,
        out_left,
        out_right,
        out_indices,
        out_ranks,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks(
        [1, 2],
        I64_MAX + 1,
        [],
        [],
        [],
        0,
        out_left,
        out_right,
        out_indices,
        out_ranks,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    err = tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks(
        [1, 2],
        2,
        None,
        None,
        None,
        1,
        out_left,
        out_right,
        out_indices,
        out_ranks,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_sorted_parity_vs_default() -> None:
    rng = random.Random(20260418_409)

    for i in range(5000):
        token_count = rng.randint(0, 80)
        rank_count = rng.randint(0, 220)

        tokens = [rng.randint(0, 180) for _ in range(token_count)]
        rows = [
            (rng.randint(0, 180), rng.randint(0, 180), rng.randint(0, 20000))
            for _ in range(rank_count)
        ]
        rows.sort(key=lambda row: (row[0], row[1]))
        rank_left = [row[0] for row in rows]
        rank_right = [row[1] for row in rows]
        rank_values = [row[2] for row in rows]

        _run_parity_case(tokens, rank_left, rank_right, rank_values, i + 1000)


if __name__ == "__main__":
    test_unsorted_rank_table_rejected_without_writes()
    test_duplicate_pair_ties_sorted_success_parity()
    test_null_and_overflow_contracts()
    test_randomized_sorted_parity_vs_default()
    print(
        "tokenizer_bpe_merge_candidates_build_checked_default_validate_ranks_reference_checks=ok"
    )
