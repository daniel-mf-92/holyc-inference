#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeApplyBestPriorityCheckedDefaultNoPartialValidateRanks."""

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
)
from test_tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks import (
    tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks,
)


def tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks(
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

    return tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
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
    token_ids: list[int] | None,
    token_count: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    seed: int,
) -> None:
    merged_init = 100000 + seed
    rank_init = 200000 + seed
    found_init = (seed % 2) == 0
    count_init = 300000 + seed

    tokens_expected = token_ids[:] if isinstance(token_ids, list) else None
    tokens_default = token_ids[:] if isinstance(token_ids, list) else None

    merged_expected = [merged_init]
    rank_expected = [rank_init]
    found_expected = [found_init]
    count_expected = [count_init]

    merged_default = [merged_init]
    rank_default = [rank_init]
    found_default = [found_init]
    count_default = [count_init]

    err_expected = tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
        tokens_expected,
        token_count,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_count,
        merged_expected,
        rank_expected,
        found_expected,
        count_expected,
    )

    err_default = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks(
        tokens_default,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        merged_default,
        rank_default,
        found_default,
        count_default,
    )

    assert err_default == err_expected
    assert tokens_default == tokens_expected
    assert merged_default == merged_expected
    assert rank_default == rank_expected
    assert found_default == found_expected
    assert count_default == count_expected


def test_known_vectors_parity_vs_explicit_validate_ranks_composition() -> None:
    run_case([10, 20, 30, 20], 4, [10, 20, 30], [20, 30, 20], [7, 4, 9], [120, 230, 320], 3, 1)
    run_case([4, 5, 6], 3, [4, 4, 5], [5, 5, 6], [12, 3, 9], [450, 403, 560], 3, 2)
    run_case([1, 2, 3], 3, [9, 8], [7, 6], [5, 4], [100, 200], 2, 3)
    run_case([7], 1, [7], [8], [1], [12], 1, 4)


def test_unsorted_and_duplicate_conflict_rejections_propagate() -> None:
    tokens = [10, 20, 30]

    merged = [111]
    rank = [222]
    found = [True]
    count = [333]
    before = tokens.copy()
    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks(
        tokens,
        len(tokens),
        [10, 9],
        [20, 40],
        [2, 1],
        [99, 88],
        2,
        merged,
        rank,
        found,
        count,
    )
    assert err != TOKENIZER_BPE_OK
    assert tokens == before
    assert merged == [111] and rank == [222] and found == [True] and count == [333]

    merged = [444]
    rank = [555]
    found = [False]
    count = [666]
    before = tokens.copy()
    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks(
        tokens,
        len(tokens),
        [10, 10],
        [20, 20],
        [1, 1],
        [101, 202],
        2,
        merged,
        rank,
        found,
        count,
    )
    assert err != TOKENIZER_BPE_OK
    assert tokens == before
    assert merged == [444] and rank == [555] and found == [False] and count == [666]


def test_exact_no_partial_write_parity_on_error_paths() -> None:
    merged = [123]
    rank = [456]
    found = [True]
    count = [789]

    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks(
        None,
        0,
        [],
        [],
        [],
        [],
        0,
        merged,
        rank,
        found,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert merged == [123] and rank == [456] and found == [True] and count == [789]

    merged = [321]
    rank = [654]
    found = [False]
    count = [987]
    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks(
        [1, 2],
        I64_MAX + 1,
        [1],
        [2],
        [3],
        [4],
        1,
        merged,
        rank,
        found,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert merged == [321] and rank == [654] and found == [False] and count == [987]


def test_randomized_parity_against_explicit_validate_ranks_composition() -> None:
    rng = random.Random(20260418_414)

    for i in range(6000):
        token_count = rng.randint(0, 40)
        tokens = [rng.randint(0, 80) for _ in range(token_count)]

        rank_count = rng.randint(0, 220)
        rows = [
            (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 8192), rng.randint(1000, 9000))
            for _ in range(rank_count)
        ]
        rows.sort(key=lambda row: (row[0], row[1]))

        # Normalize duplicate min-rank ties so generated valid tables satisfy
        # deterministic duplicate-key merge guards in the validate-ranks core.
        fixed_rows: list[tuple[int, int, int, int]] = []
        run = 0
        while run < len(rows):
            left, right, _, _ = rows[run]
            run_end = run
            while run_end < len(rows) and rows[run_end][0] == left and rows[run_end][1] == right:
                run_end += 1

            min_rank = min(rows[j][2] for j in range(run, run_end))
            min_rows = [rows[j] for j in range(run, run_end) if rows[j][2] == min_rank]
            min_merged = min_rows[0][3] if min_rows else 0

            for j in range(run, run_end):
                row = rows[j]
                if row[2] == min_rank:
                    fixed_rows.append((row[0], row[1], row[2], min_merged))
                else:
                    fixed_rows.append(row)
            run = run_end

        rank_left = [r[0] for r in fixed_rows]
        rank_right = [r[1] for r in fixed_rows]
        rank_values = [r[2] for r in fixed_rows]
        rank_merged = [r[3] for r in fixed_rows]

        run_case(
            tokens,
            len(tokens),
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            len(rank_values),
            i + 10,
        )


if __name__ == "__main__":
    test_known_vectors_parity_vs_explicit_validate_ranks_composition()
    test_unsorted_and_duplicate_conflict_rejections_propagate()
    test_exact_no_partial_write_parity_on_error_paths()
    test_randomized_parity_against_explicit_validate_ranks_composition()
    print("tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_validate_ranks_reference_checks=ok")
