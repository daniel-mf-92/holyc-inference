#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeApplyBestPriorityCheckedDefaultNoPartial."""

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
    tokenizer_bpe_merge_apply_best_priority_checked,
)


def tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial(
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

    staged_merged = [out_merged_token[0]]
    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]
    staged_count = [out_token_count[0]]

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        token_ids,
        token_count,
        derived_token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        derived_rank_table_capacity,
        staged_merged,
        staged_rank,
        staged_found,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_merged_token[0] = staged_merged[0]
    out_rank[0] = staged_rank[0]
    out_found[0] = staged_found[0]
    out_token_count[0] = staged_count[0]
    return TOKENIZER_BPE_OK


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
    merged_init = 500000 + seed
    rank_init = 600000 + seed
    found_init = (seed % 2) == 0
    count_init = 700000 + seed

    tokens_expected = token_ids[:] if isinstance(token_ids, list) else None
    tokens_default_np = token_ids[:] if isinstance(token_ids, list) else None

    merged_expected = [merged_init]
    rank_expected = [rank_init]
    found_expected = [found_init]
    count_expected = [count_init]

    merged_default_np = [merged_init]
    rank_default_np = [rank_init]
    found_default_np = [found_init]
    count_default_np = [count_init]

    # Explicit staged composition baseline:
    # 1) derive default capacities, 2) run checked core on staged outputs,
    # 3) commit outputs only if checked core succeeds.
    staged_merged = [merged_expected[0]]
    staged_rank = [rank_expected[0]]
    staged_found = [found_expected[0]]
    staged_count = [count_expected[0]]

    err_expected = tokenizer_bpe_merge_apply_best_priority_checked(
        tokens_expected,
        token_count,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_count,
        staged_merged,
        staged_rank,
        staged_found,
        staged_count,
    )
    if err_expected == TOKENIZER_BPE_OK:
        merged_expected[0] = staged_merged[0]
        rank_expected[0] = staged_rank[0]
        found_expected[0] = staged_found[0]
        count_expected[0] = staged_count[0]

    err_default_np = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial(
        tokens_default_np,
        token_count,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        merged_default_np,
        rank_default_np,
        found_default_np,
        count_default_np,
    )

    assert err_default_np == err_expected
    assert tokens_default_np == tokens_expected
    assert merged_default_np[0] == merged_expected[0]
    assert rank_default_np[0] == rank_expected[0]
    assert found_default_np[0] == found_expected[0]
    assert count_default_np[0] == count_expected[0]


def test_known_vectors_parity_vs_explicit_staged_composition() -> None:
    run_case([10, 20, 30, 20], 4, [10, 20, 30], [20, 30, 20], [7, 4, 9], [120, 230, 320], 3, 1)
    run_case([4, 5, 6], 3, [4, 4, 5], [5, 5, 6], [12, 3, 9], [450, 403, 560], 3, 2)
    run_case([1, 2, 3], 3, [9, 8], [7, 6], [5, 4], [100, 200], 2, 3)
    run_case([7], 1, [7], [8], [1], [12], 1, 4)


def test_exact_no_partial_write_parity_on_error_paths() -> None:
    merged = [111]
    rank = [222]
    found = [True]
    count = [333]

    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial(
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
    assert merged[0] == 111 and rank[0] == 222 and found[0] is True and count[0] == 333

    merged = [444]
    rank = [555]
    found = [False]
    count = [666]
    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial(
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
    assert merged[0] == 444 and rank[0] == 555 and found[0] is False and count[0] == 666

    merged = [777]
    rank = [888]
    found = [True]
    count = [999]
    err = tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial(
        [1, 2, 3],
        3,
        [1, 2],
        [2, 3],
        [10, 11],
        None,
        2,
        merged,
        rank,
        found,
        count,
    )
    assert err != TOKENIZER_BPE_OK
    assert merged[0] == 777 and rank[0] == 888 and found[0] is True and count[0] == 999


def test_randomized_parity_against_explicit_staged_composition() -> None:
    rng = random.Random(20260418_385)

    for i in range(6000):
        token_count = rng.randint(0, 40)
        tokens = [rng.randint(0, 80) for _ in range(token_count)]

        rank_count = rng.randint(0, 220)
        raw = [
            (rng.randint(0, 80), rng.randint(0, 80), rng.randint(0, 8192), rng.randint(1000, 9000))
            for _ in range(rank_count)
        ]
        raw.sort(key=lambda row: (row[0], row[1]))

        rank_left = [r[0] for r in raw]
        rank_right = [r[1] for r in raw]
        rank_values = [r[2] for r in raw]
        rank_merged = [r[3] for r in raw]

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
    test_known_vectors_parity_vs_explicit_staged_composition()
    test_exact_no_partial_write_parity_on_error_paths()
    test_randomized_parity_against_explicit_staged_composition()
    print("tokenizer_bpe_merge_apply_best_priority_checked_default_no_partial_reference_checks=ok")
