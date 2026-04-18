#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairTokenLookupCheckedDefaultNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_merge_apply_best_priority_checked import (
    TOKENIZER_BPE_ERR_BAD_PARAM,
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_merge_pair_token_lookup_checked,
)


def tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
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

    staged_merged = out_merged_token[0]
    staged_rank = out_rank[0]
    staged_found = out_found[0]

    out_merged_staged = [staged_merged]
    out_rank_staged = [staged_rank]
    out_found_staged = [staged_found]

    err = tokenizer_bpe_merge_pair_token_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        derived_capacity,
        out_merged_staged,
        out_rank_staged,
        out_found_staged,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_merged_token[0] = out_merged_staged[0]
    out_rank[0] = out_rank_staged[0]
    out_found[0] = out_found_staged[0]
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

    derived_capacity = rank_table_count

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
        derived_capacity,
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
    out_merged_default = [0x3131]
    out_rank_default = [0x4141]
    out_found_default = [False]

    out_merged_explicit = [0x3131]
    out_rank_explicit = [0x4141]
    out_found_explicit = [False]

    err_default = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
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
    err_explicit = explicit_staged_composition(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        out_merged_explicit,
        out_rank_explicit,
        out_found_explicit,
    )

    assert err_default == err_explicit
    assert out_merged_default[0] == out_merged_explicit[0]
    assert out_rank_default[0] == out_rank_explicit[0]
    assert out_found_default[0] == out_found_explicit[0]


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


def test_duplicate_rank_keeps_first_min_rank_merged_token() -> None:
    # Determinism contract for duplicate keys: choose the earliest merged token
    # among rows that share the minimum rank for the queried pair.
    rank_rows = sorted(
        [
            (12, 34, 9, 2009),
            (12, 34, 4, 2004),
            (12, 34, 4, 2005),
            (12, 34, 7, 2007),
            (12, 35, 1, 2035),
        ],
        key=lambda x: (x[0], x[1]),
    )

    rank_left = [row[0] for row in rank_rows]
    rank_right = [row[1] for row in rank_rows]
    rank_values = [row[2] for row in rank_rows]
    rank_merged = [row[3] for row in rank_rows]

    out_merged = [0x1111]
    out_rank = [0x2222]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
        12,
        34,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_rows),
        out_merged,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is True
    assert out_rank[0] == 4
    assert out_merged[0] == 2004


def test_no_partial_malformed_table_and_null_output_contracts() -> None:
    out_merged = [0x7777]
    out_rank = [0x6666]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
        1,
        2,
        None,
        [2],
        [3],
        [4],
        1,
        out_merged,
        out_rank,
        out_found,
    )
    assert err != TOKENIZER_BPE_OK
    assert out_merged[0] == 0x7777
    assert out_rank[0] == 0x6666
    assert out_found[0] is True

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
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

    err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
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

    # Truncated column vectors are malformed table input: checked core should
    # reject via bad-param/exception path and wrapper must preserve outputs.
    try:
        err = tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial(
            4,
            7,
            [4],
            [7],
            [2],
            [],
            1,
            out_merged,
            out_rank,
            out_found,
        )
        assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    except IndexError:
        pass

    assert out_merged[0] == 0x7777
    assert out_rank[0] == 0x6666
    assert out_found[0] is True


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_400)

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


if __name__ == "__main__":
    test_known_vectors_duplicate_rank_and_missing_pair()
    test_duplicate_rank_keeps_first_min_rank_merged_token()
    test_no_partial_malformed_table_and_null_output_contracts()
    test_randomized_parity_vs_explicit_staged_composition()
    print("tokenizer_bpe_merge_pair_token_lookup_checked_default_no_partial_reference_checks=ok")
