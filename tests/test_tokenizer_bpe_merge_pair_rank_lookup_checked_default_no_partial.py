#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairRankLookupCheckedDefaultNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_token_pair_rank_lookup_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_token_pair_rank_lookup_checked,
)


def tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial(
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

    derived_capacity = rank_table_count

    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]

    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_capacity,
        staged_rank,
        staged_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_rank[0] = staged_rank[0]
    out_found[0] = staged_found[0]
    return TOKENIZER_BPE_OK


def explicit_staged_composition(
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

    derived_capacity = rank_table_count

    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]

    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_capacity,
        staged_rank,
        staged_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_rank[0] = staged_rank[0]
    out_found[0] = staged_found[0]
    return TOKENIZER_BPE_OK


def run_case(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
) -> None:
    out_rank_default = [0x5252]
    out_found_default = [True]
    out_rank_explicit = [0x5252]
    out_found_explicit = [True]

    err_default = tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        out_rank_default,
        out_found_default,
    )
    err_explicit = explicit_staged_composition(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        out_rank_explicit,
        out_found_explicit,
    )

    assert err_default == err_explicit
    assert out_rank_default[0] == out_rank_explicit[0]
    assert out_found_default[0] == out_found_explicit[0]


def test_known_vectors_parity() -> None:
    triples = sorted(
        [
            (1, 2, 40),
            (1, 2, 7),
            (1, 3, 8),
            (2, 3, 5),
            (5, 8, 9),
        ],
        key=lambda x: (x[0], x[1]),
    )
    left = [it[0] for it in triples]
    right = [it[1] for it in triples]
    rank = [it[2] for it in triples]

    run_case(1, 2, left, right, rank, len(rank))
    run_case(5, 8, left, right, rank, len(rank))
    run_case(9, 9, left, right, rank, len(rank))


def test_null_and_overflow_contracts() -> None:
    out_rank = [0x7777]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial(
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
    assert out_rank[0] == 0x7777
    assert out_found[0] is False

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial(
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

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial(
        1,
        2,
        [],
        [],
        [],
        0,
        out_rank,
        None,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_no_partial_malformed_capacity_and_null_rank_table() -> None:
    out_rank = [0x6262]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial(
        3,
        4,
        None,
        [4],
        [5],
        1,
        out_rank,
        out_found,
    )
    assert err != TOKENIZER_BPE_OK
    assert out_rank[0] == 0x6262
    assert out_found[0] is True


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_398)

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

        run_case(
            query_left,
            query_right,
            rank_left,
            rank_right,
            rank_values,
            len(rows),
        )


if __name__ == "__main__":
    test_known_vectors_parity()
    test_null_and_overflow_contracts()
    test_no_partial_malformed_capacity_and_null_rank_table()
    test_randomized_parity_vs_explicit_staged_composition()
    print("tokenizer_bpe_merge_pair_rank_lookup_checked_default_no_partial_reference_checks=ok")
