#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairRankLookupCheckedDefault semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_token_pair_rank_lookup_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    tokenizer_bpe_token_pair_rank_lookup_checked,
)

TOKENIZER_BPE_OK = 0


def tokenizer_bpe_merge_pair_rank_lookup_checked_default(
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

    return tokenizer_bpe_token_pair_rank_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        derived_capacity,
        out_rank,
        out_found,
    )


def run_case(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
) -> None:
    out_rank_core = [0x5151]
    out_found_core = [False]
    out_rank_default = [0x5151]
    out_found_default = [False]

    err_core = tokenizer_bpe_token_pair_rank_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_count,
        out_rank_core,
        out_found_core,
    )
    err_default = tokenizer_bpe_merge_pair_rank_lookup_checked_default(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        out_rank_default,
        out_found_default,
    )

    assert err_default == err_core
    assert out_rank_default[0] == out_rank_core[0]
    assert out_found_default[0] == out_found_core[0]


def test_known_vectors_parity() -> None:
    rank_tuples = sorted(
        [
            (1, 2, 40),
            (1, 2, 7),
            (1, 3, 8),
            (2, 3, 5),
            (5, 8, 9),
        ],
        key=lambda x: (x[0], x[1]),
    )
    left = [it[0] for it in rank_tuples]
    right = [it[1] for it in rank_tuples]
    rank = [it[2] for it in rank_tuples]

    run_case(1, 2, left, right, rank, len(rank))
    run_case(5, 8, left, right, rank, len(rank))
    run_case(9, 9, left, right, rank, len(rank))


def test_null_and_overflow_contracts() -> None:
    out_rank = [0x7777]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default(
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
    assert out_found[0] is True

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_default(
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


def test_randomized_parity_vs_explicit_capacity_core() -> None:
    rng = random.Random(20260418_369)

    for _ in range(6000):
        n = rng.randint(0, 300)
        triples = [
            (
                rng.randint(0, 127),
                rng.randint(0, 127),
                rng.randint(0, 5000),
            )
            for _ in range(n)
        ]
        triples.sort(key=lambda x: (x[0], x[1]))

        left = [x[0] for x in triples]
        right = [x[1] for x in triples]
        ranks = [x[2] for x in triples]

        query_left = rng.randint(0, 127)
        query_right = rng.randint(0, 127)

        run_case(query_left, query_right, left, right, ranks, len(ranks))


def test_malformed_capacity_adversarial_parity() -> None:
    out_rank_default = [1234]
    out_found_default = [True]
    out_rank_core = [1234]
    out_found_core = [True]

    # Force explicit malformed-capacity behavior from core and ensure the
    # default wrapper can only reach the well-formed equivalent by construction.
    err_core = tokenizer_bpe_token_pair_rank_lookup_checked(
        7,
        7,
        [7],
        [7],
        [1],
        1,
        0,
        out_rank_core,
        out_found_core,
    )
    assert err_core != TOKENIZER_BPE_OK
    assert out_rank_core[0] == 1234
    assert out_found_core[0] is True

    err_default = tokenizer_bpe_merge_pair_rank_lookup_checked_default(
        7,
        7,
        [7],
        [7],
        [1],
        1,
        out_rank_default,
        out_found_default,
    )
    assert err_default == TOKENIZER_BPE_OK
    assert out_rank_default[0] == 1
    assert out_found_default[0] is True


if __name__ == "__main__":
    test_known_vectors_parity()
    test_null_and_overflow_contracts()
    test_randomized_parity_vs_explicit_capacity_core()
    test_malformed_capacity_adversarial_parity()
    print("tokenizer_bpe_merge_pair_rank_lookup_checked_default_reference_checks=ok")
