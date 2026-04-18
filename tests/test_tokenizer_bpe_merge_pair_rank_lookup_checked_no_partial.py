#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergePairRankLookupCheckedNoPartial semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_token_pair_rank_lookup_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_token_pair_rank_lookup_checked,
)


def tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    staged_rank = out_rank[0]
    staged_found = out_found[0]

    staged_rank_box = [staged_rank]
    staged_found_box = [staged_found]

    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_capacity,
        staged_rank_box,
        staged_found_box,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_rank[0] = staged_rank_box[0]
    out_found[0] = staged_found_box[0]
    return TOKENIZER_BPE_OK


def run_case(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
) -> None:
    out_rank_core = [0x7171]
    out_found_core = [True]
    out_rank_wrapped = [0x7171]
    out_found_wrapped = [True]

    # Explicit staged composition: mirrors no-partial wrapper contract exactly.
    staged_rank = [out_rank_core[0]]
    staged_found = [out_found_core[0]]
    err_core = tokenizer_bpe_token_pair_rank_lookup_checked(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_capacity,
        staged_rank,
        staged_found,
    )
    if err_core == TOKENIZER_BPE_OK:
        out_rank_core[0] = staged_rank[0]
        out_found_core[0] = staged_found[0]

    err_wrapped = tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial(
        left_token,
        right_token,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_capacity,
        out_rank_wrapped,
        out_found_wrapped,
    )

    assert err_wrapped == err_core
    assert out_rank_wrapped[0] == out_rank_core[0]
    assert out_found_wrapped[0] == out_found_core[0]


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

    run_case(1, 2, left, right, rank, len(rank), len(rank))
    run_case(5, 8, left, right, rank, len(rank), len(rank))
    run_case(9, 9, left, right, rank, len(rank), len(rank))


def test_null_output_adversarial_vectors() -> None:
    out_rank = [0x4444]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial(
        1,
        2,
        [1],
        [2],
        [3],
        1,
        1,
        None,
        out_found,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert out_found[0] is False

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial(
        1,
        2,
        [1],
        [2],
        [3],
        1,
        1,
        out_rank,
        None,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert out_rank[0] == 0x4444


def test_malformed_capacity_no_partial() -> None:
    out_rank = [0x8888]
    out_found = [True]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial(
        1,
        2,
        [1],
        [2],
        [3],
        3,
        2,
        out_rank,
        out_found,
    )
    assert err != TOKENIZER_BPE_OK
    assert out_rank[0] == 0x8888
    assert out_found[0] is True


def test_overflow_capacity_no_partial() -> None:
    out_rank = [0x3333]
    out_found = [False]

    err = tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial(
        1,
        2,
        [],
        [],
        [],
        0,
        I64_MAX + 1,
        out_rank,
        out_found,
    )
    assert err != TOKENIZER_BPE_OK
    assert out_rank[0] == 0x3333
    assert out_found[0] is False


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_384)

    for _ in range(7000):
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

        # Include malformed-capacity adversarial vectors regularly.
        if n > 0 and (rng.randint(0, 7) == 0):
            rank_capacity = n - 1
        else:
            rank_capacity = n

        run_case(
            query_left,
            query_right,
            left,
            right,
            ranks,
            n,
            rank_capacity,
        )


if __name__ == "__main__":
    test_known_vectors_parity()
    test_null_output_adversarial_vectors()
    test_malformed_capacity_no_partial()
    test_overflow_capacity_no_partial()
    test_randomized_parity_vs_explicit_staged_composition()
    print("tokenizer_bpe_merge_pair_rank_lookup_checked_no_partial_reference_checks=ok")
