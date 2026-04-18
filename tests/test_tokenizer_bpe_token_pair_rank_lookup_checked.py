#!/usr/bin/env python3
"""Parity harness for TokenizerBPETokenPairRankLookupChecked semantics."""

from __future__ import annotations

import random

TOKENIZER_BPE_OK = 0
TOKENIZER_BPE_ERR_NULL_PTR = 101
TOKENIZER_BPE_ERR_BAD_PARAM = 102
TOKENIZER_BPE_ERR_OVERFLOW = 103

I64_MAX = (1 << 63) - 1


def tokenizer_bpe_token_pair_rank_lookup_checked(
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

    if rank_table_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None or rank_right_tokens is None or rank_values is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if rank_table_count == 0:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    lo = 0
    hi = rank_table_count
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        mid_left = rank_left_tokens[mid]
        mid_right = rank_right_tokens[mid]

        if mid_left < left_token or (mid_left == left_token and mid_right < right_token):
            lo = mid + 1
        else:
            hi = mid

    first_match = lo
    if first_match >= rank_table_count:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    if (
        rank_left_tokens[first_match] != left_token
        or rank_right_tokens[first_match] != right_token
    ):
        out_found[0] = False
        return TOKENIZER_BPE_OK

    best_rank = rank_values[first_match]
    scan = first_match + 1
    while scan < rank_table_count:
        if rank_left_tokens[scan] != left_token:
            break
        if rank_right_tokens[scan] != right_token:
            break
        if rank_values[scan] < best_rank:
            best_rank = rank_values[scan]
        scan += 1

    out_rank[0] = best_rank
    out_found[0] = True
    return TOKENIZER_BPE_OK


def reference_rank_lookup(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
) -> tuple[bool, int]:
    hit = False
    best = 0
    for i in range(len(rank_values)):
        if rank_left_tokens[i] != left_token:
            continue
        if rank_right_tokens[i] != right_token:
            continue
        if not hit or rank_values[i] < best:
            best = rank_values[i]
            hit = True
    return hit, best


def sorted_rank_table(
    triples: list[tuple[int, int, int]],
) -> tuple[list[int], list[int], list[int]]:
    triples = sorted(triples, key=lambda item: (item[0], item[1]))
    return (
        [left for left, _, _ in triples],
        [right for _, right, _ in triples],
        [rank for _, _, rank in triples],
    )


def test_known_hit_miss_boundary_vectors() -> None:
    lefts, rights, ranks = sorted_rank_table(
        [
            (1, 2, 11),
            (1, 2, 4),
            (1, 3, 9),
            (4, 5, 2),
            (9, 9, 7),
        ]
    )

    out_rank = [999]
    out_found = [False]
    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        1,
        2,
        lefts,
        rights,
        ranks,
        len(ranks),
        len(ranks),
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is True
    assert out_rank[0] == 4

    out_rank2 = [123]
    out_found2 = [True]
    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        1,
        4,
        lefts,
        rights,
        ranks,
        len(ranks),
        len(ranks),
        out_rank2,
        out_found2,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found2[0] is False
    assert out_rank2[0] == 123

    out_rank3 = [77]
    out_found3 = [True]
    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        9,
        9,
        lefts,
        rights,
        ranks,
        len(ranks),
        len(ranks),
        out_rank3,
        out_found3,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found3[0] is True
    assert out_rank3[0] == 7


def test_parameter_contracts_and_no_partial_on_error() -> None:
    out_rank = [555]
    out_found = [True]

    assert (
        tokenizer_bpe_token_pair_rank_lookup_checked(
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
        == TOKENIZER_BPE_ERR_OVERFLOW
    )
    assert out_rank[0] == 555 and out_found[0] is True

    out_rank2 = [556]
    out_found2 = [True]
    assert (
        tokenizer_bpe_token_pair_rank_lookup_checked(
            1,
            2,
            [],
            [],
            [],
            3,
            2,
            out_rank2,
            out_found2,
        )
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )
    assert out_rank2[0] == 556 and out_found2[0] is True

    out_rank3 = [557]
    out_found3 = [True]
    assert (
        tokenizer_bpe_token_pair_rank_lookup_checked(
            1,
            2,
            None,
            [2],
            [3],
            1,
            1,
            out_rank3,
            out_found3,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )
    assert out_rank3[0] == 557 and out_found3[0] is True


def test_randomized_sorted_table_vectors() -> None:
    rng = random.Random(20260418_354)

    for _ in range(5000):
        n = rng.randint(0, 300)
        triples: list[tuple[int, int, int]] = []
        for _ in range(n):
            triples.append((rng.randint(0, 64), rng.randint(0, 64), rng.randint(0, 5000)))

        lefts, rights, ranks = sorted_rank_table(triples)

        query_left = rng.randint(0, 64)
        query_right = rng.randint(0, 64)

        out_rank = [4242]
        out_found = [False]

        err = tokenizer_bpe_token_pair_rank_lookup_checked(
            query_left,
            query_right,
            lefts,
            rights,
            ranks,
            len(ranks),
            len(ranks),
            out_rank,
            out_found,
        )
        assert err == TOKENIZER_BPE_OK

        ref_found, ref_rank = reference_rank_lookup(
            query_left,
            query_right,
            lefts,
            rights,
            ranks,
        )
        assert out_found[0] == ref_found
        if ref_found:
            assert out_rank[0] == ref_rank
        else:
            assert out_rank[0] == 4242


def test_empty_table_miss() -> None:
    out_rank = [901]
    out_found = [True]
    err = tokenizer_bpe_token_pair_rank_lookup_checked(
        3,
        4,
        [],
        [],
        [],
        0,
        0,
        out_rank,
        out_found,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is False
    assert out_rank[0] == 901


if __name__ == "__main__":
    test_known_hit_miss_boundary_vectors()
    test_parameter_contracts_and_no_partial_on_error()
    test_randomized_sorted_table_vectors()
    test_empty_table_miss()
    print("tokenizer_bpe_token_pair_rank_lookup_checked_reference_checks=ok")
