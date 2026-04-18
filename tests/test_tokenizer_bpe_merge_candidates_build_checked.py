#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeCandidatesBuildChecked semantics."""

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

    best_rank = 0
    found = False
    for i in range(rank_table_count):
        if rank_left_tokens[i] != left_token:
            continue
        if rank_right_tokens[i] != right_token:
            continue

        if (not found) or (rank_values[i] < best_rank):
            best_rank = rank_values[i]
            found = True

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_rank[0] = best_rank
    out_found[0] = True
    return TOKENIZER_BPE_OK


def tokenizer_bpe_merge_candidates_build_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_left_tokens: list[int] | None,
    out_right_tokens: list[int] | None,
    out_left_indices: list[int] | None,
    out_ranks: list[int] | None,
    out_candidate_capacity: int,
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

    if (
        token_capacity > I64_MAX
        or rank_table_capacity > I64_MAX
        or out_candidate_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if token_count > token_capacity or rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None or rank_right_tokens is None or rank_values is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count < 2:
        out_candidate_count[0] = 0
        return TOKENIZER_BPE_OK

    candidate_count = 0

    for i in range(token_count - 1):
        rank = [0]
        found = [False]
        err = tokenizer_bpe_token_pair_rank_lookup_checked(
            token_ids[i],
            token_ids[i + 1],
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_table_count,
            rank_table_capacity,
            rank,
            found,
        )
        if err != TOKENIZER_BPE_OK:
            return err

        if not found[0]:
            continue

        if candidate_count == out_candidate_capacity:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        candidate_count += 1

    candidate_count = 0
    for i in range(token_count - 1):
        rank = [0]
        found = [False]
        err = tokenizer_bpe_token_pair_rank_lookup_checked(
            token_ids[i],
            token_ids[i + 1],
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_table_count,
            rank_table_capacity,
            rank,
            found,
        )
        if err != TOKENIZER_BPE_OK:
            return err

        if not found[0]:
            continue

        if candidate_count >= out_candidate_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

        out_left_tokens[candidate_count] = token_ids[i]
        out_right_tokens[candidate_count] = token_ids[i + 1]
        out_left_indices[candidate_count] = i
        out_ranks[candidate_count] = rank[0]
        candidate_count += 1

    out_candidate_count[0] = candidate_count
    return TOKENIZER_BPE_OK


def reference_candidates(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
) -> tuple[list[int], list[int], list[int], list[int]]:
    left_tokens: list[int] = []
    right_tokens: list[int] = []
    left_indices: list[int] = []
    ranks: list[int] = []

    for i in range(len(token_ids) - 1):
        left = token_ids[i]
        right = token_ids[i + 1]

        found = False
        best_rank = 0
        for j in range(len(rank_values)):
            if rank_left_tokens[j] != left:
                continue
            if rank_right_tokens[j] != right:
                continue
            if (not found) or rank_values[j] < best_rank:
                best_rank = rank_values[j]
                found = True

        if not found:
            continue

        left_tokens.append(left)
        right_tokens.append(right)
        left_indices.append(i)
        ranks.append(best_rank)

    return left_tokens, right_tokens, left_indices, ranks


def run_known_case(
    token_ids: list[int],
    rank_left: list[int],
    rank_right: list[int],
    rank_values: list[int],
) -> None:
    cap = max(1, len(token_ids))
    out_left_tokens = [-111] * cap
    out_right_tokens = [-222] * cap
    out_left_indices = [333] * cap
    out_ranks = [444] * cap
    out_count = [777]

    err = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        len(token_ids),
        len(token_ids),
        rank_left,
        rank_right,
        rank_values,
        len(rank_values),
        len(rank_values),
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        cap,
        out_count,
    )
    assert err == TOKENIZER_BPE_OK

    ref_lt, ref_rt, ref_idx, ref_rank = reference_candidates(
        token_ids,
        rank_left,
        rank_right,
        rank_values,
    )
    assert out_count[0] == len(ref_lt)
    assert out_left_tokens[: out_count[0]] == ref_lt
    assert out_right_tokens[: out_count[0]] == ref_rt
    assert out_left_indices[: out_count[0]] == ref_idx
    assert out_ranks[: out_count[0]] == ref_rank


def test_ranked_fixture_vectors() -> None:
    run_known_case(
        [10, 20, 30, 20],
        [10, 20, 30],
        [20, 30, 20],
        [7, 4, 9],
    )

    run_known_case(
        [7, 8, 7, 8, 9],
        [7, 7, 8, 8],
        [8, 8, 7, 9],
        [12, 3, 6, 2],
    )

    run_known_case(
        [4, 5, 6],
        [4, 4, 5],
        [5, 5, 6],
        [12, 3, 9],
    )


def test_no_match_and_short_span_behavior() -> None:
    out_left_tokens = [111, 111]
    out_right_tokens = [222, 222]
    out_left_indices = [333, 333]
    out_ranks = [444, 444]
    out_count = [999]

    err = tokenizer_bpe_merge_candidates_build_checked(
        [42],
        1,
        1,
        [1],
        [2],
        [3],
        1,
        1,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        2,
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_count[0] == 0
    assert out_left_tokens == [111, 111]
    assert out_right_tokens == [222, 222]
    assert out_left_indices == [333, 333]
    assert out_ranks == [444, 444]

    out_count2 = [321]
    err = tokenizer_bpe_merge_candidates_build_checked(
        [1, 2, 3],
        3,
        3,
        [9],
        [9],
        [9],
        1,
        1,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        2,
        out_count2,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_count2[0] == 0


def test_malformed_span_boundaries_and_no_partial_on_error() -> None:
    token_ids = [10, 20, 30, 40]
    rank_left = [10, 20, 30]
    rank_right = [20, 30, 40]
    rank_values = [1, 2, 3]

    out_left_tokens = [91, 92, 93]
    out_right_tokens = [81, 82, 83]
    out_left_indices = [71, 72, 73]
    out_ranks = [61, 62, 63]
    out_count = [51]

    err = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        5,
        4,
        rank_left,
        rank_right,
        rank_values,
        3,
        3,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        3,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_count[0] == 51
    assert out_left_tokens == [91, 92, 93]
    assert out_right_tokens == [81, 82, 83]
    assert out_left_indices == [71, 72, 73]
    assert out_ranks == [61, 62, 63]

    out_count2 = [52]
    err = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        4,
        4,
        rank_left,
        rank_right,
        rank_values,
        3,
        3,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        2,
        out_count2,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_count2[0] == 52
    assert out_left_tokens == [91, 92, 93]
    assert out_right_tokens == [81, 82, 83]
    assert out_left_indices == [71, 72, 73]
    assert out_ranks == [61, 62, 63]

    out_count3 = [53]
    err = tokenizer_bpe_merge_candidates_build_checked(
        token_ids,
        4,
        I64_MAX + 1,
        rank_left,
        rank_right,
        rank_values,
        3,
        3,
        out_left_tokens,
        out_right_tokens,
        out_left_indices,
        out_ranks,
        3,
        out_count3,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_count3[0] == 53


def test_randomized_reference_parity() -> None:
    rng = random.Random(20260418_364)

    for _ in range(4000):
        token_count = rng.randint(0, 80)
        token_capacity = token_count + rng.randint(0, 8)
        rank_count = rng.randint(0, 200)
        rank_capacity = rank_count + rng.randint(0, 8)

        tokens = [rng.randint(-200, 200) for _ in range(token_capacity)]
        rank_left = [rng.randint(-200, 200) for _ in range(rank_capacity)]
        rank_right = [rng.randint(-200, 200) for _ in range(rank_capacity)]
        rank_values = [rng.randint(0, 5000) for _ in range(rank_capacity)]

        out_capacity = max(1, token_capacity)
        out_left_tokens = [-999] * out_capacity
        out_right_tokens = [-998] * out_capacity
        out_left_indices = [997] * out_capacity
        out_ranks = [996] * out_capacity
        out_count = [995]

        err = tokenizer_bpe_merge_candidates_build_checked(
            tokens,
            token_count,
            token_capacity,
            rank_left,
            rank_right,
            rank_values,
            rank_count,
            rank_capacity,
            out_left_tokens,
            out_right_tokens,
            out_left_indices,
            out_ranks,
            out_capacity,
            out_count,
        )
        assert err == TOKENIZER_BPE_OK

        ref_lt, ref_rt, ref_idx, ref_rank = reference_candidates(
            tokens[:token_count],
            rank_left[:rank_count],
            rank_right[:rank_count],
            rank_values[:rank_count],
        )

        assert out_count[0] == len(ref_lt)
        assert out_left_tokens[: out_count[0]] == ref_lt
        assert out_right_tokens[: out_count[0]] == ref_rt
        assert out_left_indices[: out_count[0]] == ref_idx
        assert out_ranks[: out_count[0]] == ref_rank


def main() -> None:
    test_ranked_fixture_vectors()
    test_no_match_and_short_span_behavior()
    test_malformed_span_boundaries_and_no_partial_on_error()
    test_randomized_reference_parity()
    print("ok")


if __name__ == "__main__":
    main()
