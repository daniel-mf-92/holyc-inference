#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeApplyBestPriorityChecked semantics."""

from __future__ import annotations

import random

TOKENIZER_BPE_OK = 0
TOKENIZER_BPE_ERR_NULL_PTR = 101
TOKENIZER_BPE_ERR_BAD_PARAM = 102
TOKENIZER_BPE_ERR_OVERFLOW = 103

I64_MAX = (1 << 63) - 1


def tokenizer_bpe_merge_pair_priority_find_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_left_index: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if token_ids is None or out_left_index is None or out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_capacity > I64_MAX or rank_table_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if token_count > token_capacity or rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None or rank_right_tokens is None or rank_values is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count < 2:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    found = False
    best_left_index = 0
    best_rank = 0

    for i in range(token_count - 1):
        left = token_ids[i]
        right = token_ids[i + 1]

        pair_rank_found = False
        pair_rank = 0
        for j in range(rank_table_count):
            if rank_left_tokens[j] != left:
                continue
            if rank_right_tokens[j] != right:
                continue
            if not pair_rank_found or rank_values[j] < pair_rank:
                pair_rank = rank_values[j]
                pair_rank_found = True

        if not pair_rank_found:
            continue

        if not found or pair_rank < best_rank:
            best_rank = pair_rank
            best_left_index = i
            found = True

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_left_index[0] = best_left_index
    out_rank[0] = best_rank
    out_found[0] = True
    return TOKENIZER_BPE_OK


def tokenizer_bpe_merge_pair_token_lookup_checked(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_merged_token: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if out_merged_token is None or out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if rank_table_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None
        or rank_right_tokens is None
        or rank_values is None
        or rank_merged_tokens is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if rank_table_count == 0:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    lo = 0
    hi = rank_table_count
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        key_mid = (rank_left_tokens[mid], rank_right_tokens[mid])
        if key_mid < (left_token, right_token):
            lo = mid + 1
        else:
            hi = mid

    first = lo
    if first >= rank_table_count:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    if (rank_left_tokens[first], rank_right_tokens[first]) != (left_token, right_token):
        out_found[0] = False
        return TOKENIZER_BPE_OK

    scan = first
    found = False
    best_rank = 0
    best_merged = 0
    while scan < rank_table_count:
        if (rank_left_tokens[scan], rank_right_tokens[scan]) != (left_token, right_token):
            break

        if not found or rank_values[scan] < best_rank:
            best_rank = rank_values[scan]
            best_merged = rank_merged_tokens[scan]
            found = True
        scan += 1

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_merged_token[0] = best_merged
    out_rank[0] = best_rank
    out_found[0] = True
    return TOKENIZER_BPE_OK


def tokenizer_bpe_merge_apply_at_index_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    left_index: int,
    merged_token: int,
    out_token_count: list[int] | None,
) -> int:
    if token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if token_count > token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if token_count < 2:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if left_index >= token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    right_index = left_index + 1
    if right_index <= left_index:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if right_index >= token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    token_ids[left_index] = merged_token

    src_index = right_index + 1
    while src_index < token_count:
        dst_index = src_index - 1
        if dst_index >= token_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW
        token_ids[dst_index] = token_ids[src_index]
        src_index += 1

    out_token_count[0] = token_count - 1
    return TOKENIZER_BPE_OK


def tokenizer_bpe_merge_apply_best_priority_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
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

    if token_capacity > I64_MAX or rank_table_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if token_count > token_capacity or rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None
        or rank_right_tokens is None
        or rank_values is None
        or rank_merged_tokens is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count < 2:
        out_found[0] = False
        out_token_count[0] = token_count
        return TOKENIZER_BPE_OK

    best_left_index = [0]
    best_rank = [0]
    best_found = [False]
    err = tokenizer_bpe_merge_pair_priority_find_checked(
        token_ids,
        token_count,
        token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_table_count,
        rank_table_capacity,
        best_left_index,
        best_rank,
        best_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if not best_found[0]:
        out_found[0] = False
        out_token_count[0] = token_count
        return TOKENIZER_BPE_OK

    best_right_index = best_left_index[0] + 1
    if best_right_index <= best_left_index[0] or best_right_index >= token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    merged_token = [0]
    pair_rank = [0]
    pair_found = [False]
    err = tokenizer_bpe_merge_pair_token_lookup_checked(
        token_ids[best_left_index[0]],
        token_ids[best_right_index],
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        merged_token,
        pair_rank,
        pair_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if not pair_found[0] or pair_rank[0] != best_rank[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_tokens = token_ids[:token_count]
    next_count = [0]
    err = tokenizer_bpe_merge_apply_at_index_checked(
        staged_tokens,
        token_count,
        token_capacity,
        best_left_index[0],
        merged_token[0],
        next_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    for i in range(next_count[0]):
        token_ids[i] = staged_tokens[i]

    out_merged_token[0] = merged_token[0]
    out_rank[0] = best_rank[0]
    out_found[0] = True
    out_token_count[0] = next_count[0]
    return TOKENIZER_BPE_OK


def two_step_reference(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    rank_merged_tokens: list[int],
) -> tuple[int, list[int], int, int, bool, int]:
    tokens = token_ids.copy()
    out_left = [777]
    out_rank = [888]
    out_found = [False]
    err = tokenizer_bpe_merge_pair_priority_find_checked(
        tokens,
        len(tokens),
        len(tokens),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        len(rank_values),
        len(rank_values),
        out_left,
        out_rank,
        out_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err, tokens, -1, -1, False, len(tokens)

    if not out_found[0]:
        return TOKENIZER_BPE_OK, tokens, -1, -1, False, len(tokens)

    merged_token = [0]
    pair_rank = [0]
    pair_found = [False]
    err = tokenizer_bpe_merge_pair_token_lookup_checked(
        tokens[out_left[0]],
        tokens[out_left[0] + 1],
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        len(rank_values),
        len(rank_values),
        merged_token,
        pair_rank,
        pair_found,
    )
    if err != TOKENIZER_BPE_OK:
        return err, tokens, -1, -1, False, len(tokens)

    if not pair_found[0] or pair_rank[0] != out_rank[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM, tokens, -1, -1, False, len(tokens)

    out_count = [0]
    err = tokenizer_bpe_merge_apply_at_index_checked(
        tokens,
        len(tokens),
        len(tokens),
        out_left[0],
        merged_token[0],
        out_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err, tokens, -1, -1, False, len(tokens)

    return TOKENIZER_BPE_OK, tokens[: out_count[0]], merged_token[0], out_rank[0], True, out_count[0]


def run_case(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    rank_merged_tokens: list[int],
) -> None:
    tokens = token_ids.copy()
    out_merged = [2222]
    out_rank = [3333]
    out_found = [False]
    out_count = [4444]

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        tokens,
        len(token_ids),
        len(token_ids),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        len(rank_values),
        len(rank_values),
        out_merged,
        out_rank,
        out_found,
        out_count,
    )

    ref_err, ref_tokens, ref_merged, ref_rank, ref_found, ref_count = two_step_reference(
        token_ids,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
    )

    assert err == ref_err
    assert out_found[0] == ref_found
    assert out_count[0] == ref_count
    assert tokens[:out_count[0]] == ref_tokens

    if ref_found:
        assert out_merged[0] == ref_merged
        assert out_rank[0] == ref_rank
    else:
        assert out_merged[0] == 2222
        assert out_rank[0] == 3333


def test_hit_miss_tie_and_duplicate_rank_vectors() -> None:
    run_case(
        [10, 20, 30, 20],
        [10, 20, 30],
        [20, 30, 20],
        [7, 4, 9],
        [120, 230, 320],
    )

    # Two candidate pairs with equal rank: leftmost merge should win.
    run_case(
        [1, 2, 3, 2],
        [1, 2],
        [2, 3],
        [5, 5],
        [12, 23],
    )

    # Duplicate pair key with different ranks: lowest rank determines merge.
    run_case(
        [4, 5, 6],
        [4, 4, 5],
        [5, 5, 6],
        [12, 3, 9],
        [450, 403, 560],
    )

    # Miss path: no ranked pairs.
    run_case(
        [9, 8, 7],
        [1, 2],
        [3, 4],
        [10, 20],
        [100, 200],
    )


def test_small_span_and_parameter_contracts() -> None:
    tokens = [77]
    out_merged = [111]
    out_rank = [222]
    out_found = [True]
    out_count = [333]

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        tokens,
        1,
        1,
        [],
        [],
        [],
        [],
        0,
        0,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_found[0] is False
    assert out_count[0] == 1
    assert out_merged[0] == 111
    assert out_rank[0] == 222

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        None,
        0,
        0,
        [],
        [],
        [],
        [],
        0,
        0,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        [1, 2],
        2,
        I64_MAX + 1,
        [1],
        [2],
        [3],
        [4],
        1,
        1,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        [1, 2],
        3,
        2,
        [1],
        [2],
        [3],
        [4],
        1,
        1,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM


def test_no_partial_outputs_and_tokens_on_error() -> None:
    tokens = [1, 2, 3, 4]
    baseline = tokens.copy()

    out_merged = [901]
    out_rank = [902]
    out_found = [True]
    out_count = [903]

    # Force BAD_PARAM via rank count > capacity.
    err = tokenizer_bpe_merge_apply_best_priority_checked(
        tokens,
        len(tokens),
        len(tokens),
        [1, 2],
        [2, 3],
        [10, 11],
        [100, 101],
        2,
        1,
        out_merged,
        out_rank,
        out_found,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert tokens == baseline
    assert out_merged[0] == 901
    assert out_rank[0] == 902
    assert out_found[0] is True
    assert out_count[0] == 903


def test_randomized_parity_against_explicit_two_step_reference() -> None:
    rng = random.Random(20260418_368)

    for _ in range(4000):
        token_count = rng.randint(0, 40)
        tokens = [rng.randint(0, 40) for _ in range(token_count)]

        rank_count = rng.randint(0, 220)
        raw = [
            (rng.randint(0, 40), rng.randint(0, 40), rng.randint(0, 4096), rng.randint(1000, 6000))
            for _ in range(rank_count)
        ]
        raw.sort(key=lambda row: (row[0], row[1]))

        rank_left = [r[0] for r in raw]
        rank_right = [r[1] for r in raw]
        rank_values = [r[2] for r in raw]
        rank_merged = [r[3] for r in raw]

        run_case(tokens, rank_left, rank_right, rank_values, rank_merged)


if __name__ == "__main__":
    test_hit_miss_tie_and_duplicate_rank_vectors()
    test_small_span_and_parameter_contracts()
    test_no_partial_outputs_and_tokens_on_error()
    test_randomized_parity_against_explicit_two_step_reference()
    print("ok")
