#!/usr/bin/env python3
"""Parity harness for TokenizerBPEMergeApplyBestPriorityCheckedNoPartialValidateRanks."""

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
    best_left = 0
    best_rank = 0
    for i in range(token_count - 1):
        left = token_ids[i]
        right = token_ids[i + 1]

        pair_found = False
        pair_rank = 0
        for j in range(rank_table_count):
            if rank_left_tokens[j] != left or rank_right_tokens[j] != right:
                continue
            if not pair_found or rank_values[j] < pair_rank:
                pair_rank = rank_values[j]
                pair_found = True

        if not pair_found:
            continue
        if not found or pair_rank < best_rank:
            best_rank = pair_rank
            best_left = i
            found = True

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_left_index[0] = best_left
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

    best_rank = rank_values[first]
    best_merged = rank_merged_tokens[first]
    scan = first + 1
    while scan < rank_table_count:
        if (rank_left_tokens[scan], rank_right_tokens[scan]) != (left_token, right_token):
            break
        if rank_values[scan] < best_rank:
            best_rank = rank_values[scan]
            best_merged = rank_merged_tokens[scan]
        scan += 1

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
    if token_count < 2 or left_index >= token_count:
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

    best_right = best_left_index[0] + 1
    if best_right <= best_left_index[0] or best_right >= token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    merged_token = [0]
    pair_rank = [0]
    pair_found = [False]
    err = tokenizer_bpe_merge_pair_token_lookup_checked(
        token_ids[best_left_index[0]],
        token_ids[best_right],
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
    next_token_count = [token_count]
    err = tokenizer_bpe_merge_apply_at_index_checked(
        staged_tokens,
        token_count,
        token_capacity,
        best_left_index[0],
        merged_token[0],
        next_token_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    for i in range(next_token_count[0]):
        token_ids[i] = staged_tokens[i]

    out_merged_token[0] = merged_token[0]
    out_rank[0] = best_rank[0]
    out_found[0] = True
    out_token_count[0] = next_token_count[0]
    return TOKENIZER_BPE_OK


def tokenizer_bpe_merge_apply_best_priority_checked_no_partial(
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

    staged_merged = [out_merged_token[0]]
    staged_rank = [out_rank[0]]
    staged_found = [out_found[0]]
    staged_count = [out_token_count[0]]

    err = tokenizer_bpe_merge_apply_best_priority_checked(
        token_ids,
        token_count,
        token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
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


def tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
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

    for i in range(1, rank_table_count):
        if rank_left_tokens[i - 1] > rank_left_tokens[i]:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if (
            rank_left_tokens[i - 1] == rank_left_tokens[i]
            and rank_right_tokens[i - 1] > rank_right_tokens[i]
        ):
            return TOKENIZER_BPE_ERR_BAD_PARAM

    run_start = 0
    while run_start < rank_table_count:
        key = (rank_left_tokens[run_start], rank_right_tokens[run_start])
        min_rank = None
        min_merged = None

        run_scan = run_start
        while run_scan < rank_table_count:
            key_scan = (rank_left_tokens[run_scan], rank_right_tokens[run_scan])
            if key_scan != key:
                break

            rank = rank_values[run_scan]
            merged = rank_merged_tokens[run_scan]
            if min_rank is None or rank < min_rank:
                min_rank = rank
                min_merged = merged
            elif rank == min_rank and merged != min_merged:
                return TOKENIZER_BPE_ERR_BAD_PARAM

            run_scan += 1

        run_start = run_scan

    return tokenizer_bpe_merge_apply_best_priority_checked_no_partial(
        token_ids,
        token_count,
        token_capacity,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        out_merged_token,
        out_rank,
        out_found,
        out_token_count,
    )


def explicit_composition(
    token_ids: list[int],
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    rank_merged_tokens: list[int],
    out_merged_token: list[int],
    out_rank: list[int],
    out_found: list[bool],
    out_token_count: list[int],
) -> tuple[int, list[int]]:
    tokens = token_ids.copy()
    err = tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
        tokens,
        len(tokens),
        len(tokens),
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        len(rank_values),
        len(rank_values),
        out_merged_token,
        out_rank,
        out_found,
        out_token_count,
    )
    return err, tokens


def test_unsorted_rank_table_rejected_no_partial() -> None:
    tokens = [10, 20, 30]
    rank_left = [10, 9]
    rank_right = [20, 40]
    rank_values = [2, 1]
    rank_merged = [99, 88]

    out_merged = [1234]
    out_rank = [5678]
    out_found = [True]
    out_count = [42]
    before_tokens = tokens.copy()

    err = tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
        tokens,
        len(tokens),
        len(tokens),
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_values),
        len(rank_values),
        out_merged,
        out_rank,
        out_found,
        out_count,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert tokens == before_tokens
    assert out_merged == [1234]
    assert out_rank == [5678]
    assert out_found == [True]
    assert out_count == [42]


def test_duplicate_min_rank_conflict_rejected_no_partial() -> None:
    tokens = [10, 20, 30]
    rank_left = [10, 10]
    rank_right = [20, 20]
    rank_values = [1, 1]
    rank_merged = [101, 202]

    out_merged = [700]
    out_rank = [800]
    out_found = [False]
    out_count = [900]
    before_tokens = tokens.copy()

    err = tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
        tokens,
        len(tokens),
        len(tokens),
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_values),
        len(rank_values),
        out_merged,
        out_rank,
        out_found,
        out_count,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert tokens == before_tokens
    assert out_merged == [700]
    assert out_rank == [800]
    assert out_found == [False]
    assert out_count == [900]


def test_sorted_duplicate_runs_success_parity() -> None:
    tokens = [10, 20, 20, 30]
    rank_left = [10, 10, 20]
    rank_right = [20, 20, 30]
    rank_values = [4, 1, 2]
    rank_merged = [700, 701, 702]

    out_merged = [0]
    out_rank = [0]
    out_found = [False]
    out_count = [0]

    err = tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
        tokens,
        len(tokens),
        len(tokens),
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        len(rank_values),
        len(rank_values),
        out_merged,
        out_rank,
        out_found,
        out_count,
    )

    assert err == TOKENIZER_BPE_OK
    assert out_found == [True]
    assert out_rank == [1]
    assert out_merged == [701]
    assert out_count == [3]
    assert tokens[: out_count[0]] == [701, 20, 30]


def test_randomized_parity_against_explicit_sorted_tables() -> None:
    random.seed(399)

    for _ in range(400):
        token_count = random.randint(2, 8)
        tokens = [random.randint(1, 12) for _ in range(token_count)]

        n = random.randint(0, 20)
        rows: list[tuple[int, int, int, int]] = []
        for _ in range(n):
            left = random.randint(1, 12)
            right = random.randint(1, 12)
            rank = random.randint(0, 15)
            merged = random.randint(20, 40)
            rows.append((left, right, rank, merged))

            if random.random() < 0.25:
                rows.append((left, right, rank, merged))

        rows.sort(key=lambda r: (r[0], r[1]))

        rank_left = [r[0] for r in rows]
        rank_right = [r[1] for r in rows]
        rank_values = [r[2] for r in rows]
        rank_merged = [r[3] for r in rows]

        direct_tokens = tokens.copy()
        out_merged_a = [random.randint(-1000, 1000)]
        out_rank_a = [random.randint(0, 500)]
        out_found_a = [bool(random.getrandbits(1))]
        out_count_a = [random.randint(0, 20)]

        out_merged_b = out_merged_a.copy()
        out_rank_b = out_rank_a.copy()
        out_found_b = out_found_a.copy()
        out_count_b = out_count_a.copy()

        err_a = tokenizer_bpe_merge_apply_best_priority_checked_no_partial_validate_ranks(
            direct_tokens,
            len(direct_tokens),
            len(direct_tokens),
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            len(rank_values),
            len(rank_values),
            out_merged_a,
            out_rank_a,
            out_found_a,
            out_count_a,
        )

        err_b, composed_tokens = explicit_composition(
            tokens,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            out_merged_b,
            out_rank_b,
            out_found_b,
            out_count_b,
        )

        assert err_a == err_b
        assert direct_tokens == composed_tokens
        assert out_merged_a == out_merged_b
        assert out_rank_a == out_rank_b
        assert out_found_a == out_found_b
        assert out_count_a == out_count_b


if __name__ == "__main__":
    test_unsorted_rank_table_rejected_no_partial()
    test_duplicate_min_rank_conflict_rejected_no_partial()
    test_sorted_duplicate_runs_success_parity()
    test_randomized_parity_against_explicit_sorted_tables()
    print("ok")
