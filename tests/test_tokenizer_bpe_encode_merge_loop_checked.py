#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodeMergeLoopChecked semantics."""

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
    best_rank = rank_values[first]
    while scan < rank_table_count:
        if (rank_left_tokens[scan], rank_right_tokens[scan]) != (left_token, right_token):
            break
        if rank_values[scan] < best_rank:
            best_rank = rank_values[scan]
        scan += 1

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
    src = right_index + 1
    while src < token_count:
        dst = src - 1
        if dst >= token_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW
        token_ids[dst] = token_ids[src]
        src += 1

    out_token_count[0] = token_count - 1
    return TOKENIZER_BPE_OK


def tokenizer_bpe_encode_merge_loop_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_token_count: list[int] | None,
) -> int:
    if token_ids is None or out_token_count is None:
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
        out_token_count[0] = token_count
        return TOKENIZER_BPE_OK

    staged = token_ids[:token_count]
    staged_count = token_count

    while staged_count >= 2:
        best_left = [0]
        best_rank = [0]
        best_found = [False]

        err = tokenizer_bpe_merge_pair_priority_find_checked(
            staged,
            staged_count,
            token_capacity,
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_table_count,
            rank_table_capacity,
            best_left,
            best_rank,
            best_found,
        )
        if err != TOKENIZER_BPE_OK:
            return err
        if not best_found[0]:
            break

        right_index = best_left[0] + 1
        if right_index <= best_left[0] or right_index >= staged_count:
            return TOKENIZER_BPE_ERR_OVERFLOW

        checked_pair_rank = [0]
        checked_pair_found = [False]
        err = tokenizer_bpe_token_pair_rank_lookup_checked(
            staged[best_left[0]],
            staged[right_index],
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_table_count,
            rank_table_capacity,
            checked_pair_rank,
            checked_pair_found,
        )
        if err != TOKENIZER_BPE_OK:
            return err
        if not checked_pair_found[0] or checked_pair_rank[0] != best_rank[0]:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        pair_merged = [0]
        pair_rank = [0]
        pair_found = [False]
        err = tokenizer_bpe_merge_pair_token_lookup_checked(
            staged[best_left[0]],
            staged[right_index],
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_merged_tokens,
            rank_table_count,
            rank_table_capacity,
            pair_merged,
            pair_rank,
            pair_found,
        )
        if err != TOKENIZER_BPE_OK:
            return err
        if not pair_found[0] or pair_rank[0] != best_rank[0]:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        next_count = [0]
        err = tokenizer_bpe_merge_apply_at_index_checked(
            staged,
            staged_count,
            token_capacity,
            best_left[0],
            pair_merged[0],
            next_count,
        )
        if err != TOKENIZER_BPE_OK:
            return err

        staged_count = next_count[0]

    for i in range(staged_count):
        token_ids[i] = staged[i]
    out_token_count[0] = staged_count
    return TOKENIZER_BPE_OK


def run_reference_two_step(
    token_ids: list[int],
    token_count: int,
    token_capacity: int,
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    rank_merged_tokens: list[int],
) -> tuple[int, list[int], int]:
    work = token_ids[:token_count]
    while len(work) >= 2:
        li = [0]
        rank = [0]
        found = [False]
        err = tokenizer_bpe_merge_pair_priority_find_checked(
            work,
            len(work),
            token_capacity,
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            len(rank_values),
            len(rank_values),
            li,
            rank,
            found,
        )
        if err != TOKENIZER_BPE_OK:
            return err, work, len(work)
        if not found[0]:
            break

        merged = [0]
        rank2 = [0]
        found2 = [False]
        checked_rank = [0]
        checked_rank_found = [False]

        err = tokenizer_bpe_token_pair_rank_lookup_checked(
            work[li[0]],
            work[li[0] + 1],
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            len(rank_values),
            len(rank_values),
            checked_rank,
            checked_rank_found,
        )
        if err != TOKENIZER_BPE_OK:
            return err, work, len(work)
        if not checked_rank_found[0] or checked_rank[0] != rank[0]:
            return TOKENIZER_BPE_ERR_BAD_PARAM, work, len(work)

        err = tokenizer_bpe_merge_pair_token_lookup_checked(
            work[li[0]],
            work[li[0] + 1],
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_merged_tokens,
            len(rank_values),
            len(rank_values),
            merged,
            rank2,
            found2,
        )
        if err != TOKENIZER_BPE_OK:
            return err, work, len(work)
        if not found2[0] or rank2[0] != rank[0]:
            return TOKENIZER_BPE_ERR_BAD_PARAM, work, len(work)

        out_count = [0]
        err = tokenizer_bpe_merge_apply_at_index_checked(
            work,
            len(work),
            token_capacity,
            li[0],
            merged[0],
            out_count,
        )
        if err != TOKENIZER_BPE_OK:
            return err, work, len(work)
        work = work[: out_count[0]]

    return TOKENIZER_BPE_OK, work, len(work)


def test_known_trace_to_fixpoint() -> None:
    # Merge table (sorted by pair key) with ranks that force a deterministic trace:
    # [65,66,67,68] -> (65,66)->300 -> [300,67,68]
    # [300,67]->400 -> [400,68]
    # [400,68]->500 -> [500]
    left = [65, 300, 400]
    right = [66, 67, 68]
    ranks = [1, 2, 3]
    merged = [300, 400, 500]

    tokens = [65, 66, 67, 68, -1, -1]
    out_count = [999]

    err = tokenizer_bpe_encode_merge_loop_checked(
        tokens,
        4,
        6,
        left,
        right,
        ranks,
        merged,
        3,
        3,
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_count[0] == 1
    assert tokens[:1] == [500]


def test_no_partial_writes_on_error_contract() -> None:
    base = [10, 11, 12, 13, 99, 98]

    out_count = [777]
    tokens = base.copy()
    err = tokenizer_bpe_encode_merge_loop_checked(
        tokens,
        4,
        6,
        None,
        [11],
        [1],
        [100],
        1,
        1,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert tokens == base
    assert out_count[0] == 777

    out_count2 = [778]
    tokens2 = base.copy()
    err = tokenizer_bpe_encode_merge_loop_checked(
        tokens2,
        4,
        I64_MAX + 1,
        [10],
        [11],
        [1],
        [100],
        1,
        1,
        out_count2,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert tokens2 == base
    assert out_count2[0] == 778

    out_count3 = [779]
    tokens3 = base.copy()
    err = tokenizer_bpe_encode_merge_loop_checked(
        tokens3,
        7,
        6,
        [10],
        [11],
        [1],
        [100],
        1,
        1,
        out_count3,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert tokens3 == base
    assert out_count3[0] == 779


def test_adversarial_rank_desync_reports_bad_param() -> None:
    # Priority scan is linear and still sees a merge candidate, while pair lookup
    # uses binary-search over lexicographic keys and can fail on unsorted tables.
    # The merge loop must treat this desync as malformed table input.
    left = [2, 1, 1]
    right = [3, 2, 2]
    ranks = [5, 1, 2]
    merged = [12, 10, 11]

    tokens = [1, 2, 3, -1]
    out_count = [123]
    original = tokens.copy()

    err = tokenizer_bpe_encode_merge_loop_checked(
        tokens,
        3,
        4,
        left,
        right,
        ranks,
        merged,
        3,
        3,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert tokens == original
    assert out_count[0] == 123


def test_randomized_parity_vs_explicit_two_step_reference() -> None:
    rng = random.Random(20260418_365)

    for _ in range(2000):
        token_count = rng.randint(1, 40)
        slack = rng.randint(0, 6)
        token_capacity = token_count + slack

        tokens = [rng.randint(1, 40) for _ in range(token_capacity)]
        initial = tokens.copy()

        n_pairs = rng.randint(0, 140)
        pairs = []
        for _ in range(n_pairs):
            l = rng.randint(1, 40)
            r = rng.randint(1, 40)
            rank = rng.randint(0, 300)
            merged = rng.randint(100, 400)
            pairs.append((l, r, rank, merged))

        # The checked pair-lookup requires lexicographic sorting by (left,right).
        pairs.sort(key=lambda item: (item[0], item[1]))

        left = [p[0] for p in pairs]
        right = [p[1] for p in pairs]
        ranks = [p[2] for p in pairs]
        merged = [p[3] for p in pairs]

        out_count = [0]
        err = tokenizer_bpe_encode_merge_loop_checked(
            tokens,
            token_count,
            token_capacity,
            left,
            right,
            ranks,
            merged,
            len(pairs),
            len(pairs),
            out_count,
        )
        assert err == TOKENIZER_BPE_OK

        ref_err, ref_tokens, ref_count = run_reference_two_step(
            initial,
            token_count,
            token_capacity,
            left,
            right,
            ranks,
            merged,
        )
        assert ref_err == TOKENIZER_BPE_OK

        assert out_count[0] == ref_count
        assert tokens[:ref_count] == ref_tokens


def test_small_counts_and_zero_rank_table() -> None:
    # Count 0
    tokens0 = [7, 8, 9]
    out0 = [999]
    assert (
        tokenizer_bpe_encode_merge_loop_checked(
            tokens0,
            0,
            3,
            [],
            [],
            [],
            [],
            0,
            0,
            out0,
        )
        == TOKENIZER_BPE_OK
    )
    assert out0[0] == 0

    # Count 1
    tokens1 = [42, 99]
    out1 = [888]
    assert (
        tokenizer_bpe_encode_merge_loop_checked(
            tokens1,
            1,
            2,
            [],
            [],
            [],
            [],
            0,
            0,
            out1,
        )
        == TOKENIZER_BPE_OK
    )
    assert out1[0] == 1
    assert tokens1[0] == 42

    # Zero rank table with larger count should be no-op.
    tokens2 = [5, 6, 7, 8]
    out2 = [777]
    assert (
        tokenizer_bpe_encode_merge_loop_checked(
            tokens2,
            4,
            4,
            [],
            [],
            [],
            [],
            0,
            0,
            out2,
        )
        == TOKENIZER_BPE_OK
    )
    assert out2[0] == 4
    assert tokens2[:4] == [5, 6, 7, 8]


def test_duplicate_pair_uses_lowest_rank_trace() -> None:
    # Duplicate (1,2) entries appear contiguously; lowest rank=3 must win.
    # Trace: [1,2,3] -> (1,2)->99 -> [99,3] -> (99,3)->55 -> [55]
    left = [1, 1, 99]
    right = [2, 2, 3]
    ranks = [7, 3, 9]
    merged = [77, 99, 55]

    tokens = [1, 2, 3, -1]
    out_count = [0]
    err = tokenizer_bpe_encode_merge_loop_checked(
        tokens,
        3,
        4,
        left,
        right,
        ranks,
        merged,
        3,
        3,
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert out_count[0] == 1
    assert tokens[:1] == [55]
