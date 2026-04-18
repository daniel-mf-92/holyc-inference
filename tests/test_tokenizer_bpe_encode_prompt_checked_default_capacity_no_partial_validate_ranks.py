#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedDefaultCapacityNoPartialValidateRanks."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_prompt_checked import (
    I32_BYTES,
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)
from test_tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial import (
    tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial,
)


def tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    prompt_nbytes: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_token_ids: list[int] | None,
    out_token_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if byte_len > I64_MAX or prompt_nbytes > I64_MAX or rank_table_capacity > I64_MAX:
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

    # Guard 1: keys must be lexicographically sorted by (left,right).
    for i in range(1, rank_table_count):
        prev_left = rank_left_tokens[i - 1]  # type: ignore[index]
        prev_right = rank_right_tokens[i - 1]  # type: ignore[index]
        cur_left = rank_left_tokens[i]  # type: ignore[index]
        cur_right = rank_right_tokens[i]  # type: ignore[index]

        if prev_left > cur_left:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if prev_left == cur_left and prev_right > cur_right:
            return TOKENIZER_BPE_ERR_BAD_PARAM

    # Guard 2: for duplicate keys, all minimum-rank rows must map
    # to one deterministic merged token.
    run_start = 0
    while run_start < rank_table_count:
        key_left = rank_left_tokens[run_start]  # type: ignore[index]
        key_right = rank_right_tokens[run_start]  # type: ignore[index]

        min_rank_set = False
        min_merged_set = False
        min_rank = 0
        min_merged = 0

        run_scan = run_start
        while run_scan < rank_table_count:
            if rank_left_tokens[run_scan] != key_left:  # type: ignore[index]
                break
            if rank_right_tokens[run_scan] != key_right:  # type: ignore[index]
                break

            candidate_rank = rank_values[run_scan]  # type: ignore[index]
            candidate_merged = rank_merged_tokens[run_scan]  # type: ignore[index]

            if (not min_rank_set) or candidate_rank < min_rank:
                min_rank = candidate_rank
                min_merged = candidate_merged
                min_rank_set = True
                min_merged_set = True
            elif candidate_rank == min_rank:
                if not min_merged_set:
                    min_merged = candidate_merged
                    min_merged_set = True
                elif candidate_merged != min_merged:
                    return TOKENIZER_BPE_ERR_BAD_PARAM

            run_scan += 1

        run_start = run_scan

    return tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
        data,
        byte_len,
        io_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        out_token_ids,
        out_token_count,
    )


def _baseline_with_snapshot(
    payload: list[int] | None,
    byte_len: int,
    cursor_start: int,
    prompt_nbytes: int,
    left: list[int] | None,
    right: list[int] | None,
    ranks: list[int] | None,
    merged: list[int] | None,
    rank_count: int,
    rank_capacity: int,
    out_seed: int,
    count_seed: int,
):
    out = [out_seed] * 320
    out_count = [count_seed]
    cursor = [cursor_start]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
        payload,
        byte_len,
        cursor,
        prompt_nbytes,
        left,
        right,
        ranks,
        merged,
        rank_count,
        rank_capacity,
        out,
        out_count,
    )
    return err, cursor, out_count, out


def test_unsorted_rank_table_rejected_without_commit() -> None:
    payload = list(b"abc")

    left = [1, 1, 0]
    right = [2, 3, 9]
    ranks = [3, 1, 0]
    merged = [100, 101, 102]

    err, cursor, count, out = _baseline_with_snapshot(
        payload,
        len(payload),
        0,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(left),
        len(left),
        0x7A7A,
        0xBEEF,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert count[0] == 0xBEEF
    assert out == [0x7A7A] * 320


def test_duplicate_min_rank_tie_with_mismatched_merged_rejected_no_partial() -> None:
    payload = list(b"aba")

    # Duplicate key (10,11) with same min rank but conflicting merged token.
    left = [10, 10, 10, 11]
    right = [11, 11, 11, 12]
    ranks = [7, 2, 2, 9]
    merged = [1000, 2000, 2001, 3000]

    err, cursor, count, out = _baseline_with_snapshot(
        payload,
        len(payload),
        0,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(left),
        len(left),
        0x6262,
        0x8181,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert count[0] == 0x8181
    assert out == [0x6262] * 320


def test_sorted_table_success_parity_with_base_default_capacity_no_partial() -> None:
    rank_entries = sorted(
        [
            (97, 98, 3, 901),
            (98, 99, 2, 902),
            (901, 99, 1, 903),
            (32, 32, 8, 910),
            (49, 50, 6, 920),
            (920, 51, 4, 921),
            (22909, 25105, 1, 1300),
            # duplicate key with same minimum-rank merged token (deterministic)
            (97, 98, 1, 999),
            (97, 98, 1, 999),
        ],
        key=lambda row: (row[0], row[1]),
    )

    left = [row[0] for row in rank_entries]
    right = [row[1] for row in rank_entries]
    ranks = [row[2] for row in rank_entries]
    merged = [row[3] for row in rank_entries]

    payload = list("abc abc 123 世界".encode("utf-8"))

    out_a = [0x5151] * 320
    out_b = [0x5151] * 320
    count_a = [0x1234]
    count_b = [0x1234]
    cursor_a = [0]
    cursor_b = [0]

    err_a = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
        payload,
        len(payload),
        cursor_a,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(left),
        len(left),
        out_a,
        count_a,
    )
    err_b = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
        payload,
        len(payload),
        cursor_b,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(left),
        len(left),
        out_b,
        count_b,
    )

    assert err_a == TOKENIZER_BPE_OK
    assert err_b == TOKENIZER_BPE_OK
    assert err_a == err_b
    assert cursor_a[0] == cursor_b[0]
    assert count_a[0] == count_b[0]
    assert out_a == out_b


def test_null_rank_arrays_with_nonzero_rank_count_returns_null_ptr() -> None:
    out = [0x1111] * 64
    out_count = [0x2222]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
        [ord("a")],
        1,
        cursor,
        1,
        None,
        [],
        [],
        [],
        1,
        1,
        out,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert cursor[0] == 0
    assert out_count[0] == 0x2222
    assert out == [0x1111] * 64


def test_randomized_sorted_parity_and_unsorted_rejection() -> None:
    rng = random.Random(20260418_404)

    corpus = [
        "alpha beta gamma",
        "TempleOS_1996",
        "Καλημέρα κόσμε",
        "世界 hello",
        "a_b c-d 123",
        "🙂🙃 text",
    ]

    for _ in range(600):
        text = rng.choice(corpus)
        payload = list(text.encode("utf-8"))
        byte_len = len(payload)

        cursor_start = rng.randint(0, byte_len)
        prompt_nbytes = rng.randint(0, byte_len - cursor_start)

        pair_count = rng.randint(0, 24)
        entries = []
        for _ in range(pair_count):
            left_token = rng.randint(0, 512)
            right_token = rng.randint(0, 512)
            rank = rng.randint(0, 255)
            merged_token = rng.randint(513, 4096)
            entries.append((left_token, right_token, rank, merged_token))

            if rng.random() < 0.2:
                entries.append((left_token, right_token, rank, merged_token))

        entries_sorted = sorted(entries, key=lambda row: (row[0], row[1]))
        left = [row[0] for row in entries_sorted]
        right = [row[1] for row in entries_sorted]
        ranks = [row[2] for row in entries_sorted]
        merged = [row[3] for row in entries_sorted]

        out_a = [0x4444] * 320
        out_b = [0x4444] * 320
        count_a = [0x8888]
        count_b = [0x8888]
        cursor_a = [cursor_start]
        cursor_b = [cursor_start]

        err_a = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
            payload,
            byte_len,
            cursor_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(entries_sorted),
            len(entries_sorted),
            out_a,
            count_a,
        )
        err_b = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
            payload,
            byte_len,
            cursor_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(entries_sorted),
            len(entries_sorted),
            out_b,
            count_b,
        )

        assert err_a == err_b
        assert cursor_a[0] == cursor_b[0]
        assert count_a[0] == count_b[0]
        assert out_a == out_b

        if len(entries_sorted) >= 2:
            left_unsorted = left[:]
            right_unsorted = right[:]
            ranks_unsorted = ranks[:]
            merged_unsorted = merged[:]

            # Force strict lexicographic-order violation deterministically.
            # Keep row payload sizes unchanged but make key[1] < key[0].
            left_unsorted[1] = left_unsorted[0] - 1
            right_unsorted[1] = right_unsorted[0]

            out_u = [0x9999] * 320
            count_u = [0xABCD]
            cursor_u = [cursor_start]

            err_u = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
                payload,
                byte_len,
                cursor_u,
                prompt_nbytes,
                left_unsorted,
                right_unsorted,
                ranks_unsorted,
                merged_unsorted,
                len(entries_sorted),
                len(entries_sorted),
                out_u,
                count_u,
            )

            assert err_u == TOKENIZER_BPE_ERR_BAD_PARAM
            assert cursor_u[0] == cursor_start
            assert count_u[0] == 0xABCD
            assert out_u == [0x9999] * 320


def test_rank_count_capacity_mismatch_no_partial() -> None:
    out = [0xCAFE] * 16
    out_count = [0xBABE]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_validate_ranks(
        [ord("x")],
        1,
        cursor,
        1,
        [1],
        [2],
        [3],
        [4],
        2,
        1,
        out,
        out_count,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_count[0] == 0xBABE
    assert out == [0xCAFE] * 16


if __name__ == "__main__":
    test_unsorted_rank_table_rejected_without_commit()
    test_duplicate_min_rank_tie_with_mismatched_merged_rejected_no_partial()
    test_sorted_table_success_parity_with_base_default_capacity_no_partial()
    test_null_rank_arrays_with_nonzero_rank_count_returns_null_ptr()
    test_randomized_sorted_parity_and_unsorted_rejection()
    test_rank_count_capacity_mismatch_no_partial()
    print("ok")
