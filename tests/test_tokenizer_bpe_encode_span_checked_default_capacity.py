#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodeSpanCheckedDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_span_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    tokenizer_bpe_encode_span_checked,
)


def tokenizer_bpe_encode_span_checked_default_capacity(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    span_nbytes: int,
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

    if span_nbytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = span_nbytes
    return tokenizer_bpe_encode_span_checked(
        data,
        byte_len,
        io_cursor,
        span_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        out_token_ids,
        derived_capacity,
        out_token_count,
    )


def run_case(
    payload: list[int] | None,
    byte_len: int,
    cursor_start: int,
    span_nbytes: int,
    rank_left: list[int] | None,
    rank_right: list[int] | None,
    rank_values: list[int] | None,
    rank_merged: list[int] | None,
    rank_count: int,
    rank_capacity: int,
) -> None:
    out_core = [0x51515151] * 64
    out_def = [0x51515151] * 64
    count_core = [0xABCD]
    count_def = [0xABCD]
    cursor_core = [cursor_start]
    cursor_def = [cursor_start]

    err_core = tokenizer_bpe_encode_span_checked(
        payload,
        byte_len,
        cursor_core,
        span_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_capacity,
        out_core,
        span_nbytes,
        count_core,
    )
    err_def = tokenizer_bpe_encode_span_checked_default_capacity(
        payload,
        byte_len,
        cursor_def,
        span_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_capacity,
        out_def,
        count_def,
    )

    assert err_def == err_core
    assert cursor_def[0] == cursor_core[0]
    assert count_def[0] == count_core[0]
    assert out_def == out_core


def test_known_merge_fixture_parity() -> None:
    rank_entries = sorted(
        [
            (108, 108, 1, 300),
            (200, 300, 2, 400),
            (104, 101, 3, 200),
            (300, 111, 4, 401),
            (400, 111, 0, 500),
        ],
        key=lambda item: (item[0], item[1]),
    )
    left = [item[0] for item in rank_entries]
    right = [item[1] for item in rank_entries]
    ranks = [item[2] for item in rank_entries]
    merged = [item[3] for item in rank_entries]

    run_case(list(b"hello"), 5, 0, 5, left, right, ranks, merged, len(ranks), len(ranks))


def test_empty_span_parity() -> None:
    run_case(list(b"abc"), 3, 1, 0, [], [], [], [], 0, 0)


def test_error_no_partial_output_parity() -> None:
    out = [0x7A7A7A7A] * 8
    out_count = [0x2468]
    cursor = [0]

    err = tokenizer_bpe_encode_span_checked_default_capacity(
        None,
        0,
        cursor,
        0,
        [],
        [],
        [],
        [],
        0,
        0,
        out,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert cursor[0] == 0
    assert out_count[0] == 0x2468
    assert out == [0x7A7A7A7A] * 8


def test_overflow_guard_for_default_capacity() -> None:
    out = [0] * 8
    out_count = [123]
    cursor = [0]
    err = tokenizer_bpe_encode_span_checked_default_capacity(
        [],
        0,
        cursor,
        I64_MAX + 1,
        [],
        [],
        [],
        [],
        0,
        0,
        out,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out_count[0] == 123


def test_randomized_parity_against_explicit_capacity_core() -> None:
    rng = random.Random(20260418_367)

    for _ in range(3000):
        payload_len = rng.randint(0, 48)
        payload = [rng.randint(0, 255) for _ in range(payload_len)]
        byte_len = payload_len

        cursor_start = rng.randint(0, byte_len)
        span_nbytes = rng.randint(0, byte_len - cursor_start)

        rank_count = rng.randint(0, 80)
        rank_left = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 1000) for _ in range(rank_count)]
        rank_merged = [rng.randint(0, 1024) for _ in range(rank_count)]

        run_case(
            payload,
            byte_len,
            cursor_start,
            span_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_count,
        )


def test_rank_table_count_capacity_mismatch_parity() -> None:
    out = [0x1111] * 16
    out_count = [0x2222]
    cursor = [0]

    err = tokenizer_bpe_encode_span_checked_default_capacity(
        list(b"abc"),
        3,
        cursor,
        3,
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
    assert out_count[0] == 0x2222
    assert out == [0x1111] * 16


if __name__ == "__main__":
    test_known_merge_fixture_parity()
    test_empty_span_parity()
    test_error_no_partial_output_parity()
    test_overflow_guard_for_default_capacity()
    test_randomized_parity_against_explicit_capacity_core()
    test_rank_table_count_capacity_mismatch_parity()
    print("ok")
