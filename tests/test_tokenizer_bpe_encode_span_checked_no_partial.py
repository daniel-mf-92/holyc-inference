#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodeSpanCheckedNoPartial."""

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
    TOKENIZER_UTF8_ERR_TRUNCATED,
    tokenizer_bpe_encode_span_checked,
)


def tokenizer_bpe_encode_span_checked_no_partial(
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
    out_token_capacity: int,
    out_token_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if out_token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_cursor = [io_cursor[0]]
    staged_count = [out_token_count[0]]

    staged_capacity = out_token_capacity
    if staged_capacity == 0:
        staged_capacity = 1
    staged_tokens = [0] * staged_capacity

    err = tokenizer_bpe_encode_span_checked(
        data,
        byte_len,
        staged_cursor,
        span_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        staged_tokens,
        out_token_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_count[0]):
        out_token_ids[idx] = staged_tokens[idx]

    out_token_count[0] = staged_count[0]
    io_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


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
    out_capacity: int,
) -> None:
    out_core = [0x51515151] * 96
    out_np = [0x51515151] * 96
    count_core = [0xABCD]
    count_np = [0xABCD]
    cursor_core = [cursor_start]
    cursor_np = [cursor_start]

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
        out_capacity,
        count_core,
    )
    err_np = tokenizer_bpe_encode_span_checked_no_partial(
        payload,
        byte_len,
        cursor_np,
        span_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_capacity,
        out_np,
        out_capacity,
        count_np,
    )

    assert err_np == err_core
    assert cursor_np[0] == cursor_core[0]
    assert count_np[0] == count_core[0]
    assert out_np == out_core


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

    run_case(list(b"hello"), 5, 0, 5, left, right, ranks, merged, len(ranks), len(ranks), 8)


def test_no_partial_write_on_error_paths() -> None:
    out = [0x7A7A7A7A] * 12
    out_count = [0x2468]
    cursor = [0]

    err = tokenizer_bpe_encode_span_checked_no_partial(
        [0x61, 0x62, 0x63],
        3,
        cursor,
        3,
        [1],
        [2],
        [3],
        [4],
        1,
        1,
        out,
        2,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_count[0] == 0x2468
    assert out == [0x7A7A7A7A] * 12

    out = [0x3333] * 6
    out_count = [0x4444]
    cursor = [0]
    err = tokenizer_bpe_encode_span_checked_no_partial(
        [0xE2, 0x82],
        2,
        cursor,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        out,
        6,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_TRUNCATED
    assert cursor[0] == 0
    assert out_count[0] == 0x4444
    assert out == [0x3333] * 6


def test_null_and_overflow_contract() -> None:
    out = [0x11] * 4
    out_count = [0x22]
    cursor = [0]

    err = tokenizer_bpe_encode_span_checked_no_partial(
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
        1,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert cursor[0] == 0 and out_count[0] == 0x22 and out == [0x11] * 4

    err = tokenizer_bpe_encode_span_checked_no_partial(
        [],
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
        I64_MAX + 1,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert cursor[0] == 0 and out_count[0] == 0x22 and out == [0x11] * 4


def test_randomized_parity_against_core() -> None:
    rng = random.Random(20260418_371)

    for _ in range(3000):
        payload_len = rng.randint(0, 48)
        payload = [rng.randint(0, 255) for _ in range(payload_len)]
        byte_len = payload_len

        cursor_start = rng.randint(0, byte_len)
        span_nbytes = rng.randint(0, byte_len - cursor_start)
        out_capacity = rng.randint(0, 80)

        rank_count = rng.randint(0, 60)
        rank_left = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 1000) for _ in range(rank_count)]
        rank_merged = [rng.randint(0, 2048) for _ in range(rank_count)]

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
            out_capacity,
        )


if __name__ == "__main__":
    test_known_merge_fixture_parity()
    test_no_partial_write_on_error_paths()
    test_null_and_overflow_contract()
    test_randomized_parity_against_core()
    print("ok")
