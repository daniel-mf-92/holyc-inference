#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedDefaultNoPartial."""

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
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
    TOKENIZER_UTF8_ERR_TRUNCATED,
)
from test_tokenizer_bpe_encode_prompt_checked_no_partial import (
    tokenizer_bpe_encode_prompt_checked_no_partial,
)


def tokenizer_bpe_encode_prompt_checked_default_no_partial(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    prompt_nbytes: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    out_token_ids: list[int] | None,
    out_token_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if byte_len > I64_MAX or prompt_nbytes > I64_MAX or rank_table_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    derived_rank_table_capacity = rank_table_count
    derived_out_token_capacity = prompt_nbytes

    staged_cursor = [cursor]
    staged_count = [out_token_count[0]]

    staged_capacity = derived_out_token_capacity
    if staged_capacity == 0:
        staged_capacity = 1

    if staged_capacity > (I64_MAX // I32_BYTES):
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_tokens = [0] * staged_capacity

    err = tokenizer_bpe_encode_prompt_checked_no_partial(
        data,
        byte_len,
        staged_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        derived_rank_table_capacity,
        staged_tokens,
        derived_out_token_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > derived_out_token_capacity:
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
    prompt_nbytes: int,
    rank_left: list[int] | None,
    rank_right: list[int] | None,
    rank_values: list[int] | None,
    rank_merged: list[int] | None,
    rank_count: int,
) -> None:
    out_expected = [0x51515151] * 256
    out_wrapper = [0x51515151] * 256
    count_expected = [0xABCD]
    count_wrapper = [0xABCD]
    cursor_expected = [cursor_start]
    cursor_wrapper = [cursor_start]

    staged_expected_cursor = [cursor_expected[0]]
    staged_expected_count = [count_expected[0]]
    staged_expected_tokens = [0] * max(1, prompt_nbytes)

    err_expected = tokenizer_bpe_encode_prompt_checked_no_partial(
        payload,
        byte_len,
        staged_expected_cursor,
        prompt_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_count,
        staged_expected_tokens,
        prompt_nbytes,
        staged_expected_count,
    )
    if err_expected == TOKENIZER_BPE_OK:
        for idx in range(staged_expected_count[0]):
            out_expected[idx] = staged_expected_tokens[idx]
        count_expected[0] = staged_expected_count[0]
        cursor_expected[0] = staged_expected_cursor[0]

    err_wrapper = tokenizer_bpe_encode_prompt_checked_default_no_partial(
        payload,
        byte_len,
        cursor_wrapper,
        prompt_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        out_wrapper,
        count_wrapper,
    )

    assert err_wrapper == err_expected
    assert cursor_wrapper[0] == cursor_expected[0]
    assert count_wrapper[0] == count_expected[0]
    assert out_wrapper == out_expected


def test_known_prompt_fixture_parity_vs_explicit_staged_composition() -> None:
    rank_entries = sorted(
        [
            (108, 108, 1, 300),
            (200, 300, 2, 400),
            (104, 101, 3, 200),
            (400, 111, 0, 500),
            (119, 111, 1, 210),
            (210, 114, 2, 220),
            (220, 108, 3, 230),
            (230, 100, 0, 501),
            (49, 50, 1, 310),
            (310, 51, 0, 502),
            (103, 111, 0, 503),
        ],
        key=lambda item: (item[0], item[1]),
    )
    left = [item[0] for item in rank_entries]
    right = [item[1] for item in rank_entries]
    ranks = [item[2] for item in rank_entries]
    merged = [item[3] for item in rank_entries]

    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    run_case(payload, len(payload), 0, len(payload), left, right, ranks, merged, len(ranks))


def test_malformed_utf8_prompt_tail_no_partial_parity() -> None:
    payload = [ord("o"), ord("k"), ord(" "), 0xE2, 0x82]
    out_ids = [0xC0DE] * 24
    out_count = [0x7788]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_no_partial(
        payload,
        len(payload),
        cursor,
        len(payload),
        [],
        [],
        [],
        [],
        0,
        out_ids,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_TRUNCATED
    assert cursor[0] == 0
    assert out_count[0] == 0x7788
    assert out_ids == [0xC0DE] * 24


def test_exact_no_partial_write_on_error_paths() -> None:
    out = [0x7A7A7A7A] * 24
    out_count = [0x2468]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_no_partial(
        None,
        0,
        cursor,
        0,
        [],
        [],
        [],
        [],
        0,
        out,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert cursor[0] == 0 and out_count[0] == 0x2468 and out == [0x7A7A7A7A] * 24

    out = [0x2222] * 16
    out_count = [0x3333]
    cursor = [5]
    err = tokenizer_bpe_encode_prompt_checked_default_no_partial(
        [ord("x")] * 5,
        5,
        cursor,
        1,
        [],
        [],
        [],
        [],
        0,
        out,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 5 and out_count[0] == 0x3333 and out == [0x2222] * 16

    out = [0x1111] * 16
    out_count = [0x9999]
    cursor = [3]
    err = tokenizer_bpe_encode_prompt_checked_default_no_partial(
        [ord("a"), ord("b"), ord("c")],
        3,
        cursor,
        1,
        [],
        [],
        [],
        [],
        0,
        out,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 3 and out_count[0] == 0x9999 and out == [0x1111] * 16


def test_randomized_prompt_parity() -> None:
    rng = random.Random(402)

    rank_entries = sorted(
        [
            (97, 98, 1, 1001),
            (98, 99, 2, 1002),
            (1001, 99, 0, 2001),
            (32, 32, 1, 3001),
            (49, 50, 1, 3002),
            (3002, 51, 0, 3003),
        ],
        key=lambda x: (x[0], x[1]),
    )
    left = [item[0] for item in rank_entries]
    right = [item[1] for item in rank_entries]
    rank_values = [item[2] for item in rank_entries]
    merged = [item[3] for item in rank_entries]

    corpus = [
        "abc abc",
        "abc123",
        "A_B\t12",
        "Καλημέρα κόσμε",
        "世界 123 abc",
        "mix: abc 世界 123",
    ]

    for _ in range(120):
        text = rng.choice(corpus)
        payload = list(text.encode("utf-8"))
        if not payload:
            continue
        start = rng.randrange(0, len(payload))
        span = rng.randrange(0, len(payload) - start + 1)
        run_case(
            payload,
            len(payload),
            start,
            span,
            left,
            right,
            rank_values,
            merged,
            len(rank_entries),
        )


if __name__ == "__main__":
    test_known_prompt_fixture_parity_vs_explicit_staged_composition()
    test_malformed_utf8_prompt_tail_no_partial_parity()
    test_exact_no_partial_write_on_error_paths()
    test_randomized_prompt_parity()
    print("ok")
