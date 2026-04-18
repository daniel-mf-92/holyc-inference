#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    TOKENIZER_UTF8_ERR_TRUNCATED,
    tokenizer_bpe_encode_prompt_checked,
)


def tokenizer_bpe_encode_prompt_checked_default_capacity(
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

    if prompt_nbytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = prompt_nbytes
    return tokenizer_bpe_encode_prompt_checked(
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
        derived_capacity,
        out_token_count,
    )


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
    rank_capacity: int,
) -> None:
    out_core = [0x51515151] * 192
    out_def = [0x51515151] * 192
    count_core = [0xABCD]
    count_def = [0xABCD]
    cursor_core = [cursor_start]
    cursor_def = [cursor_start]

    err_core = tokenizer_bpe_encode_prompt_checked(
        payload,
        byte_len,
        cursor_core,
        prompt_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_capacity,
        out_core,
        prompt_nbytes,
        count_core,
    )
    err_def = tokenizer_bpe_encode_prompt_checked_default_capacity(
        payload,
        byte_len,
        cursor_def,
        prompt_nbytes,
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


def test_known_prompt_fixture_parity() -> None:
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
    run_case(payload, len(payload), 0, len(payload), left, right, ranks, merged, len(ranks), len(ranks))


def test_empty_prompt_span_parity() -> None:
    run_case(list(b"abc"), 3, 2, 0, [], [], [], [], 0, 0)


def test_error_no_partial_output_parity() -> None:
    out = [0x7A7A7A7A] * 24
    out_count = [0x2468]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity(
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
    assert out == [0x7A7A7A7A] * 24


def test_overflow_guard_for_default_capacity() -> None:
    out = [0] * 16
    out_count = [123]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity(
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


def test_capacity_adversarial_rank_table_mismatch() -> None:
    out = [0x3333] * 16
    out_count = [0x4444]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity(
        [ord("a"), ord("b"), ord("c"), ord(" "), ord("d")],
        5,
        cursor,
        5,
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
    assert out_count[0] == 0x4444
    assert out == [0x3333] * 16


def test_malformed_utf8_parity() -> None:
    out_core = [0x5555] * 16
    out_def = [0x5555] * 16
    count_core = [0x6666]
    count_def = [0x6666]
    cursor_core = [0]
    cursor_def = [0]

    payload = [0xE2, 0x82]
    err_core = tokenizer_bpe_encode_prompt_checked(
        payload,
        2,
        cursor_core,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        out_core,
        2,
        count_core,
    )
    err_def = tokenizer_bpe_encode_prompt_checked_default_capacity(
        payload,
        2,
        cursor_def,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        out_def,
        count_def,
    )

    assert err_core == TOKENIZER_UTF8_ERR_TRUNCATED
    assert err_def == err_core
    assert cursor_def[0] == cursor_core[0] == 0
    assert count_def[0] == count_core[0] == 0x6666
    assert out_def == out_core == [0x5555] * 16


def test_randomized_parity_against_explicit_capacity_core() -> None:
    rng = random.Random(20260418_375)

    corpus = [
        "alpha BETA_12",
        "Русский текст",
        "漢字かな交じり",
        "🙂🙃",
        "TempleOS 1996!",
        "a_b c-d e+f",
        "mañana déjà vu",
        "γειά σου κόσμε",
    ]

    for _ in range(3000):
        text = rng.choice(corpus)
        if rng.randint(0, 1):
            text += chr(rng.randint(32, 126))

        payload = list(text.encode("utf-8"))
        byte_len = len(payload)

        cursor_start = rng.randint(0, byte_len)
        prompt_nbytes = rng.randint(0, byte_len - cursor_start)

        rank_count = rng.randint(0, 80)
        rank_left = [rng.randint(0, 8192) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 8192) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 20000) for _ in range(rank_count)]
        rank_merged = [rng.randint(0, 20000) for _ in range(rank_count)]

        run_case(
            payload,
            byte_len,
            cursor_start,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_count,
        )


if __name__ == "__main__":
    test_known_prompt_fixture_parity()
    test_empty_prompt_span_parity()
    test_error_no_partial_output_parity()
    test_overflow_guard_for_default_capacity()
    test_capacity_adversarial_rank_table_mismatch()
    test_malformed_utf8_parity()
    test_randomized_parity_against_explicit_capacity_core()
    print("ok")
