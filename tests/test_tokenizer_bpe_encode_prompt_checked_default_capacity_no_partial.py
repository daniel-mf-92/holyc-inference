#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedDefaultCapacityNoPartial."""

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
    tokenizer_bpe_encode_prompt_checked,
)


def tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
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

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    derived_capacity = prompt_nbytes

    staged_cursor = [cursor]
    staged_count = [out_token_count[0]]

    staged_capacity = derived_capacity
    if staged_capacity == 0:
        staged_capacity = 1

    if staged_capacity > (I64_MAX // I32_BYTES):
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_tokens = [0] * staged_capacity

    err = tokenizer_bpe_encode_prompt_checked(
        data,
        byte_len,
        staged_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        staged_tokens,
        derived_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > derived_capacity:
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
    rank_capacity: int,
) -> None:
    out_expected = [0x51515151] * 224
    out_wrapper = [0x51515151] * 224
    count_expected = [0xABCD]
    count_wrapper = [0xABCD]
    cursor_expected = [cursor_start]
    cursor_wrapper = [cursor_start]

    staged_expected_cursor = [cursor_expected[0]]
    staged_expected_count = [count_expected[0]]
    staged_expected_tokens = [0] * max(1, prompt_nbytes)

    err_expected = tokenizer_bpe_encode_prompt_checked(
        payload,
        byte_len,
        staged_expected_cursor,
        prompt_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_capacity,
        staged_expected_tokens,
        prompt_nbytes,
        staged_expected_count,
    )
    if err_expected == TOKENIZER_BPE_OK:
        for idx in range(staged_expected_count[0]):
            out_expected[idx] = staged_expected_tokens[idx]
        count_expected[0] = staged_expected_count[0]
        cursor_expected[0] = staged_expected_cursor[0]

    err_wrapper = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
        payload,
        byte_len,
        cursor_wrapper,
        prompt_nbytes,
        rank_left,
        rank_right,
        rank_values,
        rank_merged,
        rank_count,
        rank_capacity,
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
    run_case(payload, len(payload), 0, len(payload), left, right, ranks, merged, len(ranks), len(ranks))


def test_exact_no_partial_write_on_error_paths() -> None:
    out = [0x7A7A7A7A] * 24
    out_count = [0x2468]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
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
    assert cursor[0] == 0 and out_count[0] == 0x2468 and out == [0x7A7A7A7A] * 24

    out = [0x2222] * 16
    out_count = [0x3333]
    cursor = [5]
    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
        [ord("a"), ord("b"), ord("c")],
        3,
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
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 5 and out_count[0] == 0x3333 and out == [0x2222] * 16


def test_rank_count_capacity_mismatch_rejected_without_commit() -> None:
    out = [0xABAB] * 16
    out_count = [0xCDCD]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
        [ord("a")],
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
    assert cursor[0] == 0 and out_count[0] == 0xCDCD and out == [0xABAB] * 16


def test_out_of_bounds_prompt_span_no_partial() -> None:
    out = [0x1111] * 16
    out_count = [0x2222]
    cursor = [3]

    err = tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial(
        [ord("a"), ord("b"), ord("c"), ord("d")],
        4,
        cursor,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        out,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 3 and out_count[0] == 0x2222 and out == [0x1111] * 16


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_389)

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

    for _ in range(4000):
        text = rng.choice(corpus)
        if rng.randint(0, 1):
            text += chr(rng.randint(32, 126))

        payload = list(text.encode("utf-8"))
        byte_len = len(payload)

        cursor_start = rng.randint(0, byte_len)
        prompt_nbytes = rng.randint(0, byte_len - cursor_start)

        rank_count = rng.randint(0, 96)
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
    test_known_prompt_fixture_parity_vs_explicit_staged_composition()
    test_exact_no_partial_write_on_error_paths()
    test_rank_count_capacity_mismatch_rejected_without_commit()
    test_out_of_bounds_prompt_span_no_partial()
    test_randomized_parity_vs_explicit_staged_composition()
    print("tokenizer_bpe_encode_prompt_checked_default_capacity_no_partial_reference_checks=ok")
