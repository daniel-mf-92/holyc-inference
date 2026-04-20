#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece."""

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
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
)
from test_tokenizer_bpe_encode_prompt_checked_no_partial import (
    tokenizer_bpe_encode_prompt_checked_no_partial,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
    max_piece_len: int,
    out_token_ids: list[int] | None,
    out_token_capacity: int,
    out_token_count: list[int] | None,
    out_required_token_capacity: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_token_ids is None
        or out_token_count is None
        or out_required_token_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_capacity > I64_MAX
        or rank_table_count > I64_MAX
        or out_token_capacity > I64_MAX
        or max_piece_len > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    if prompt_nbytes and max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    required_token_capacity = prompt_nbytes

    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    required_merge_workspace_bytes = prompt_nbytes * max_piece_len
    if required_merge_workspace_bytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = required_token_capacity

    if required_token_capacity > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_encode_prompt_checked_no_partial(
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
        out_token_capacity,
        out_token_count,
    )


def staged_composition(
    data: list[int],
    byte_len: int,
    io_cursor: list[int],
    prompt_nbytes: int,
    rank_left_tokens: list[int],
    rank_right_tokens: list[int],
    rank_values: list[int],
    rank_merged_tokens: list[int],
    rank_table_count: int,
    rank_table_capacity: int,
    max_piece_len: int,
    out_token_ids: list[int],
    out_token_capacity: int,
    out_token_count: list[int],
    out_required_token_capacity: list[int],
) -> int:
    if prompt_nbytes and max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if prompt_nbytes * max_piece_len > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = prompt_nbytes
    out_required_token_capacity[0] = derived_capacity

    if derived_capacity > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_encode_prompt_checked_no_partial(
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
        out_token_capacity,
        out_token_count,
    )


def test_source_contains_helper_and_no_partial_delegate() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(" in source
    body = source.split("I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(", 1)[1].split(
        "I32 TokenizerBPEDecodeTokenSpanChecked", 1
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoPartial(" in body
    assert "required_token_capacity = prompt_nbytes" in body


def test_success_fixture_and_required_capacity_reporting() -> None:
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
    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x5151] * 512
    out_b = [0x5151] * 512
    count_a = [0xABCD]
    count_b = [0xABCD]
    req_a = [0xDEAD]
    req_b = [0xDEAD]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
        payload,
        len(payload),
        cursor_a,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_a,
        len(payload),
        count_a,
        req_a,
    )
    err_b = staged_composition(
        payload,
        len(payload),
        cursor_b,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_b,
        len(payload),
        count_b,
        req_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert req_a[0] == req_b[0] == len(payload)
    assert cursor_a[0] == cursor_b[0]
    assert count_a[0] == count_b[0]
    assert out_a == out_b


def test_adversarial_no_partial_and_error_vectors() -> None:
    payload = [ord("o"), ord("k")]
    out = [0xAAAA] * 16
    count = [0x3333]
    cursor = [0]
    required = [0x7777]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
        payload,
        len(payload),
        cursor,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        0,
        out,
        16,
        count,
        required,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert count[0] == 0x3333
    assert out == [0xAAAA] * 16
    assert required[0] == 0x7777

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
        payload,
        len(payload),
        [0],
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        I64_MAX,
        [0] * 8,
        8,
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    out = [0xBBBB] * 8
    count = [0x9999]
    cursor = [0]
    required = [0x4444]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
        payload,
        len(payload),
        cursor,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        1,
        out,
        1,
        count,
        required,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert required[0] == 2
    assert cursor[0] == 0
    assert count[0] == 0x9999
    assert out == [0xBBBB] * 8


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_682)

    for _ in range(2500):
        payload_len = rng.randint(0, 96)
        payload = [rng.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor_seed)

        rank_count = rng.randint(0, 48)
        pairs = []
        for _ in range(rank_count):
            left = rng.randint(0, 255)
            right = rng.randint(0, 255)
            rank = rng.randint(0, 32)
            merged = rng.randint(256, 4096)
            pairs.append((left, right, rank, merged))
        pairs.sort(key=lambda item: (item[0], item[1]))

        rank_left = [item[0] for item in pairs]
        rank_right = [item[1] for item in pairs]
        rank_values = [item[2] for item in pairs]
        rank_merged = [item[3] for item in pairs]

        rank_capacity = rank_count + rng.randint(0, 5)
        max_piece_len = rng.randint(0, 16)
        out_capacity = rng.randint(0, 128)

        out_a = [0x61] * 256
        out_b = [0x61] * 256
        count_a = [0x1234]
        count_b = [0x1234]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]
        req_a = [0xABC]
        req_b = [0xABC]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
            payload,
            payload_len,
            cursor_a,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_capacity,
            max_piece_len,
            out_a,
            out_capacity,
            count_a,
            req_a,
        )
        err_b = staged_composition(
            payload,
            payload_len,
            cursor_b,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_capacity,
            max_piece_len,
            out_b,
            out_capacity,
            count_b,
            req_b,
        )

        assert err_a == err_b
        assert cursor_a[0] == cursor_b[0]
        assert count_a[0] == count_b[0]
        assert out_a == out_b
        assert req_a[0] == req_b[0]


if __name__ == "__main__":
    test_source_contains_helper_and_no_partial_delegate()
    test_success_fixture_and_required_capacity_reporting()
    test_adversarial_no_partial_and_error_vectors()
    test_randomized_parity_vs_explicit_checked_composition()
    print("ok")
