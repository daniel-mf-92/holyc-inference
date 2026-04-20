#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromLens."""

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
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
    out_token_ids: list[int] | None,
    out_token_capacity: int,
    out_token_count: list[int] | None,
    out_required_token_capacity: list[int] | None,
    out_max_piece_len: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_token_ids is None
        or out_token_count is None
        or out_required_token_capacity is None
        or out_max_piece_len is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
        or out_token_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    max_piece_len = 0
    for piece_index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[piece_index]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > max_piece_len:
            max_piece_len = piece_len

    out_max_piece_len[0] = max_piece_len

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
        max_piece_len,
        out_token_ids,
        out_token_capacity,
        out_token_count,
        out_required_token_capacity,
    )


def explicit_checked_max_piece_then_encode(
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
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
    out_token_ids: list[int] | None,
    out_token_capacity: int,
    out_token_count: list[int] | None,
    out_required_token_capacity: list[int] | None,
    out_max_piece_len: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_token_ids is None
        or out_token_count is None
        or out_required_token_capacity is None
        or out_max_piece_len is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
        or out_token_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    max_piece_len = 0
    for piece_index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[piece_index]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > max_piece_len:
            max_piece_len = piece_len

    out_max_piece_len[0] = max_piece_len

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
        max_piece_len,
        out_token_ids,
        out_token_capacity,
        out_token_count,
        out_required_token_capacity,
    )


def _build_rank_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    entries = sorted(
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
    left = [item[0] for item in entries]
    right = [item[1] for item in entries]
    ranks = [item[2] for item in entries]
    merged = [item[3] for item in entries]
    return left, right, ranks, merged


def test_source_contains_from_lens_helper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens("
    assert sig in source

    body = source.split(sig, 1)[1]
    assert "if (vocab_piece_count > vocab_piece_capacity)" in body
    assert "while (piece_index < vocab_piece_count)" in body
    assert "*out_max_piece_len = max_piece_len;" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(" in body


def test_success_fixture_matches_explicit_composition() -> None:
    left, right, ranks, merged = _build_rank_tables()
    vocab_lens = [1, 2, 4, 3, 8, 5, 1]

    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))

    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x5151] * 512
    out_b = [0x5151] * 512
    count_a = [0xABCD]
    count_b = [0xABCD]
    req_a = [0xDEAD]
    req_b = [0xDEAD]
    max_a = [0xEEEE]
    max_b = [0xEEEE]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_a,
        len(payload),
        count_a,
        req_a,
        max_a,
    )
    err_b = explicit_checked_max_piece_then_encode(
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
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_b,
        len(payload),
        count_b,
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert cursor_a[0] == cursor_b[0] == len(payload)
    assert count_a[0] == count_b[0]
    assert req_a[0] == req_b[0] == len(payload)
    assert max_a[0] == max_b[0] == 8
    assert out_a == out_b


def test_error_vectors_and_no_partial_parity() -> None:
    payload = [ord("o"), ord("k")]

    out_a = [0xAAAA] * 8
    out_b = [0xAAAA] * 8
    count_a = [0x3333]
    count_b = [0x3333]
    req_a = [0x7777]
    req_b = [0x7777]
    max_a = [0x9999]
    max_b = [0x9999]
    cursor_a = [0]
    cursor_b = [0]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
        payload,
        len(payload),
        cursor_a,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        [1, 3, 7],
        3,
        3,
        out_a,
        1,
        count_a,
        req_a,
        max_a,
    )
    err_b = explicit_checked_max_piece_then_encode(
        payload,
        len(payload),
        cursor_b,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        [1, 3, 7],
        3,
        3,
        out_b,
        1,
        count_b,
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor_a == cursor_b == [0]
    assert count_a == count_b == [0x3333]
    assert req_a == req_b == [2]
    assert max_a == max_b == [7]
    assert out_a == out_b == [0xAAAA] * 8

    max_out = [0xBEEF]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        [1, I64_MAX + 1],
        2,
        2,
        [0] * 8,
        8,
        [0xAAAA],
        [0xBBBB],
        max_out,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert max_out[0] == 0xBEEF

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        [1],
        2,
        1,
        [0] * 8,
        8,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
        None,
        0,
        [0],
        0,
        [],
        [],
        [],
        [],
        0,
        0,
        [1],
        1,
        1,
        [0],
        1,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260421_686)

    for _ in range(3000):
        payload_len = rng.randint(0, 80)
        payload = [rng.randint(0, 127) for _ in range(payload_len)]

        cursor_seed = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor_seed)

        rank_capacity = rng.randint(0, 40)
        rank_count = rng.randint(0, rank_capacity)
        rank_left = [rng.randint(0, 400) for _ in range(rank_capacity)]
        rank_right = [rng.randint(0, 400) for _ in range(rank_capacity)]
        rank_values = [rng.randint(0, 6) for _ in range(rank_capacity)]
        rank_merged = [rng.randint(0, 600) for _ in range(rank_capacity)]

        vocab_capacity = rng.randint(0, 40)
        vocab_count = rng.randint(0, vocab_capacity)
        vocab_lens = [rng.randint(0, 12) for _ in range(vocab_capacity)]

        out_capacity = rng.randint(0, 96)

        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]
        out_a = [0x6161] * 128
        out_b = [0x6161] * 128
        count_a = [0x4242]
        count_b = [0x4242]
        req_a = [0x7373]
        req_b = [0x7373]
        max_a = [0x5151]
        max_b = [0x5151]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
            payload,
            len(payload),
            cursor_a,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_capacity,
            vocab_lens,
            vocab_count,
            vocab_capacity,
            out_a,
            out_capacity,
            count_a,
            req_a,
            max_a,
        )
        err_b = explicit_checked_max_piece_then_encode(
            payload,
            len(payload),
            cursor_b,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_capacity,
            vocab_lens,
            vocab_count,
            vocab_capacity,
            out_b,
            out_capacity,
            count_b,
            req_b,
            max_b,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert out_a == out_b
        assert count_a == count_b
        assert req_a == req_b
        assert max_a == max_b


def test_adversarial_overflow_vectors() -> None:
    payload = [1, 2, 3]
    out = [0xAAAA] * 8
    count = [0x1111]
    req = [0x2222]
    max_piece = [0x3333]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
        payload,
        len(payload),
        [0],
        2,
        [],
        [],
        [],
        [],
        I64_MAX + 1,
        I64_MAX + 1,
        [1],
        1,
        1,
        out,
        8,
        count,
        req,
        max_piece,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert count == [0x1111]
    assert req == [0x2222]
    assert max_piece == [0x3333]
    assert out == [0xAAAA] * 8


if __name__ == "__main__":
    test_source_contains_from_lens_helper()
    test_success_fixture_matches_explicit_composition()
    test_error_vectors_and_no_partial_parity()
    test_randomized_parity_vs_explicit_composition()
    test_adversarial_overflow_vectors()
    print("ok")
