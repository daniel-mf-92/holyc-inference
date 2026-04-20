#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromLens (IQ-686)."""

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

    staged_max_piece_len = 0
    for index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[index]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > staged_max_piece_len:
            staged_max_piece_len = piece_len

    staged_required = [0x12345678]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
        staged_max_piece_len,
        out_token_ids,
        out_token_capacity,
        out_token_count,
        staged_required,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len
    return TOKENIZER_BPE_OK


def explicit_checked_composition(
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

    staged_max_piece_len = 0
    for index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[index]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > staged_max_piece_len:
            staged_max_piece_len = piece_len

    staged_required = [0xCAFED00D]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
        staged_max_piece_len,
        out_token_ids,
        out_token_capacity,
        out_token_count,
        staged_required,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len
    return TOKENIZER_BPE_OK


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


def test_source_contains_helper_and_staged_publish() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "staged_max_piece_len = 0;" in body
    assert "staged_required_token_capacity" in body
    assert "err = TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(" in body
    assert "if (err != TOKENIZER_BPE_OK)" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_max_piece_len = staged_max_piece_len;" in body


def test_success_fixture_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 8, 5, 1]

    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x5151] * 512
    out_b = [0x5151] * 512
    count_a = [0xABCD]
    count_b = [0xABCD]
    req_a = [0xDEAD]
    req_b = [0xDEAD]
    max_a = [0xBEEF]
    max_b = [0xBEEF]

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
    err_b = explicit_checked_composition(
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
    assert cursor_a[0] == cursor_b[0]
    assert count_a[0] == count_b[0]
    assert req_a[0] == req_b[0] == len(payload)
    assert max_a[0] == max_b[0] == 8
    assert out_a == out_b


def test_downstream_error_keeps_staged_outputs_unpublished() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("TempleOS".encode("utf-8"))
    vocab_lens = [1, 2, 6]

    out = [0x9A9A] * 32
    count = [0xAAAA]
    cursor = [0]
    required = [0x11111111]
    max_piece = [0x22222222]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
        payload,
        len(payload),
        cursor,
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
        out,
        len(payload) - 1,
        count,
        required,
        max_piece,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert count[0] == 0xAAAA
    assert required[0] == 0x11111111
    assert max_piece[0] == 0x22222222
    assert out == [0x9A9A] * 32


def test_adversarial_parity_randomized() -> None:
    random.seed(686)

    for _ in range(250):
        byte_len = random.randint(0, 48)
        payload = [random.randint(0, 255) for _ in range(byte_len)]
        cursor0 = random.randint(0, byte_len)
        remain = byte_len - cursor0
        prompt_nbytes = random.randint(0, remain)

        rank_table_capacity = random.randint(0, 24)
        rank_table_count = random.randint(0, rank_table_capacity)
        left = [random.randint(0, 300) for _ in range(rank_table_capacity)]
        right = [random.randint(0, 300) for _ in range(rank_table_capacity)]
        ranks = [random.randint(0, 7) for _ in range(rank_table_capacity)]
        merged = [random.randint(301, 700) for _ in range(rank_table_capacity)]

        vocab_piece_capacity = random.randint(0, 20)
        vocab_piece_count = random.randint(0, vocab_piece_capacity)
        vocab_piece_lens = [random.randint(0, 12) for _ in range(vocab_piece_capacity)]

        out_token_capacity = random.randint(0, 64)
        out_a = [0xA5A5] * max(1, out_token_capacity + 8)
        out_b = [0xA5A5] * max(1, out_token_capacity + 8)

        cursor_a = [cursor0]
        cursor_b = [cursor0]
        count_a = [0x1BAD]
        count_b = [0x1BAD]
        req_a = [0x2222]
        req_b = [0x2222]
        max_a = [0x3333]
        max_b = [0x3333]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
            payload,
            byte_len,
            cursor_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_table_count,
            rank_table_capacity,
            vocab_piece_lens,
            vocab_piece_count,
            vocab_piece_capacity,
            out_a,
            out_token_capacity,
            count_a,
            req_a,
            max_a,
        )
        err_b = explicit_checked_composition(
            payload,
            byte_len,
            cursor_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_table_count,
            rank_table_capacity,
            vocab_piece_lens,
            vocab_piece_count,
            vocab_piece_capacity,
            out_b,
            out_token_capacity,
            count_b,
            req_b,
            max_b,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert count_a == count_b
        assert req_a == req_b
        assert max_a == max_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_helper_and_staged_publish()
    test_success_fixture_parity()
    test_downstream_error_keeps_staged_outputs_unpublished()
    test_adversarial_parity_randomized()
    print("ok")
