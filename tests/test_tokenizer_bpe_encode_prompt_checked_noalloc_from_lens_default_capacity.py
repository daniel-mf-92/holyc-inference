#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacity."""

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
    for index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[index]
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


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
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
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_token_capacity = prompt_nbytes

    snapshot_data = data
    snapshot_byte_len = byte_len
    snapshot_prompt_nbytes = prompt_nbytes
    snapshot_rank_left_tokens = rank_left_tokens
    snapshot_rank_right_tokens = rank_right_tokens
    snapshot_rank_values = rank_values
    snapshot_rank_merged_tokens = rank_merged_tokens
    snapshot_rank_table_count = rank_table_count
    snapshot_rank_table_capacity = rank_table_capacity
    snapshot_vocab_piece_lens = vocab_piece_lens
    snapshot_vocab_piece_count = vocab_piece_count
    snapshot_vocab_piece_capacity = vocab_piece_capacity
    snapshot_out_token_ids = out_token_ids
    snapshot_cursor = io_cursor[0]

    staged_cursor = [snapshot_cursor]
    staged_token_count = [0]
    staged_required_token_capacity = [0]
    staged_max_piece_len = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        out_token_ids,
        derived_out_token_capacity,
        staged_token_count,
        staged_required_token_capacity,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if (
        snapshot_data is not data
        or snapshot_byte_len != byte_len
        or snapshot_prompt_nbytes != prompt_nbytes
        or snapshot_rank_left_tokens is not rank_left_tokens
        or snapshot_rank_right_tokens is not rank_right_tokens
        or snapshot_rank_values is not rank_values
        or snapshot_rank_merged_tokens is not rank_merged_tokens
        or snapshot_rank_table_count != rank_table_count
        or snapshot_rank_table_capacity != rank_table_capacity
        or snapshot_vocab_piece_lens is not vocab_piece_lens
        or snapshot_vocab_piece_count != vocab_piece_count
        or snapshot_vocab_piece_capacity != vocab_piece_capacity
        or snapshot_out_token_ids is not out_token_ids
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    io_cursor[0] = staged_cursor[0]
    out_token_count[0] = staged_token_count[0]
    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
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
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_token_capacity = prompt_nbytes

    staged_cursor = [io_cursor[0]]
    staged_token_count = [0]
    staged_required_token_capacity = [0]
    staged_max_piece_len = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        out_token_ids,
        derived_out_token_capacity,
        staged_token_count,
        staged_required_token_capacity,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    io_cursor[0] = staged_cursor[0]
    out_token_count[0] = staged_token_count[0]
    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    return TOKENIZER_BPE_OK


def test_source_contains_lens_default_capacity_helpers() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")

    sig_lens = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens("
    assert sig_lens in source
    body_lens = source.split(sig_lens, 1)[1]
    assert "if (vocab_piece_count > vocab_piece_capacity)" in body_lens
    assert "while (piece_index < vocab_piece_count)" in body_lens
    assert "if (err != TOKENIZER_BPE_OK)" in body_lens
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body_lens
    assert "*out_max_piece_len = staged_max_piece_len;" in body_lens
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(" in body_lens

    sig_default = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacity("
    assert sig_default in source
    body_default = source.split(sig_default, 1)[1]
    assert "derived_out_token_capacity = prompt_nbytes;" in body_default
    assert "IQ-821 default-capacity contract" in body_default
    assert "snapshot_cursor = *io_cursor;" in body_default
    assert "staged_cursor = snapshot_cursor;" in body_default
    assert "if (err != TOKENIZER_BPE_OK)" in body_default
    assert "*io_cursor = staged_cursor;" in body_default
    assert "*out_token_count = staged_token_count;" in body_default
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLens(" in body_default


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


def test_success_fixture_parity() -> None:
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

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
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
        count_b,
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert req_a == req_b == [len(payload)]
    assert max_a == max_b == [8]
    assert cursor_a == cursor_b
    assert count_a == count_b
    assert out_a == out_b


def test_null_overflow_and_no_partial() -> None:
    payload = [ord("o"), ord("k")]
    left: list[int] = []
    right: list[int] = []
    ranks: list[int] = []
    merged: list[int] = []
    vocab_lens = [1, 2, 3]

    sent_out = [0xAAAA] * 8
    sent_count = [0xBBBB]
    sent_req = [0xCCCC]
    sent_max = [0xDDDD]
    sent_cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
        payload,
        len(payload),
        sent_cursor,
        2,
        left,
        right,
        ranks,
        merged,
        0,
        0,
        None,
        0,
        0,
        sent_out,
        sent_count,
        sent_req,
        sent_max,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert sent_cursor == [0]
    assert sent_count == [0xBBBB]
    assert sent_req == [0xCCCC]
    assert sent_max == [0xDDDD]
    assert sent_out == [0xAAAA] * 8

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
        payload,
        I64_MAX + 1,
        [0],
        1,
        left,
        right,
        ranks,
        merged,
        0,
        0,
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        [0] * 8,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
        payload,
        len(payload),
        [0],
        1,
        left,
        right,
        ranks,
        merged,
        0,
        0,
        [1, I64_MAX + 1],
        2,
        2,
        [0] * 8,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW


def test_cursor_oob_rejects_without_partial_publish() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("TempleOS".encode("utf-8"))
    vocab_lens = [1, 2, 6]

    cursor_a = [len(payload) + 1]
    cursor_b = [len(payload) + 1]
    out_a = [0xA1A1] * 32
    out_b = [0xA1A1] * 32
    count_a = [0xB2B2]
    count_b = [0xB2B2]
    req_a = [0xC3C3]
    req_b = [0xC3C3]
    max_a = [0xD4D4]
    max_b = [0xD4D4]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
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
        count_b,
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor_a == cursor_b == [len(payload) + 1]
    assert count_a == count_b == [0xB2B2]
    assert req_a == req_b == [0xC3C3]
    assert max_a == max_b == [0xD4D4]
    assert out_a == out_b == [0xA1A1] * 32


def test_randomized_parity() -> None:
    rng = random.Random(20260420_691)

    for _ in range(2200):
        byte_len = rng.randint(0, 64)
        payload = [rng.randint(0, 255) for _ in range(byte_len)]

        prompt_nbytes = rng.randint(0, byte_len)
        cursor = rng.randint(0, byte_len)

        rank_table_count = rng.randint(0, 12)
        rank_table_capacity = rank_table_count + rng.randint(0, 3)

        left = [rng.randint(0, 700) for _ in range(rank_table_capacity)]
        right = [rng.randint(0, 700) for _ in range(rank_table_capacity)]
        ranks = [rng.randint(0, 4) for _ in range(rank_table_capacity)]
        merged = [rng.randint(0, 700) for _ in range(rank_table_capacity)]

        vocab_piece_count = rng.randint(0, 16)
        vocab_piece_capacity = vocab_piece_count + rng.randint(0, 3)
        vocab_lens = [rng.randint(0, 32) for _ in range(vocab_piece_capacity)]

        out_a = [0x9191] * max(1, byte_len + 4)
        out_b = [0x9191] * max(1, byte_len + 4)
        count_a = [0x3131]
        count_b = [0x3131]
        req_a = [0x4141]
        req_b = [0x4141]
        max_a = [0x5151]
        max_b = [0x5151]

        cursor_a = [cursor]
        cursor_b = [cursor]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity(
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
            vocab_lens,
            vocab_piece_count,
            vocab_piece_capacity,
            out_a,
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
            vocab_lens,
            vocab_piece_count,
            vocab_piece_capacity,
            out_b,
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
    test_source_contains_lens_default_capacity_helpers()
    test_success_fixture_parity()
    test_null_overflow_and_no_partial()
    test_cursor_oob_rejects_without_partial_publish()
    test_randomized_parity()
    print("ok")
