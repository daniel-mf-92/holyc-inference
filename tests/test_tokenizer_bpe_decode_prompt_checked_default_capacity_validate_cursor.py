#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursor."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_capacity import (
    tokenizer_bpe_decode_prompt_checked_default_capacity,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
)


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
    out_byte_capacity: int,
    out_byte_count: list[int] | None,
) -> int:
    if (
        token_ids is None
        or io_token_cursor is None
        or vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or out_byte_count is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_count = token_count - cursor
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_prompt_checked_default_capacity(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_capacity,
        out_byte_count,
    )


def explicit_guarded_composition(
    token_ids: list[int],
    token_count: int,
    io_token_cursor: list[int],
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    out_bytes: list[int],
    out_byte_capacity: int,
    out_byte_count: list[int],
) -> int:
    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_count = token_count - cursor
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_prompt_checked_default_capacity(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_capacity,
        out_byte_count,
    )


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes], out_capacity: int) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x59] * 2048
    out_wrapper = out_ref.copy()
    count_ref = [0xBABA]
    count_wrapper = [0xBABA]
    cursor_ref = [token_cursor]
    cursor_wrapper = [token_cursor]

    err_ref = explicit_guarded_composition(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_ref,
        out_capacity,
        count_ref,
    )
    err_wrapper = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
        token_ids,
        len(token_ids),
        cursor_wrapper,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_wrapper,
        out_capacity,
        count_wrapper,
    )

    assert err_wrapper == err_ref
    assert cursor_wrapper[0] == cursor_ref[0]
    assert count_wrapper[0] == count_ref[0]
    assert out_wrapper == out_ref


def test_multilingual_success_parity_vs_explicit_guarded_composition() -> None:
    pieces = [
        b"hello",
        b" ",
        b"world",
        b"!",
        " Κα".encode("utf-8"),
        "λη".encode("utf-8"),
        "μέ".encode("utf-8"),
        "ρα".encode("utf-8"),
        " 世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]
    token_ids = [0, 1, 2, 3, 1, 4, 5, 6, 7, 1, 8, 1, 9, 10]

    run_case(token_ids, 0, pieces, out_capacity=256)
    run_case(token_ids, 6, pieces, out_capacity=256)


def test_cursor_window_adversarial_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x22] * 64
    count = [0x7070]

    cursor = [5]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out,
        len(out),
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 5 and count[0] == 0x7070
    assert out == [0x22] * 64

    cursor = [0]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
        [0, 1],
        2,
        cursor,
        blob,
        I64_MAX + 1,
        offsets,
        lens,
        len(pieces),
        out,
        len(out),
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW


def test_randomized_cursor_and_capacity_parity() -> None:
    rng = random.Random(438)
    pieces = [
        b"a",
        b"bc",
        b"def",
        b"ghi",
        "μ".encode("utf-8"),
        "🙂".encode("utf-8"),
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(250):
        token_count = rng.randint(0, 20)
        token_ids = [rng.randrange(len(pieces)) for _ in range(token_count)]
        cursor0 = rng.randint(0, token_count + 2)
        out_capacity = rng.randint(0, 128)

        out_ref = [0x33] * 256
        out_wrapper = out_ref.copy()
        count_ref = [0x1111]
        count_wrapper = [0x1111]
        cursor_ref = [cursor0]
        cursor_wrapper = [cursor0]

        err_ref = explicit_guarded_composition(
            token_ids,
            token_count,
            cursor_ref,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_ref,
            out_capacity,
            count_ref,
        )
        err_wrapper = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
            token_ids,
            token_count,
            cursor_wrapper,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_wrapper,
            out_capacity,
            count_wrapper,
        )

        assert err_wrapper == err_ref
        assert cursor_wrapper[0] == cursor_ref[0]
        assert count_wrapper[0] == count_ref[0]
        assert out_wrapper == out_ref
