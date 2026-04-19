#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromMaxPieceDefaultSpan helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece import (
    tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_span(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
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

    if token_count > I64_MAX or vocab_piece_bytes_len > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    span_token_count = token_count - cursor
    return tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        token_ids,
        token_count,
        io_token_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
    )


def explicit_default_span_composition(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
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

    if token_count > I64_MAX or vocab_piece_bytes_len > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        token_ids,
        token_count,
        io_token_cursor,
        token_count - cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
    )


def test_source_contains_helper_signature_and_default_span_derivation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceDefaultSpan("
        in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceDefaultSpan(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight",
        1,
    )[0]
    assert "if (cursor > token_count)" in body
    assert "span_token_count = token_count - cursor;" in body
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiece(" in body


def test_multilingual_and_adversarial_prompt_tail_parity() -> None:
    pieces = [
        b"",
        b"hello",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    token_ids = [1, 2, 3, 4, 5]

    out_a = [0x77] * 256
    out_b = out_a.copy()
    count_a = [0xABCD]
    count_b = [0xABCD]
    cursor_a = [2]
    cursor_b = [2]

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_span(
        token_ids,
        len(token_ids),
        cursor_a,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out_a,
        count_a,
    )
    err_b = explicit_default_span_composition(
        token_ids,
        len(token_ids),
        cursor_b,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out_b,
        count_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert cursor_a[0] == cursor_b[0] == len(token_ids)
    assert count_a[0] == count_b[0]
    assert out_a == out_b

    seeded_out = [0x33] * 64
    seeded_count = [0x4545]
    bad_cursor = [9]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_span(
        [0, 1, 2],
        3,
        bad_cursor,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        seeded_out,
        seeded_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert bad_cursor[0] == 9
    assert seeded_out == [0x33] * 64
    assert seeded_count[0] == 0x4545


def test_null_and_overflow_rejects() -> None:
    out = [0x44] * 32
    count = [0x2222]

    assert (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_span(
            None,
            0,
            [0],
            [0],
            1,
            [0],
            [0],
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_span(
            [0],
            I64_MAX + 1,
            [0],
            [0],
            1,
            [0],
            [0],
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_prompt_tail_parity() -> None:
    rng = random.Random(20260419_530)
    pieces = [
        b"",
        b"a",
        b"bc",
        b"DEF",
        "🙂".encode("utf-8"),
        "中".encode("utf-8"),
        b"\n",
        b" ",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(4000):
        token_count = rng.randint(0, 64)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        if token_count and rng.random() < 0.12:
            token_ids[rng.randrange(token_count)] = len(pieces) + rng.randint(1, 9)
        if token_count and rng.random() < 0.06:
            token_ids[rng.randrange(token_count)] = -1

        cursor_seed = rng.randint(0, token_count + 4)

        out_a = [0x5A] * 1024
        out_b = out_a.copy()
        count_a = [0x2626]
        count_b = [0x2626]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_span(
            token_ids,
            token_count,
            cursor_a,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_a,
            count_a,
        )
        err_b = explicit_default_span_composition(
            token_ids,
            token_count,
            cursor_b,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_b,
            count_b,
        )

        assert err_a == err_b
        assert cursor_a[0] == cursor_b[0]
        assert count_a[0] == count_b[0]
        assert out_a == out_b

