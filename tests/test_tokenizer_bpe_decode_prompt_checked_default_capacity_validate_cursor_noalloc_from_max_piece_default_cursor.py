#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromMaxPieceDefaultCursor helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece import (
    tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor(
    token_ids: list[int] | None,
    token_count: int,
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
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = [0]
    return tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        token_ids,
        token_count,
        cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
    )


def explicit_default_cursor_composition(
    token_ids: list[int] | None,
    token_count: int,
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
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = [0]
    return tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        token_ids,
        token_count,
        cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
    )


def test_source_contains_helper_signature_and_default_cursor_delegate() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceDefaultCursor("
        in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceDefaultCursor(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceViaCapacityHelper(",
        1,
    )[0]
    assert "if (!token_ids || !vocab_piece_bytes || !vocab_piece_offsets ||" in body
    assert "cursor = 0;" in body
    assert (
        "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiece(" in body
    )


def test_null_and_overflow_failures_preserve_out_byte_count() -> None:
    out_count = [0x5757]

    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor(
        None,
        0,
        [0],
        1,
        [0],
        [0],
        1,
        [0],
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert out_count[0] == 0x5757

    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor(
        [0],
        I64_MAX + 1,
        [0],
        1,
        [0],
        [0],
        1,
        [0],
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_count[0] == 0x5757


def test_multilingual_and_adversarial_prompt_parity() -> None:
    pieces = [
        b"hello",
        b" ",
        b"world",
        " Κα".encode("utf-8"),
        "λη".encode("utf-8"),
        "μέ".encode("utf-8"),
        "ρα".encode("utf-8"),
        " 世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    token_ids = [0, 1, 2, 1, 3, 4, 5, 6, 1, 7, 1, 8, 9]

    out_a = [0xA5] * 2048
    out_b = out_a.copy()
    count_a = [0x1234]
    count_b = [0x1234]

    err_a = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor(
        token_ids,
        len(token_ids),
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out_a,
        count_a,
    )
    err_b = explicit_default_cursor_composition(
        token_ids,
        len(token_ids),
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out_b,
        count_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert count_a[0] == count_b[0]
    assert out_a == out_b

    bad_offsets = offsets.copy()
    bad_offsets[3] = len(blob) + 5
    out_a = [0xB6] * 256
    out_b = out_a.copy()
    count_a = [0x2222]
    count_b = [0x2222]

    err_a = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor(
        token_ids,
        len(token_ids),
        blob,
        len(blob),
        bad_offsets,
        lens,
        len(lens),
        out_a,
        count_a,
    )
    err_b = explicit_default_cursor_composition(
        token_ids,
        len(token_ids),
        blob,
        len(blob),
        bad_offsets,
        lens,
        len(lens),
        out_b,
        count_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_ERR_BAD_PARAM
    assert count_a[0] == count_b[0] == 0x2222
    assert out_a == out_b


def test_randomized_equivalence_against_explicit_composition() -> None:
    rng = random.Random(20260419_542)

    for _ in range(6000):
        token_count = rng.randint(0, 128)
        if rng.random() < 0.06:
            token_count = I64_MAX + rng.randint(1, 32)

        vocab_piece_count = rng.randint(0, 96)
        lens = [rng.randint(0, 96) for _ in range(vocab_piece_count)]
        if vocab_piece_count and rng.random() < 0.08:
            lens[rng.randrange(vocab_piece_count)] = I64_MAX + rng.randint(1, 16)

        blob_len = sum(length for length in lens if 0 <= length <= I64_MAX)
        blob = [0] * max(1, blob_len)

        offsets = []
        running = 0
        for length in lens:
            offsets.append(running)
            if 0 <= length <= I64_MAX:
                running += length

        if vocab_piece_count:
            token_ids = [rng.randrange(vocab_piece_count) for _ in range(token_count if token_count <= I64_MAX else 0)]
        else:
            token_ids = [0 for _ in range(token_count if token_count <= I64_MAX else 0)]

        if vocab_piece_count and rng.random() < 0.06:
            idx = rng.randrange(vocab_piece_count)
            offsets[idx] = len(blob) + rng.randint(1, 24)

        out_a = [0xCC] * 32768
        out_b = out_a.copy()
        count_a = [0x1111]
        count_b = [0x1111]

        err_a = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor(
            token_ids,
            token_count,
            blob,
            len(blob),
            offsets,
            lens,
            vocab_piece_count,
            out_a,
            count_a,
        )

        err_b = explicit_default_cursor_composition(
            token_ids,
            token_count,
            blob,
            len(blob),
            offsets,
            lens,
            vocab_piece_count,
            out_b,
            count_b,
        )

        assert err_a == err_b
        assert count_a[0] == count_b[0]
        assert out_a == out_b


def run() -> None:
    test_source_contains_helper_signature_and_default_cursor_delegate()
    test_null_and_overflow_failures_preserve_out_byte_count()
    test_multilingual_and_adversarial_prompt_parity()
    test_randomized_equivalence_against_explicit_composition()
    print(
        "tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_default_cursor=ok"
    )


if __name__ == "__main__":
    run()
