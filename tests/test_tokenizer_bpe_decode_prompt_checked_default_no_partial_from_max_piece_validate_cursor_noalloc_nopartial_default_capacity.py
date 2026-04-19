#!/usr/bin/env python3
"""Parity harness for ...NoAllocNoPartialDefaultCapacity prompt decode wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial import (
    tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial_default_capacity(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    max_piece_bytes: int,
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

    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    # Canonical default-capacity derivation in HolyC wrapper.
    derived_vocab_piece_capacity = vocab_piece_count

    return tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        derived_vocab_piece_capacity,
        max_piece_bytes,
        out_bytes,
        out_byte_count,
    )


def staged_default_capacity_composition(
    token_ids: list[int],
    token_count: int,
    io_token_cursor: list[int],
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    max_piece_bytes: int,
    out_bytes: list[int],
    out_byte_count: list[int],
) -> int:
    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    return tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        max_piece_bytes,
        out_bytes,
        out_byte_count,
    )


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes], max_piece_bytes: int) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0xA4] * 4096
    out_new = out_ref.copy()
    count_ref = [0x7373]
    count_new = [0x7373]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    err_ref = staged_default_capacity_composition(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_ref,
        count_ref,
    )
    err_new = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial_default_capacity(
        token_ids,
        len(token_ids),
        cursor_new,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_new,
        count_new,
    )

    assert err_new == err_ref
    assert out_new == out_ref
    assert cursor_new[0] == cursor_ref[0]
    assert count_new[0] == count_ref[0]


def test_source_contains_wrapper_derivation_and_delegation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAllocNoPartialDefaultCapacity"
        in source
    )

    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAllocNoPartialDefaultCapacity",
        1,
    )[1].split("I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialValidateCursor", 1)[0]
    assert "derived_vocab_piece_capacity = vocab_piece_count" in body
    assert (
        "TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAllocNoPartial(" in body
    )


def test_multilingual_parity_vectors() -> None:
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

    run_case(token_ids, 0, pieces, max_piece_bytes=16)
    run_case(token_ids, 6, pieces, max_piece_bytes=16)


def test_adversarial_no_partial_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x6A] * 128
    count = [0x9191]

    cursor = [8]
    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial_default_capacity(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        8,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 8 and count[0] == 0x9191
    assert out == [0x6A] * 128

    cursor = [0]
    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial_default_capacity(
        [1],
        1,
        cursor,
        blob,
        len(blob),
        offsets,
        [1, len(blob) + 7, 3],
        3,
        8,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x9191
    assert out == [0x6A] * 128


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x22] * 16
    count = [0xABAB]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial_default_capacity(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial_default_capacity(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_regression() -> None:
    rng = random.Random(492)
    pieces = [
        b"a",
        b"bc",
        "δ".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"xyz",
        b" ",
    ]

    for _ in range(200):
        count = rng.randint(0, 24)
        token_ids = [rng.randrange(0, len(pieces)) for _ in range(count)]
        cursor = rng.randint(0, count)
        max_piece_bytes = rng.randint(0, 10)
        run_case(token_ids, cursor, pieces, max_piece_bytes=max_piece_bytes)


if __name__ == "__main__":
    test_source_contains_wrapper_derivation_and_delegation()
    test_multilingual_parity_vectors()
    test_adversarial_no_partial_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity_regression()
    print(
        "ok: decode prompt default no-partial from-max-piece validate-cursor "
        "noalloc no-partial default-capacity parity"
    )
