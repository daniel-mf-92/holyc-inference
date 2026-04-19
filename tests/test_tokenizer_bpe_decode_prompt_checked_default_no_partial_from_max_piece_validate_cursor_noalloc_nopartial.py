#!/usr/bin/env python3
"""Parity harness for ...FromMaxPieceValidateCursorNoAllocNoPartial wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc import (
    tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
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
        or vocab_piece_capacity > I64_MAX
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_cursor = [io_token_cursor[0]]
    staged_count = [out_byte_count[0]]
    staged_out = out_bytes.copy()

    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
        token_ids,
        token_count,
        staged_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        max_piece_bytes,
        staged_out,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    out_bytes[:] = staged_out
    io_token_cursor[0] = staged_cursor[0]
    out_byte_count[0] = staged_count[0]
    return TOKENIZER_BPE_OK


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes], max_piece_bytes: int) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x5E] * 4096
    out_new = out_ref.copy()
    count_ref = [0x3131]
    count_new = [0x3131]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    err_ref = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        max_piece_bytes,
        out_ref,
        count_ref,
    )
    err_new = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
        token_ids,
        len(token_ids),
        cursor_new,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        max_piece_bytes,
        out_new,
        count_new,
    )

    assert err_new == err_ref
    assert out_new == out_ref
    assert cursor_new[0] == cursor_ref[0]
    assert count_new[0] == count_ref[0]


def test_source_contains_wrapper_without_malloc() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAllocNoPartial" in source
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAllocNoPartial", 1
    )[1].split("I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialValidateCursor", 1)[0]
    assert "MAlloc(" not in body
    assert "TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAlloc(" in body


def test_multilingual_success_parity() -> None:
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
    run_case(token_ids, 7, pieces, max_piece_bytes=16)


def test_no_partial_adversarial_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x77] * 96
    count = [0xBABA]

    cursor = [6]
    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        8,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 6
    assert count[0] == 0xBABA
    assert out == [0x77] * 96

    cursor = [0]
    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
        [2, 2, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        0,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert count[0] == 0xBABA
    assert out == [0x77] * 96


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x19] * 16
    count = [0x5C5C]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity() -> None:
    rng = random.Random(20260419_477)
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

    for _ in range(3000):
        token_count = rng.randint(0, 48)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        token_cursor = rng.randint(0, token_count + 5)
        max_piece_bytes = rng.randint(0, 40)
        run_case(token_ids, token_cursor, pieces, max_piece_bytes)


def run() -> None:
    test_source_contains_wrapper_without_malloc()
    test_multilingual_success_parity()
    test_no_partial_adversarial_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity()
    print("tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc_nopartial=ok")


if __name__ == "__main__":
    run()
