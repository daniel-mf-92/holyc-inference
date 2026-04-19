#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiece."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial import (
    tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
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

    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    max_piece_bytes = 0
    for i in range(vocab_piece_count):
        piece_len = vocab_piece_lens[i]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > max_piece_bytes:
            max_piece_bytes = piece_len

    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_capacity = token_count * max_piece_bytes

    return tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        derived_out_capacity,
        out_byte_count,
    )


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes]) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x92] * 2048
    out_new = out_ref.copy()
    count_ref = [0x4545]
    count_new = [0x4545]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    max_piece_bytes = max((len(piece) for piece in pieces), default=0)

    err_ref = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_ref,
        len(token_ids) * max_piece_bytes,
        count_ref,
    )

    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
        token_ids,
        len(token_ids),
        cursor_new,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_new,
        count_new,
    )

    assert err_new == err_ref
    assert cursor_new[0] == cursor_ref[0]
    assert count_new[0] == count_ref[0]
    assert out_new == out_ref


def test_source_contains_wrapper_and_delegation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiece" in source
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiece(", 1
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefault", 1
    )[0]
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiecePreflight" in body
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocNoPartial" in body


def test_multilingual_success_vectors() -> None:
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

    run_case(token_ids, 0, pieces)
    run_case(token_ids, 5, pieces)


def test_no_partial_on_adversarial_failures() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x66] * 128
    count = [0x7E7E]

    cursor = [8]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 8 and count[0] == 0x7E7E
    assert out == [0x66] * 128

    cursor = [0]
    bad_lens = lens.copy()
    bad_lens[1] = len(blob) + 9
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
        [1],
        1,
        cursor,
        blob,
        len(blob),
        offsets,
        bad_lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x7E7E
    assert out == [0x66] * 128


def test_null_and_overflow_domains() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x12] * 16
    count = [0x3A3A]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_against_staged_composition() -> None:
    rng = random.Random(20260419_482)
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

    for _ in range(3000):
        token_count = rng.randint(0, 48)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 3)

        out_ref = [0x5A] * 1024
        out_new = out_ref.copy()
        count_ref = [0x2626]
        count_new = [0x2626]
        cursor_ref = [cursor_seed]
        cursor_new = [cursor_seed]

        max_piece_bytes = max((len(piece) for piece in pieces), default=0)

        err_ref = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial(
            token_ids,
            token_count,
            cursor_ref,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_ref,
            token_count * max_piece_bytes,
            count_ref,
        )
        err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece(
            token_ids,
            token_count,
            cursor_new,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_new,
            count_new,
        )

        assert err_new == err_ref
        assert cursor_new[0] == cursor_ref[0]
        assert count_new[0] == count_ref[0]
        assert out_new == out_ref


def run() -> None:
    test_source_contains_wrapper_and_delegation()
    test_multilingual_success_vectors()
    test_no_partial_on_adversarial_failures()
    test_null_and_overflow_domains()
    test_randomized_parity_against_staged_composition()
    print("tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece=ok")


if __name__ == "__main__":
    run()
