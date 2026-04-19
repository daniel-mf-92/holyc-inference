#!/usr/bin/env python3
"""Parity harness for ...FromMaxPieceValidateCursorNoAlloc wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor import (
    tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
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

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_count = token_count - cursor
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_capacity = token_count * max_piece_bytes

    staged_cursor = [cursor]
    staged_count = [out_byte_count[0]]

    err = tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
        token_ids,
        token_count,
        staged_cursor,
        decode_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        max_piece_bytes,
        out_bytes,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_cursor[0] != token_count or staged_count[0] > derived_out_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


from test_tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc import (
    tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc,
)


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes], max_piece_bytes: int) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0xA9] * 2048
    out_noalloc = out_ref.copy()
    count_ref = [0x5151]
    count_noalloc = [0x5151]
    cursor_ref = [token_cursor]
    cursor_noalloc = [token_cursor]

    err_ref = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor(
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
    err_noalloc = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
        token_ids,
        len(token_ids),
        cursor_noalloc,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        max_piece_bytes,
        out_noalloc,
        count_noalloc,
    )

    assert err_noalloc == err_ref
    assert cursor_noalloc[0] == cursor_ref[0]
    assert count_noalloc[0] == count_ref[0]
    assert out_noalloc == out_ref


def test_source_contains_noalloc_wrapper_without_malloc() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAlloc" in source
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialFromMaxPieceValidateCursorNoAlloc", 1
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultNoPartialValidateCursor", 1
    )[0]
    assert "MAlloc(" not in body
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultNoPartialValidateCursorNoAlloc(" not in body


def test_multilingual_success_parity_against_allocating_wrapper() -> None:
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


def test_cursor_and_capacity_adversarial_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x7A] * 64
    count = [0xABCD]

    cursor = [5]
    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
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
    assert cursor[0] == 5 and count[0] == 0xABCD
    assert out == [0x7A] * 64

    cursor = [0]
    err = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
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
    assert cursor[0] == 0 and count[0] == 0xABCD
    assert out == [0x7A] * 64


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x19] * 16
    count = [0x5C5C]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
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
        tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
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


def test_randomized_parity_against_allocating_wrapper() -> None:
    rng = random.Random(20260419_461)
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

    for _ in range(2500):
        token_count = rng.randint(0, 40)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 3)
        max_piece_bytes = rng.randint(0, 32)

        out_ref = [0xDA] * 512
        out_noalloc = out_ref.copy()
        count_ref = [0x7878]
        count_noalloc = [0x7878]
        cursor_ref = [cursor_seed]
        cursor_noalloc = [cursor_seed]

        err_ref = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor(
            token_ids,
            token_count,
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
        err_noalloc = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
            token_ids,
            token_count,
            cursor_noalloc,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            max_piece_bytes,
            out_noalloc,
            count_noalloc,
        )

        assert err_noalloc == err_ref
        assert cursor_noalloc[0] == cursor_ref[0]
        assert count_noalloc[0] == count_ref[0]
        assert out_noalloc == out_ref


def run() -> None:
    test_source_contains_noalloc_wrapper_without_malloc()
    test_multilingual_success_parity_against_allocating_wrapper()
    test_cursor_and_capacity_adversarial_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity_against_allocating_wrapper()
    print("tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc=ok")


if __name__ == "__main__":
    run()
