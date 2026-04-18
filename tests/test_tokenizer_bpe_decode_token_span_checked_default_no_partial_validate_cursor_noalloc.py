#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeTokenSpanCheckedDefaultNoPartialValidateCursorNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor import (
    tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    span_token_count: int,
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
        or span_token_count > I64_MAX
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

    decode_count = span_token_count
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_capacity = token_count * max_piece_bytes

    decode_end = cursor + decode_count
    if decode_end < cursor or decode_end > token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_out_count = 0
    scan = cursor
    while scan < decode_end:
        token_id = token_ids[scan]
        if token_id < 0:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if token_id >= vocab_piece_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        piece_offset = vocab_piece_offsets[token_id]
        piece_len = vocab_piece_lens[token_id]

        if piece_offset > vocab_piece_bytes_len:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if piece_len > vocab_piece_bytes_len - piece_offset:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if piece_len > derived_out_capacity - staged_out_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        staged_out_count += piece_len
        if staged_out_count > derived_out_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

        scan += 1

    write_cursor = 0
    scan = cursor
    while scan < decode_end:
        token_id = token_ids[scan]
        piece_offset = vocab_piece_offsets[token_id]
        piece_len = vocab_piece_lens[token_id]

        for i in range(piece_len):
            out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + i]
            write_cursor += 1
            if write_cursor > staged_out_count:
                return TOKENIZER_BPE_ERR_OVERFLOW

        scan += 1

    if write_cursor != staged_out_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_out_count
    io_token_cursor[0] = decode_end
    return TOKENIZER_BPE_OK


def run_case(
    token_ids: list[int],
    token_cursor: int,
    span_count: int,
    pieces: list[bytes],
    max_piece_bytes: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0xB5] * 2048
    out_noalloc = out_ref.copy()
    count_ref = [0x3131]
    count_noalloc = [0x3131]
    cursor_ref = [token_cursor]
    cursor_noalloc = [token_cursor]

    err_ref = tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor(
        token_ids,
        len(token_ids),
        cursor_ref,
        span_count,
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

    err_noalloc = tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
        token_ids,
        len(token_ids),
        cursor_noalloc,
        span_count,
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

    run_case(token_ids, 0, len(token_ids), pieces, max_piece_bytes=16)
    run_case(token_ids, 4, 7, pieces, max_piece_bytes=16)


def test_no_partial_on_bad_cursor_window_and_capacity() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x91] * 64
    count = [0x4444]

    cursor = [4]
    err = tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
        [0, 1, 2],
        3,
        cursor,
        0,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        4,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 4 and count[0] == 0x4444 and out == [0x91] * 64

    cursor = [2]
    err = tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
        [0, 1, 2],
        3,
        cursor,
        2,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        4,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 2 and count[0] == 0x4444 and out == [0x91] * 64


def test_contract_overflow_and_null_guards() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x33] * 16
    count = [0x6666]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
            [0],
            I64_MAX + 1,
            cursor,
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            8,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )

    assert (
        tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
            None,
            0,
            cursor,
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            8,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    cursor = [0]
    err = tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc(
        [0, -1],
        2,
        cursor,
        2,
        blob,
        len(blob),
        offsets,
        lens,
        1,
        1,
        8,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x6666 and out == [0x33] * 16


def test_randomized_parity_against_allocating_wrapper() -> None:
    rng = random.Random(20260419_445)
    pieces = [
        b"a",
        b"bc",
        b"def",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
    ]

    for _ in range(4000):
        token_count = rng.randint(0, 64)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor = rng.randint(0, token_count)
        span = rng.randint(0, token_count - cursor)
        max_piece_bytes = rng.randint(0, 16)
        run_case(token_ids, cursor, span, pieces, max_piece_bytes)


if __name__ == "__main__":
    test_multilingual_success_parity_against_allocating_wrapper()
    test_no_partial_on_bad_cursor_window_and_capacity()
    test_contract_overflow_and_null_guards()
    test_randomized_parity_against_allocating_wrapper()
    print("test_tokenizer_bpe_decode_token_span_checked_default_no_partial_validate_cursor_noalloc: ok")
