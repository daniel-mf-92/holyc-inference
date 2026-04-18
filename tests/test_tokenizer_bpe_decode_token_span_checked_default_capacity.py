#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeTokenSpanCheckedDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked import (
    build_vocab_tables,
    tokenizer_bpe_decode_token_span_checked,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    span_token_count: int,
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
        or span_token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    return tokenizer_bpe_decode_token_span_checked(
        token_ids,
        token_count,
        io_token_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        out_bytes,
        out_byte_capacity,
        out_byte_count,
    )


def run_case(token_ids: list[int], token_cursor: int, span_count: int, pieces: list[bytes]) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_explicit = [0x6A] * 1024
    out_wrapper = out_explicit.copy()
    count_explicit = [0x9191]
    count_wrapper = [0x9191]
    cursor_explicit = [token_cursor]
    cursor_wrapper = [token_cursor]

    err_explicit = tokenizer_bpe_decode_token_span_checked(
        token_ids,
        len(token_ids),
        cursor_explicit,
        span_count,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        out_explicit,
        len(out_explicit),
        count_explicit,
    )

    err_wrapper = tokenizer_bpe_decode_token_span_checked_default_capacity(
        token_ids,
        len(token_ids),
        cursor_wrapper,
        span_count,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_wrapper,
        len(out_wrapper),
        count_wrapper,
    )

    assert err_wrapper == err_explicit
    assert cursor_wrapper[0] == cursor_explicit[0]
    assert count_wrapper[0] == count_explicit[0]
    assert out_wrapper == out_explicit


def test_multilingual_parity_vs_explicit_capacity_call() -> None:
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

    run_case(token_ids, 0, len(token_ids), pieces)
    run_case(token_ids, 5, 6, pieces)


def test_guard_contracts_and_no_mutation_on_error() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0xCC] * 32
    count = [0x7A7A]
    cursor = [0]

    err = tokenizer_bpe_decode_token_span_checked_default_capacity(
        [0, 1, 2],
        3,
        cursor,
        3,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        2,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x7A7A
    assert out == [0xCC] * 32

    err = tokenizer_bpe_decode_token_span_checked_default_capacity(
        None,
        0,
        cursor,
        0,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        len(out),
        count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_decode_token_span_checked_default_capacity(
        [0],
        I64_MAX + 1,
        cursor,
        0,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        len(out),
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW


def test_randomized_parity_vs_explicit_capacity_call() -> None:
    rng = random.Random(20260418_425)
    pieces = [
        b"a",
        b"bb",
        b"ccc",
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]

    for _ in range(3000):
        token_ids = [rng.randrange(len(pieces)) for _ in range(rng.randint(0, 48))]
        cursor = rng.randint(0, len(token_ids))
        span = rng.randint(0, len(token_ids) - cursor)
        run_case(token_ids, cursor, span, pieces)


if __name__ == "__main__":
    test_multilingual_parity_vs_explicit_capacity_call()
    test_guard_contracts_and_no_mutation_on_error()
    test_randomized_parity_vs_explicit_capacity_call()
    print("test_tokenizer_bpe_decode_token_span_checked_default_capacity: ok")
