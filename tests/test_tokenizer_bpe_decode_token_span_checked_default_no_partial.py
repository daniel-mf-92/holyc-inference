#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeTokenSpanCheckedDefaultNoPartial."""

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
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_no_partial(
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

    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_capacity = token_count * max_piece_bytes

    staged_cursor = [io_token_cursor[0]]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(derived_out_capacity, 1)

    err = tokenizer_bpe_decode_token_span_checked(
        token_ids,
        token_count,
        staged_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        staged_bytes,
        derived_out_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > derived_out_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_count[0]):
        out_bytes[idx] = staged_bytes[idx]

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(
    token_ids: list[int],
    token_cursor: int,
    span_count: int,
    pieces: list[bytes],
    max_piece_bytes: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_explicit = [0xA7] * 1024
    out_wrapper = out_explicit.copy()
    count_explicit = [0x5151]
    count_wrapper = [0x5151]
    cursor_explicit = [token_cursor]
    cursor_wrapper = [token_cursor]

    if len(token_ids) and max_piece_bytes > I64_MAX // len(token_ids):
        err_explicit = TOKENIZER_BPE_ERR_OVERFLOW
    else:
        derived_capacity = len(token_ids) * max_piece_bytes
        staged_out = [0] * max(derived_capacity, 1)
        staged_count = [count_explicit[0]]
        staged_cursor = [cursor_explicit[0]]

        err_explicit = tokenizer_bpe_decode_token_span_checked(
            token_ids,
            len(token_ids),
            staged_cursor,
            span_count,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            staged_out,
            derived_capacity,
            staged_count,
        )
        if err_explicit == TOKENIZER_BPE_OK:
            for idx in range(staged_count[0]):
                out_explicit[idx] = staged_out[idx]
            count_explicit[0] = staged_count[0]
            cursor_explicit[0] = staged_cursor[0]

    err_wrapper = tokenizer_bpe_decode_token_span_checked_default_no_partial(
        token_ids,
        len(token_ids),
        cursor_wrapper,
        span_count,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        max_piece_bytes,
        out_wrapper,
        count_wrapper,
    )

    assert err_wrapper == err_explicit
    assert cursor_wrapper[0] == cursor_explicit[0]
    assert count_wrapper[0] == count_explicit[0]
    assert out_wrapper == out_explicit


def test_multilingual_parity_vs_explicit_staged_composition() -> None:
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


def test_adversarial_capacity_and_malformed_token_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0xCC] * 64
    count = [0x8080]
    cursor = [0]

    err = tokenizer_bpe_decode_token_span_checked_default_no_partial(
        [0, 1, 2],
        3,
        cursor,
        3,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        1,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x8080
    assert out == [0xCC] * 64

    err = tokenizer_bpe_decode_token_span_checked_default_no_partial(
        [0, -1, 2],
        3,
        cursor,
        3,
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
    assert cursor[0] == 0 and count[0] == 0x8080
    assert out == [0xCC] * 64


def test_overflow_and_null_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x7272]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_token_span_checked_default_no_partial(
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
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_token_span_checked_default_no_partial(
            [0],
            I64_MAX,
            cursor,
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            2,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_415)
    pieces = [
        b"a",
        b"bb",
        b"ccc",
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]

    for _ in range(2500):
        token_ids = [rng.randrange(len(pieces)) for _ in range(rng.randint(0, 40))]
        cursor = rng.randint(0, len(token_ids))
        span = rng.randint(0, len(token_ids) - cursor)
        max_piece_bytes = rng.randint(0, 8)
        run_case(token_ids, cursor, span, pieces, max_piece_bytes=max_piece_bytes)


if __name__ == "__main__":
    test_multilingual_parity_vs_explicit_staged_composition()
    test_adversarial_capacity_and_malformed_token_vectors()
    test_overflow_and_null_contracts()
    test_randomized_parity_vs_explicit_staged_composition()
    print("test_tokenizer_bpe_decode_token_span_checked_default_no_partial: ok")
