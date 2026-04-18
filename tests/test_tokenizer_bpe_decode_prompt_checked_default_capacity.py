#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodePromptCheckedDefaultCapacity."""

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


def tokenizer_bpe_decode_prompt_checked(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
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
        or vocab_piece_capacity > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_token_span_checked(
        token_ids,
        token_count,
        io_token_cursor,
        token_count - cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        out_bytes,
        out_byte_capacity,
        out_byte_count,
    )


def tokenizer_bpe_decode_prompt_checked_default_capacity(
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

    return tokenizer_bpe_decode_prompt_checked(
        token_ids,
        token_count,
        io_token_cursor,
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


def run_case(
    token_ids: list[int],
    token_cursor: int,
    pieces: list[bytes],
    out_capacity: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_explicit = [0xA1] * 2048
    out_wrapper = out_explicit.copy()
    count_explicit = [0x1313]
    count_wrapper = [0x1313]
    cursor_explicit = [token_cursor]
    cursor_wrapper = [token_cursor]

    err_explicit = tokenizer_bpe_decode_prompt_checked(
        token_ids,
        len(token_ids),
        cursor_explicit,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        out_explicit,
        out_capacity,
        count_explicit,
    )

    err_wrapper = tokenizer_bpe_decode_prompt_checked_default_capacity(
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

    assert err_wrapper == err_explicit
    assert cursor_wrapper[0] == cursor_explicit[0]
    assert count_wrapper[0] == count_explicit[0]
    assert out_wrapper == out_explicit


def test_multilingual_parity_vs_explicit_capacity_decode() -> None:
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


def test_cursor_and_malformed_capacity_adversarial_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x66] * 64
    count = [0x4040]

    cursor = [5]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity(
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
    assert cursor[0] == 5 and count[0] == 0x4040
    assert out == [0x66] * 64

    # Explicit malformed-capacity adversary: wrapper semantics must equal
    # explicit-capacity decode only when capacity == piece_count.
    cursor_good = [0]
    count_good = [0x4040]
    out_good = [0x66] * 64
    err_good = tokenizer_bpe_decode_prompt_checked_default_capacity(
        [0, 1, 2],
        3,
        cursor_good,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_good,
        len(out_good),
        count_good,
    )
    assert err_good == 0

    cursor_bad = [0]
    count_bad = [0x4040]
    out_bad = [0x66] * 64
    err_bad = tokenizer_bpe_decode_prompt_checked(
        [0, 1, 2],
        3,
        cursor_bad,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces) - 1,
        out_bad,
        len(out_bad),
        count_bad,
    )
    assert err_bad == TOKENIZER_BPE_ERR_BAD_PARAM


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x12] * 16
    count = [0x7777]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            len(out),
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            len(out),
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_vs_explicit_capacity_decode() -> None:
    rng = random.Random(20260419_437)
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
        out_capacity = rng.randint(0, 256)
        run_case(token_ids, cursor, pieces, out_capacity)


if __name__ == "__main__":
    test_multilingual_parity_vs_explicit_capacity_decode()
    test_cursor_and_malformed_capacity_adversarial_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity_vs_explicit_capacity_decode()
    print("test_tokenizer_bpe_decode_prompt_checked_default_capacity: ok")
