#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeSingleTokenCheckedDefault."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_single_token_checked import (
    build_vocab_tables,
    tokenizer_bpe_decode_single_token_checked,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_single_token_checked_default(
    token_id: int,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    max_piece_bytes: int,
    out_bytes: list[int] | None,
    io_out_cursor: list[int] | None,
    out_byte_count: list[int] | None,
) -> int:
    if (
        vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or io_out_cursor is None
        or out_byte_count is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_cursor = io_out_cursor[0]
    if out_cursor > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if max_piece_bytes > I64_MAX - out_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_byte_capacity = out_cursor + max_piece_bytes

    return tokenizer_bpe_decode_single_token_checked(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        out_bytes,
        derived_out_byte_capacity,
        io_out_cursor,
        out_byte_count,
    )


def test_multilingual_fixture_explicit_parity() -> None:
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
    blob, offsets, lens = build_vocab_tables(pieces)

    token_id = 8
    cursor0 = 9
    max_piece_bytes = 16

    out_default = [0x9E] * 256
    out_explicit = out_default.copy()
    count_default = [0x6161]
    count_explicit = [0x6161]
    cursor_default = [cursor0]
    cursor_explicit = [cursor0]

    err_default = tokenizer_bpe_decode_single_token_checked_default(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_default,
        cursor_default,
        count_default,
    )

    derived_capacity = cursor0 + max_piece_bytes
    err_explicit = tokenizer_bpe_decode_single_token_checked(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        out_explicit,
        derived_capacity,
        cursor_explicit,
        count_explicit,
    )

    assert err_default == TOKENIZER_BPE_OK
    assert err_default == err_explicit
    assert out_default == out_explicit
    assert count_default[0] == count_explicit[0]
    assert cursor_default[0] == cursor_explicit[0]


def test_token_range_failures_and_small_default_capacity() -> None:
    pieces = [b"A", b"BC", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    seed = [0x44] * 32

    for bad_token in (-1, 99):
        out = seed.copy()
        count = [0x5252]
        cursor = [3]

        err = tokenizer_bpe_decode_single_token_checked_default(
            bad_token,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            8,
            out,
            cursor,
            count,
        )
        assert err == TOKENIZER_BPE_ERR_BAD_PARAM
        assert out == seed
        assert count[0] == 0x5252
        assert cursor[0] == 3

    out = seed.copy()
    count = [0x5353]
    cursor = [1]
    err = tokenizer_bpe_decode_single_token_checked_default(
        2,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        1,
        out,
        cursor,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == seed
    assert count[0] == 0x5353
    assert cursor[0] == 1


def test_overflow_and_null_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x7070]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_single_token_checked_default(
            0,
            None,
            len(blob),
            offsets,
            lens,
            1,
            8,
            out,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_default(
            0,
            blob,
            I64_MAX + 1,
            offsets,
            lens,
            1,
            8,
            out,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )

    cursor_over = [I64_MAX]
    assert (
        tokenizer_bpe_decode_single_token_checked_default(
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            cursor_over,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_explicit_default_parity() -> None:
    rng = random.Random(20260418_421)
    pieces = [b"a", b"bb", b"ccc", "Κα".encode("utf-8"), "世界".encode("utf-8"), "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(6000):
        token_id = rng.randint(-2, len(pieces) + 2)
        out_default = [0xD7] * 128
        out_explicit = out_default.copy()

        cursor0 = rng.randint(0, 120)
        cursor_default = [cursor0]
        cursor_explicit = [cursor0]

        count_default = [0x3B3B]
        count_explicit = [0x3B3B]

        max_piece_bytes = rng.randint(0, 16)

        err_default = tokenizer_bpe_decode_single_token_checked_default(
            token_id,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            max_piece_bytes,
            out_default,
            cursor_default,
            count_default,
        )

        derived_capacity = cursor0 + max_piece_bytes
        err_explicit = tokenizer_bpe_decode_single_token_checked(
            token_id,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            out_explicit,
            derived_capacity,
            cursor_explicit,
            count_explicit,
        )

        assert err_default == err_explicit
        assert out_default == out_explicit
        assert count_default[0] == count_explicit[0]
        assert cursor_default[0] == cursor_explicit[0]


if __name__ == "__main__":
    test_multilingual_fixture_explicit_parity()
    test_token_range_failures_and_small_default_capacity()
    test_overflow_and_null_contracts()
    test_randomized_explicit_default_parity()
    print("test_tokenizer_bpe_decode_single_token_checked_default: ok")
