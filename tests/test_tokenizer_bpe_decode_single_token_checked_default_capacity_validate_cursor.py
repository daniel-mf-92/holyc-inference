#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityValidateCursor."""

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


def tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
    token_id: int,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
    out_byte_capacity: int,
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
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_out_cursor[0]
    if cursor > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_single_token_checked(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        out_bytes,
        out_byte_capacity,
        io_out_cursor,
        out_byte_count,
    )


def explicit_composition(
    token_id: int,
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    out_bytes: list[int],
    out_byte_capacity: int,
    io_out_cursor: list[int],
    out_byte_count: list[int],
) -> int:
    cursor = io_out_cursor[0]
    if cursor > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_single_token_checked(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        out_bytes,
        out_byte_capacity,
        io_out_cursor,
        out_byte_count,
    )


def run_case(
    token_id: int,
    pieces: list[bytes],
    out_byte_capacity: int,
    cursor0: int,
    count0: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_wrapper = [0xA5] * max(out_byte_capacity + 16, 64)
    out_explicit = out_wrapper.copy()
    cursor_wrapper = [cursor0]
    cursor_explicit = [cursor0]
    count_wrapper = [count0]
    count_explicit = [count0]

    err_wrapper = tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_wrapper,
        out_byte_capacity,
        cursor_wrapper,
        count_wrapper,
    )

    err_explicit = explicit_composition(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_explicit,
        out_byte_capacity,
        cursor_explicit,
        count_explicit,
    )

    assert err_wrapper == err_explicit
    assert out_wrapper == out_explicit
    assert cursor_wrapper[0] == cursor_explicit[0]
    assert count_wrapper[0] == count_explicit[0]


def test_multilingual_success_and_cursor_guard() -> None:
    pieces = [
        b"hello",
        b" ",
        b"world",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
    ]

    run_case(token_id=4, pieces=pieces, out_byte_capacity=64, cursor0=9, count0=0x1111)
    run_case(token_id=5, pieces=pieces, out_byte_capacity=64, cursor0=12, count0=0x2222)

    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x66] * 32
    cursor = [33]
    count = [0xAAAA]
    err = tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
        0,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out,
        32,
        cursor,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == [0x66] * 32
    assert cursor[0] == 33
    assert count[0] == 0xAAAA


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x22] * 8
    cursor = [0]
    count = [0x9090]

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
            0,
            None,
            len(blob),
            offsets,
            lens,
            1,
            out,
            8,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
            0,
            blob,
            I64_MAX + 1,
            offsets,
            lens,
            1,
            out,
            8,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            I64_MAX + 1,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity() -> None:
    rng = random.Random(20260419_456)
    pieces = [b"a", b"bb", b"ccc", "Κα".encode("utf-8"), "世界".encode("utf-8"), "🙂".encode("utf-8")]

    for _ in range(7000):
        out_cap = rng.randint(0, 160)
        cursor0 = rng.randint(0, 170)
        token_id = rng.randint(-3, len(pieces) + 3)
        count0 = rng.randint(0, 0xFFFF)
        run_case(token_id, pieces, out_cap, cursor0, count0)


if __name__ == "__main__":
    test_multilingual_success_and_cursor_guard()
    test_null_and_overflow_contracts()
    test_randomized_parity()
    print("ok")
