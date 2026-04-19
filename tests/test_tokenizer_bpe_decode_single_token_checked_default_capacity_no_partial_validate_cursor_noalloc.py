#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityNoPartialValidateCursorNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_single_token_checked import build_vocab_tables
from test_tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor import (
    tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc(
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
    if piece_len > out_byte_capacity - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_out_count = piece_len
    if staged_out_count > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if staged_out_count > I64_MAX - cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    staged_cursor = cursor + staged_out_count
    if staged_cursor < cursor or staged_cursor > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    write_cursor = cursor
    for idx in range(staged_out_count):
        out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + idx]

        write_cursor += 1
        if write_cursor <= cursor:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if write_cursor > staged_cursor:
            return TOKENIZER_BPE_ERR_OVERFLOW

    if write_cursor != staged_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_out_count
    io_out_cursor[0] = staged_cursor
    return TOKENIZER_BPE_OK


def run_case(
    token_id: int,
    pieces: list[bytes],
    cursor0: int,
    count0: int,
    out_byte_capacity: int,
    seed_size: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_alloc = [0xA5] * seed_size
    out_noalloc = out_alloc.copy()
    count_alloc = [count0]
    count_noalloc = [count0]
    cursor_alloc = [cursor0]
    cursor_noalloc = [cursor0]

    err_alloc = tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_alloc,
        out_byte_capacity,
        cursor_alloc,
        count_alloc,
    )

    err_noalloc = tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_noalloc,
        out_byte_capacity,
        cursor_noalloc,
        count_noalloc,
    )

    assert err_noalloc == err_alloc
    assert out_noalloc == out_alloc
    assert count_noalloc[0] == count_alloc[0]
    assert cursor_noalloc[0] == cursor_alloc[0]


def test_source_contains_noalloc_wrapper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityNoPartialValidateCursorNoAlloc" in source
    assert "MAlloc(" not in source.split(
        "TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityNoPartialValidateCursorNoAlloc", 1
    )[1].split("I32 TokenizerBPEDecodeSingleTokenCheckedDefaultNoPartialValidateCursor", 1)[0]


def test_multilingual_parity_against_allocating_wrapper() -> None:
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

    run_case(token_id=0, pieces=pieces, cursor0=0, count0=0x1111, out_byte_capacity=64, seed_size=256)
    run_case(token_id=8, pieces=pieces, cursor0=9, count0=0x2222, out_byte_capacity=96, seed_size=256)
    run_case(token_id=9, pieces=pieces, cursor0=31, count0=0x3333, out_byte_capacity=192, seed_size=512)


def test_adversarial_cursor_capacity_vectors() -> None:
    pieces = [b"A", b"BC", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)
    seed = [0x4E] * 96

    for bad_token in (-1, 99):
        out = seed.copy()
        count = [0xBEEF]
        cursor = [4]

        err = tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc(
            bad_token,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out,
            len(out),
            cursor,
            count,
        )
        assert err == TOKENIZER_BPE_ERR_BAD_PARAM
        assert out == seed
        assert count[0] == 0xBEEF
        assert cursor[0] == 4

    out = seed.copy()
    count = [0xCAFE]
    cursor = [97]
    err = tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc(
        0,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out,
        96,
        cursor,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == seed
    assert count[0] == 0xCAFE
    assert cursor[0] == 97


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x11] * 16
    count = [0x7070]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc(
            0,
            None,
            len(blob),
            offsets,
            lens,
            1,
            out,
            len(out),
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc(
            0,
            blob,
            I64_MAX + 1,
            offsets,
            lens,
            1,
            out,
            len(out),
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity() -> None:
    rng = random.Random(20260419_459)
    pieces = [b"a", b"bb", b"ccc", "Κα".encode("utf-8"), "世界".encode("utf-8"), "🙂".encode("utf-8")]

    for _ in range(7000):
        token_id = rng.randint(-2, len(pieces) + 2)
        cursor0 = rng.randint(0, 196)
        out_cap = rng.randint(0, 192)

        run_case(
            token_id=token_id,
            pieces=pieces,
            cursor0=cursor0,
            count0=0x3A3A,
            out_byte_capacity=out_cap,
            seed_size=192,
        )


if __name__ == "__main__":
    test_source_contains_noalloc_wrapper()
    test_multilingual_parity_against_allocating_wrapper()
    test_adversarial_cursor_capacity_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity()
    print("test_tokenizer_bpe_decode_single_token_checked_default_capacity_no_partial_validate_cursor_noalloc: ok")
