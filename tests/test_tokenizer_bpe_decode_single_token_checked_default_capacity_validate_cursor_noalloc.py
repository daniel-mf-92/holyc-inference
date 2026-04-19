#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityValidateCursorNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_single_token_checked import build_vocab_tables
from test_tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor import (
    tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
)


def tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
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

    return tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
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
    seed_size: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_old = [0xA5] * seed_size
    out_new = out_old.copy()
    cursor_old = [cursor0]
    cursor_new = [cursor0]
    count_old = [count0]
    count_new = [count0]

    err_old = tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_old,
        out_byte_capacity,
        cursor_old,
        count_old,
    )

    err_new = tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_new,
        out_byte_capacity,
        cursor_new,
        count_new,
    )

    assert err_new == err_old
    assert out_new == out_old
    assert cursor_new[0] == cursor_old[0]
    assert count_new[0] == count_old[0]


def test_source_contains_wrapper_and_no_malloc() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityValidateCursorNoAlloc" in source
    section = source.split(
        "I32 TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityValidateCursorNoAlloc", 1
    )[1].split("I32 TokenizerBPEDecodeSingleTokenCheckedDefaultCapacityValidateCursorNoPartial", 1)[0]
    assert "MAlloc(" not in section


def test_multilingual_parity() -> None:
    pieces = [
        b"hello",
        b" ",
        b"world",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]

    run_case(token_id=0, pieces=pieces, out_byte_capacity=96, cursor0=0, count0=0x1111, seed_size=256)
    run_case(token_id=4, pieces=pieces, out_byte_capacity=96, cursor0=9, count0=0x2222, seed_size=256)
    run_case(token_id=5, pieces=pieces, out_byte_capacity=128, cursor0=17, count0=0x3333, seed_size=256)


def test_adversarial_vectors() -> None:
    pieces = [b"A", b"BC", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)
    seed = [0x61] * 80

    for bad_token in (-1, 99):
        out = seed.copy()
        cursor = [5]
        count = [0xBEEF]
        err = tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
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
        assert cursor[0] == 5
        assert count[0] == 0xBEEF

    out = seed.copy()
    cursor = [81]
    count = [0xCAFE]
    err = tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
        0,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out,
        80,
        cursor,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == seed
    assert cursor[0] == 81
    assert count[0] == 0xCAFE


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x44] * 16
    cursor = [0]
    count = [0x9090]

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
            0,
            None,
            len(blob),
            offsets,
            lens,
            1,
            out,
            16,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
            0,
            blob,
            I64_MAX + 1,
            offsets,
            lens,
            1,
            out,
            16,
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc(
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
    rng = random.Random(20260419_476)
    pieces = [b"a", b"bb", b"ccc", "Κα".encode("utf-8"), "世界".encode("utf-8"), "🙂".encode("utf-8")]

    for _ in range(8000):
        token_id = rng.randint(-2, len(pieces) + 2)
        out_cap = rng.randint(0, 192)
        cursor0 = rng.randint(0, 196)

        run_case(
            token_id=token_id,
            pieces=pieces,
            out_byte_capacity=out_cap,
            cursor0=cursor0,
            count0=0x5A5A,
            seed_size=192,
        )


if __name__ == "__main__":
    test_source_contains_wrapper_and_no_malloc()
    test_multilingual_parity()
    test_adversarial_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity()
    print("test_tokenizer_bpe_decode_single_token_checked_default_capacity_validate_cursor_noalloc: ok")
