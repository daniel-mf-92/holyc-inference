#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeSingleTokenCheckedDefaultNoPartialValidateCursor."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_single_token_checked import build_vocab_tables
from test_tokenizer_bpe_decode_single_token_checked_default_no_partial import (
    tokenizer_bpe_decode_single_token_checked_default_no_partial,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
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

    cursor = io_out_cursor[0]
    if cursor > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if max_piece_bytes > I64_MAX - cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_byte_capacity = cursor + max_piece_bytes
    if cursor > derived_out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_cursor = [cursor]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(derived_out_byte_capacity, 1)

    err = tokenizer_bpe_decode_single_token_checked_default_no_partial(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        max_piece_bytes,
        staged_bytes,
        staged_cursor,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if (
        staged_count[0] > derived_out_byte_capacity
        or staged_cursor[0] < cursor
        or staged_cursor[0] > derived_out_byte_capacity
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    committed = staged_cursor[0] - cursor
    if committed != staged_count[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(committed):
        out_bytes[cursor + idx] = staged_bytes[cursor + idx]

    out_byte_count[0] = staged_count[0]
    io_out_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def explicit_staged_no_partial_composition(
    token_id: int,
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    max_piece_bytes: int,
    out_bytes: list[int],
    io_out_cursor: list[int],
    out_byte_count: list[int],
) -> int:
    if (
        vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_out_cursor[0]
    if cursor > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if max_piece_bytes > I64_MAX - cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_byte_capacity = cursor + max_piece_bytes
    if cursor > derived_out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_cursor = [cursor]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(derived_out_byte_capacity, 1)

    err = tokenizer_bpe_decode_single_token_checked_default_no_partial(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        max_piece_bytes,
        staged_bytes,
        staged_cursor,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if (
        staged_count[0] > derived_out_byte_capacity
        or staged_cursor[0] < cursor
        or staged_cursor[0] > derived_out_byte_capacity
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    committed = staged_cursor[0] - cursor
    if committed != staged_count[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(committed):
        out_bytes[cursor + idx] = staged_bytes[cursor + idx]

    out_byte_count[0] = staged_count[0]
    io_out_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(
    token_id: int,
    pieces: list[bytes],
    cursor0: int,
    count0: int,
    max_piece_bytes: int,
    seed_size: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_wrapper = [0xA7] * seed_size
    out_ref = out_wrapper.copy()
    count_wrapper = [count0]
    count_ref = [count0]
    cursor_wrapper = [cursor0]
    cursor_ref = [cursor0]

    err_wrapper = tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_wrapper,
        cursor_wrapper,
        count_wrapper,
    )

    err_ref = explicit_staged_no_partial_composition(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_ref,
        cursor_ref,
        count_ref,
    )

    assert err_wrapper == err_ref
    assert out_wrapper == out_ref
    assert count_wrapper[0] == count_ref[0]
    assert cursor_wrapper[0] == cursor_ref[0]


def test_multilingual_parity_vs_explicit_staged_no_partial() -> None:
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

    run_case(token_id=0, pieces=pieces, cursor0=0, count0=0x1111, max_piece_bytes=16, seed_size=512)
    run_case(token_id=8, pieces=pieces, cursor0=9, count0=0x2222, max_piece_bytes=24, seed_size=512)
    run_case(token_id=9, pieces=pieces, cursor0=31, count0=0x3333, max_piece_bytes=16, seed_size=1024)


def test_adversarial_cursor_and_token_vectors() -> None:
    pieces = [b"A", b"BC", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)
    seed = [0x4E] * 96

    for bad_token in (-1, 99):
        out = seed.copy()
        count = [0xBEEF]
        cursor = [4]

        err = tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
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
        assert count[0] == 0xBEEF
        assert cursor[0] == 4

    out = seed.copy()
    count = [0xCAFE]
    cursor = [I64_MAX]
    err = tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
        0,
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
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out == seed
    assert count[0] == 0xCAFE
    assert cursor[0] == I64_MAX


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x11] * 16
    count = [0x7070]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
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
        tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
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


def test_randomized_parity() -> None:
    rng = random.Random(20260419_440)
    pieces = [b"a", b"bb", b"ccc", "Κα".encode("utf-8"), "世界".encode("utf-8"), "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(7000):
        token_id = rng.randint(-2, len(pieces) + 2)
        out_wrapper = [0xD5] * 192
        out_ref = out_wrapper.copy()

        cursor0 = rng.randint(0, 180)
        cursor_wrapper = [cursor0]
        cursor_ref = [cursor0]

        count_wrapper = [0x3A3A]
        count_ref = [0x3A3A]

        max_piece_bytes = rng.randint(0, 24)

        err_wrapper = tokenizer_bpe_decode_single_token_checked_default_no_partial_validate_cursor(
            token_id,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            max_piece_bytes,
            out_wrapper,
            cursor_wrapper,
            count_wrapper,
        )

        err_ref = explicit_staged_no_partial_composition(
            token_id,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            max_piece_bytes,
            out_ref,
            cursor_ref,
            count_ref,
        )

        assert err_wrapper == err_ref
        assert out_wrapper == out_ref
        assert count_wrapper[0] == count_ref[0]
        assert cursor_wrapper[0] == cursor_ref[0]


if __name__ == "__main__":
    test_multilingual_parity_vs_explicit_staged_no_partial()
    test_adversarial_cursor_and_token_vectors()
    test_null_and_overflow_contracts()
    test_randomized_parity()
    print("ok")
