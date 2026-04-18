#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodePromptCheckedDefaultCapacityNoPartialValidateCursor."""

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


def tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
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
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

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
    staged_bytes = [0] * max(derived_out_capacity, 1)

    err = tokenizer_bpe_decode_token_span_checked(
        token_ids,
        token_count,
        staged_cursor,
        decode_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        staged_bytes,
        derived_out_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_cursor[0] != token_count or staged_count[0] > derived_out_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_count[0]):
        out_bytes[idx] = staged_bytes[idx]

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def explicit_staged_core_composition(
    token_ids: list[int],
    token_count: int,
    io_token_cursor: list[int],
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    max_piece_bytes: int,
    out_bytes: list[int],
    out_byte_count: list[int],
) -> int:
    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or max_piece_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

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
    staged_bytes = [0] * max(derived_out_capacity, 1)

    err = tokenizer_bpe_decode_token_span_checked(
        token_ids,
        token_count,
        staged_cursor,
        decode_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        staged_bytes,
        derived_out_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_cursor[0] != token_count or staged_count[0] > derived_out_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_count[0]):
        out_bytes[idx] = staged_bytes[idx]

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes], max_piece_bytes: int) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x51] * 1024
    out_wrapped = out_ref.copy()
    count_ref = [0xACAC]
    count_wrapped = [0xACAC]
    cursor_ref = [token_cursor]
    cursor_wrapped = [token_cursor]

    err_ref = explicit_staged_core_composition(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_ref,
        count_ref,
    )
    err_wrapped = tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor(
        token_ids,
        len(token_ids),
        cursor_wrapped,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        max_piece_bytes,
        out_wrapped,
        count_wrapped,
    )

    assert err_wrapped == err_ref
    assert cursor_wrapped[0] == cursor_ref[0]
    assert count_wrapped[0] == count_ref[0]
    assert out_wrapped == out_ref


def test_multilingual_success_parity_vs_explicit_staged_core_composition() -> None:
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


def test_cursor_overflow_and_mul_overflow_reject_no_partial() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x37] * 64
    count = [0x9393]

    cursor = [4]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        8,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 4 and count[0] == 0x9393
    assert out == [0x37] * 64

    cursor = [2]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        I64_MAX,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert cursor[0] == 2 and count[0] == 0x9393
    assert out == [0x37] * 64


def test_null_and_domain_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x99] * 16
    count = [0x7A7A]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_against_explicit_staged_core_composition() -> None:
    rng = random.Random(20260418_429)
    pieces = [
        b"a",
        b"bc",
        b"def",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
    ]

    for _ in range(3000):
        token_count = rng.randint(0, 64)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor = rng.randint(0, token_count)
        max_piece_bytes = rng.randint(0, 32)
        run_case(token_ids, cursor, pieces, max_piece_bytes)


if __name__ == "__main__":
    test_multilingual_success_parity_vs_explicit_staged_core_composition()
    test_cursor_overflow_and_mul_overflow_reject_no_partial()
    test_null_and_domain_contracts()
    test_randomized_parity_against_explicit_staged_core_composition()
    print("test_tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial_validate_cursor: ok")
