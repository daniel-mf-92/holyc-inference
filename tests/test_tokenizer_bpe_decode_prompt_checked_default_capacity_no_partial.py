#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodePromptCheckedDefaultCapacityNoPartial."""

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


def tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial(
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

    staged_cursor = [io_token_cursor[0]]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(out_byte_capacity, 1)

    err = tokenizer_bpe_decode_prompt_checked(
        token_ids,
        token_count,
        staged_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        staged_bytes,
        out_byte_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > out_byte_capacity:
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
    out_bytes: list[int],
    out_byte_capacity: int,
    out_byte_count: list[int],
) -> int:
    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_cursor = [io_token_cursor[0]]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(out_byte_capacity, 1)

    err = tokenizer_bpe_decode_prompt_checked(
        token_ids,
        token_count,
        staged_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_count,
        staged_bytes,
        out_byte_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_count[0]):
        out_bytes[idx] = staged_bytes[idx]

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(
    token_ids: list[int],
    token_cursor: int,
    pieces: list[bytes],
    out_capacity: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0xA5] * 2048
    out_wrapped = out_ref.copy()
    count_ref = [0x5151]
    count_wrapped = [0x5151]
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
        out_ref,
        out_capacity,
        count_ref,
    )

    err_wrapped = tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial(
        token_ids,
        len(token_ids),
        cursor_wrapped,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_wrapped,
        out_capacity,
        count_wrapped,
    )

    assert err_wrapped == err_ref
    assert cursor_wrapped[0] == cursor_ref[0]
    assert count_wrapped[0] == count_ref[0]
    assert out_wrapped == out_ref


def test_multilingual_parity_vs_explicit_staged_core_composition() -> None:
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

    run_case(token_ids, 0, pieces, out_capacity=512)
    run_case(token_ids, 5, pieces, out_capacity=512)


def test_malformed_cursor_and_token_vectors_no_partial() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0xCC] * 64
    count = [0x8A8A]
    cursor = [4]

    err = tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        len(out),
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 4 and count[0] == 0x8A8A
    assert out == [0xCC] * 64

    cursor = [0]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial(
        [0, -1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        len(out),
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x8A8A
    assert out == [0xCC] * 64


def test_capacity_and_null_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x7272]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial(
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
        tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial(
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


def test_randomized_parity_vs_explicit_staged_core_composition() -> None:
    rng = random.Random(20260418_427)
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
    test_multilingual_parity_vs_explicit_staged_core_composition()
    test_malformed_cursor_and_token_vectors_no_partial()
    test_capacity_and_null_contracts()
    test_randomized_parity_vs_explicit_staged_core_composition()
    print("test_tokenizer_bpe_decode_prompt_checked_default_capacity_no_partial: ok")
