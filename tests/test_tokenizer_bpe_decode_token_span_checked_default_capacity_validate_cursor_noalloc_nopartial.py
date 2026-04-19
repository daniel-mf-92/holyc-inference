#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityValidateCursorNoAllocNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc import (
    tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
)

TOKENIZER_BPE_OK = 0


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
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

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_count = span_token_count
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_cursor = [cursor]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(out_byte_capacity, 1)

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc(
        token_ids,
        token_count,
        staged_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        staged_bytes,
        out_byte_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_count[0]):
        out_bytes[i] = staged_bytes[i]

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def explicit_staged_composition(
    token_ids: list[int],
    token_count: int,
    io_token_cursor: list[int],
    span_token_count: int,
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
        or span_token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_count = span_token_count
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_cursor = [cursor]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(out_byte_capacity, 1)

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc(
        token_ids,
        token_count,
        staged_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        staged_bytes,
        out_byte_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_count[0]):
        out_bytes[i] = staged_bytes[i]

    out_byte_count[0] = staged_count[0]
    io_token_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(
    token_ids: list[int],
    token_cursor: int,
    span_count: int,
    pieces: list[bytes],
    out_capacity: int,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0xC7] * 2048
    out_wrapped = out_ref.copy()
    count_ref = [0x6E6E]
    count_wrapped = [0x6E6E]
    cursor_ref = [token_cursor]
    cursor_wrapped = [token_cursor]

    err_ref = explicit_staged_composition(
        token_ids,
        len(token_ids),
        cursor_ref,
        span_count,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_ref,
        out_capacity,
        count_ref,
    )

    err_wrapped = (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
            token_ids,
            len(token_ids),
            cursor_wrapped,
            span_count,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_wrapped,
            out_capacity,
            count_wrapped,
        )
    )

    assert err_wrapped == err_ref
    assert cursor_wrapped[0] == cursor_ref[0]
    assert count_wrapped[0] == count_ref[0]
    assert out_wrapped == out_ref


def test_source_contains_expected_wrapper_shape() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocNoPartial" in source
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoPartialNoAlloc(" in source


def test_multilingual_success_parity_vs_explicit_staged_composition() -> None:
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

    run_case(token_ids, 0, len(token_ids), pieces, out_capacity=512)
    run_case(token_ids, 4, 7, pieces, out_capacity=512)


def test_no_partial_on_cursor_window_and_capacity_rejects() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x2A] * 64
    count = [0x5D5D]

    cursor = [4]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
        [0, 1, 2],
        3,
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
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 4 and count[0] == 0x5D5D
    assert out == [0x2A] * 64

    cursor = [2]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
        [0, 1, 2],
        3,
        cursor,
        2,
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
    assert cursor[0] == 2 and count[0] == 0x5D5D
    assert out == [0x2A] * 64

    cursor = [0]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
        [2, 2, 2],
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
    assert cursor[0] == 0 and count[0] == 0x5D5D
    assert out == [0x2A] * 64


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x11] * 8
    count = [0x4242]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
            None,
            0,
            cursor,
            0,
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
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial(
            [0],
            I64_MAX + 1,
            cursor,
            0,
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


def test_randomized_parity_against_explicit_staged_composition() -> None:
    rng = random.Random(0xC0D495)

    for _ in range(220):
        piece_count = rng.randint(1, 20)
        pieces: list[bytes] = []
        for _ in range(piece_count):
            if rng.random() < 0.28:
                text = rng.choice(["🙂", "λ", "Ж", "界", "ñ"]) * rng.randint(1, 2)
                pieces.append(text.encode("utf-8"))
            else:
                ln = rng.randint(0, 6)
                pieces.append(bytes(rng.randrange(32, 127) for _ in range(ln)))

        token_len = rng.randint(0, 36)
        token_ids = [rng.randrange(piece_count) for _ in range(token_len)]
        cursor = rng.randint(0, token_len)
        max_span = token_len - cursor
        span = rng.randint(0, max_span)

        worst_piece = max((len(piece) for piece in pieces), default=0)
        full_cap = token_len * max(worst_piece, 1)
        out_capacity = rng.randint(0, max(1, full_cap + 6))

        run_case(token_ids, cursor, span, pieces, out_capacity)


if __name__ == "__main__":
    test_source_contains_expected_wrapper_shape()
    test_multilingual_success_parity_vs_explicit_staged_composition()
    test_no_partial_on_cursor_window_and_capacity_rejects()
    test_null_and_overflow_contracts()
    test_randomized_parity_against_explicit_staged_composition()
    print("ok")
