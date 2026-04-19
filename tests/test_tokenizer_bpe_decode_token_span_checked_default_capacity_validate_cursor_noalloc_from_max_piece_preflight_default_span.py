#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromMaxPiecePreflightDefaultSpan helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight import (
    explicit_inline_preflight,
    tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
    out_byte_count: list[int] | None,
    out_max_piece_bytes: list[int] | None,
    out_derived_out_byte_capacity: list[int] | None,
) -> int:
    if (
        token_ids is None
        or io_token_cursor is None
        or vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or out_byte_count is None
        or out_max_piece_bytes is None
        or out_derived_out_byte_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_bytes_len > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    span_token_count = token_count - cursor
    return tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
        token_ids,
        token_count,
        io_token_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
        out_max_piece_bytes,
        out_derived_out_byte_capacity,
    )


def explicit_default_span_composition(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
    out_byte_count: list[int] | None,
    out_max_piece_bytes: list[int] | None,
    out_derived_out_byte_capacity: list[int] | None,
) -> int:
    if (
        token_ids is None
        or io_token_cursor is None
        or vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or out_byte_count is None
        or out_max_piece_bytes is None
        or out_derived_out_byte_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_bytes_len > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    span_token_count = token_count - cursor
    return explicit_inline_preflight(
        token_ids,
        token_count,
        io_token_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
        out_max_piece_bytes,
        out_derived_out_byte_capacity,
    )


def test_source_contains_helper_signature_and_default_span_derivation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflightDefaultSpan("
        in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflightDefaultSpan(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiecePreflight",
        1,
    )[0]
    assert "if (cursor > token_count)" in body
    assert "span_token_count = token_count - cursor;" in body
    assert (
        "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight("
        in body
    )


def test_null_and_overflow_failures_preserve_outputs() -> None:
    out_max = [777]
    out_cap = [888]

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span(
        None,
        0,
        [0],
        [0],
        1,
        [0],
        [0],
        1,
        [0],
        [0],
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert out_max[0] == 777
    assert out_cap[0] == 888

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span(
        [0],
        I64_MAX + 1,
        [0],
        [0],
        1,
        [0],
        [0],
        1,
        [0],
        [0],
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_max[0] == 777
    assert out_cap[0] == 888


def test_multilingual_and_adversarial_prompt_tail_parity() -> None:
    pieces = [
        b"",
        b"hello",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    max_a = [123]
    cap_a = [456]
    max_b = [789]
    cap_b = [987]

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span(
        [1, 2, 3, 4, 5],
        5,
        [2],
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 256,
        [0],
        max_a,
        cap_a,
    )
    err_b = explicit_default_span_composition(
        [1, 2, 3, 4, 5],
        5,
        [2],
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 256,
        [0],
        max_b,
        cap_b,
    )
    assert err_a == err_b == TOKENIZER_BPE_OK
    assert max_a[0] == max_b[0] == max(lens)
    assert cap_a[0] == cap_b[0] == (5 - 2) * max(lens)

    seeded_max = [0xA1A1]
    seeded_cap = [0xB2B2]
    bad_cursor = [9]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span(
        [0, 1, 2],
        3,
        bad_cursor,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 32,
        [0],
        seeded_max,
        seeded_cap,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert bad_cursor[0] == 9
    assert seeded_max[0] == 0xA1A1
    assert seeded_cap[0] == 0xB2B2


def test_randomized_equivalence_against_explicit_default_span_composition() -> None:
    rng = random.Random(20260419_527)

    for _ in range(6000):
        token_count = rng.randint(0, 128)
        if rng.random() < 0.05:
            token_count = I64_MAX + rng.randint(1, 32)

        cursor = rng.randint(0, 128)
        if token_count <= I64_MAX and rng.random() < 0.70:
            cursor = rng.randint(0, token_count)

        vocab_piece_count = rng.randint(0, 96)
        lens = [rng.randint(0, 160) for _ in range(vocab_piece_count)]
        if vocab_piece_count and rng.random() < 0.08:
            lens[rng.randrange(vocab_piece_count)] = I64_MAX + rng.randint(1, 16)

        blob = [0] * max(1, sum(length if length > 0 and length <= I64_MAX else 0 for length in lens))
        offsets = []
        running = 0
        for length in lens:
            offsets.append(running)
            if 0 <= length <= I64_MAX:
                running += length

        out_max_a = [0x1111]
        out_cap_a = [0x2222]
        out_max_b = [0x3333]
        out_cap_b = [0x4444]

        cursor_a = [cursor]
        cursor_b = [cursor]

        err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span(
            [0] * (token_count if token_count <= I64_MAX else 0),
            token_count,
            cursor_a,
            blob,
            len(blob),
            offsets,
            lens,
            vocab_piece_count,
            [0] * 64,
            [0],
            out_max_a,
            out_cap_a,
        )

        err_b = explicit_default_span_composition(
            [0] * (token_count if token_count <= I64_MAX else 0),
            token_count,
            cursor_b,
            blob,
            len(blob),
            offsets,
            lens,
            vocab_piece_count,
            [0] * 64,
            [0],
            out_max_b,
            out_cap_b,
        )

        assert err_a == err_b
        assert cursor_a[0] == cursor_b[0]
        if err_a == TOKENIZER_BPE_OK:
            assert out_max_a[0] == out_max_b[0]
            assert out_cap_a[0] == out_cap_b[0]


def run() -> None:
    test_source_contains_helper_signature_and_default_span_derivation()
    test_null_and_overflow_failures_preserve_outputs()
    test_multilingual_and_adversarial_prompt_tail_parity()
    test_randomized_equivalence_against_explicit_default_span_composition()
    print(
        "tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight_default_span=ok"
    )


if __name__ == "__main__":
    run()
