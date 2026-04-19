#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromMaxPieceNoPartialDefaultSpan helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial import (
    tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
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

    if token_count > I64_MAX or vocab_piece_bytes_len > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    span_token_count = token_count - cursor
    return tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
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

    if token_count > I64_MAX or vocab_piece_bytes_len > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
        token_ids,
        token_count,
        io_token_cursor,
        token_count - cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        out_byte_count,
    )


def test_source_contains_helper_signature_and_default_span_derivation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialDefaultSpan("
        in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialDefaultSpan(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodePromptChecked",
        1,
    )[0]
    assert "if (cursor > token_count)" in body
    assert "span_token_count = token_count - *io_token_cursor;" in body
    assert (
        "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial("
        in body
    )


def run_case(token_ids: list[int], cursor: int, pieces: list[bytes]) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_a = [0x5A] * 2048
    out_b = out_a.copy()
    count_a = [0x2121]
    count_b = [0x2121]
    cursor_a = [cursor]
    cursor_b = [cursor]

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
        token_ids,
        len(token_ids),
        cursor_a,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out_a,
        count_a,
    )
    err_b = explicit_default_span_composition(
        token_ids,
        len(token_ids),
        cursor_b,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out_b,
        count_b,
    )

    assert err_a == err_b
    assert cursor_a[0] == cursor_b[0]
    assert count_a[0] == count_b[0]
    assert out_a == out_b


def test_multilingual_and_adversarial_prompt_tail_parity() -> None:
    pieces = [
        b"",
        b"hello",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\\x00A\\n",
    ]

    token_ids = [1, 2, 3, 4, 5, 0, 1, 4, 2, 5]
    run_case(token_ids, 0, pieces)
    run_case(token_ids, 4, pieces)
    run_case(token_ids, len(token_ids), pieces)


def test_no_partial_failures_preserve_outputs() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x4C] * 128
    count = [0x6D6D]

    bad_cursor = [7]
    out_seed = out.copy()
    count_seed = count[0]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
        [0, 1, 2],
        3,
        bad_cursor,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert bad_cursor[0] == 7
    assert count[0] == count_seed
    assert out == out_seed

    out = [0x2E] * 16
    count = [0x1234]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
        [0],
        I64_MAX + 1,
        [0],
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert count[0] == 0x1234
    assert out == [0x2E] * 16


def test_randomized_default_span_parity() -> None:
    random.seed(531)

    base = [b"A", b"BB", b"CCC", b"DDDD", "🙂".encode("utf-8"), b" ", b"\\n"]
    blob, offsets, lens = build_vocab_tables(base)

    for _ in range(200):
        token_count = random.randint(0, 32)
        token_ids = [random.randrange(len(base)) for _ in range(token_count)]
        cursor = random.randint(0, token_count)

        out_a = [0x7F] * 2048
        out_b = out_a.copy()
        count_a = [random.randint(0, 17)]
        count_b = [count_a[0]]
        cursor_a = [cursor]
        cursor_b = [cursor]

        err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
            token_ids,
            token_count,
            cursor_a,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_a,
            count_a,
        )
        err_b = explicit_default_span_composition(
            token_ids,
            token_count,
            cursor_b,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_b,
            count_b,
        )

        assert err_a == err_b == TOKENIZER_BPE_OK
        assert cursor_a[0] == cursor_b[0] == token_count
        assert count_a[0] == count_b[0]
        assert out_a == out_b


def test_error_state_parity_matrix_against_explicit_composition() -> None:
    random.seed(546)

    pieces = [b"A", b"BC", "世界".encode("utf-8"), b"\x00", b"tail"]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(220):
        token_count = random.randint(0, 24)
        token_ids = [random.randrange(len(pieces)) for _ in range(token_count)]
        cursor = random.randint(0, token_count + 4)

        out_a = [0x4A] * 256
        out_b = out_a.copy()
        count_a = [random.randint(0, 99)]
        count_b = [count_a[0]]
        cursor_a = [cursor]
        cursor_b = [cursor]

        err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
            token_ids,
            token_count,
            cursor_a,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_a,
            count_a,
        )
        err_b = explicit_default_span_composition(
            token_ids,
            token_count,
            cursor_b,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_b,
            count_b,
        )

        assert err_a == err_b
        assert cursor_a[0] == cursor_b[0]
        assert count_a[0] == count_b[0]
        assert out_a == out_b

    overflow_count = [0x55]
    overflow_out = [0x22] * 32
    overflow_cursor = [0]
    overflow_err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_default_span(
        [0],
        I64_MAX + 1,
        overflow_cursor,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        overflow_out,
        overflow_count,
    )
    assert overflow_err == TOKENIZER_BPE_ERR_OVERFLOW
    assert overflow_cursor[0] == 0
    assert overflow_count[0] == 0x55
    assert overflow_out == [0x22] * 32


if __name__ == "__main__":
    test_source_contains_helper_signature_and_default_span_derivation()
    test_multilingual_and_adversarial_prompt_tail_parity()
    test_no_partial_failures_preserve_outputs()
    test_randomized_default_span_parity()
    test_error_state_parity_matrix_against_explicit_composition()
    print("ok")
