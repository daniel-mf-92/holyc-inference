#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromMaxPieceNoPartialPreflight helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight import (
    explicit_inline_preflight,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
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

    if (
        token_count > I64_MAX
        or span_token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if span_token_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    lens = vocab_piece_lens
    if vocab_piece_count == 0:
        max_piece = 0
    else:
        max_piece = 0
        for i in range(vocab_piece_count):
            value = lens[i]
            if value > I64_MAX:
                return TOKENIZER_BPE_ERR_OVERFLOW
            if value > max_piece:
                max_piece = value

    if span_token_count and max_piece > I64_MAX // span_token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_max_piece_bytes[0] = max_piece
    out_derived_out_byte_capacity[0] = span_token_count * max_piece
    return TOKENIZER_BPE_OK


def test_source_contains_from_max_piece_nopartial_preflight_and_wrapper_use() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight(" in source
    )

    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial",
        1,
    )[0]
    assert "if (cursor > token_count)" in body
    assert "if (span_token_count > token_count - cursor)" in body
    assert "TokenizerBPEComputeMaxPieceBytesChecked(vocab_piece_lens," in body
    assert "if (span_token_count &&" in body
    assert "max_piece_bytes > 0x7FFFFFFFFFFFFFFF / span_token_count" in body

    wrapper_body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial(",
        1,
    )[1].split("I32 TokenizerBPEDecodePromptChecked", 1)[0]
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight(" in wrapper_body


def test_null_overflow_and_cursor_span_error_surfaces() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    assert (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
            None,
            0,
            [0],
            0,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            [0],
            [0],
            [0],
            [0],
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    out_max = [0xAA]
    out_cap = [0xBB]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
        [0],
        I64_MAX + 1,
        [0],
        0,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0],
        [0],
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_max[0] == 0xAA
    assert out_cap[0] == 0xBB

    out_max = [0xCC]
    out_cap = [0xDD]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
        [0, 1, 2],
        3,
        [4],
        0,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0],
        [0],
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_max[0] == 0xCC
    assert out_cap[0] == 0xDD

    out_max = [0x11]
    out_cap = [0x22]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
        [0, 1, 2],
        3,
        [2],
        2,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0],
        [0],
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_max[0] == 0x11
    assert out_cap[0] == 0x22


def test_multilingual_and_randomized_inline_parity() -> None:
    pieces = [
        b"",
        b"hello",
        " Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    err_new = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
        [1, 2, 3, 4],
        4,
        [1],
        3,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 64,
        [0],
        [123],
        [456],
    )
    err_ref = explicit_inline_preflight(
        [1, 2, 3, 4],
        4,
        [1],
        3,
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 64,
        [0],
        [789],
        [987],
    )
    assert err_new == err_ref == TOKENIZER_BPE_OK

    rng = random.Random(20260419_514)
    dense_out = [0] * 2048
    for _ in range(5000):
        token_count = rng.randint(0, 64)
        span_token_count = rng.randint(0, token_count + 8)
        cursor = rng.randint(0, token_count + 5)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]

        max_new = [0x1234]
        cap_new = [0x5678]
        max_ref = [0x1111]
        cap_ref = [0x2222]

        err_new = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight(
            token_ids,
            token_count,
            [cursor],
            span_token_count,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            dense_out,
            [0],
            max_new,
            cap_new,
        )
        err_ref = explicit_inline_preflight(
            token_ids,
            token_count,
            [cursor],
            span_token_count,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            dense_out,
            [0],
            max_ref,
            cap_ref,
        )

        assert err_new == err_ref
        if err_new == TOKENIZER_BPE_OK:
            assert max_new[0] == max_ref[0]
            assert cap_new[0] == cap_ref[0]


def run() -> None:
    test_source_contains_from_max_piece_nopartial_preflight_and_wrapper_use()
    test_null_overflow_and_cursor_span_error_surfaces()
    test_multilingual_and_randomized_inline_parity()
    print(
        "tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial_preflight=ok"
    )


if __name__ == "__main__":
    run()
