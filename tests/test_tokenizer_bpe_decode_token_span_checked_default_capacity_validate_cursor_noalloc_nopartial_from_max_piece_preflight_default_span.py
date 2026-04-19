#!/usr/bin/env python3
"""Parity harness for ...NoAllocNoPartialFromMaxPiecePreflightDefaultSpan helper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight import (
    explicit_inline_preflight,
    tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
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
    return tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiecePreflightDefaultSpan("
        in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiecePreflightDefaultSpan(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight",
        1,
    )[0]
    assert "if (cursor > token_count)" in body
    assert "span_token_count = token_count - cursor;" in body
    assert (
        "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight("
        in body
    )


def test_null_and_overflow_failures_preserve_outputs() -> None:
    out_max = [777]
    out_cap = [888]

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
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

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
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
        b"\\x00A\\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    token_ids = [1, 2, 3, 4, 5, 0]
    token_count = len(token_ids)
    cursor = [2]

    out_max_a = [0xAB]
    out_cap_a = [0xCD]
    out_max_b = [0x11]
    out_cap_b = [0x22]

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
        token_ids,
        token_count,
        cursor.copy(),
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 128,
        [0],
        out_max_a,
        out_cap_a,
    )
    err_b = explicit_default_span_composition(
        token_ids,
        token_count,
        cursor.copy(),
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0] * 128,
        [0],
        out_max_b,
        out_cap_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert out_max_a[0] == out_max_b[0] == max(lens)
    assert out_cap_a[0] == out_cap_b[0] == (token_count - cursor[0]) * max(lens)


def test_cursor_out_of_range_and_capacity_overflow_parity() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out_max_a = [0xAA]
    out_cap_a = [0xBB]
    out_max_b = [0xCC]
    out_cap_b = [0xDD]

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
        [0, 1, 2],
        3,
        [4],
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0],
        [0],
        out_max_a,
        out_cap_a,
    )
    err_b = explicit_default_span_composition(
        [0, 1, 2],
        3,
        [4],
        blob,
        len(blob),
        offsets,
        lens,
        len(lens),
        [0],
        [0],
        out_max_b,
        out_cap_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_max_a[0] == 0xAA
    assert out_cap_a[0] == 0xBB
    assert out_max_b[0] == 0xCC
    assert out_cap_b[0] == 0xDD

    huge_lens = [0, (I64_MAX // 2) + 1]
    out_max_a = [0x33]
    out_cap_a = [0x44]
    out_max_b = [0x55]
    out_cap_b = [0x66]

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
        [0, 1],
        2,
        [0],
        [0, 1, 2],
        3,
        [0, 1],
        huge_lens,
        2,
        [0],
        [0],
        out_max_a,
        out_cap_a,
    )
    err_b = explicit_default_span_composition(
        [0, 1],
        2,
        [0],
        [0, 1, 2],
        3,
        [0, 1],
        huge_lens,
        2,
        [0],
        [0],
        out_max_b,
        out_cap_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_max_a[0] == 0x33
    assert out_cap_a[0] == 0x44
    assert out_max_b[0] == 0x55
    assert out_cap_b[0] == 0x66


def test_randomized_default_span_parity() -> None:
    rng = random.Random(20260419_528)
    pieces = [
        b"",
        b"a",
        b"bc",
        b"DEF",
        "🙂".encode("utf-8"),
        "中".encode("utf-8"),
        b"\\n",
        b" ",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(5000):
        token_count = rng.randint(0, 64)
        cursor = rng.randint(0, token_count + 6)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        out_count = [rng.randint(0, 8)]

        max_new = [0x1010]
        cap_new = [0x2020]
        max_ref = [0x3030]
        cap_ref = [0x4040]

        err_new = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight_default_span(
            token_ids,
            token_count,
            [cursor],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            [0] * 512,
            out_count.copy(),
            max_new,
            cap_new,
        )
        err_ref = explicit_default_span_composition(
            token_ids,
            token_count,
            [cursor],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            [0] * 512,
            out_count.copy(),
            max_ref,
            cap_ref,
        )

        assert err_new == err_ref
        if err_new == TOKENIZER_BPE_OK:
            assert max_new[0] == max_ref[0]
            assert cap_new[0] == cap_ref[0]
        else:
            assert max_new[0] == 0x1010
            assert cap_new[0] == 0x2020


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(
            [
                __file__,
                "-q",
            ]
        )
    )
