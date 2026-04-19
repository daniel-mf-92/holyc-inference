#!/usr/bin/env python3
"""Parity harness for ...NoAllocNoPartialFromMaxPiecePreflight helper."""

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


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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


def test_source_contains_helper_signature_and_delegate() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert (
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiecePreflight(" in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiecePreflight(",
        1,
    )[1].split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight",
        1,
    )[0]
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight(" in body



def test_wrapper_uses_nopartial_preflight() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    wrapper_body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial(",
        1,
    )[1].split("I32 TokenizerBPEDecodePromptChecked", 1)[0]
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartialPreflight(" in wrapper_body



def test_multilingual_parity_against_inline_preflight() -> None:
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

    err_a = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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
        max_a,
        cap_a,
    )
    err_b = explicit_inline_preflight(
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
        max_b,
        cap_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert max_a[0] == max_b[0] == max(lens)
    assert cap_a[0] == cap_b[0] == 3 * max(lens)



def test_adversarial_null_cursor_span_overflow() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    assert (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
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

    out_max = [0x33]
    out_cap = [0x44]
    huge_lens = [0, (I64_MAX // 2) + 1]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
        [0, 1],
        2,
        [0],
        2,
        [0, 1, 2],
        3,
        [0, 1],
        huge_lens,
        2,
        [0],
        [0],
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_max[0] == 0x33
    assert out_cap[0] == 0x44



def test_randomized_parity_against_inline_preflight() -> None:
    rng = random.Random(20260419_510)
    pieces = [
        b"",
        b"a",
        b"bc",
        b"DEF",
        "🙂".encode("utf-8"),
        "中".encode("utf-8"),
        b"\n",
        b" ",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(5000):
        token_count = rng.randint(0, 64)
        span_token_count = rng.randint(0, token_count + 8)
        cursor = rng.randint(0, token_count + 5)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]

        max_new = [0x1010]
        cap_new = [0x2020]
        max_ref = [0x3030]
        cap_ref = [0x4040]

        err_new = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight(
            token_ids,
            token_count,
            [cursor],
            span_token_count,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            [0] * 2048,
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
            [0] * 2048,
            [0],
            max_ref,
            cap_ref,
        )

        assert err_new == err_ref
        if err_new == TOKENIZER_BPE_OK:
            assert max_new[0] == max_ref[0]
            assert cap_new[0] == cap_ref[0]



def run() -> None:
    test_source_contains_helper_signature_and_delegate()
    test_wrapper_uses_nopartial_preflight()
    test_multilingual_parity_against_inline_preflight()
    test_adversarial_null_cursor_span_overflow()
    test_randomized_parity_against_inline_preflight()
    print(
        "tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_nopartial_from_max_piece_preflight=ok"
    )


if __name__ == "__main__":
    run()
