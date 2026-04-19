#!/usr/bin/env python3
"""Parity harness for TokenizerBPEComputePromptDecodeCapacityFromLensCheckedDefault."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked import (
    tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_prompt_out_byte_capacity: list[int] | None,
) -> int:
    if vocab_piece_lens is None or out_prompt_out_byte_capacity is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    vocab_piece_capacity = vocab_piece_count
    return tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
        token_count,
        vocab_piece_lens,
        vocab_piece_capacity,
        out_prompt_out_byte_capacity,
    )


def explicit_default_composition(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_prompt_out_byte_capacity: list[int] | None,
) -> int:
    if vocab_piece_lens is None or out_prompt_out_byte_capacity is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    return tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
        token_count,
        vocab_piece_lens,
        vocab_piece_count,
        out_prompt_out_byte_capacity,
    )


def test_source_contains_default_wrapper_signature_and_delegate() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "I32 TokenizerBPEComputePromptDecodeCapacityFromLensCheckedDefault(" in source
    body = source.split(
        "I32 TokenizerBPEComputePromptDecodeCapacityFromLensCheckedDefault(", 1
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoPartialNoAllocFromMaxPiece",
        1,
    )[0]
    assert "vocab_piece_capacity = vocab_piece_count;" in body
    assert "TokenizerBPEComputePromptDecodeCapacityFromLensChecked(" in body


def test_null_and_overflow_edges() -> None:
    out = [555]
    assert (
        tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
            1, None, 1, out
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )
    assert (
        tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
            1, [1], 1, None
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    out = [111]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
        I64_MAX + 1,
        [1],
        1,
        out,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 111

    out = [222]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
        1,
        [1],
        I64_MAX + 1,
        out,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 222


def test_multilingual_and_randomized_equivalence() -> None:
    pieces = [
        b"hello",
        b" ",
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        " Κα".encode("utf-8"),
        b"",
    ]
    lens = [len(piece) for piece in pieces]

    out_a = [0xABCD]
    out_b = [0x1234]
    token_count = 15

    err_a = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
        token_count,
        lens,
        len(lens),
        out_a,
    )
    err_b = explicit_default_composition(
        token_count,
        lens,
        len(lens),
        out_b,
    )
    assert err_a == err_b == TOKENIZER_BPE_OK
    assert out_a[0] == out_b[0]

    rng = random.Random(20260419_508)
    for _ in range(5000):
        vocab_piece_count = rng.randint(0, 120)
        lens = [rng.randint(0, 128) for _ in range(vocab_piece_count)]
        token_count = rng.randint(0, 256)

        out_new = [0xAAAA]
        out_ref = [0xBBBB]

        err_new = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default(
            token_count,
            lens,
            vocab_piece_count,
            out_new,
        )
        err_ref = explicit_default_composition(
            token_count,
            lens,
            vocab_piece_count,
            out_ref,
        )
        assert err_new == err_ref
        if err_new == TOKENIZER_BPE_OK:
            assert out_new[0] == out_ref[0]


def run() -> None:
    test_source_contains_default_wrapper_signature_and_delegate()
    test_null_and_overflow_edges()
    test_multilingual_and_randomized_equivalence()
    print("tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked_default=ok")


if __name__ == "__main__":
    run()
