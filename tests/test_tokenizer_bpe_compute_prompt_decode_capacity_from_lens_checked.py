#!/usr/bin/env python3
"""Parity harness for TokenizerBPEComputePromptDecodeCapacityFromLensChecked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_compute_max_piece_bytes_checked import (
    tokenizer_bpe_compute_max_piece_bytes_checked,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_prompt_out_byte_capacity: list[int] | None,
) -> int:
    if vocab_piece_lens is None or out_prompt_out_byte_capacity is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    max_piece = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if token_count and max_piece[0] > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_prompt_out_byte_capacity[0] = token_count * max_piece[0]
    return TOKENIZER_BPE_OK


def explicit_staged_composition(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_prompt_out_byte_capacity: list[int] | None,
) -> int:
    if vocab_piece_lens is None or out_prompt_out_byte_capacity is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    max_piece = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if token_count and max_piece[0] > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_prompt_out_byte_capacity[0] = token_count * max_piece[0]
    return TOKENIZER_BPE_OK


def via_decode_wrapper_capacity_math(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
) -> tuple[int, int]:
    if vocab_piece_lens is None:
        return TOKENIZER_BPE_ERR_NULL_PTR, 0

    max_piece = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err, 0

    if token_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW, 0
    if token_count and max_piece[0] > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW, 0

    return TOKENIZER_BPE_OK, token_count * max_piece[0]


def test_source_contains_helper_signature_and_core_composition() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "I32 TokenizerBPEComputePromptDecodeCapacityFromLensChecked(" in source
    body = source.split(
        "I32 TokenizerBPEComputePromptDecodeCapacityFromLensChecked(", 1
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoPartialNoAllocFromMaxPiece",
        1,
    )[0]
    assert "TokenizerBPEComputePromptDecodeCapacityFromMaxPieceChecked(" in body
    assert "TokenizerBPEComputePromptDecodeCapacityFromMaxPieceChecked(" in body
    assert "out_prompt_out_byte_capacity" in body


def test_empty_and_multilingual_vectors() -> None:
    out = [777]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(0, [], 0, out)
    assert err == TOKENIZER_BPE_OK
    assert out[0] == 0

    pieces = [
        b"hello",
        b" ",
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        " Κα".encode("utf-8"),
        b"",
    ]
    _, _, lens = build_vocab_tables(pieces)

    out_a = [0xAAAA]
    out_b = [0xBBBB]
    token_count = 13

    err_a = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
        token_count,
        lens,
        len(lens),
        out_a,
    )
    err_b = explicit_staged_composition(
        token_count,
        lens,
        len(lens),
        out_b,
    )
    assert err_a == err_b == TOKENIZER_BPE_OK
    assert out_a[0] == out_b[0] == token_count * max(lens)


def test_adversarial_null_and_overflow_edges() -> None:
    out = [111]
    assert (
        tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(1, None, 1, out)
        == TOKENIZER_BPE_ERR_NULL_PTR
    )
    assert (
        tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(1, [1], 1, None)
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    out = [222]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
        I64_MAX + 1,
        [1],
        1,
        out,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 222

    out = [333]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
        1,
        [1],
        I64_MAX + 1,
        out,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 333

    out = [444]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
        3,
        [0, (I64_MAX // 2) + 1],
        2,
        out,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 444


def test_randomized_scalar_and_wrapper_composition_equivalence() -> None:
    rng = random.Random(20260419_494)

    for _ in range(4000):
        vocab_piece_count = rng.randint(0, 96)
        lens = [rng.randint(0, 64) for _ in range(vocab_piece_count)]
        token_count = rng.randint(0, 128)

        out_new = [0x55AA]
        out_ref = [0xA55A]

        err_new = tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked(
            token_count,
            lens,
            vocab_piece_count,
            out_new,
        )
        err_ref = explicit_staged_composition(
            token_count,
            lens,
            vocab_piece_count,
            out_ref,
        )
        assert err_new == err_ref
        if err_new == TOKENIZER_BPE_OK:
            assert out_new[0] == out_ref[0]

        err_wrap, derived = via_decode_wrapper_capacity_math(
            token_count,
            lens,
            vocab_piece_count,
        )
        assert err_wrap == err_new
        if err_new == TOKENIZER_BPE_OK:
            assert derived == out_new[0]


def run() -> None:
    test_source_contains_helper_signature_and_core_composition()
    test_empty_and_multilingual_vectors()
    test_adversarial_null_and_overflow_edges()
    test_randomized_scalar_and_wrapper_composition_equivalence()
    print("tokenizer_bpe_compute_prompt_decode_capacity_from_lens_checked=ok")


if __name__ == "__main__":
    run()
