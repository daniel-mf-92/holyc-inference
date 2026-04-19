#!/usr/bin/env python3
"""Parity harness for TokenizerBPEComputePromptDecodeCapacityFromMaxPieceCheckedDefault."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked import (
    tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_max_piece_bytes: list[int] | None,
    out_prompt_out_byte_capacity: list[int] | None,
) -> int:
    if (
        vocab_piece_lens is None
        or out_max_piece_bytes is None
        or out_prompt_out_byte_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    vocab_piece_capacity = vocab_piece_count
    return tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked(
        token_count,
        vocab_piece_lens,
        vocab_piece_capacity,
        out_max_piece_bytes,
        out_prompt_out_byte_capacity,
    )


def explicit_default_composition(
    token_count: int,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_max_piece_bytes: list[int] | None,
    out_prompt_out_byte_capacity: list[int] | None,
) -> int:
    if (
        vocab_piece_lens is None
        or out_max_piece_bytes is None
        or out_prompt_out_byte_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if token_count > I64_MAX or vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    return tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked(
        token_count,
        vocab_piece_lens,
        vocab_piece_count,
        out_max_piece_bytes,
        out_prompt_out_byte_capacity,
    )


def test_source_contains_default_wrapper_signature_and_delegate() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "I32 TokenizerBPEComputePromptDecodeCapacityFromMaxPieceCheckedDefault(" in source

    body = source.split(
        "I32 TokenizerBPEComputePromptDecodeCapacityFromMaxPieceCheckedDefault(", 1
    )[1].split(
        "I32 TokenizerBPEComputePromptDecodeCapacityFromLensChecked", 1
    )[0]

    assert "vocab_piece_capacity = vocab_piece_count;" in body
    assert "TokenizerBPEComputePromptDecodeCapacityFromMaxPieceChecked(" in body


def test_null_and_overflow_edges_leave_outputs_unchanged() -> None:
    out_max = [111]
    out_cap = [222]

    err = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default(
        1,
        None,
        1,
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert out_max[0] == 111
    assert out_cap[0] == 222

    err = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default(
        I64_MAX + 1,
        [1],
        1,
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_max[0] == 111
    assert out_cap[0] == 222


def test_multilingual_and_adversarial_equivalence() -> None:
    lens = [
        len(b"hello"),
        len(" Κα".encode("utf-8")),
        len("世界".encode("utf-8")),
        len("🙂".encode("utf-8")),
        0,
    ]

    out_max_new = [0xAAAA]
    out_cap_new = [0xBBBB]
    out_max_ref = [0xCCCC]
    out_cap_ref = [0xDDDD]

    err_new = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default(
        17,
        lens,
        len(lens),
        out_max_new,
        out_cap_new,
    )
    err_ref = explicit_default_composition(
        17,
        lens,
        len(lens),
        out_max_ref,
        out_cap_ref,
    )
    assert err_new == err_ref == TOKENIZER_BPE_OK
    assert out_max_new[0] == out_max_ref[0] == max(lens)
    assert out_cap_new[0] == out_cap_ref[0] == 17 * max(lens)

    out_max = [777]
    out_cap = [888]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default(
        3,
        [I64_MAX // 2 + 1],
        1,
        out_max,
        out_cap,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_max[0] == 777
    assert out_cap[0] == 888


def test_randomized_equivalence_against_explicit_composition() -> None:
    rng = random.Random(20260419_526)

    for _ in range(6000):
        vocab_piece_count = rng.randint(0, 128)
        lens = [rng.randint(0, 192) for _ in range(vocab_piece_count)]
        if vocab_piece_count and rng.random() < 0.08:
            lens[rng.randrange(vocab_piece_count)] = I64_MAX + rng.randint(1, 32)

        token_count = rng.randint(0, 512)
        if rng.random() < 0.05:
            token_count = I64_MAX + rng.randint(1, 32)

        out_max_new = [0x1234]
        out_cap_new = [0x5678]
        out_max_ref = [0x9ABC]
        out_cap_ref = [0xDEF0]

        err_new = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default(
            token_count,
            lens,
            vocab_piece_count,
            out_max_new,
            out_cap_new,
        )
        err_ref = explicit_default_composition(
            token_count,
            lens,
            vocab_piece_count,
            out_max_ref,
            out_cap_ref,
        )

        assert err_new == err_ref
        if err_new == TOKENIZER_BPE_OK:
            assert out_max_new[0] == out_max_ref[0]
            assert out_cap_new[0] == out_cap_ref[0]


def run() -> None:
    test_source_contains_default_wrapper_signature_and_delegate()
    test_null_and_overflow_edges_leave_outputs_unchanged()
    test_multilingual_and_adversarial_equivalence()
    test_randomized_equivalence_against_explicit_composition()
    print("tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked_default=ok")


if __name__ == "__main__":
    run()
