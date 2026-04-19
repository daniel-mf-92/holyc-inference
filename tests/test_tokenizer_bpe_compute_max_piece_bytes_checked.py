#!/usr/bin/env python3
"""Parity harness for TokenizerBPEComputeMaxPieceBytesChecked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece import (
    tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_compute_max_piece_bytes_checked(
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_max_piece_bytes: list[int] | None,
) -> int:
    if vocab_piece_lens is None or out_max_piece_bytes is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if vocab_piece_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    scan = 0
    max_piece_bytes = 0
    while scan < vocab_piece_count:
        piece_len = vocab_piece_lens[scan]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > max_piece_bytes:
            max_piece_bytes = piece_len

        scan += 1
        if scan == 0:
            return TOKENIZER_BPE_ERR_BAD_PARAM

    out_max_piece_bytes[0] = max_piece_bytes
    return TOKENIZER_BPE_OK


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece_via_helper(
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

    max_piece = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if token_count > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if io_token_cursor[0] > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if token_count and max_piece[0] > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = token_count * max_piece[0]
    decode_end = token_count
    cursor = io_token_cursor[0]

    staged_out_count = 0
    scan = cursor
    while scan < decode_end:
        token_id = token_ids[scan]
        if token_id < 0 or token_id >= vocab_piece_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        piece_offset = vocab_piece_offsets[token_id]
        piece_len = vocab_piece_lens[token_id]
        if piece_offset > vocab_piece_bytes_len:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if piece_len > vocab_piece_bytes_len - piece_offset:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if piece_len > derived_capacity - staged_out_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        staged_out_count += piece_len
        if staged_out_count > derived_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

        scan += 1

    write_cursor = 0
    scan = cursor
    while scan < decode_end:
        token_id = token_ids[scan]
        piece_offset = vocab_piece_offsets[token_id]
        piece_len = vocab_piece_lens[token_id]

        for i in range(piece_len):
            out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + i]
            write_cursor += 1

        scan += 1

    if write_cursor != staged_out_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_out_count
    io_token_cursor[0] = decode_end
    return TOKENIZER_BPE_OK


def test_source_contains_new_helper_and_integration_call() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "I32 TokenizerBPEComputeMaxPieceBytesChecked(" in source

    wrapper = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoPartialNoAllocFromMaxPiece", 1
    )[1].split("I32 TokenizerBPEDecodePromptCheckedDefault", 1)[0]
    assert (
        "TokenizerBPEComputeMaxPieceBytesChecked(" in wrapper
        or "TokenizerBPEComputePromptDecodeCapacityFromMaxPieceChecked(" in wrapper
    )


def test_scalar_empty_and_multilingual_vectors() -> None:
    out = [999]
    err = tokenizer_bpe_compute_max_piece_bytes_checked([], 0, out)
    assert err == TOKENIZER_BPE_OK and out[0] == 0

    lens = [5, 1, len(" Κα".encode("utf-8")), len("世界".encode("utf-8")), len("🙂".encode("utf-8")), 0]
    out = [123]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(lens, len(lens), out)
    assert err == TOKENIZER_BPE_OK
    assert out[0] == max(lens)


def test_scalar_adversarial_and_contract_edges() -> None:
    out = [111]
    assert tokenizer_bpe_compute_max_piece_bytes_checked(None, 0, out) == TOKENIZER_BPE_ERR_NULL_PTR
    assert tokenizer_bpe_compute_max_piece_bytes_checked([], 0, None) == TOKENIZER_BPE_ERR_NULL_PTR

    out = [222]
    err = tokenizer_bpe_compute_max_piece_bytes_checked([1], I64_MAX + 1, out)
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 222

    out = [333]
    err = tokenizer_bpe_compute_max_piece_bytes_checked([0, I64_MAX + 1], 2, out)
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out[0] == 333


def test_integration_parity_against_prompt_noalloc_from_max_piece_wrapper() -> None:
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
    blob, offsets, lens = build_vocab_tables(pieces)
    token_ids = [0, 1, 2, 3, 1, 4, 5, 6, 7, 1, 8, 1, 9, 10]

    out_ref = [0xA1] * 2048
    out_new = out_ref.copy()
    count_ref = [0xDEAD]
    count_new = [0xDEAD]
    cursor_ref = [3]
    cursor_new = [3]

    err_ref = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_ref,
        count_ref,
    )
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece_via_helper(
        token_ids,
        len(token_ids),
        cursor_new,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_new,
        count_new,
    )

    assert err_new == err_ref
    assert cursor_new[0] == cursor_ref[0]
    assert count_new[0] == count_ref[0]
    assert out_new == out_ref


def test_randomized_scalar_and_integration_parity() -> None:
    rng = random.Random(20260419_479)

    for _ in range(3000):
        lens_count = rng.randint(0, 96)
        lens = [rng.randint(0, 48) for _ in range(lens_count)]

        out = [0x4D4D]
        err = tokenizer_bpe_compute_max_piece_bytes_checked(lens, lens_count, out)
        assert err == TOKENIZER_BPE_OK
        assert out[0] == (max(lens) if lens else 0)

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

    for _ in range(2500):
        token_count = rng.randint(0, 48)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 3)

        out_ref = [0x5D] * 1024
        out_new = out_ref.copy()
        count_ref = [0x3131]
        count_new = [0x3131]
        cursor_ref = [cursor_seed]
        cursor_new = [cursor_seed]

        err_ref = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
            token_ids,
            token_count,
            cursor_ref,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_ref,
            count_ref,
        )
        err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece_via_helper(
            token_ids,
            token_count,
            cursor_new,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_new,
            count_new,
        )

        assert err_new == err_ref
        assert cursor_new[0] == cursor_ref[0]
        assert count_new[0] == count_ref[0]
        assert out_new == out_ref


def run() -> None:
    test_source_contains_new_helper_and_integration_call()
    test_scalar_empty_and_multilingual_vectors()
    test_scalar_adversarial_and_contract_edges()
    test_integration_parity_against_prompt_noalloc_from_max_piece_wrapper()
    test_randomized_scalar_and_integration_parity()
    print("tokenizer_bpe_compute_max_piece_bytes_checked=ok")


if __name__ == "__main__":
    run()
