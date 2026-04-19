#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityValidateCursorNoPartialNoAllocFromMaxPiece."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc import (
    tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
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

    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    max_piece_bytes = 0
    for i in range(vocab_piece_count):
        piece_len = vocab_piece_lens[i]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > max_piece_bytes:
            max_piece_bytes = piece_len

    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_capacity = token_count * max_piece_bytes

    decode_end = token_count

    staged_out_count = 0
    scan = cursor
    while scan < decode_end:
        token_id = token_ids[scan]
        if token_id < 0:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if token_id >= vocab_piece_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        piece_offset = vocab_piece_offsets[token_id]
        piece_len = vocab_piece_lens[token_id]

        if piece_offset > vocab_piece_bytes_len:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if piece_len > vocab_piece_bytes_len - piece_offset:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        if piece_len > derived_out_capacity - staged_out_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        staged_out_count += piece_len
        if staged_out_count > derived_out_capacity:
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
            if write_cursor > staged_out_count:
                return TOKENIZER_BPE_ERR_OVERFLOW

        scan += 1

    if write_cursor != staged_out_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_out_count
    io_token_cursor[0] = decode_end
    return TOKENIZER_BPE_OK


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes]) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x93] * 2048
    out_new = out_ref.copy()
    count_ref = [0x4343]
    count_new = [0x4343]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    max_piece_bytes = max((len(piece) for piece in pieces), default=0)

    err_ref = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
        token_ids,
        len(token_ids),
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        max_piece_bytes,
        out_ref,
        count_ref,
    )
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
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


def test_source_contains_wrapper_and_is_noalloc() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoPartialNoAllocFromMaxPiece" in source
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoPartialNoAllocFromMaxPiece", 1
    )[1].split("I32 TokenizerBPEDecodePromptCheckedDefault", 1)[0]
    assert "MAlloc(" not in body
    assert "decode_end = token_count;" in body


def test_multilingual_success_vectors() -> None:
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

    run_case(token_ids, 0, pieces)
    run_case(token_ids, 5, pieces)


def test_no_partial_on_adversarial_failures() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x55] * 128
    count = [0x6D6D]

    cursor = [9]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 9 and count[0] == 0x6D6D
    assert out == [0x55] * 128

    cursor = [0]
    bad_offsets = offsets.copy()
    bad_offsets[2] = len(blob) + 1
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
        [2],
        1,
        cursor,
        blob,
        len(blob),
        bad_offsets,
        lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0x6D6D
    assert out == [0x55] * 128


def test_null_and_overflow_domains() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x2D2D]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_against_staged_composition() -> None:
    rng = random.Random(20260419_463)
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

    for _ in range(3000):
        token_count = rng.randint(0, 48)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 3)

        out_ref = [0x5D] * 1024
        out_new = out_ref.copy()
        count_ref = [0x3131]
        count_new = [0x3131]
        cursor_ref = [cursor_seed]
        cursor_new = [cursor_seed]

        max_piece_bytes = max((len(piece) for piece in pieces), default=0)

        err_ref = tokenizer_bpe_decode_prompt_checked_default_no_partial_from_max_piece_validate_cursor_noalloc(
            token_ids,
            token_count,
            cursor_ref,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            max_piece_bytes,
            out_ref,
            count_ref,
        )
        err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece(
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
    test_source_contains_wrapper_and_is_noalloc()
    test_multilingual_success_vectors()
    test_no_partial_on_adversarial_failures()
    test_null_and_overflow_domains()
    test_randomized_parity_against_staged_composition()
    print("tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_no_partial_noalloc_from_max_piece=ok")


if __name__ == "__main__":
    run()
