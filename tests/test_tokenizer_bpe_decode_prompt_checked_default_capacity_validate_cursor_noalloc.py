#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAlloc."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor import (
    tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    out_bytes: list[int] | None,
    out_byte_capacity: int,
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
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_count = token_count - cursor
    if decode_count > token_count - cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_end = cursor + decode_count
    if decode_end < cursor or decode_end > token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

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
        if piece_len > out_byte_capacity - staged_out_count:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        staged_out_count += piece_len
        if staged_out_count > out_byte_capacity:
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


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes], out_capacity: int) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_alloc = [0x61] * 2048
    out_noalloc = out_alloc.copy()
    count_alloc = [0x1234]
    count_noalloc = [0x1234]
    cursor_alloc = [token_cursor]
    cursor_noalloc = [token_cursor]

    err_alloc = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
        token_ids,
        len(token_ids),
        cursor_alloc,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_alloc,
        out_capacity,
        count_alloc,
    )
    err_noalloc = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
        token_ids,
        len(token_ids),
        cursor_noalloc,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_noalloc,
        out_capacity,
        count_noalloc,
    )

    assert err_noalloc == err_alloc
    assert cursor_noalloc[0] == cursor_alloc[0]
    assert count_noalloc[0] == count_alloc[0]
    assert out_noalloc == out_alloc


def test_source_contains_noalloc_wrapper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAlloc" in source
    assert "MAlloc(" not in source.split("TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAlloc", 1)[1].split("I32 TokenizerBPEDecodePromptCheckedDefaultCapacityNoPartial", 1)[0]


def test_multilingual_success_parity_against_allocating_wrapper() -> None:
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

    run_case(token_ids, 0, pieces, 256)
    run_case(token_ids, 6, pieces, 256)


def test_cursor_capacity_adversarial_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x44] * 64
    count = [0xCAFE]

    cursor = [5]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
        [0, 1, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        16,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 5 and count[0] == 0xCAFE

    cursor = [0]
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
        [2, 2, 2],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        2,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and count[0] == 0xCAFE


def test_null_and_overflow_surfaces() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0xEE] * 16
    count = [0x9B9B]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
            None,
            0,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            1,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
            [0],
            I64_MAX + 1,
            cursor,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            out,
            1,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_failure_keeps_outputs_unchanged() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x55] * 32
    out_before = out.copy()
    count = [0xBEEF]
    cursor = [1]

    # token id 9 is out of vocab; pass1 must fail before any caller writes.
    err = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
        [0, 9, 1],
        3,
        cursor,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        32,
        count,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 1
    assert count[0] == 0xBEEF
    assert out == out_before


def test_randomized_parity_against_allocating_wrapper() -> None:
    rng = random.Random(20260419_453)
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
        token_count = rng.randint(0, 40)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 3)
        out_capacity = rng.randint(0, 128)

        out_alloc = [0xAB] * 256
        out_noalloc = out_alloc.copy()
        count_alloc = [0x1111]
        count_noalloc = [0x1111]
        cursor_alloc = [cursor_seed]
        cursor_noalloc = [cursor_seed]

        err_alloc = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor(
            token_ids,
            token_count,
            cursor_alloc,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_alloc,
            out_capacity,
            count_alloc,
        )
        err_noalloc = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
            token_ids,
            token_count,
            cursor_noalloc,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_noalloc,
            out_capacity,
            count_noalloc,
        )

        assert err_noalloc == err_alloc
        assert cursor_noalloc[0] == cursor_alloc[0]
        assert count_noalloc[0] == count_alloc[0]
        assert out_noalloc == out_alloc


def run() -> None:
    test_source_contains_noalloc_wrapper()
    test_multilingual_success_parity_against_allocating_wrapper()
    test_cursor_capacity_adversarial_vectors()
    test_null_and_overflow_surfaces()
    test_failure_keeps_outputs_unchanged()
    test_randomized_parity_against_allocating_wrapper()
    print("tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc=ok")


if __name__ == "__main__":
    run()
