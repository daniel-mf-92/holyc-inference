#!/usr/bin/env python3
"""Parity harness for ...DefaultCapacityValidateCursorNoAllocFromMaxPiece."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_compute_max_piece_bytes_checked import (
    tokenizer_bpe_compute_max_piece_bytes_checked,
)
from test_tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc import (
    tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
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

    max_piece_holder = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece_holder,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    max_piece_bytes = max_piece_holder[0]
    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW
    derived_out_capacity = token_count * max_piece_bytes

    return tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        derived_out_capacity,
        out_byte_count,
    )


def staged_composition(
    token_ids: list[int],
    token_count: int,
    io_token_cursor: list[int],
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    out_bytes: list[int],
    out_byte_count: list[int],
) -> int:
    max_piece_holder = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece_holder,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    max_piece_bytes = max_piece_holder[0]
    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    return tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
        token_ids,
        token_count,
        io_token_cursor,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        out_bytes,
        token_count * max_piece_bytes,
        out_byte_count,
    )


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes]) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x8A] * 2048
    out_new = out_ref.copy()
    count_ref = [0x1212]
    count_new = [0x1212]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    err_ref = staged_composition(
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
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
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


def test_source_contains_wrapper_and_delegation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiece" in source
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiece", 1
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocNoPartialFromMaxPiece", 1
    )[0]
    assert "TokenizerBPEComputeMaxPieceBytesChecked" in body
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAlloc(" in body


def test_multilingual_success_parity_against_staged_composition() -> None:
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
    run_case(token_ids, 6, pieces)


def test_adversarial_vectors_match_staged_composition() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x72] * 128
    out_new = out_ref.copy()
    count_ref = [0x2222]
    count_new = [0x2222]

    cursor_ref = [7]
    cursor_new = [7]
    err_ref = staged_composition(
        [0, 1, 2],
        3,
        cursor_ref,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out_ref,
        count_ref,
    )
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        [0, 1, 2],
        3,
        cursor_new,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out_new,
        count_new,
    )
    assert err_new == err_ref == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor_new[0] == cursor_ref[0] == 7
    assert count_new[0] == count_ref[0] == 0x2222

    out_ref = [0x73] * 128
    out_new = out_ref.copy()
    count_ref = [0x3333]
    count_new = [0x3333]
    bad_lens = lens.copy()
    bad_lens[2] = len(blob) + 9

    cursor_ref = [0]
    cursor_new = [0]
    err_ref = staged_composition(
        [2],
        1,
        cursor_ref,
        blob,
        len(blob),
        offsets,
        bad_lens,
        3,
        out_ref,
        count_ref,
    )
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        [2],
        1,
        cursor_new,
        blob,
        len(blob),
        offsets,
        bad_lens,
        3,
        out_new,
        count_new,
    )
    assert err_new == err_ref == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor_new[0] == cursor_ref[0] == 0
    assert count_new[0] == count_ref[0] == 0x3333


def test_null_and_overflow_domains() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x4C4C]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
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
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
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
    rng = random.Random(20260419_490)
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

        out_ref = [0x58] * 1024
        out_new = out_ref.copy()
        count_ref = [0xA1A1]
        count_new = [0xA1A1]
        cursor_ref = [cursor_seed]
        cursor_new = [cursor_seed]

        if rng.random() < 0.2 and token_count:
            token_ids[rng.randrange(token_count)] = len(pieces) + rng.randint(0, 4)

        offsets_case = offsets.copy()
        lens_case = lens.copy()

        if rng.random() < 0.1 and pieces:
            slot = rng.randrange(len(pieces))
            offsets_case[slot] = len(blob) + rng.randint(1, 8)

        if rng.random() < 0.1 and pieces:
            slot = rng.randrange(len(pieces))
            lens_case[slot] = len(blob) + rng.randint(1, 8)

        err_ref = staged_composition(
            token_ids,
            token_count,
            cursor_ref,
            blob,
            len(blob),
            offsets_case,
            lens_case,
            len(pieces),
            out_ref,
            count_ref,
        )
        err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
            token_ids,
            token_count,
            cursor_new,
            blob,
            len(blob),
            offsets_case,
            lens_case,
            len(pieces),
            out_new,
            count_new,
        )

        assert err_new == err_ref
        assert cursor_new[0] == cursor_ref[0]
        assert count_new[0] == count_ref[0]
        assert out_new == out_ref
