#!/usr/bin/env python3
"""Parity harness for ...FromMaxPieceViaCapacityHelper wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked import (
    tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked,
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


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper(
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

    out_max_piece = [0]
    out_capacity = [0]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked(
        token_count,
        vocab_piece_lens,
        vocab_piece_count,
        out_max_piece,
        out_capacity,
    )
    if err != TOKENIZER_BPE_OK:
        return err

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
        out_capacity[0],
        out_byte_count,
    )


def explicit_inline_composition(
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
    out_max_piece = [0]
    out_capacity = [0]
    err = tokenizer_bpe_compute_prompt_decode_capacity_from_max_piece_checked(
        token_count,
        vocab_piece_lens,
        vocab_piece_count,
        out_max_piece,
        out_capacity,
    )
    if err != TOKENIZER_BPE_OK:
        return err

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
        out_capacity[0],
        out_byte_count,
    )


def run_case(token_ids: list[int], token_cursor: int, pieces: list[bytes]) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x44] * 4096
    out_new = out_ref.copy()
    count_ref = [0x7777]
    count_new = [0x7777]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    err_ref = explicit_inline_composition(
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
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper(
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


def test_source_contains_via_capacity_helper_wrapper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    marker = "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceViaCapacityHelper("
    assert marker in source

    body = source.split(marker, 1)[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight(", 1
    )[0]

    assert "TokenizerBPEComputePromptDecodeCapacityFromMaxPieceChecked(" in body
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAlloc(" in body
    assert "TokenizerBPEComputeMaxPieceBytesChecked" not in body


def test_multilingual_success_parity_against_inline_composition() -> None:
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


def test_adversarial_vectors_match_inline_composition() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0x12] * 128
    out_new = out_ref.copy()
    count_ref = [0xABAB]
    count_new = [0xABAB]
    cursor_ref = [4]
    cursor_new = [4]

    err_ref = explicit_inline_composition(
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
    err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper(
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
    assert cursor_new[0] == cursor_ref[0] == 4
    assert count_new[0] == count_ref[0] == 0xABAB
    assert out_new == out_ref


def test_null_and_overflow_domains() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x4C4C]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper(
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
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper(
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


def test_randomized_parity_against_inline_composition() -> None:
    rng = random.Random(20260419_539)
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
        token_count = rng.randint(0, 56)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 3)

        out_ref = [0xAA] * 1536
        out_new = out_ref.copy()
        count_ref = [0x3A3A]
        count_new = [0x3A3A]
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

        err_ref = explicit_inline_composition(
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
        err_new = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper(
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


def run() -> None:
    test_source_contains_via_capacity_helper_wrapper()
    test_multilingual_success_parity_against_inline_composition()
    test_adversarial_vectors_match_inline_composition()
    test_null_and_overflow_domains()
    test_randomized_parity_against_inline_composition()
    print(
        "tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_via_capacity_helper=ok"
    )


if __name__ == "__main__":
    run()
