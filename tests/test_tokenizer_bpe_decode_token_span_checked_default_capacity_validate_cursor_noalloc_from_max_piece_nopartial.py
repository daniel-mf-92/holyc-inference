#!/usr/bin/env python3
"""Parity harness for ...FromMaxPieceNoPartial token-span decode wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_compute_max_piece_bytes_checked import (
    tokenizer_bpe_compute_max_piece_bytes_checked,
)
from test_tokenizer_bpe_decode_token_span_checked import build_vocab_tables
from test_tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece import (
    tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
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
        or span_token_count > I64_MAX
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
    if span_token_count and max_piece_bytes > I64_MAX // span_token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = span_token_count * max_piece_bytes
    staged_cursor = [io_token_cursor[0]]
    staged_count = [out_byte_count[0]]
    staged_out = [0] * max(1, derived_capacity)

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        token_ids,
        token_count,
        staged_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        staged_out,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > derived_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_count[0]):
        out_bytes[i] = staged_out[i]

    io_token_cursor[0] = staged_cursor[0]
    out_byte_count[0] = staged_count[0]
    return TOKENIZER_BPE_OK


def staged_composition(
    token_ids: list[int],
    token_count: int,
    io_token_cursor: list[int],
    span_token_count: int,
    vocab_piece_bytes: list[int],
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int],
    vocab_piece_lens: list[int],
    vocab_piece_count: int,
    out_bytes: list[int],
    out_byte_count: list[int],
) -> int:
    if token_count > I64_MAX or span_token_count > I64_MAX:
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
    if span_token_count and max_piece_bytes > I64_MAX // span_token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_capacity = span_token_count * max_piece_bytes
    staged_cursor = [io_token_cursor[0]]
    staged_count = [out_byte_count[0]]
    staged_out = [0] * max(1, derived_capacity)

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece(
        token_ids,
        token_count,
        staged_cursor,
        span_token_count,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        staged_out,
        staged_count,
    )

    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > derived_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_count[0]):
        out_bytes[i] = staged_out[i]

    io_token_cursor[0] = staged_cursor[0]
    out_byte_count[0] = staged_count[0]
    return TOKENIZER_BPE_OK


def test_source_contains_wrapper_and_staged_shape() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial" in source
    body = source.split(
        "I32 TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial", 1
    )[1].split("I32 TokenizerBPEDecodePromptChecked", 1)[0]
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight(" in body
    assert "TokenizerBPEDecodeTokenSpanCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiece(" in body
    assert "MAlloc(" in body
    assert "if (staged_out_count > 0x7FFFFFFFFFFFFFFF)" in body
    assert "if (staged_cursor < cursor)" in body
    assert "if (staged_cursor > token_count)" in body


def run_case(
    token_ids: list[int],
    token_cursor: int,
    span_count: int,
    pieces: list[bytes],
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_ref = [0xA5] * 2048
    out_new = out_ref.copy()
    count_ref = [0x3434]
    count_new = [0x3434]
    cursor_ref = [token_cursor]
    cursor_new = [token_cursor]

    err_ref = staged_composition(
        token_ids,
        len(token_ids),
        cursor_ref,
        span_count,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        out_ref,
        count_ref,
    )

    err_new = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
        token_ids,
        len(token_ids),
        cursor_new,
        span_count,
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


def test_multilingual_success_parity() -> None:
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

    run_case(token_ids, 0, len(token_ids), pieces)
    run_case(token_ids, 4, 7, pieces)


def test_no_partial_on_failure_vectors() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x66] * 128
    count = [0x7E7E]

    bad_cursor = [5]
    seeded_out = out.copy()
    seeded_count = count[0]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
        [0, 1, 2],
        3,
        bad_cursor,
        0,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert bad_cursor[0] == 5
    assert count[0] == seeded_count
    assert out == seeded_out


def test_zero_span_commits_empty_and_preserves_cursor() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x4E] * 64
    count = [0x5151]
    cursor = [2]

    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
        [0, 1, 2],
        3,
        cursor,
        0,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == 2
    assert count[0] == 0
    assert out == [0x4E] * 64

    # Force span token id failure after wrapper preflight to verify no commit.
    bad_ids = [0, 17]
    bad_cursor = [0]
    seeded_out = out.copy()
    seeded_count = count[0]
    err = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
        bad_ids,
        2,
        bad_cursor,
        2,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        out,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert bad_cursor[0] == 0
    assert count[0] == seeded_count
    assert out == seeded_out


def test_null_and_overflow_domains() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x12] * 16
    count = [0x3A3A]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
            None,
            0,
            cursor,
            0,
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
        tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
            [0],
            I64_MAX + 1,
            cursor,
            0,
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
    rng = random.Random(20260419_497)
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

    for _ in range(3500):
        token_count = rng.randint(0, 64)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        if token_count and rng.random() < 0.10:
            token_ids[rng.randrange(token_count)] = len(pieces) + rng.randint(1, 9)
        if token_count and rng.random() < 0.05:
            token_ids[rng.randrange(token_count)] = -1

        cursor_seed = rng.randint(0, token_count + 3)
        span_seed = rng.randint(0, token_count + 6)

        out_ref = [0x5A] * 1024
        out_new = out_ref.copy()
        count_ref = [0x2626]
        count_new = [0x2626]
        cursor_ref = [cursor_seed]
        cursor_new = [cursor_seed]

        err_ref = staged_composition(
            token_ids,
            token_count,
            cursor_ref,
            span_seed,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            out_ref,
            count_ref,
        )
        err_new = tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial(
            token_ids,
            token_count,
            cursor_new,
            span_seed,
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
    test_source_contains_wrapper_and_staged_shape()
    test_multilingual_success_parity()
    test_no_partial_on_failure_vectors()
    test_zero_span_commits_empty_and_preserves_cursor()
    test_null_and_overflow_domains()
    test_randomized_parity_against_staged_composition()
    print("tokenizer_bpe_decode_token_span_checked_default_capacity_validate_cursor_noalloc_from_max_piece_nopartial=ok")


if __name__ == "__main__":
    run()
