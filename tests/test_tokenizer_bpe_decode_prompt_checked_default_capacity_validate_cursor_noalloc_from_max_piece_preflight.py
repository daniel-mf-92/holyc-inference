#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromMaxPiecePreflight helper."""

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


def tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
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
    *,
    out_max_piece_bytes_present: bool = True,
    out_derived_out_byte_capacity_present: bool = True,
) -> tuple[int, int, int]:
    if (
        token_ids is None
        or io_token_cursor is None
        or vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or out_byte_count is None
        or not out_max_piece_bytes_present
        or not out_derived_out_byte_capacity_present
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR, 0, 0

    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW, 0, 0

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM, 0, 0

    max_piece_holder = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece_holder,
    )
    if err != TOKENIZER_BPE_OK:
        return err, 0, 0

    max_piece_bytes = max_piece_holder[0]
    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW, 0, 0

    return TOKENIZER_BPE_OK, max_piece_bytes, token_count * max_piece_bytes


def inline_preflight_equivalent(
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
    *,
    out_max_piece_bytes_present: bool = True,
    out_derived_out_byte_capacity_present: bool = True,
) -> tuple[int, int, int]:
    if (
        token_ids is None
        or io_token_cursor is None
        or vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or out_byte_count is None
        or not out_max_piece_bytes_present
        or not out_derived_out_byte_capacity_present
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR, 0, 0

    if (
        token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW, 0, 0

    cursor = io_token_cursor[0]
    if cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM, 0, 0

    max_piece_holder = [0]
    err = tokenizer_bpe_compute_max_piece_bytes_checked(
        vocab_piece_lens,
        vocab_piece_count,
        max_piece_holder,
    )
    if err != TOKENIZER_BPE_OK:
        return err, 0, 0

    max_piece_bytes = max_piece_holder[0]
    if token_count and max_piece_bytes > I64_MAX // token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW, 0, 0

    return TOKENIZER_BPE_OK, max_piece_bytes, token_count * max_piece_bytes


def wrapper_using_preflight(
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
    err, _max_piece_bytes, derived_capacity = (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            token_ids,
            token_count,
            io_token_cursor,
            vocab_piece_bytes,
            vocab_piece_bytes_len,
            vocab_piece_offsets,
            vocab_piece_lens,
            vocab_piece_count,
            out_bytes,
            out_byte_count,
        )
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
        derived_capacity,
        out_byte_count,
    )


def test_source_contains_preflight_and_wrapper_delegation() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    assert "TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight" in source
    assert (
        "status = TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight("
        in source
    )
    body = source.split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPiecePreflight", 1
    )[1].split(
        "I32 TokenizerBPEDecodePromptCheckedDefaultCapacityValidateCursorNoAllocFromMaxPieceNoPartial", 1
    )[0]
    assert "!out_max_piece_bytes" in body
    assert "!out_derived_out_byte_capacity" in body


def test_null_cursor_overflow_and_output_ptr_surfaces() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)
    token_ids = [0, 1]
    out = [0x77] * 128
    out_count = [0x4444]

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            None,
            2,
            [0],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out,
            out_count,
        )[0]
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            token_ids,
            2,
            [0],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out,
            out_count,
            out_max_piece_bytes_present=False,
        )[0]
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            token_ids,
            I64_MAX + 1,
            [0],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out,
            out_count,
        )[0]
        == TOKENIZER_BPE_ERR_OVERFLOW
    )

    assert (
        tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            token_ids,
            2,
            [3],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out,
            out_count,
        )[0]
        == TOKENIZER_BPE_ERR_BAD_PARAM
    )


def test_multilingual_and_adversarial_preflight_parity() -> None:
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
    out = [0x91] * 4096
    out_count = [0x6767]

    for cursor in [0, 4, len(token_ids), len(token_ids) + 1]:
        got = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            token_ids,
            len(token_ids),
            [cursor],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out,
            out_count,
        )
        ref = inline_preflight_equivalent(
            token_ids,
            len(token_ids),
            [cursor],
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out,
            out_count,
        )
        assert got == ref


def test_randomized_preflight_and_wrapper_parity() -> None:
    rng = random.Random(20260419_506)
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

    for _ in range(4000):
        token_count = rng.randint(0, 48)
        token_ids = [rng.randint(0, len(pieces) - 1) for _ in range(token_count)]
        cursor_seed = rng.randint(0, token_count + 4)

        out_a = [0xAB] * 2048
        out_b = out_a.copy()
        count_a = [0x5151]
        count_b = [0x5151]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        got_pre = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight(
            token_ids,
            token_count,
            cursor_a,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_a,
            count_a,
        )
        ref_pre = inline_preflight_equivalent(
            token_ids,
            token_count,
            cursor_b,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_b,
            count_b,
        )
        assert got_pre == ref_pre

        cursor_wrapped = [cursor_seed]
        cursor_staged = [cursor_seed]
        out_wrapped = [0xCD] * 2048
        out_staged = out_wrapped.copy()
        count_wrapped = [0x6262]
        count_staged = [0x6262]

        err_wrapped = wrapper_using_preflight(
            token_ids,
            token_count,
            cursor_wrapped,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_wrapped,
            count_wrapped,
        )

        err, _max_piece_bytes, derived_capacity = inline_preflight_equivalent(
            token_ids,
            token_count,
            cursor_staged,
            blob,
            len(blob),
            offsets,
            lens,
            len(lens),
            out_staged,
            count_staged,
        )
        if err != TOKENIZER_BPE_OK:
            err_staged = err
        else:
            err_staged = tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc(
                token_ids,
                token_count,
                cursor_staged,
                blob,
                len(blob),
                offsets,
                lens,
                len(lens),
                out_staged,
                derived_capacity,
                count_staged,
            )

        assert err_wrapped == err_staged
        assert cursor_wrapped[0] == cursor_staged[0]
        assert count_wrapped[0] == count_staged[0]
        assert out_wrapped == out_staged


def run() -> None:
    test_source_contains_preflight_and_wrapper_delegation()
    test_null_cursor_overflow_and_output_ptr_surfaces()
    test_multilingual_and_adversarial_preflight_parity()
    test_randomized_preflight_and_wrapper_parity()
    print("tokenizer_bpe_decode_prompt_checked_default_capacity_validate_cursor_noalloc_from_max_piece_preflight=ok")


if __name__ == "__main__":
    run()
