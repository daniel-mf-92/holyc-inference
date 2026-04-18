#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeSingleTokenCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_decode_single_token_checked import (
    build_vocab_tables,
    tokenizer_bpe_decode_single_token_checked,
)
from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_single_token_checked_no_partial(
    token_id: int,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
    out_bytes: list[int] | None,
    out_byte_capacity: int,
    io_out_cursor: list[int] | None,
    out_byte_count: list[int] | None,
) -> int:
    if (
        vocab_piece_bytes is None
        or vocab_piece_offsets is None
        or vocab_piece_lens is None
        or out_bytes is None
        or io_out_cursor is None
        or out_byte_count is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_out_cursor[0]
    staged_cursor = [cursor]
    staged_count = [out_byte_count[0]]
    staged_bytes = [0] * max(out_byte_capacity, 1)

    err = tokenizer_bpe_decode_single_token_checked(
        token_id,
        vocab_piece_bytes,
        vocab_piece_bytes_len,
        vocab_piece_offsets,
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        staged_bytes,
        out_byte_capacity,
        staged_cursor,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if (
        staged_count[0] > out_byte_capacity
        or staged_cursor[0] < cursor
        or staged_cursor[0] > out_byte_capacity
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    committed_count = staged_cursor[0] - cursor
    if committed_count != staged_count[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(committed_count):
        out_bytes[cursor + idx] = staged_bytes[cursor + idx]

    out_byte_count[0] = staged_count[0]
    io_out_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(
    token_id: int,
    pieces: list[bytes],
    out_capacity: int,
    cursor0: int,
    count0: int = 0x6161,
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)

    out_wrapper = [0xA7] * 256
    out_explicit = out_wrapper.copy()

    cursor_wrapper = [cursor0]
    cursor_explicit = [cursor0]

    count_wrapper = [count0]
    count_explicit = [count0]

    err_wrapper = tokenizer_bpe_decode_single_token_checked_no_partial(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        out_wrapper,
        out_capacity,
        cursor_wrapper,
        count_wrapper,
    )

    staged_out = [0] * max(out_capacity, 1)
    staged_cursor = [cursor0]
    staged_count = [count0]

    err_explicit = tokenizer_bpe_decode_single_token_checked(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        staged_out,
        out_capacity,
        staged_cursor,
        staged_count,
    )
    if err_explicit == TOKENIZER_BPE_OK:
        committed = staged_cursor[0] - cursor0
        if committed < 0 or committed != staged_count[0] or staged_cursor[0] > out_capacity:
            err_explicit = TOKENIZER_BPE_ERR_BAD_PARAM
        else:
            for idx in range(committed):
                out_explicit[cursor0 + idx] = staged_out[cursor0 + idx]
            count_explicit[0] = staged_count[0]
            cursor_explicit[0] = staged_cursor[0]

    assert err_wrapper == err_explicit
    assert out_wrapper == out_explicit
    assert count_wrapper[0] == count_explicit[0]
    assert cursor_wrapper[0] == cursor_explicit[0]


def test_multilingual_fixture_parity_vs_explicit_staged_composition() -> None:
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

    run_case(token_id=8, pieces=pieces, out_capacity=256, cursor0=11)
    run_case(token_id=9, pieces=pieces, out_capacity=256, cursor0=0)


def test_error_paths_preserve_cursor_count_and_output() -> None:
    pieces = [b"A", b"BC", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    seed = [0x5A] * 48

    def run_fail(token_id: int, capacity: int, cursor0: int) -> None:
        out = seed.copy()
        count = [0xBEEF]
        cursor = [cursor0]

        err = tokenizer_bpe_decode_single_token_checked_no_partial(
            token_id,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            out,
            capacity,
            cursor,
            count,
        )

        assert err == TOKENIZER_BPE_ERR_BAD_PARAM
        assert out == seed
        assert count[0] == 0xBEEF
        assert cursor[0] == cursor0

    run_fail(-1, len(seed), 0)
    run_fail(99, len(seed), 0)
    run_fail(2, 2, 0)
    run_fail(1, len(seed), len(seed) + 1)


def test_overflow_and_null_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 16
    count = [0x7070]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_single_token_checked_no_partial(
            0,
            None,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            len(out),
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_single_token_checked_no_partial(
            0,
            blob,
            I64_MAX + 1,
            offsets,
            lens,
            1,
            1,
            out,
            len(out),
            cursor,
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


def test_randomized_parity_against_explicit_staged_reference() -> None:
    rng = random.Random(20260418_428)
    pieces = [b"a", b"bb", b"ccc", "Κα".encode("utf-8"), "世界".encode("utf-8"), "🙂".encode("utf-8")]

    for _ in range(6000):
        token_id = rng.randint(-2, len(pieces) + 2)
        out_capacity = rng.randint(0, 128)
        cursor0 = rng.randint(0, 132)
        run_case(token_id=token_id, pieces=pieces, out_capacity=out_capacity, cursor0=cursor0)


if __name__ == "__main__":
    test_multilingual_fixture_parity_vs_explicit_staged_composition()
    test_error_paths_preserve_cursor_count_and_output()
    test_overflow_and_null_contracts()
    test_randomized_parity_against_explicit_staged_reference()
    print("test_tokenizer_bpe_decode_single_token_checked_no_partial: ok")
