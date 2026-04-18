#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeSingleTokenChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)


def tokenizer_bpe_decode_single_token_checked(
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

    if vocab_piece_capacity < vocab_piece_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_cursor = io_out_cursor[0]
    if out_cursor > out_byte_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if token_id < 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    token_id_u64 = token_id
    if token_id_u64 >= vocab_piece_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    piece_offset = vocab_piece_offsets[token_id_u64]
    piece_len = vocab_piece_lens[token_id_u64]

    if piece_offset > vocab_piece_bytes_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if piece_len > vocab_piece_bytes_len - piece_offset:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if piece_len > out_byte_capacity - out_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_out_count = piece_len
    write_cursor = out_cursor

    for idx in range(piece_len):
        out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + idx]
        write_cursor += 1
        if write_cursor <= out_cursor:
            return TOKENIZER_BPE_ERR_OVERFLOW

    if write_cursor != out_cursor + staged_out_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_out_count
    io_out_cursor[0] = write_cursor
    return TOKENIZER_BPE_OK


def build_vocab_tables(pieces: list[bytes]) -> tuple[list[int], list[int], list[int]]:
    blob: list[int] = []
    offsets: list[int] = []
    lens: list[int] = []
    cursor = 0
    for piece in pieces:
        offsets.append(cursor)
        lens.append(len(piece))
        blob.extend(piece)
        cursor += len(piece)
    return blob, offsets, lens


def test_multilingual_known_fixture_byte_exact() -> None:
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

    out = [0xCC] * 256
    out_count = [0x7777]
    out_cursor = [7]

    token_id = 8
    err = tokenizer_bpe_decode_single_token_checked(
        token_id,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        out,
        len(out),
        out_cursor,
        out_count,
    )
    assert err == TOKENIZER_BPE_OK

    expected = pieces[token_id]
    start = 7
    assert out_count[0] == len(expected)
    assert bytes(out[start : start + out_count[0]]) == expected
    assert out_cursor[0] == start + len(expected)


def test_error_paths_preserve_no_partial_cursor_count_and_output() -> None:
    pieces = [b"A", b"BC", "🙂".encode("utf-8")]
    blob, offsets, lens = build_vocab_tables(pieces)

    seed_out = [0xAA] * 32

    def run_fail(token_id: int, cap: int, cursor0: int) -> None:
        out = seed_out.copy()
        count = [0x5151]
        cursor = [cursor0]
        err = tokenizer_bpe_decode_single_token_checked(
            token_id,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            out,
            cap,
            cursor,
            count,
        )
        assert err == TOKENIZER_BPE_ERR_BAD_PARAM
        assert out == seed_out
        assert count[0] == 0x5151
        assert cursor[0] == cursor0

    run_fail(-1, len(seed_out), 0)
    run_fail(99, len(seed_out), 0)
    run_fail(2, 2, 0)
    run_fail(1, len(seed_out), len(seed_out) + 1)


def test_malformed_vocab_tables_rejected_without_output_mutation() -> None:
    pieces = [b"x", b"yz"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x44] * 32
    count = [0xABCD]
    cursor = [0]

    bad_offsets = offsets.copy()
    bad_offsets[1] = len(blob) + 1
    err = tokenizer_bpe_decode_single_token_checked(
        1,
        blob,
        len(blob),
        bad_offsets,
        lens,
        len(pieces),
        len(pieces),
        out,
        len(out),
        cursor,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == [0x44] * 32 and count[0] == 0xABCD and cursor[0] == 0

    bad_lens = lens.copy()
    bad_lens[1] = len(blob) + 3
    err = tokenizer_bpe_decode_single_token_checked(
        1,
        blob,
        len(blob),
        offsets,
        bad_lens,
        len(pieces),
        len(pieces),
        out,
        len(out),
        cursor,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == [0x44] * 32 and count[0] == 0xABCD and cursor[0] == 0


def test_overflow_and_null_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x11] * 8
    count = [0x9999]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_single_token_checked(
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
        tokenizer_bpe_decode_single_token_checked(
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


def test_randomized_parity_against_reference_piece_copy() -> None:
    rng = random.Random(20260418_416)
    pieces = [
        b"a",
        b"bb",
        b"ccc",
        "Κα".encode("utf-8"),
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(5000):
        out = [0xD3] * 128
        count = [0x3333]
        cursor0 = rng.randint(0, 110)
        cursor = [cursor0]

        token = rng.randrange(len(pieces))
        capacity = rng.randint(cursor0, len(out))

        expected = pieces[token]
        if len(expected) > capacity - cursor0:
            err_expected = TOKENIZER_BPE_ERR_BAD_PARAM
        else:
            err_expected = TOKENIZER_BPE_OK

        err = tokenizer_bpe_decode_single_token_checked(
            token,
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

        assert err == err_expected
        if err == TOKENIZER_BPE_OK:
            assert count[0] == len(expected)
            assert bytes(out[cursor0 : cursor0 + count[0]]) == expected
            assert cursor[0] == cursor0 + len(expected)
        else:
            assert count[0] == 0x3333
            assert cursor[0] == cursor0
            assert out == [0xD3] * 128


if __name__ == "__main__":
    test_multilingual_known_fixture_byte_exact()
    test_error_paths_preserve_no_partial_cursor_count_and_output()
    test_malformed_vocab_tables_rejected_without_output_mutation()
    test_overflow_and_null_contracts()
    test_randomized_parity_against_reference_piece_copy()
    print("test_tokenizer_bpe_decode_single_token_checked: ok")
