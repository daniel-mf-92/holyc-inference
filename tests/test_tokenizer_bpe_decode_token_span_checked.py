#!/usr/bin/env python3
"""Parity harness for TokenizerBPEDecodeTokenSpanChecked semantics."""

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


def tokenizer_bpe_decode_token_span_checked(
    token_ids: list[int] | None,
    token_count: int,
    io_token_cursor: list[int] | None,
    span_token_count: int,
    vocab_piece_bytes: list[int] | None,
    vocab_piece_bytes_len: int,
    vocab_piece_offsets: list[int] | None,
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
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
        or span_token_count > I64_MAX
        or vocab_piece_bytes_len > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
        or out_byte_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    token_cursor = io_token_cursor[0]
    if token_cursor > token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if vocab_piece_capacity < vocab_piece_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if span_token_count > token_count - token_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    decode_end = token_cursor + span_token_count
    if decode_end < token_cursor or decode_end > token_count:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_out_count = 0
    for scan in range(token_cursor, decode_end):
        token_id = token_ids[scan]
        if token_id < 0 or token_id >= vocab_piece_count:
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

    write_cursor = 0
    for scan in range(token_cursor, decode_end):
        token_id = token_ids[scan]
        piece_offset = vocab_piece_offsets[token_id]
        piece_len = vocab_piece_lens[token_id]

        for i in range(piece_len):
            out_bytes[write_cursor] = vocab_piece_bytes[piece_offset + i]
            write_cursor += 1
            if write_cursor > staged_out_count:
                return TOKENIZER_BPE_ERR_OVERFLOW

    if write_cursor != staged_out_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_byte_count[0] = staged_out_count
    io_token_cursor[0] = decode_end
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


def decode_reference(
    token_ids: list[int],
    cursor: int,
    span_count: int,
    pieces: list[bytes],
) -> bytes:
    out = bytearray()
    end = cursor + span_count
    for token_id in token_ids[cursor:end]:
        out.extend(pieces[token_id])
    return bytes(out)


def run_success_case(
    token_ids: list[int],
    token_cursor: int,
    span_count: int,
    pieces: list[bytes],
) -> None:
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0xCC] * 512
    out_count = [0x7777]
    cursor = [token_cursor]

    err = tokenizer_bpe_decode_token_span_checked(
        token_ids,
        len(token_ids),
        cursor,
        span_count,
        blob,
        len(blob),
        offsets,
        lens,
        len(pieces),
        len(pieces),
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_OK

    expected = decode_reference(token_ids, token_cursor, span_count, pieces)
    assert out_count[0] == len(expected)
    assert bytes(out[: out_count[0]]) == expected
    assert cursor[0] == token_cursor + span_count


def test_known_multilingual_fixture_byte_exact() -> None:
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

    run_success_case(token_ids, 0, len(token_ids), pieces)
    run_success_case(token_ids, 5, 6, pieces)
    run_success_case(token_ids, 12, 2, pieces)


def test_capacity_and_cursor_guards_preserve_no_partial_outputs() -> None:
    pieces = [b"A", b"BC", b"DEF"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0xAA] * 8
    out_count = [0x5555]
    cursor = [1]

    err = tokenizer_bpe_decode_token_span_checked(
        [0, 1, 2],
        3,
        cursor,
        2,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        out,
        2,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 1
    assert out_count[0] == 0x5555
    assert out == [0xAA] * 8

    err = tokenizer_bpe_decode_token_span_checked(
        [0, 1, 2],
        3,
        [4],
        0,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM


def test_rejects_malformed_token_and_vocab_tables() -> None:
    pieces = [b"a", b"bb", b"ccc"]
    blob, offsets, lens = build_vocab_tables(pieces)

    out = [0x11] * 16
    out_count = [0x2222]
    cursor = [0]

    err = tokenizer_bpe_decode_token_span_checked(
        [0, -1, 2],
        3,
        cursor,
        3,
        blob,
        len(blob),
        offsets,
        lens,
        3,
        3,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and out_count[0] == 0x2222 and out == [0x11] * 16

    bad_offsets = offsets[:]
    bad_offsets[1] = len(blob) + 1
    err = tokenizer_bpe_decode_token_span_checked(
        [1],
        1,
        cursor,
        1,
        blob,
        len(blob),
        bad_offsets,
        lens,
        3,
        3,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and out_count[0] == 0x2222 and out == [0x11] * 16

    bad_lens = lens[:]
    bad_lens[2] = len(blob) + 5
    err = tokenizer_bpe_decode_token_span_checked(
        [2],
        1,
        cursor,
        1,
        blob,
        len(blob),
        offsets,
        bad_lens,
        3,
        3,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0 and out_count[0] == 0x2222 and out == [0x11] * 16


def test_zero_span_commits_empty_decode_without_writing_bytes() -> None:
    pieces = [b"alpha"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0x9A] * 12
    out_count = [0xABCD]
    cursor = [0]

    err = tokenizer_bpe_decode_token_span_checked(
        [0],
        1,
        cursor,
        0,
        blob,
        len(blob),
        offsets,
        lens,
        1,
        1,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == 0
    assert out_count[0] == 0
    assert out == [0x9A] * 12


def test_randomized_reference_decoder_parity() -> None:
    random.seed(410)

    pieces = [
        b"h",
        b"e",
        b"ll",
        b"o",
        b" ",
        "世界".encode("utf-8"),
        "🙂".encode("utf-8"),
        b"!",
        b"\n",
    ]
    blob, offsets, lens = build_vocab_tables(pieces)

    for _ in range(200):
        token_ids = [random.randrange(len(pieces)) for _ in range(random.randint(1, 32))]
        cursor = random.randrange(len(token_ids) + 1)
        span_count = random.randrange(len(token_ids) - cursor + 1)

        out = [0x42] * 512
        out_count = [0x9009]
        cursor_box = [cursor]

        err = tokenizer_bpe_decode_token_span_checked(
            token_ids,
            len(token_ids),
            cursor_box,
            span_count,
            blob,
            len(blob),
            offsets,
            lens,
            len(pieces),
            len(pieces),
            out,
            len(out),
            out_count,
        )
        assert err == TOKENIZER_BPE_OK

        expected = decode_reference(token_ids, cursor, span_count, pieces)
        assert out_count[0] == len(expected)
        assert bytes(out[: out_count[0]]) == expected
        assert cursor_box[0] == cursor + span_count


def test_null_and_overflow_contracts() -> None:
    pieces = [b"x"]
    blob, offsets, lens = build_vocab_tables(pieces)
    out = [0] * 4
    count = [0]
    cursor = [0]

    assert (
        tokenizer_bpe_decode_token_span_checked(
            None,
            0,
            cursor,
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            len(out),
            count,
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    assert (
        tokenizer_bpe_decode_token_span_checked(
            [0],
            I64_MAX + 1,
            cursor,
            0,
            blob,
            len(blob),
            offsets,
            lens,
            1,
            1,
            out,
            len(out),
            count,
        )
        == TOKENIZER_BPE_ERR_OVERFLOW
    )


if __name__ == "__main__":
    test_known_multilingual_fixture_byte_exact()
    test_capacity_and_cursor_guards_preserve_no_partial_outputs()
    test_rejects_malformed_token_and_vocab_tables()
    test_zero_span_commits_empty_decode_without_writing_bytes()
    test_randomized_reference_decoder_parity()
    test_null_and_overflow_contracts()
    print("ok")
