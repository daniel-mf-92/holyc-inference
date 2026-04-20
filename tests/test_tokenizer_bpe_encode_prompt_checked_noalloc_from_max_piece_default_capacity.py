#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacity (IQ-685)."""

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
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    prompt_nbytes: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    max_piece_len: int,
    out_token_ids: list[int] | None,
    out_token_count: list[int] | None,
    out_required_token_capacity: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_token_ids is None
        or out_token_count is None
        or out_required_token_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or max_piece_len > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_token_capacity = prompt_nbytes

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
        data,
        byte_len,
        io_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        max_piece_len,
        out_token_ids,
        derived_out_token_capacity,
        out_token_count,
        out_required_token_capacity,
    )


def explicit_checked_composition(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    prompt_nbytes: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    max_piece_len: int,
    out_token_ids: list[int] | None,
    out_token_count: list[int] | None,
    out_required_token_capacity: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_token_ids is None
        or out_token_count is None
        or out_required_token_capacity is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or max_piece_len > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_token_capacity = prompt_nbytes

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
        data,
        byte_len,
        io_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        max_piece_len,
        out_token_ids,
        derived_out_token_capacity,
        out_token_count,
        out_required_token_capacity,
    )


def _build_rank_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    entries = sorted(
        [
            (108, 108, 1, 300),
            (200, 300, 2, 400),
            (104, 101, 3, 200),
            (400, 111, 0, 500),
            (119, 111, 1, 210),
            (210, 114, 2, 220),
            (220, 108, 3, 230),
            (230, 100, 0, 501),
            (49, 50, 1, 310),
            (310, 51, 0, 502),
            (103, 111, 0, 503),
        ],
        key=lambda item: (item[0], item[1]),
    )
    left = [item[0] for item in entries]
    right = [item[1] for item in entries]
    ranks = [item[2] for item in entries]
    merged = [item[3] for item in entries]
    return left, right, ranks, merged


def test_source_contains_max_piece_default_capacity_wrapper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")

    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacity("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "derived_out_token_capacity = prompt_nbytes;" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(" in body


def test_success_fixture_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()

    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))

    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x5151] * 512
    out_b = [0x5151] * 512
    count_a = [0xABCD]
    count_b = [0xABCD]
    req_a = [0xDEAD]
    req_b = [0xDEAD]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
        payload,
        len(payload),
        cursor_a,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_a,
        count_a,
        req_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cursor_b,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_b,
        count_b,
        req_b,
    )

    assert err_a == err_b
    assert cursor_a == cursor_b
    assert count_a == count_b
    assert req_a == req_b
    assert out_a == out_b


def test_randomized_parity() -> None:
    rng = random.Random(20260420_685)

    for _ in range(1800):
        payload_len = rng.randint(0, 96)
        payload = [rng.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor_seed)

        rank_count = rng.randint(0, 48)
        pairs = []
        for _ in range(rank_count):
            left = rng.randint(0, 255)
            right = rng.randint(0, 255)
            rank = rng.randint(0, 32)
            merged = rng.randint(256, 4096)
            pairs.append((left, right, rank, merged))
        pairs.sort(key=lambda item: (item[0], item[1]))

        rank_left = [item[0] for item in pairs]
        rank_right = [item[1] for item in pairs]
        rank_values = [item[2] for item in pairs]
        rank_merged = [item[3] for item in pairs]

        rank_capacity = rank_count + rng.randint(0, 5)
        max_piece_len = rng.randint(0, 16)

        out_a = [0x61] * 256
        out_b = [0x61] * 256
        count_a = [0x1234]
        count_b = [0x1234]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]
        req_a = [0xABC]
        req_b = [0xABC]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
            payload,
            payload_len,
            cursor_a,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_capacity,
            max_piece_len,
            out_a,
            count_a,
            req_a,
        )

        err_b = explicit_checked_composition(
            payload,
            payload_len,
            cursor_b,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_values,
            rank_merged,
            rank_count,
            rank_capacity,
            max_piece_len,
            out_b,
            count_b,
            req_b,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert count_a == count_b
        assert req_a == req_b
        assert out_a == out_b


def test_adversarial_utf8_cursor_max_piece_vectors() -> None:
    left, right, ranks, merged = _build_rank_tables()

    vectors = [
        # Valid UTF-8 prompt with mixed-width scalars.
        ("ASCII Καλημέρα 世界 😊".encode("utf-8"), 0, 8),
        # Cursor lands on continuation byte.
        ("Ωmega".encode("utf-8"), 1, 4),
        # Prompt length exactly to end from non-zero cursor.
        ("hello世界".encode("utf-8"), 5, 6),
        # Zero-length prompt from middle cursor.
        ("abcdef".encode("utf-8"), 3, 1),
        # Invalid UTF-8 byte payload still must parity-match error behavior.
        (bytes([0xF0, 0x28, 0x8C, 0x28, 0x61, 0x62]), 0, 5),
    ]

    for payload_bytes, cursor_seed, max_piece_len in vectors:
        payload = list(payload_bytes)
        byte_len = len(payload)
        prompt_nbytes = byte_len - cursor_seed

        out_a = [0x88] * 256
        out_b = [0x88] * 256
        count_a = [0x4444]
        count_b = [0x4444]
        req_a = [0x9999]
        req_b = [0x9999]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
            payload,
            byte_len,
            cursor_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            max_piece_len,
            out_a,
            count_a,
            req_a,
        )
        err_b = explicit_checked_composition(
            payload,
            byte_len,
            cursor_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            max_piece_len,
            out_b,
            count_b,
            req_b,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert count_a == count_b
        assert req_a == req_b
        assert out_a == out_b


def test_no_partial_commit_on_capacity_or_bounds_error() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("token stream".encode("utf-8"))

    cursor_a = [2]
    cursor_b = [2]
    count_a = [0xBEEF]
    count_b = [0xBEEF]
    req_a = [0xCAFE]
    req_b = [0xCAFE]
    out_a = [0x5A] * 32
    out_b = [0x5A] * 32

    # Force deterministic out-of-bounds prompt slice.
    prompt_nbytes = len(payload) - 1
    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
        payload,
        len(payload),
        cursor_a,
        prompt_nbytes,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_a,
        count_a,
        req_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cursor_b,
        prompt_nbytes,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_b,
        count_b,
        req_b,
    )

    assert err_a == err_b == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor_a == cursor_b == [2]
    assert count_a == count_b == [0xBEEF]
    assert out_a == out_b == [0x5A] * 32

    # Wrapper path retains no-partial semantics on success-capacity buffers.
    cursor_a = [0]
    cursor_b = [0]
    count_a = [0xAAAA]
    count_b = [0xAAAA]
    req_a = [0x1111]
    req_b = [0x1111]
    out_a = [0x7C] * 32
    out_b = [0x7C] * 32

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
        payload,
        len(payload),
        cursor_a,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_a,
        count_a,
        req_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cursor_b,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        8,
        out_b,
        count_b,
        req_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert req_a[0] == req_b[0] == len(payload)
    assert cursor_a == cursor_b == [len(payload)]
    assert count_a == count_b
    assert out_a == out_b


def test_max_piece_overflow_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = [ord("x")] * 4
    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x20] * 16
    out_b = [0x20] * 16
    count_a = [0x22]
    count_b = [0x22]
    req_a = [0x33]
    req_b = [0x33]

    huge = I64_MAX

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
        payload,
        len(payload),
        cursor_a,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        huge,
        out_a,
        count_a,
        req_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cursor_b,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        huge,
        out_b,
        count_b,
        req_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_ERR_OVERFLOW
    assert cursor_a == cursor_b == [0]
    assert count_a == count_b == [0x22]
    assert out_a == out_b == [0x20] * 16


def test_empty_prompt_success_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abc".encode("utf-8"))
    cursor_a = [2]
    cursor_b = [2]
    out_a = [0x44] * 8
    out_b = [0x44] * 8
    count_a = [0x2A2A]
    count_b = [0x2A2A]
    req_a = [0x1717]
    req_b = [0x1717]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity(
        payload,
        len(payload),
        cursor_a,
        0,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        0,
        out_a,
        count_a,
        req_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cursor_b,
        0,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        0,
        out_b,
        count_b,
        req_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert cursor_a == cursor_b == [2]
    assert count_a == count_b == [0]
    assert req_a == req_b == [0]
    assert out_a == out_b == [0x44] * 8


if __name__ == "__main__":
    test_source_contains_max_piece_default_capacity_wrapper()
    test_success_fixture_parity()
    test_randomized_parity()
    test_adversarial_utf8_cursor_max_piece_vectors()
    test_no_partial_commit_on_capacity_or_bounds_error()
    test_max_piece_overflow_parity()
    test_empty_prompt_success_parity()
    print("tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity=ok")
