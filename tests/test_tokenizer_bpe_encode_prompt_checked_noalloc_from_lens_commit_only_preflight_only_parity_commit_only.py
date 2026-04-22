#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyPreflightOnlyParityCommitOnly (IQ-1054)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only(
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
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
    out_token_capacity: int,
    out_required_token_capacity: list[int] | None,
    out_required_token_bytes: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_required_token_capacity is None
        or out_required_token_bytes is None
        or out_next_cursor is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        out_required_token_capacity is out_required_token_bytes
        or out_required_token_capacity is out_next_cursor
        or out_required_token_bytes is out_next_cursor
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
        or out_token_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    snapshot_data = data
    snapshot_byte_len = byte_len
    snapshot_cursor = io_cursor[0]

    staged_required = [0]
    staged_required_bytes = [0]
    staged_next_cursor = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(
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
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        out_token_capacity,
        staged_required,
        staged_required_bytes,
        staged_next_cursor,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    canonical_required = [0]
    canonical_max_piece = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only(
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
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        out_token_capacity,
        canonical_required,
        canonical_max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if canonical_required[0] > I64_MAX // 4:
        return TOKENIZER_BPE_ERR_OVERFLOW
    canonical_required_bytes = canonical_required[0] * 4

    if snapshot_cursor > byte_len or prompt_nbytes > (byte_len - snapshot_cursor):
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    canonical_next_cursor = snapshot_cursor + prompt_nbytes

    if (
        staged_required[0] != canonical_required[0]
        or staged_required_bytes[0] != canonical_required_bytes
        or staged_next_cursor[0] != canonical_next_cursor
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if snapshot_data is not data or snapshot_byte_len != byte_len or snapshot_cursor != io_cursor[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if canonical_max_piece[0] > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = canonical_required[0]
    out_required_token_bytes[0] = canonical_required_bytes
    out_next_cursor[0] = canonical_next_cursor
    return TOKENIZER_BPE_OK


def explicit_checked_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only(*args)


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
        ],
        key=lambda item: (item[0], item[1]),
    )
    left = [item[0] for item in entries]
    right = [item[1] for item in entries]
    ranks = [item[2] for item in entries]
    merged = [item[3] for item in entries]
    return left, right, ranks, merged


def test_source_contains_signature_and_composition_chain() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = (
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens"
        "CommitOnlyPreflightOnlyParityCommitOnly("
    )
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytes(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnlyParity(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnly(" in body
    assert "if (snapshot_bytes != bytes || snapshot_byte_len != byte_len ||" in body


def test_known_vector() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    req = [0xAAAA]
    req_bytes = [0xBBBB]
    next_cursor = [0xCCCC]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only(
        payload,
        len(payload),
        cursor,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        len(payload),
        req,
        req_bytes,
        next_cursor,
    )

    assert err == TOKENIZER_BPE_OK
    assert req == [len(payload)]
    assert req_bytes == [len(payload) * 4]
    assert next_cursor == [len(payload)]
    assert cursor == [0]


def test_error_no_write() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    req = [0x1111]
    req_bytes = [0x2222]
    next_cursor = [0x3333]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only(
        payload,
        len(payload),
        cursor,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        len(payload) - 1,
        req,
        req_bytes,
        next_cursor,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert req == [0x1111]
    assert req_bytes == [0x2222]
    assert next_cursor == [0x3333]
    assert cursor == [0]


def test_fuzz_parity() -> None:
    random.seed(20260422_1054)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(256):
        nbytes = random.randint(0, 48)
        payload = [random.randint(0, 255) for _ in range(nbytes)]
        cursor = random.randint(0, nbytes) if random.random() > 0.2 else random.randint(nbytes + 1, nbytes + 3)
        prompt_nbytes = random.randint(0, max(0, nbytes - min(cursor, nbytes)))

        vocab_count = random.randint(0, 10)
        vocab_capacity = vocab_count + random.randint(0, 2)
        if random.random() < 0.1:
            vocab_capacity = max(0, vocab_count - 1)
        vocab_lens = [random.randint(0, 10) for _ in range(vocab_count)]

        out_cap = random.randint(0, nbytes + 3)

        req_a = [0x1234]
        req_b = [0x1234]
        req_bytes_a = [0x5678]
        req_bytes_b = [0x5678]
        next_a = [0x9ABC]
        next_b = [0x9ABC]
        cur_a = [cursor]
        cur_b = [cursor]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only(
            payload,
            nbytes,
            cur_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            vocab_lens,
            vocab_count,
            vocab_capacity,
            out_cap,
            req_a,
            req_bytes_a,
            next_a,
        )

        err_b = explicit_checked_composition(
            payload,
            nbytes,
            cur_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            vocab_lens,
            vocab_count,
            vocab_capacity,
            out_cap,
            req_b,
            req_bytes_b,
            next_b,
        )

        assert err_a == err_b
        assert req_a == req_b
        assert req_bytes_a == req_bytes_b
        assert next_a == next_b
        assert cur_a == cur_b
