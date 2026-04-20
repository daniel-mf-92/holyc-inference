#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromLensCommitOnlyRequiredBytesPreflightOnly."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
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
    out_max_piece_len: list[int] | None,
    out_required_merge_workspace_bytes: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_required_token_capacity is None
        or out_max_piece_len is None
        or out_required_merge_workspace_bytes is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if out_token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_required = [0]
    staged_max_piece_len = [0]
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
        staged_required,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required[0] and staged_max_piece_len[0] > I64_MAX // staged_required[0]:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_bytes = staged_required[0] * staged_max_piece_len[0]
    if staged_bytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    out_required_merge_workspace_bytes[0] = staged_bytes
    return TOKENIZER_BPE_OK


def explicit_checked_composition(*args):
    (
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
        out_required_token_capacity,
        out_max_piece_len,
        out_required_merge_workspace_bytes,
    ) = args

    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_required_token_capacity is None
        or out_max_piece_len is None
        or out_required_merge_workspace_bytes is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if out_token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_required = [0]
    staged_max_piece_len = [0]
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
        staged_required,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required[0] and staged_max_piece_len[0] > I64_MAX // staged_required[0]:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_bytes = staged_required[0] * staged_max_piece_len[0]
    if staged_bytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    out_required_merge_workspace_bytes[0] = staged_bytes
    return TOKENIZER_BPE_OK


def test_source_contains_required_bytes_preflight_only_signature_and_atomic_publish() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytes(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnly(" in body
    assert "staged_required_merge_workspace_bytes =" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_max_piece_len = staged_max_piece_len;" in body
    assert "*out_required_merge_workspace_bytes = staged_required_merge_workspace_bytes;" in body


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


def test_success_and_capacity_error_no_write() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    req = [0xAAAA]
    max_piece = [0xBBBB]
    required_bytes = [0xCCCC]
    cursor = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
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
        max_piece,
        required_bytes,
    )
    assert err == TOKENIZER_BPE_OK
    assert req == [len(payload)]
    assert max_piece == [max(vocab_lens)]
    assert required_bytes == [len(payload) * max(vocab_lens)]
    assert cursor == [0]

    req_fail = [0x1111]
    max_fail = [0x2222]
    bytes_fail = [0x3333]
    cursor_fail = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
        payload,
        len(payload),
        cursor_fail,
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
        req_fail,
        max_fail,
        bytes_fail,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert req_fail == [0x1111]
    assert max_fail == [0x2222]
    assert bytes_fail == [0x3333]
    assert cursor_fail == [0]


def test_workspace_bytes_overflow_no_write() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = [65, 66]
    vocab_lens = [I64_MAX]

    req = [0x1111]
    max_piece = [0x2222]
    required_bytes = [0x3333]
    cursor = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
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
        max_piece,
        required_bytes,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert req == [0x1111]
    assert max_piece == [0x2222]
    assert required_bytes == [0x3333]
    assert cursor == [0]


def test_fuzz_parity() -> None:
    random.seed(700)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(128):
        nbytes = random.randint(0, 32)
        payload = [random.randint(0, 255) for _ in range(nbytes)]
        cursor = (
            random.randint(0, nbytes)
            if random.random() > 0.25
            else random.randint(nbytes + 1, nbytes + 3)
        )
        prompt_nbytes = random.randint(0, max(0, nbytes - min(cursor, nbytes)))
        vocab_count = random.randint(0, 8)
        vocab_capacity = vocab_count + random.randint(0, 1)
        if random.random() < 0.1:
            vocab_capacity = max(0, vocab_count - 1)
        vocab_lens = [random.randint(0, 8) for _ in range(vocab_count)]
        out_cap = random.randint(0, nbytes + 2)

        req_a = [0x1234]
        req_b = [0x1234]
        max_a = [0x5678]
        max_b = [0x5678]
        bytes_a = [0x9ABC]
        bytes_b = [0x9ABC]
        cur_a = [cursor]
        cur_b = [cursor]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
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
            max_a,
            bytes_a,
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
            max_b,
            bytes_b,
        )

        assert err_a == err_b
        assert req_a == req_b
        assert max_a == max_b
        assert bytes_a == bytes_b
        assert cur_a == cur_b
