#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromLensCommitOnlyRequiredBytesDefaultCapacity."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_prompt_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
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

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    derived_out_token_capacity = prompt_nbytes

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes(
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
        derived_out_token_capacity,
        out_required_token_capacity,
        out_max_piece_len,
        out_required_merge_workspace_bytes,
    )


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

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or vocab_piece_count > I64_MAX
        or vocab_piece_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes(
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
        prompt_nbytes,
        out_required_token_capacity,
        out_max_piece_len,
        out_required_merge_workspace_bytes,
    )


def test_source_contains_default_capacity_required_bytes_signature() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesDefaultCapacity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnly(", 1
    )[0]
    assert "derived_out_token_capacity = prompt_nbytes;" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytes(" in body


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


def test_success_and_overflow_guard_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    req_a = [0xAAAA]
    max_a = [0xBBBB]
    bytes_a = [0xCCCC]
    cur_a = [0]

    req_b = [0xAAAA]
    max_b = [0xBBBB]
    bytes_b = [0xCCCC]
    cur_b = [0]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
        payload,
        len(payload),
        cur_a,
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
        req_a,
        max_a,
        bytes_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cur_b,
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
        req_b,
        max_b,
        bytes_b,
    )

    assert err_a == TOKENIZER_BPE_OK
    assert err_a == err_b
    assert req_a == req_b == [len(payload)]
    assert max_a == max_b == [max(vocab_lens)]
    assert bytes_a == bytes_b == [len(payload) * max(vocab_lens)]
    assert cur_a == cur_b == [0]


def test_null_and_domain_overflow() -> None:
    left, right, ranks, merged = _build_rank_tables()
    vocab_lens = [1, 2, 3]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
        None,
        0,
        [0],
        0,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
        [65],
        I64_MAX + 1,
        [0],
        1,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        [0x11],
        [0x22],
        [0x33],
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW


def test_fuzz_parity() -> None:
    random.seed(699)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(160):
        nbytes = random.randint(0, 48)
        payload = [random.randint(0, 255) for _ in range(nbytes)]
        cursor = (
            random.randint(0, nbytes)
            if random.random() > 0.30
            else random.randint(nbytes + 1, nbytes + 3)
        )
        prompt_nbytes = random.randint(0, max(0, nbytes - min(cursor, nbytes)))
        vocab_count = random.randint(0, 10)
        vocab_capacity = vocab_count + random.randint(0, 1)
        if random.random() < 0.10:
            vocab_capacity = max(0, vocab_count - 1)
        vocab_lens = [random.randint(0, 16) for _ in range(vocab_count)]

        req_a = [0x1111]
        req_b = [0x1111]
        max_a = [0x2222]
        max_b = [0x2222]
        bytes_a = [0x3333]
        bytes_b = [0x3333]
        cur_a = [cursor]
        cur_b = [cursor]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
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
            req_b,
            max_b,
            bytes_b,
        )

        assert err_a == err_b
        assert req_a == req_b
        assert max_a == max_b
        assert bytes_a == bytes_b
        assert cur_a == cur_b
