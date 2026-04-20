#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesCommitOnly."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only(
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
    out_token_ids: list[int] | None,
    out_token_capacity: int,
    required_token_capacity: int,
    required_max_piece_len: int,
    required_merge_workspace_bytes: int,
    out_token_count: list[int] | None,
    out_required_token_capacity: list[int] | None,
    out_max_piece_len: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_token_ids is None
        or out_token_count is None
        or out_required_token_capacity is None
        or out_max_piece_len is None
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
        or required_token_capacity > I64_MAX
        or required_max_piece_len > I64_MAX
        or required_merge_workspace_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    canonical_required = [0]
    canonical_max_piece = [0]
    canonical_required_bytes = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes(
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
        canonical_required_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if required_token_capacity != canonical_required[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if required_max_piece_len != canonical_max_piece[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if required_merge_workspace_bytes != canonical_required_bytes[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
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
        out_token_ids,
        out_token_capacity,
        out_token_count,
        out_required_token_capacity,
        out_max_piece_len,
    )


def explicit_checked_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only(
        *args
    )


def test_source_contains_required_bytes_commit_only_signature_and_parity_gates() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesDefaultCapacity(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesPreflightOnly(" in body
    assert "if (required_token_capacity != canonical_required_token_capacity)" in body
    assert "if (required_max_piece_len != canonical_max_piece_len)" in body
    assert "if (required_merge_workspace_bytes != canonical_required_merge_workspace_bytes)" in body
    assert "return TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnly(" in body


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


def _canonical_diags(
    payload: list[int],
    prompt_nbytes: int,
    left: list[int],
    right: list[int],
    ranks: list[int],
    merged: list[int],
    vocab_lens: list[int],
    out_token_capacity: int,
) -> tuple[int, int, int]:
    req = [0]
    max_piece = [0]
    req_bytes = [0]
    cursor = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes(
        payload,
        len(payload),
        cursor,
        prompt_nbytes,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_token_capacity,
        req,
        max_piece,
        req_bytes,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor == [0]
    return req[0], max_piece[0], req_bytes[0]


def test_success_fixture_parity_and_atomic_commit() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂 xyz".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    out_capacity = len(payload)
    req, max_piece, req_bytes = _canonical_diags(
        payload, len(payload), left, right, ranks, merged, vocab_lens, out_capacity
    )

    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x1111] * 512
    out_b = [0x1111] * 512
    count_a = [0x1234]
    count_b = [0x1234]
    req_a = [0x5678]
    req_b = [0x5678]
    max_a = [0x7777]
    max_b = [0x7777]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only(
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
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_a,
        out_capacity,
        req,
        max_piece,
        req_bytes,
        count_a,
        req_a,
        max_a,
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
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_b,
        out_capacity,
        req,
        max_piece,
        req_bytes,
        count_b,
        req_b,
        max_b,
    )

    assert err_a == TOKENIZER_BPE_OK
    assert err_b == TOKENIZER_BPE_OK
    assert err_a == err_b
    assert cursor_a == cursor_b
    assert count_a == count_b
    assert req_a == req_b
    assert max_a == max_b
    assert out_a[: count_a[0]] == out_b[: count_b[0]]


def test_diag_mismatch_is_no_partial_for_outputs() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abcdef".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    out = [0x9999] * 64
    cursor = [0]
    token_count = [0x1111]
    required = [0x2222]
    max_piece = [0x3333]

    req, mp, req_bytes = _canonical_diags(
        payload, len(payload), left, right, ranks, merged, vocab_lens, len(payload)
    )

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only(
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
        out,
        len(payload),
        req + 1,
        mp,
        req_bytes,
        token_count,
        required,
        max_piece,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor == [0]
    assert token_count == [0x1111]
    assert required == [0x2222]
    assert max_piece == [0x3333]
    assert out == [0x9999] * 64


def test_fuzz_parity() -> None:
    random.seed(701)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(128):
        nbytes = random.randint(0, 48)
        payload = [random.randint(0, 255) for _ in range(nbytes)]
        cursor = random.randint(0, nbytes)
        prompt = random.randint(0, nbytes - cursor)

        piece_count = random.randint(1, 12)
        vocab_lens = [random.randint(1, 16) for _ in range(piece_count)]

        out_capacity = random.randint(max(1, prompt), prompt + 8)
        out_len = max(1, out_capacity, prompt + 8)
        out_a = [0xA5A5] * out_len
        out_b = [0xA5A5] * out_len

        req = [0]
        max_piece = [0]
        req_bytes = [0]
        pre_cursor = [cursor]
        err_diag = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes(
            payload,
            nbytes,
            pre_cursor,
            prompt,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            vocab_lens,
            piece_count,
            piece_count,
            out_capacity,
            req,
            max_piece,
            req_bytes,
        )
        if err_diag != TOKENIZER_BPE_OK:
            continue

        cursor_a = [cursor]
        cursor_b = [cursor]
        count_a = [random.randint(0, 10)]
        count_b = [count_a[0]]
        req_a = [0x1234]
        req_b = [0x1234]
        max_a = [0x5678]
        max_b = [0x5678]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only(
            payload,
            nbytes,
            cursor_a,
            prompt,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            vocab_lens,
            piece_count,
            piece_count,
            out_a,
            out_capacity,
            req[0],
            max_piece[0],
            req_bytes[0],
            count_a,
            req_a,
            max_a,
        )
        err_b = explicit_checked_composition(
            payload,
            nbytes,
            cursor_b,
            prompt,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            vocab_lens,
            piece_count,
            piece_count,
            out_b,
            out_capacity,
            req[0],
            max_piece[0],
            req_bytes[0],
            count_b,
            req_b,
            max_b,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert count_a == count_b
        assert req_a == req_b
        assert max_a == max_b
        assert out_a == out_b


if __name__ == "__main__":
    test_source_contains_required_bytes_commit_only_signature_and_parity_gates()
    test_success_fixture_parity_and_atomic_commit()
    test_diag_mismatch_is_no_partial_for_outputs()
    test_fuzz_parity()
    print("ok")
