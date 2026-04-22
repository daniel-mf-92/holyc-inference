#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnlyParity."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(
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

    snapshot_cursor = io_cursor[0]

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

    staged_cursor = [snapshot_cursor]
    staged_tokens = [0] * max(1, out_token_capacity)
    staged_token_count = [0]
    staged_required = [0]
    staged_max_piece = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
        data,
        byte_len,
        staged_cursor,
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
        staged_tokens,
        out_token_capacity,
        staged_token_count,
        staged_required,
        staged_max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if snapshot_cursor > byte_len or prompt_nbytes > byte_len - snapshot_cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    staged_expected_next_cursor = snapshot_cursor + prompt_nbytes

    if staged_required[0] > I64_MAX // 4:
        return TOKENIZER_BPE_ERR_OVERFLOW
    staged_required_bytes = staged_required[0] * 4

    if (
        staged_required[0] != canonical_required[0]
        or staged_required_bytes != canonical_required_bytes
        or staged_max_piece[0] != canonical_max_piece[0]
        or staged_cursor[0] != staged_expected_next_cursor
        or staged_required[0] > out_token_capacity
        or staged_token_count[0] > out_token_capacity
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if io_cursor[0] != snapshot_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = canonical_required[0]
    out_required_token_bytes[0] = canonical_required_bytes
    out_next_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def explicit_checked_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(*args)


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


def test_source_contains_signature_and_required_composition() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytes(",
        1,
    )[0]

    assert "IQ-1047 diagnostics parity gate for no-alloc lens encode preflight/commit." in source
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnly(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnly(" in body
    assert "if (out_required_token_capacity_ptr == out_required_token_bytes_ptr ||" in body
    assert "canonical_required_token_bytes = canonical_required_token_capacity * sizeof(I32);" in body
    assert "staged_required_token_bytes = staged_required_token_capacity * sizeof(I32);" in body
    assert "staged_cursor != staged_expected_next_cursor" in body
    assert "if (*io_cursor != snapshot_cursor)" in body
    assert "*out_required_token_capacity = canonical_required_token_capacity;" in body
    assert "*out_required_token_bytes = canonical_required_token_bytes;" in body
    assert "*out_next_cursor = staged_cursor;" in body


def test_success_and_no_partial_failure_paths() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    req = [0xAAAA]
    req_bytes = [0xBBBB]
    next_cursor = [0xCCCC]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(
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

    req_fail = [0x1111]
    req_bytes_fail = [0x2222]
    next_cursor_fail = [0x3333]
    cursor_fail = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(
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
        req_bytes_fail,
        next_cursor_fail,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert req_fail == [0x1111]
    assert req_bytes_fail == [0x2222]
    assert next_cursor_fail == [0x3333]
    assert cursor_fail == [0]


def test_alias_guard() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abc".encode("utf-8"))
    vocab_lens = [1, 2]

    alias = [777]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(
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
        alias,
        alias,
        [888],
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert alias == [777]
    assert cursor == [0]


def test_randomized_parity_vs_explicit_composition() -> None:
    rng = random.Random(20260422_1047)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(400):
        nbytes = rng.randint(0, 32)
        payload = [rng.randint(0, 255) for _ in range(nbytes)]

        if rng.random() < 0.22:
            cursor_val = rng.randint(nbytes + 1, nbytes + 4)
        else:
            cursor_val = rng.randint(0, nbytes)

        if cursor_val <= nbytes:
            prompt_nbytes = rng.randint(0, nbytes - cursor_val)
        else:
            prompt_nbytes = rng.randint(0, nbytes)

        vocab_count = rng.randint(0, 8)
        vocab_capacity = vocab_count + rng.randint(0, 1)
        if rng.random() < 0.12:
            vocab_capacity = max(0, vocab_count - 1)
        vocab_lens = [rng.randint(0, 8) for _ in range(vocab_count)]

        out_cap = rng.randint(0, nbytes + 2)

        req_a = [111]
        reqb_a = [222]
        next_a = [333]
        req_b = [111]
        reqb_b = [222]
        next_b = [333]
        cur_a = [cursor_val]
        cur_b = [cursor_val]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity(
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
            reqb_a,
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
            reqb_b,
            next_b,
        )

        assert err_a == err_b
        assert cur_a == cur_b
        assert req_a == req_b
        assert reqb_a == reqb_b
        assert next_a == next_b


if __name__ == "__main__":
    test_source_contains_signature_and_required_composition()
    test_success_and_no_partial_failure_paths()
    test_alias_guard()
    test_randomized_parity_vs_explicit_composition()
    print("tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity=ok")
