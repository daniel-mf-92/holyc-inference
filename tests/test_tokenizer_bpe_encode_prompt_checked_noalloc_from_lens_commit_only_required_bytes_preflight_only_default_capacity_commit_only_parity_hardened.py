#!/usr/bin/env python3
"""Parity harness for ...PreflightOnlyDefaultCapacityCommitOnlyParityHardened."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity_preflight_only_noalloc import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity_preflight_only_noalloc,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity_commit_only_parity_hardened(
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

    initial_cursor = io_cursor[0]
    canonical_required = [0]
    canonical_max_piece = [0]
    canonical_required_bytes = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
        data,
        byte_len,
        [initial_cursor],
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
        canonical_required,
        canonical_max_piece,
        canonical_required_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err
    parity_required = [0]
    parity_max_piece = [0]
    parity_required_bytes = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity_preflight_only_noalloc(
        data,
        byte_len,
        [initial_cursor],
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
        parity_required,
        parity_max_piece,
        parity_required_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err
    if initial_cursor != io_cursor[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if parity_required[0] != canonical_required[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if parity_max_piece[0] != canonical_max_piece[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if parity_required_bytes[0] != canonical_required_bytes[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = canonical_required[0]
    out_max_piece_len[0] = canonical_max_piece[0]
    out_required_merge_workspace_bytes[0] = canonical_required_bytes[0]
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

    canonical_required = [0]
    canonical_max_piece = [0]
    canonical_required_bytes = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
        data,
        byte_len,
        [io_cursor[0]],
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
        canonical_required,
        canonical_max_piece,
        canonical_required_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    parity_required = [0]
    parity_max_piece = [0]
    parity_required_bytes = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity_preflight_only_noalloc(
        data,
        byte_len,
        [io_cursor[0]],
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
        parity_required,
        parity_max_piece,
        parity_required_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if parity_required[0] != canonical_required[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if parity_max_piece[0] != canonical_max_piece[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if parity_required_bytes[0] != canonical_required_bytes[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = canonical_required[0]
    out_max_piece_len[0] = canonical_max_piece[0]
    out_required_merge_workspace_bytes[0] = canonical_required_bytes[0]
    return TOKENIZER_BPE_OK


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


def test_source_contains_signature_and_hardening_calls() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesPreflightOnlyDefaultCapacityCommitOnlyParityHardened("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesPreflightOnlyDefaultCapacity(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesDefaultCapacity(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesCommitOnlyDefaultCapacityPreflightOnly(" in body
    assert "initial_cursor = *io_cursor;" in body
    assert "staged_cursor = initial_cursor;" in body
    assert "if (staged_cursor != initial_cursor)" in body
    assert "if (parity_cursor != initial_cursor)" in body
    assert "if (parity_required_token_capacity != canonical_required_token_capacity)" in body
    assert "*out_required_merge_workspace_bytes = canonical_required_merge_workspace_bytes;" in body


def test_success_error_no_write_parity_and_sentinel_preservation() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂 xyz".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    req_a = [0x1111]
    max_a = [0x2222]
    bytes_a = [0x3333]
    req_b = [0x1111]
    max_b = [0x2222]
    bytes_b = [0x3333]
    cur_a = [3]
    cur_b = [3]

    good_prompt_nbytes = len(payload) - cur_a[0]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity_commit_only_parity_hardened(
        payload,
        len(payload),
        cur_a,
        good_prompt_nbytes,
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
        good_prompt_nbytes,
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
    assert cur_a == cur_b == [3]
    assert req_a == req_b
    assert max_a == max_b
    assert bytes_a == bytes_b

    bad_req_a = [0xAAAA]
    bad_max_a = [0xBBBB]
    bad_bytes_a = [0xCCCC]
    bad_req_b = [0xAAAA]
    bad_max_b = [0xBBBB]
    bad_bytes_b = [0xCCCC]
    bad_cur_a = [3]
    bad_cur_b = [3]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity_commit_only_parity_hardened(
        payload,
        len(payload),
        bad_cur_a,
        good_prompt_nbytes + 1,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        bad_req_a,
        bad_max_a,
        bad_bytes_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        bad_cur_b,
        good_prompt_nbytes + 1,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        bad_req_b,
        bad_max_b,
        bad_bytes_b,
    )

    assert err_a == err_b
    assert err_a != TOKENIZER_BPE_OK
    assert bad_cur_a == bad_cur_b == [3]
    assert bad_req_a == bad_req_b == [0xAAAA]
    assert bad_max_a == bad_max_b == [0xBBBB]
    assert bad_bytes_a == bad_bytes_b == [0xCCCC]


def test_null_overflow_and_fuzz_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity_commit_only_parity_hardened(
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
        [1, 2, 5],
        3,
        3,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity_commit_only_parity_hardened(
        [],
        I64_MAX + 1,
        [0],
        0,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        [1, 2, 5],
        3,
        3,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    rng = random.Random(705)
    for _ in range(250):
        payload_len = rng.randint(0, 48)
        payload = [rng.randint(0, 255) for _ in range(payload_len)]
        cursor0 = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor0)

        vocab_len = rng.randint(1, 14)
        vocab_lens = [rng.randint(0, 12) for _ in range(vocab_len)]
        rank_count = len(ranks)
        rank_capacity = rank_count + rng.randint(0, 4)
        vocab_capacity = vocab_len + rng.randint(0, 4)

        req_a = [0x111]
        max_a = [0x222]
        bytes_a = [0x333]
        req_b = [0x111]
        max_b = [0x222]
        bytes_b = [0x333]
        cur_a = [cursor0]
        cur_b = [cursor0]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity_commit_only_parity_hardened(
            payload,
            payload_len,
            cur_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_count,
            rank_capacity,
            vocab_lens,
            vocab_len,
            vocab_capacity,
            req_a,
            max_a,
            bytes_a,
        )
        err_b = explicit_checked_composition(
            payload,
            payload_len,
            cur_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_count,
            rank_capacity,
            vocab_lens,
            vocab_len,
            vocab_capacity,
            req_b,
            max_b,
            bytes_b,
        )

        assert err_a == err_b
        assert cur_a == cur_b
        assert req_a == req_b
        assert max_a == max_b
        assert bytes_a == bytes_b


if __name__ == "__main__":
    test_source_contains_signature_and_hardening_calls()
    test_success_error_no_write_parity_and_sentinel_preservation()
    test_null_overflow_and_fuzz_parity()
    print("ok")
