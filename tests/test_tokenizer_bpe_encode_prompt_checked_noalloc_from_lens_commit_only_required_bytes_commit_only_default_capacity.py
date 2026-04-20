#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromLensCommitOnlyRequiredBytesCommitOnlyDefaultCapacity."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity(
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
        or required_token_capacity > I64_MAX
        or required_max_piece_len > I64_MAX
        or required_merge_workspace_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    canonical_required = [0]
    canonical_max_piece = [0]
    canonical_required_bytes = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
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

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only(
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
        out_token_count,
        out_required_token_capacity,
        out_max_piece_len,
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
        out_token_ids,
        required_token_capacity,
        required_max_piece_len,
        required_merge_workspace_bytes,
        out_token_count,
        out_required_token_capacity,
        out_max_piece_len,
    ) = args

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
        or required_token_capacity > I64_MAX
        or required_max_piece_len > I64_MAX
        or required_merge_workspace_bytes > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    req = [0]
    max_piece = [0]
    req_bytes = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
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
        req,
        max_piece,
        req_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if required_token_capacity != req[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if required_max_piece_len != max_piece[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if required_merge_workspace_bytes != req_bytes[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only(
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
        out_token_count,
        out_required_token_capacity,
        out_max_piece_len,
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


def test_source_contains_signature_and_default_capacity_parity_gates() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesCommitOnlyDefaultCapacity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesCommitOnly(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesDefaultCapacity(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityCommitOnly(" in body
    assert "if (required_token_capacity != canonical_required_token_capacity)" in body
    assert "if (required_max_piece_len != canonical_max_piece_len)" in body
    assert "if (required_merge_workspace_bytes != canonical_required_merge_workspace_bytes)" in body


def test_success_and_bad_diagnostic_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂 xyz".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    req = [0]
    max_piece = [0]
    req_bytes = [0]
    cur = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
        payload,
        len(payload),
        cur,
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
        req,
        max_piece,
        req_bytes,
    )
    assert err == TOKENIZER_BPE_OK

    out_tokens_a = [0x1111] * 512
    out_tokens_b = [0x1111] * 512
    count_a = [0x2222]
    count_b = [0x2222]
    req_a = [0x3333]
    req_b = [0x3333]
    max_a = [0x4444]
    max_b = [0x4444]
    cur_a = [0]
    cur_b = [0]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity(
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
        out_tokens_a,
        req[0],
        max_piece[0],
        req_bytes[0],
        count_a,
        req_a,
        max_a,
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
        out_tokens_b,
        req[0],
        max_piece[0],
        req_bytes[0],
        count_b,
        req_b,
        max_b,
    )

    assert err_a == TOKENIZER_BPE_OK
    assert err_a == err_b
    assert cur_a == cur_b
    assert count_a == count_b
    assert req_a == req_b
    assert max_a == max_b
    assert out_tokens_a[: count_a[0]] == out_tokens_b[: count_b[0]]

    sentinel = [0x7777] * 512
    bad_count = [0xA1]
    bad_req = [0xA2]
    bad_max = [0xA3]
    bad_cur = [0]
    err_bad = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity(
        payload,
        len(payload),
        bad_cur,
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
        sentinel,
        req[0] + 1,
        max_piece[0],
        req_bytes[0],
        bad_count,
        bad_req,
        bad_max,
    )
    assert err_bad == TOKENIZER_BPE_ERR_BAD_PARAM
    assert bad_cur == [0]
    assert bad_count == [0xA1]
    assert bad_req == [0xA2]
    assert bad_max == [0xA3]
    assert sentinel == [0x7777] * 512


def test_randomized_parity() -> None:
    random.seed(0x702D)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(300):
        payload_len = random.randint(0, 48)
        payload = [random.randint(0, 255) for _ in range(payload_len)]
        cursor0 = random.randint(0, payload_len)
        prompt_nbytes = random.randint(0, payload_len - cursor0)

        vocab_len = random.randint(1, 14)
        vocab_lens = [random.randint(0, 12) for _ in range(vocab_len)]
        rank_count = len(ranks)
        rank_capacity = rank_count + random.randint(0, 4)
        vocab_capacity = vocab_len + random.randint(0, 4)

        req = [0]
        max_piece = [0]
        req_bytes = [0]
        cur_diag = [cursor0]
        diag_err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_default_capacity(
            payload,
            payload_len,
            cur_diag,
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
            req,
            max_piece,
            req_bytes,
        )
        if diag_err != TOKENIZER_BPE_OK:
            continue

        out_a = [0x99] * 128
        out_b = [0x99] * 128
        count_a = [0x55]
        count_b = [0x55]
        req_a = [0x66]
        req_b = [0x66]
        max_a = [0x77]
        max_b = [0x77]
        cur_a = [cursor0]
        cur_b = [cursor0]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_commit_only_default_capacity(
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
            out_a,
            req[0],
            max_piece[0],
            req_bytes[0],
            count_a,
            req_a,
            max_a,
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
            out_b,
            req[0],
            max_piece[0],
            req_bytes[0],
            count_b,
            req_b,
            max_b,
        )

        assert err_a == err_b
        assert cur_a == cur_b
        assert count_a == count_b
        assert req_a == req_b
        assert max_a == max_b
        assert out_a[: count_a[0]] == out_b[: count_b[0]]


if __name__ == "__main__":
    test_source_contains_signature_and_default_capacity_parity_gates()
    test_success_and_bad_diagnostic_parity()
    test_randomized_parity()
    print("ok")
