#!/usr/bin/env python3
"""Parity harness for ...RequiredBytesPreflightOnlyDefaultCapacity."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
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
    staged_cursor = io_cursor[0]

    staged_required = [0]
    staged_max_piece_len = [0]
    staged_required_merge_workspace_bytes = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
        data,
        byte_len,
        [staged_cursor],
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
        staged_required,
        staged_max_piece_len,
        staged_required_merge_workspace_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if (
        staged_required[0] > I64_MAX
        or staged_max_piece_len[0] > I64_MAX
        or staged_required_merge_workspace_bytes[0] > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    out_required_merge_workspace_bytes[0] = staged_required_merge_workspace_bytes[0]
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

    staged_required = [0]
    staged_max_piece_len = [0]
    staged_required_merge_workspace_bytes = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only(
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
        prompt_nbytes,
        staged_required,
        staged_max_piece_len,
        staged_required_merge_workspace_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if (
        staged_required[0] > I64_MAX
        or staged_max_piece_len[0] > I64_MAX
        or staged_required_merge_workspace_bytes[0] > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    out_required_merge_workspace_bytes[0] = staged_required_merge_workspace_bytes[0]
    return TOKENIZER_BPE_OK


def _build_rank_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    entries = sorted(
        [
            (97, 98, 4, 1000),
            (1000, 99, 1, 1001),
            (1001, 100, 0, 1002),
            (101, 102, 3, 1003),
            (1003, 103, 2, 1004),
            (104, 105, 5, 1005),
        ],
        key=lambda item: (item[0], item[1]),
    )
    left = [item[0] for item in entries]
    right = [item[1] for item in entries]
    ranks = [item[2] for item in entries]
    merged = [item[3] for item in entries]
    return left, right, ranks, merged


def test_source_contains_required_bytes_preflight_only_default_capacity_signature() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesPreflightOnlyDefaultCapacity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnly(",
        1,
    )[0]
    assert "derived_out_token_capacity = prompt_nbytes;" in body
    assert "staged_cursor = *io_cursor;" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesPreflightOnly(" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_max_piece_len = staged_max_piece_len;" in body
    assert "*out_required_merge_workspace_bytes = staged_required_merge_workspace_bytes;" in body


def test_success_and_error_and_no_write_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abcdef".encode("utf-8"))
    vocab_lens = [1, 2, 5]

    req_a = [0xAAAA]
    max_a = [0xBBBB]
    merge_a = [0xCCCC]
    cur_a = [0]

    req_b = [0xAAAA]
    max_b = [0xBBBB]
    merge_b = [0xCCCC]
    cur_b = [0]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
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
        merge_a,
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
        merge_b,
    )

    assert err_a == TOKENIZER_BPE_OK
    assert err_a == err_b
    assert req_a == req_b == [len(payload)]
    assert max_a == max_b == [max(vocab_lens)]
    assert merge_a == merge_b == [len(payload) * max(vocab_lens)]
    assert cur_a == cur_b == [0]

    req_a = [0x1]
    max_a = [0x2]
    merge_a = [0x3]
    cur_a = [0]

    req_b = [0x1]
    max_b = [0x2]
    merge_b = [0x3]
    cur_b = [0]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
        payload,
        len(payload),
        cur_a,
        len(payload) + 1,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        [1, 2, 5],
        3,
        3,
        req_a,
        max_a,
        merge_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cur_b,
        len(payload) + 1,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        [1, 2, 5],
        3,
        3,
        req_b,
        max_b,
        merge_b,
    )

    assert err_a == err_b
    assert err_a != TOKENIZER_BPE_OK
    assert req_a == [0x1]
    assert req_b == [0x1]
    assert max_a == [0x2]
    assert max_b == [0x2]
    assert merge_a == [0x3]
    assert merge_b == [0x3]
    assert cur_a == [0]
    assert cur_b == [0]


def test_null_and_overflow_and_fuzz_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    vocab_lens = [1, 3, 6]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
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

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
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
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW

    rng = random.Random(703)
    for _ in range(250):
        payload_len = rng.randint(0, 14)
        payload = [rng.randint(0, 255) for _ in range(payload_len)]

        prompt_nbytes = rng.randint(0, payload_len)
        cursor0 = rng.randint(0, payload_len)

        rank_cap = rng.randint(len(ranks), len(ranks) + 2)
        vocab_count = rng.randint(1, 5)
        vocab_capacity = rng.randint(vocab_count, vocab_count + 2)
        sampled_lens = [rng.randint(1, 10) for _ in range(vocab_count)]

        req_a = [0xABCD]
        max_a = [0xBCDE]
        merge_a = [0xCDEF]
        cur_a = [cursor0]

        req_b = [0xABCD]
        max_b = [0xBCDE]
        merge_b = [0xCDEF]
        cur_b = [cursor0]

        args = (
            payload,
            payload_len,
            cur_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            rank_cap,
            sampled_lens,
            vocab_count,
            vocab_capacity,
            req_a,
            max_a,
            merge_a,
        )
        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_required_bytes_preflight_only_default_capacity(
            *args
        )

        args_b = (
            payload,
            payload_len,
            cur_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            rank_cap,
            sampled_lens,
            vocab_count,
            vocab_capacity,
            req_b,
            max_b,
            merge_b,
        )
        err_b = explicit_checked_composition(*args_b)

        assert err_a == err_b
        assert cur_a == cur_b
        assert req_a == req_b
        assert max_a == max_b
        assert merge_a == merge_b


if __name__ == "__main__":
    test_source_contains_required_bytes_preflight_only_default_capacity_signature()
    test_success_and_error_and_no_write_parity()
    test_null_and_overflow_and_fuzz_parity()
    print("ok")
