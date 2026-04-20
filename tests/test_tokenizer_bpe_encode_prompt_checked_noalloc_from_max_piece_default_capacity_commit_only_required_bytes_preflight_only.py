#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyRequiredBytesPreflightOnly (IQ-826)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes_preflight_only(
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
    out_required_token_capacity: list[int] | None,
    out_required_merge_workspace_bytes: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_required_token_capacity is None
        or out_required_merge_workspace_bytes is None
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

    snapshot_byte_len = byte_len
    snapshot_cursor = io_cursor[0]
    snapshot_prompt_nbytes = prompt_nbytes
    snapshot_rank_table_count = rank_table_count
    snapshot_rank_table_capacity = rank_table_capacity
    snapshot_max_piece_len = max_piece_len

    if snapshot_cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_required_token_capacity = [out_required_token_capacity[0]]
    staged_cursor = [snapshot_cursor]
    staged_token_count = [0]
    staged_required_token_capacity_after_commit = [out_required_token_capacity[0]]

    staged_token_capacity = max(1, prompt_nbytes)
    if staged_token_capacity > I64_MAX // 4:
        return TOKENIZER_BPE_ERR_OVERFLOW
    staged_tokens = [0] * staged_token_capacity

    staged_required_merge_workspace_bytes = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes(
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
        max_piece_len,
        staged_tokens,
        staged_token_count,
        staged_required_token_capacity_after_commit,
        staged_required_merge_workspace_bytes,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if snapshot_byte_len != byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_cursor != io_cursor[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_prompt_nbytes != prompt_nbytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_rank_table_count != rank_table_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_rank_table_capacity != rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_max_piece_len != max_piece_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if staged_required_token_capacity_after_commit[0] != prompt_nbytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_required_token_capacity_after_commit[0] != staged_required_token_capacity[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if staged_required_merge_workspace_bytes[0] != prompt_nbytes * max_piece_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required_token_capacity_after_commit[0]
    out_required_merge_workspace_bytes[0] = staged_required_merge_workspace_bytes[0]
    return TOKENIZER_BPE_OK


def explicit_checked_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes_preflight_only(
        *args
    )


def test_source_contains_commit_only_required_bytes_preflight_only_wrapper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyRequiredBytesPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyRequiredBytes(" in body
    assert "staged_tokens = MAlloc(staged_alloc_bytes);" in body
    assert "if (snapshot_cursor != *io_cursor)" in body
    assert "*out_required_token_capacity = staged_required_token_capacity_after_commit;" in body
    assert "*out_required_merge_workspace_bytes = staged_required_merge_workspace_bytes;" in body


def test_success_and_no_publish_on_failure() -> None:
    payload = list(b"holycode")

    req = [4]
    reqb = [0x2222]
    cursor = [2]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes_preflight_only(
        payload,
        len(payload),
        cursor,
        4,
        [],
        [],
        [],
        [],
        0,
        0,
        7,
        req,
        reqb,
    )
    assert err == TOKENIZER_BPE_OK
    assert req == [4]
    assert reqb == [28]
    assert cursor == [2]

    req_fail = [0xAAAA]
    reqb_fail = [0xBBBB]
    cursor_fail = [3]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes_preflight_only(
        payload,
        len(payload),
        cursor_fail,
        3,
        [],
        [],
        [],
        [],
        0,
        0,
        I64_MAX,
        req_fail,
        reqb_fail,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert req_fail == [0xAAAA]
    assert reqb_fail == [0xBBBB]
    assert cursor_fail == [3]


def test_fuzz_parity() -> None:
    random.seed(20260421_826)

    for _ in range(2200):
        payload_len = random.randint(0, 96)
        payload = [random.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = random.randint(0, payload_len)
        prompt_nbytes = random.randint(0, payload_len - cursor_seed)

        rank_count = random.randint(0, 20)
        rank_capacity = rank_count + random.randint(0, 3)

        rank_left = [random.randint(0, 255) for _ in range(rank_count)]
        rank_right = [random.randint(0, 255) for _ in range(rank_count)]
        rank_values = [random.randint(0, 64) for _ in range(rank_count)]
        rank_merged = [random.randint(256, 8192) for _ in range(rank_count)]

        rank_left_arg = None if rank_count and random.random() < 0.08 else rank_left
        rank_right_arg = None if rank_count and random.random() < 0.08 else rank_right
        rank_values_arg = None if rank_count and random.random() < 0.08 else rank_values
        rank_merged_arg = None if rank_count and random.random() < 0.08 else rank_merged

        max_piece_len = random.randint(0, 192)

        req_a = [0x3333]
        req_b = [0x3333]
        reqb_a = [0x4444]
        reqb_b = [0x4444]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_required_bytes_preflight_only(
            payload,
            payload_len,
            cursor_a,
            prompt_nbytes,
            rank_left_arg,
            rank_right_arg,
            rank_values_arg,
            rank_merged_arg,
            rank_count,
            rank_capacity,
            max_piece_len,
            req_a,
            reqb_a,
        )
        err_b = explicit_checked_composition(
            payload,
            payload_len,
            cursor_b,
            prompt_nbytes,
            rank_left_arg,
            rank_right_arg,
            rank_values_arg,
            rank_merged_arg,
            rank_count,
            rank_capacity,
            max_piece_len,
            req_b,
            reqb_b,
        )

        assert err_a == err_b
        assert req_a == req_b
        assert reqb_a == reqb_b
        assert cursor_a == cursor_b


if __name__ == "__main__":
    test_source_contains_commit_only_required_bytes_preflight_only_wrapper()
    test_success_and_no_publish_on_failure()
    test_fuzz_parity()
    print("ok")
