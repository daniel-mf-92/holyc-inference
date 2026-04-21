#!/usr/bin/env python3
"""Parity harness for ...CommitOnlyPreflightOnlyRequiredBytes (IQ-815)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes(
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

    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    staged_required_merge_workspace_bytes = prompt_nbytes * max_piece_len

    staged_required_token_capacity = [out_required_token_capacity[0]]
    staged_cursor = [io_cursor[0]]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only(
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
        staged_required_token_capacity,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_cursor[0] != io_cursor[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_required_token_capacity[0] != prompt_nbytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_required_merge_workspace_bytes[0] = staged_required_merge_workspace_bytes
    return TOKENIZER_BPE_OK


def explicit_checked_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes(
        *args
    )


def test_source_contains_required_bytes_helper_and_atomic_publish() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytes("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnly(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnly(" in body
    assert "staged_required_merge_workspace_bytes = prompt_nbytes * max_piece_len;" in body
    assert "if (snapshot_bytes != bytes)" in body
    assert "if (snapshot_io_cursor_ptr != io_cursor)" in body
    assert "if (snapshot_out_required_merge_workspace_bytes_ptr !=" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_required_merge_workspace_bytes = staged_required_merge_workspace_bytes;" in body


def test_required_bytes_success_and_no_partial_on_failure() -> None:
    payload = list("holycode".encode("utf-8"))

    required_token_capacity = [0x4444]
    required_merge_workspace_bytes = [0x5555]
    cursor = [2]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes(
        payload,
        len(payload),
        cursor,
        3,
        [],
        [],
        [],
        [],
        0,
        0,
        6,
        required_token_capacity,
        required_merge_workspace_bytes,
    )
    assert err == TOKENIZER_BPE_OK
    assert required_token_capacity == [3]
    assert required_merge_workspace_bytes == [18]
    assert cursor == [2]

    required_token_capacity_fail = [0xAAAA]
    required_merge_workspace_bytes_fail = [0xBBBB]
    cursor_fail = [5]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes(
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
        required_token_capacity_fail,
        required_merge_workspace_bytes_fail,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert required_token_capacity_fail == [0xAAAA]
    assert required_merge_workspace_bytes_fail == [0xBBBB]
    assert cursor_fail == [5]


def test_required_bytes_fuzz_parity() -> None:
    random.seed(20260421_815)

    for _ in range(2600):
        payload_len = random.randint(0, 128)
        payload = [random.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = random.randint(0, payload_len)
        prompt_nbytes = random.randint(0, payload_len - cursor_seed)

        rank_count = random.randint(0, 32)
        rank_capacity = rank_count + random.randint(0, 4)

        rank_left = [random.randint(0, 255) for _ in range(rank_count)]
        rank_right = [random.randint(0, 255) for _ in range(rank_count)]
        rank_values = [random.randint(0, 64) for _ in range(rank_count)]
        rank_merged = [random.randint(256, 8192) for _ in range(rank_count)]

        rank_left_arg = None if rank_count and random.random() < 0.09 else rank_left
        rank_right_arg = None if rank_count and random.random() < 0.09 else rank_right
        rank_values_arg = None if rank_count and random.random() < 0.09 else rank_values
        rank_merged_arg = None if rank_count and random.random() < 0.09 else rank_merged

        max_piece_len = random.randint(0, 256)

        req_tok_a = [0x1111]
        req_tok_b = [0x1111]
        req_mw_a = [0x2222]
        req_mw_b = [0x2222]
        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes(
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
            req_tok_a,
            req_mw_a,
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
            req_tok_b,
            req_mw_b,
        )

        assert err_a == err_b
        assert req_tok_a == req_tok_b
        assert req_mw_a == req_mw_b
        assert cursor_a == cursor_b


if __name__ == "__main__":
    test_source_contains_required_bytes_helper_and_atomic_publish()
    test_required_bytes_success_and_no_partial_on_failure()
    test_required_bytes_fuzz_parity()
    print("ok")
