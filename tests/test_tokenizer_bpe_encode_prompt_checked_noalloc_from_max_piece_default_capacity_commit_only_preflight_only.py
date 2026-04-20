#!/usr/bin/env python3
"""Parity harness for ...FromMaxPieceDefaultCapacityCommitOnlyPreflightOnly (IQ-800)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only(
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
) -> int:
    if data is None or io_cursor is None or out_required_token_capacity is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_count > I64_MAX
        or rank_table_capacity > I64_MAX
        or max_piece_len > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_required = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only(
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
        max_piece_len,
        staged_required,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required[0] != prompt_nbytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required[0]
    return TOKENIZER_BPE_OK


def explicit_checked_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only(
        *args
    )


def test_source_contains_commit_only_preflight_only_signature_and_atomic_publish() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnly(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityPreflightOnly(" in body
    assert "if (staged_required_token_capacity != prompt_nbytes)" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body


def test_commit_only_calls_commit_only_preflight_only() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnly("
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnly(" in body


def test_success_and_no_write_on_failure() -> None:
    payload = list("holy".encode("utf-8"))

    required = [0x7777]
    cursor = [1]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only(
        payload,
        len(payload),
        cursor,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        8,
        required,
    )
    assert err == TOKENIZER_BPE_OK
    assert required == [2]
    assert cursor == [1]

    required_fail = [0x9999]
    cursor_fail = [2]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only(
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
        2,
        required_fail,
    )
    assert err != TOKENIZER_BPE_OK
    assert required_fail == [0x9999]
    assert cursor_fail == [2]


def test_fuzz_parity() -> None:
    random.seed(20260421_800)

    for _ in range(2400):
        payload_len = random.randint(0, 96)
        payload = [random.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = random.randint(0, payload_len)
        prompt_nbytes = random.randint(0, payload_len - cursor_seed)

        rank_count = random.randint(0, 48)
        rank_capacity = rank_count + random.randint(0, 4)

        rank_left = [random.randint(0, 255) for _ in range(rank_count)]
        rank_right = [random.randint(0, 255) for _ in range(rank_count)]
        rank_values = [random.randint(0, 32) for _ in range(rank_count)]
        rank_merged = [random.randint(256, 4096) for _ in range(rank_count)]

        rank_left_arg = None if rank_count and random.random() < 0.12 else rank_left
        rank_right_arg = None if rank_count and random.random() < 0.12 else rank_right
        rank_values_arg = None if rank_count and random.random() < 0.12 else rank_values
        rank_merged_arg = None if rank_count and random.random() < 0.12 else rank_merged

        max_piece_len = random.randint(0, 24)

        req_a = [0x1234]
        req_b = [0x1234]
        cur_a = [cursor_seed]
        cur_b = [cursor_seed]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only(
            payload,
            payload_len,
            cur_a,
            prompt_nbytes,
            rank_left_arg,
            rank_right_arg,
            rank_values_arg,
            rank_merged_arg,
            rank_count,
            rank_capacity,
            max_piece_len,
            req_a,
        )
        err_b = explicit_checked_composition(
            payload,
            payload_len,
            cur_b,
            prompt_nbytes,
            rank_left_arg,
            rank_right_arg,
            rank_values_arg,
            rank_merged_arg,
            rank_count,
            rank_capacity,
            max_piece_len,
            req_b,
        )

        assert err_a == err_b
        assert req_a == req_b
        assert cur_a == cur_b


if __name__ == "__main__":
    test_source_contains_commit_only_preflight_only_signature_and_atomic_publish()
    test_commit_only_calls_commit_only_preflight_only()
    test_success_and_no_write_on_failure()
    test_fuzz_parity()
    print("ok")
