#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityPreflightOnly (IQ-797)."""

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


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only(
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

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    if prompt_nbytes and max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    required_token_capacity = prompt_nbytes
    derived_out_token_capacity = prompt_nbytes

    if required_token_capacity != derived_out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    required_merge_workspace_bytes = prompt_nbytes * max_piece_len
    if required_merge_workspace_bytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_left_tokens is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_right_tokens is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_values is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_merged_tokens is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR

    out_required_token_capacity[0] = required_token_capacity
    return TOKENIZER_BPE_OK


def explicit_checked_composition(
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

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    if prompt_nbytes and max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    required_merge_workspace_bytes = prompt_nbytes * max_piece_len
    if required_merge_workspace_bytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_left_tokens is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_right_tokens is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_values is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_merged_tokens is None and rank_table_count:
        return TOKENIZER_BPE_ERR_NULL_PTR

    out_required_token_capacity[0] = prompt_nbytes
    return TOKENIZER_BPE_OK


def test_source_contains_default_capacity_preflight_only_wrapper() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")

    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]
    assert "derived_out_token_capacity = prompt_nbytes;" in body
    assert "required_merge_workspace_bytes = prompt_nbytes * max_piece_len;" in body
    assert "*out_required_token_capacity = required_token_capacity;" in body


def test_no_write_contract_on_failure() -> None:
    payload = [ord("o"), ord("k")]
    cursor = [1]
    required = [0x8888]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only(
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
        0,
        required,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor == [1]
    assert required == [0x8888]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only(
        payload,
        len(payload),
        [0],
        1,
        None,
        [],
        [],
        [],
        1,
        1,
        1,
        required,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert required == [0x8888]


def test_parity_vs_explicit_checked_composition_randomized() -> None:
    rng = random.Random(20260420_797)

    for _ in range(2200):
        payload_len = rng.randint(0, 96)
        payload = [rng.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor_seed)

        rank_count = rng.randint(0, 48)
        rank_capacity = rank_count + rng.randint(0, 4)

        rank_left = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_values = [rng.randint(0, 32) for _ in range(rank_count)]
        rank_merged = [rng.randint(256, 4096) for _ in range(rank_count)]

        if rank_count and rng.random() < 0.12:
            rank_left_arg = None
        else:
            rank_left_arg = rank_left

        if rank_count and rng.random() < 0.12:
            rank_right_arg = None
        else:
            rank_right_arg = rank_right

        if rank_count and rng.random() < 0.12:
            rank_values_arg = None
        else:
            rank_values_arg = rank_values

        if rank_count and rng.random() < 0.12:
            rank_merged_arg = None
        else:
            rank_merged_arg = rank_merged

        max_piece_len = rng.randint(0, 24)

        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]
        req_a = [0x1234]
        req_b = [0x1234]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only(
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
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert req_a == req_b


if __name__ == "__main__":
    test_source_contains_default_capacity_preflight_only_wrapper()
    test_no_write_contract_on_failure()
    test_parity_vs_explicit_checked_composition_randomized()
    print("tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_preflight_only=ok")
