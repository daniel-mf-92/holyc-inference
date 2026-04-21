#!/usr/bin/env python3
"""Parity-preflight harness for ...RequiredBytesParityPreflightOnly (IQ-859)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only(
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
    if prompt_nbytes > byte_len - snapshot_cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    staged_required_token_capacity = [out_required_token_capacity[0]]
    staged_required_merge_workspace_bytes = [out_required_merge_workspace_bytes[0]]
    staged_cursor = [snapshot_cursor]

    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity(
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
        staged_required_merge_workspace_bytes,
    )
    if status != TOKENIZER_BPE_OK:
        return status
    if staged_cursor[0] != snapshot_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_last_piece_end = snapshot_cursor + staged_required_token_capacity[0]
    if staged_last_piece_end < snapshot_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if staged_last_piece_end > byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    canonical_required_token_capacity = prompt_nbytes
    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    canonical_required_merge_workspace_bytes = prompt_nbytes * max_piece_len

    canonical_last_piece_end = snapshot_cursor + canonical_required_token_capacity
    if canonical_last_piece_end < snapshot_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if canonical_last_piece_end > byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

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

    if staged_required_token_capacity[0] != canonical_required_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_required_merge_workspace_bytes[0] != canonical_required_merge_workspace_bytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_last_piece_end != canonical_last_piece_end:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required_token_capacity[0]
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
        max_piece_len,
        out_required_token_capacity,
        out_required_merge_workspace_bytes,
    ) = args

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

    snapshot_cursor = io_cursor[0]
    if snapshot_cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if prompt_nbytes > byte_len - snapshot_cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    staged_required_token_capacity = [out_required_token_capacity[0]]
    staged_required_merge_workspace_bytes = [out_required_merge_workspace_bytes[0]]
    staged_cursor = [snapshot_cursor]

    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity(
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
        staged_required_merge_workspace_bytes,
    )
    if status != TOKENIZER_BPE_OK:
        return status
    if staged_cursor[0] != snapshot_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_last_piece_end = snapshot_cursor + staged_required_token_capacity[0]
    if staged_last_piece_end < snapshot_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if staged_last_piece_end > byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    canonical_required_token_capacity = prompt_nbytes
    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    canonical_required_merge_workspace_bytes = prompt_nbytes * max_piece_len

    canonical_last_piece_end = snapshot_cursor + canonical_required_token_capacity
    if canonical_last_piece_end < snapshot_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if canonical_last_piece_end > byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    if staged_required_token_capacity[0] != canonical_required_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_required_merge_workspace_bytes[0] != canonical_required_merge_workspace_bytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if staged_last_piece_end != canonical_last_piece_end:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_required_merge_workspace_bytes[0] = staged_required_merge_workspace_bytes[0]
    return TOKENIZER_BPE_OK


def test_source_contains_signature_and_parity_preflight_only_logic() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytesParityPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens(", 1
    )[0]

    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytesParity(" in body
    assert "staged_last_piece_end = snapshot_cursor + staged_required_token_capacity;" in body
    assert "canonical_required_token_capacity = prompt_nbytes;" in body
    assert "canonical_required_merge_workspace_bytes = prompt_nbytes * max_piece_len;" in body
    assert "canonical_last_piece_end = snapshot_cursor + canonical_required_token_capacity;" in body
    assert "if (staged_required_token_capacity != canonical_required_token_capacity)" in body
    assert "if (staged_required_merge_workspace_bytes !=" in body
    assert "if (staged_last_piece_end != canonical_last_piece_end)" in body


def test_known_vector_and_no_partial_failure() -> None:
    payload = list(b"holycode")

    out_req = [3]
    out_ws = [15]
    cursor = [2]
    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only(
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
        5,
        out_req,
        out_ws,
    )
    assert status == TOKENIZER_BPE_OK
    assert out_req == [3]
    assert out_ws == [15]
    assert cursor == [2]

    out_req_fail = [5]
    out_ws_fail = [0x1234]
    cursor_fail = [6]
    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only(
        payload,
        len(payload),
        cursor_fail,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        I64_MAX,
        out_req_fail,
        out_ws_fail,
    )
    assert status == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_req_fail == [5]
    assert out_ws_fail == [0x1234]
    assert cursor_fail == [6]


def test_fuzz_parity_and_no_partial_publish() -> None:
    rng = random.Random(20260421_859)

    for _ in range(2200):
        payload_len = rng.randint(0, 128)
        payload = [rng.randint(0, 127) for _ in range(payload_len)]

        cursor_seed = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor_seed)

        rank_count = rng.randint(0, 24)
        rank_capacity = rank_count + rng.randint(0, 2)

        rank_left = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_right = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_vals = [rng.randint(0, 255) for _ in range(rank_count)]
        rank_merged = [rng.randint(0, 4095) for _ in range(rank_count)]

        if rank_count and rng.random() < 0.05:
            rank_left = None
        if rank_count and rng.random() < 0.05:
            rank_right = None
        if rank_count and rng.random() < 0.05:
            rank_vals = None
        if rank_count and rng.random() < 0.05:
            rank_merged = None

        max_piece_len = rng.randint(0, 256)
        if rng.random() < 0.06:
            max_piece_len = I64_MAX

        out_req_a = [0x91]
        out_ws_a = [0xA2]
        out_req_b = [0x91]
        out_ws_b = [0xA2]

        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        status_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only(
            payload,
            payload_len,
            cursor_a,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_vals,
            rank_merged,
            rank_count,
            rank_capacity,
            max_piece_len,
            out_req_a,
            out_ws_a,
        )
        status_b = explicit_checked_composition(
            payload,
            payload_len,
            cursor_b,
            prompt_nbytes,
            rank_left,
            rank_right,
            rank_vals,
            rank_merged,
            rank_count,
            rank_capacity,
            max_piece_len,
            out_req_b,
            out_ws_b,
        )

        assert status_a == status_b
        assert out_req_a == out_req_b
        assert out_ws_a == out_ws_b
        assert cursor_a == cursor_b


if __name__ == "__main__":
    test_source_contains_signature_and_parity_preflight_only_logic()
    test_known_vector_and_no_partial_failure()
    test_fuzz_parity_and_no_partial_publish()
    print("ok")
