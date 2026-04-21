#!/usr/bin/env python3
"""Parity gate harness for ...RequiredBytesParityPreflightOnlyCommitOnlyParity (IQ-865)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only_parity(
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

    snapshot_data = data
    snapshot_io_cursor = io_cursor
    snapshot_rank_left_tokens = rank_left_tokens
    snapshot_rank_right_tokens = rank_right_tokens
    snapshot_rank_values = rank_values
    snapshot_rank_merged_tokens = rank_merged_tokens
    snapshot_out_required_token_capacity = out_required_token_capacity
    snapshot_out_required_merge_workspace_bytes = out_required_merge_workspace_bytes

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

    preflight_required_token_capacity = [out_required_token_capacity[0]]
    preflight_required_merge_workspace_bytes = [out_required_merge_workspace_bytes[0]]
    preflight_cursor = [snapshot_cursor]
    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only(
        data,
        byte_len,
        preflight_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        max_piece_len,
        preflight_required_token_capacity,
        preflight_required_merge_workspace_bytes,
    )
    if status != TOKENIZER_BPE_OK:
        return status

    commit_required_token_capacity = [out_required_token_capacity[0]]
    commit_required_merge_workspace_bytes = [out_required_merge_workspace_bytes[0]]
    commit_cursor = [snapshot_cursor]
    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only(
        data,
        byte_len,
        commit_cursor,
        prompt_nbytes,
        rank_left_tokens,
        rank_right_tokens,
        rank_values,
        rank_merged_tokens,
        rank_table_count,
        rank_table_capacity,
        max_piece_len,
        commit_required_token_capacity,
        commit_required_merge_workspace_bytes,
    )
    if status != TOKENIZER_BPE_OK:
        return status

    if preflight_cursor[0] != snapshot_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if commit_cursor[0] != snapshot_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if preflight_required_token_capacity[0] != commit_required_token_capacity[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if preflight_required_merge_workspace_bytes[0] != commit_required_merge_workspace_bytes[0]:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    preflight_last_piece_end = snapshot_cursor + preflight_required_token_capacity[0]
    if preflight_last_piece_end < snapshot_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if preflight_last_piece_end > byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    commit_last_piece_end = snapshot_cursor + commit_required_token_capacity[0]
    if commit_last_piece_end < snapshot_cursor:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if commit_last_piece_end > byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    if preflight_last_piece_end != commit_last_piece_end:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if preflight_required_token_capacity[0] != prompt_nbytes:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if prompt_nbytes and max_piece_len > I64_MAX // prompt_nbytes:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if preflight_required_merge_workspace_bytes[0] != prompt_nbytes * max_piece_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if snapshot_data is not data:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_io_cursor is not io_cursor:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_rank_left_tokens is not rank_left_tokens:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_rank_right_tokens is not rank_right_tokens:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_rank_values is not rank_values:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_rank_merged_tokens is not rank_merged_tokens:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if snapshot_out_required_token_capacity is not out_required_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if (
        snapshot_out_required_merge_workspace_bytes
        is not out_required_merge_workspace_bytes
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

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

    out_required_token_capacity[0] = preflight_required_token_capacity[0]
    out_required_merge_workspace_bytes[0] = preflight_required_merge_workspace_bytes[0]
    return TOKENIZER_BPE_OK


def explicit_parity_composition(*args):
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only_parity(*args)


def test_source_contains_signature_and_parity_contract() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytesParityPreflightOnlyCommitOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens(", 1
    )[0]

    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytesParityPreflightOnly(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPieceDefaultCapacityCommitOnlyPreflightOnlyRequiredBytesParityPreflightOnlyCommitOnly(" in body
    assert "if (preflight_only_required_token_capacity !=" in body
    assert "commit_only_required_token_capacity)" in body
    assert "if (preflight_only_required_merge_workspace_bytes !=" in body
    assert "commit_only_required_merge_workspace_bytes)" in body
    assert "if (snapshot_bytes != bytes)" in body
    assert "if (snapshot_io_cursor_ptr != io_cursor)" in body
    assert "if (snapshot_rank_left_tokens != rank_left_tokens)" in body
    assert "if (snapshot_rank_right_tokens != rank_right_tokens)" in body
    assert "if (snapshot_rank_values != rank_values)" in body
    assert "if (snapshot_rank_merged_tokens != rank_merged_tokens)" in body
    assert "if (snapshot_out_required_token_capacity_ptr !=" in body
    assert "if (snapshot_out_required_merge_workspace_bytes_ptr !=" in body


def test_known_vector_and_failure_no_partial() -> None:
    payload = list(b"holycode")

    out_req = [3]
    out_ws = [15]
    cursor = [2]
    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only_parity(
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

    fail_req = [11]
    fail_ws = [0x9999]
    fail_cursor = [6]
    status = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only_parity(
        payload,
        len(payload),
        fail_cursor,
        2,
        [],
        [],
        [],
        [],
        0,
        0,
        I64_MAX,
        fail_req,
        fail_ws,
    )
    assert status == TOKENIZER_BPE_ERR_OVERFLOW
    assert fail_req == [11]
    assert fail_ws == [0x9999]
    assert fail_cursor == [6]


def test_fuzz_parity_and_no_partial_publish() -> None:
    rng = random.Random(20260421_865)

    for _ in range(2500):
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

        out_req_a = [rng.randint(0, 512)]
        out_ws_a = [rng.randint(0, 4096)]
        out_req_b = out_req_a.copy()
        out_ws_b = out_ws_a.copy()

        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]

        status_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only_parity(
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
        status_b = explicit_parity_composition(
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
        assert cursor_a == cursor_b
        assert out_req_a == out_req_b
        assert out_ws_a == out_ws_b


def main() -> None:
    test_source_contains_signature_and_parity_contract()
    test_known_vector_and_failure_no_partial()
    test_fuzz_parity_and_no_partial_publish()
    print(
        "tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece_default_capacity_commit_only_preflight_only_required_bytes_parity_preflight_only_commit_only_parity=ok"
    )


if __name__ == "__main__":
    main()
