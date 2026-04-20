#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnly."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
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
    out_token_capacity: int,
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
        or out_token_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    canonical_max_piece_len = 0
    for piece_index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[piece_index]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > canonical_max_piece_len:
            canonical_max_piece_len = piece_len

    canonical_required_token_capacity = prompt_nbytes

    if prompt_nbytes and canonical_max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if canonical_required_token_capacity > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_cursor = [cursor]
    staged_token_count = [out_token_count[0]]
    staged_required_token_capacity = [out_required_token_capacity[0]]

    stage_capacity = max(1, out_token_capacity)
    if stage_capacity > I64_MAX // 4:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_tokens = [0] * stage_capacity
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
        canonical_max_piece_len,
        staged_tokens,
        out_token_capacity,
        staged_token_count,
        staged_required_token_capacity,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required_token_capacity[0] != canonical_required_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if staged_token_count[0] > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for index in range(staged_token_count[0]):
        out_token_ids[index] = staged_tokens[index]

    out_token_count[0] = staged_token_count[0]
    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_max_piece_len[0] = canonical_max_piece_len
    io_cursor[0] = staged_cursor[0]
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
    vocab_piece_lens: list[int] | None,
    vocab_piece_count: int,
    vocab_piece_capacity: int,
    out_token_ids: list[int] | None,
    out_token_capacity: int,
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
        or out_token_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if vocab_piece_count > vocab_piece_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    canonical_max_piece_len = 0
    for piece_index in range(vocab_piece_count):
        piece_len = vocab_piece_lens[piece_index]
        if piece_len > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if piece_len > canonical_max_piece_len:
            canonical_max_piece_len = piece_len

    canonical_required_token_capacity = prompt_nbytes

    if prompt_nbytes and canonical_max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if canonical_required_token_capacity > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_cursor = [cursor]
    staged_token_count = [out_token_count[0]]
    staged_required_token_capacity = [out_required_token_capacity[0]]
    stage_capacity = max(1, out_token_capacity)
    staged_tokens = [0] * stage_capacity

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_max_piece(
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
        canonical_max_piece_len,
        staged_tokens,
        out_token_capacity,
        staged_token_count,
        staged_required_token_capacity,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required_token_capacity[0] != canonical_required_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if staged_token_count[0] > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for index in range(staged_token_count[0]):
        out_token_ids[index] = staged_tokens[index]

    out_token_count[0] = staged_token_count[0]
    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_max_piece_len[0] = canonical_max_piece_len
    io_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def test_source_contains_lens_commit_only_signature_and_staged_delegate() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityCommitOnly(",
        1,
    )[0]
    assert "canonical_max_piece_len = 0;" in body
    assert "canonical_required_token_capacity = prompt_nbytes;" in body
    assert "staged_cursor = cursor;" in body
    assert "staged_token_count = *out_token_count;" in body
    assert "staged_required_token_capacity = *out_required_token_capacity;" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromMaxPiece(" in body
    assert "if (staged_required_token_capacity != canonical_required_token_capacity)" in body
    assert "for (i = 0; i < staged_token_count; i++)" in body
    assert "*out_token_count = staged_token_count;" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_max_piece_len = canonical_max_piece_len;" in body
    assert "*io_cursor = staged_cursor;" in body


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


def test_success_fixture_parity_and_atomic_commit() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂 xyz".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x4444] * 512
    out_b = [0x4444] * 512
    count_a = [0x1234]
    count_b = [0x1234]
    req_a = [0x5678]
    req_b = [0x5678]
    max_a = [0x7777]
    max_b = [0x7777]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
        payload,
        len(payload),
        cursor_a,
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
        out_a,
        len(payload),
        count_a,
        req_a,
        max_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        cursor_b,
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
        out_b,
        len(payload),
        count_b,
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert cursor_a[0] == cursor_b[0] == len(payload)
    assert count_a[0] == count_b[0]
    assert req_a[0] == req_b[0] == len(payload)
    assert max_a[0] == max_b[0] == 9
    assert out_a == out_b


def test_adversarial_error_vectors_preserve_no_partial_outputs() -> None:
    payload = [ord("o"), ord("k"), ord("!")]
    left: list[int] = []
    right: list[int] = []
    ranks: list[int] = []
    merged: list[int] = []

    out = [0xAAAA] * 16
    count = [0x5151]
    req = [0x6161]
    max_piece = [0x7171]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
        payload,
        len(payload),
        cursor,
        3,
        left,
        right,
        ranks,
        merged,
        0,
        0,
        [0, 0, 0],
        3,
        3,
        out,
        16,
        count,
        req,
        max_piece,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == [0xAAAA] * 16
    assert count[0] == 0x5151
    assert req[0] == 0x6161
    assert max_piece[0] == 0x7171
    assert cursor[0] == 0

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
        payload,
        len(payload),
        [1],
        3,
        left,
        right,
        ranks,
        merged,
        0,
        0,
        [1],
        1,
        1,
        [0] * 8,
        8,
        [0],
        [0],
        [0],
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    out = [0xBBBB] * 8
    count = [0x2121]
    req = [0x3131]
    max_piece = [0x4141]
    cursor = [0]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
        payload,
        len(payload),
        cursor,
        3,
        left,
        right,
        ranks,
        merged,
        0,
        0,
        [1, 2, 4],
        3,
        3,
        out,
        1,
        count,
        req,
        max_piece,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out == [0xBBBB] * 8
    assert count[0] == 0x2121
    assert req[0] == 0x3131
    assert max_piece[0] == 0x4141
    assert cursor[0] == 0


def test_randomized_parity_vs_explicit_checked_composition() -> None:
    rng = random.Random(20260420_696)

    for _ in range(2500):
        payload_len = rng.randint(0, 96)
        payload = [rng.randint(0, 127) for _ in range(payload_len)]
        cursor_seed = rng.randint(0, payload_len)
        prompt_nbytes = rng.randint(0, payload_len - cursor_seed)

        rank_count = rng.randint(0, 32)
        left = [rng.randint(0, 600) for _ in range(rank_count)]
        right = [rng.randint(0, 600) for _ in range(rank_count)]
        ranks = [rng.randint(0, 64) for _ in range(rank_count)]
        merged = [rng.randint(0, 1024) for _ in range(rank_count)]

        vocab_count = rng.randint(0, 32)
        vocab_lens = [rng.randint(0, 16) for _ in range(vocab_count)]

        out_cap = rng.randint(0, 96)
        out_a = [0x7A7A] * max(1, out_cap + 8)
        out_b = [0x7A7A] * max(1, out_cap + 8)

        cursor_a = [cursor_seed]
        cursor_b = [cursor_seed]
        count_a = [rng.randint(0, 1000)]
        count_b = [count_a[0]]
        req_a = [rng.randint(0, 1000)]
        req_b = [req_a[0]]
        max_a = [rng.randint(0, 1000)]
        max_b = [max_a[0]]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only(
            payload,
            payload_len,
            cursor_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_count,
            rank_count,
            vocab_lens,
            vocab_count,
            vocab_count,
            out_a,
            out_cap,
            count_a,
            req_a,
            max_a,
        )
        err_b = explicit_checked_composition(
            payload,
            payload_len,
            cursor_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_count,
            rank_count,
            vocab_lens,
            vocab_count,
            vocab_count,
            out_b,
            out_cap,
            count_b,
            req_b,
            max_b,
        )

        assert err_a == err_b
        assert cursor_a == cursor_b
        assert count_a == count_b
        assert req_a == req_b
        assert max_a == max_b
        assert out_a == out_b


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
