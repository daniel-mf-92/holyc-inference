#!/usr/bin/env python3
"""Harness for ...ParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-1085)."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
)
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    out_token_capacity: int,
    out_required_token_capacity: list[int] | None,
    out_required_token_bytes: list[int] | None,
    out_next_cursor: list[int] | None,
    staged_fn=tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only,
    canonical_fn=tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity,
    preflight_fn=tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_required_token_capacity is None
        or out_required_token_bytes is None
        or out_next_cursor is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        out_required_token_capacity is out_required_token_bytes
        or out_required_token_capacity is out_next_cursor
        or out_required_token_bytes is out_next_cursor
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

    snapshot_data = data
    snapshot_byte_len = byte_len
    snapshot_cursor = io_cursor[0]
    snapshot_prompt_nbytes = prompt_nbytes
    snapshot_rank_table_count = rank_table_count
    snapshot_rank_table_capacity = rank_table_capacity
    snapshot_vocab_piece_count = vocab_piece_count
    snapshot_vocab_piece_capacity = vocab_piece_capacity
    snapshot_out_token_capacity = out_token_capacity

    snapshot_out_required = out_required_token_capacity
    snapshot_out_required_bytes = out_required_token_bytes
    snapshot_out_next_cursor = out_next_cursor

    staged_required = [0]
    staged_required_bytes = [0]
    staged_next_cursor = [0]
    err = staged_fn(
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
        out_token_capacity,
        staged_required,
        staged_required_bytes,
        staged_next_cursor,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    canonical_required = [0]
    canonical_required_bytes = [0]
    canonical_next_cursor = [0]
    err = canonical_fn(
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
        out_token_capacity,
        canonical_required,
        canonical_required_bytes,
        canonical_next_cursor,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    preflight_required = [0]
    preflight_max_piece = [0]
    err = preflight_fn(
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
        out_token_capacity,
        preflight_required,
        preflight_max_piece,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if preflight_required[0] > I64_MAX // 4:
        return TOKENIZER_BPE_ERR_OVERFLOW
    preflight_required_bytes = preflight_required[0] * 4

    if snapshot_cursor > byte_len or prompt_nbytes > (byte_len - snapshot_cursor):
        return TOKENIZER_BPE_ERR_BAD_PARAM
    preflight_next_cursor = snapshot_cursor + prompt_nbytes

    if (
        snapshot_data is not data
        or snapshot_byte_len != byte_len
        or snapshot_cursor != io_cursor[0]
        or snapshot_prompt_nbytes != prompt_nbytes
        or snapshot_rank_table_count != rank_table_count
        or snapshot_rank_table_capacity != rank_table_capacity
        or snapshot_vocab_piece_count != vocab_piece_count
        or snapshot_vocab_piece_capacity != vocab_piece_capacity
        or snapshot_out_token_capacity != out_token_capacity
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if (
        snapshot_out_required is not out_required_token_capacity
        or snapshot_out_required_bytes is not out_required_token_bytes
        or snapshot_out_next_cursor is not out_next_cursor
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if (
        staged_required[0] != canonical_required[0]
        or staged_required_bytes[0] != canonical_required_bytes[0]
        or staged_next_cursor[0] != canonical_next_cursor[0]
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if (
        canonical_required[0] != preflight_required[0]
        or canonical_required_bytes[0] != preflight_required_bytes
        or canonical_next_cursor[0] != preflight_next_cursor
    ):
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if preflight_max_piece[0] > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    out_required_token_capacity[0] = staged_required[0]
    out_required_token_bytes[0] = staged_required_bytes[0]
    out_next_cursor[0] = staged_next_cursor[0]
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
        ],
        key=lambda item: (item[0], item[1]),
    )
    return [x[0] for x in entries], [x[1] for x in entries], [x[2] for x in entries], [x[3] for x in entries]


def test_source_contains_signature_and_composition_chain() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = (
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLens"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    )
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyRequiredBytesCommitOnly(",
        1,
    )[0]
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLensCommitOnlyPreflightOnly(" in body


def test_known_vector_success() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello".encode("utf-8"))
    vocab_lens = [1, 2, 4]

    req = [0xAAAA]
    req_bytes = [0xBBBB]
    next_cursor = [0xCCCC]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        payload,
        len(payload),
        cursor,
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
        len(payload),
        req,
        req_bytes,
        next_cursor,
    )

    assert err == TOKENIZER_BPE_OK
    assert req == [len(payload)]
    assert req_bytes == [len(payload) * 4]
    assert next_cursor[0] == len(payload)
    assert cursor[0] == 0


def test_no_partial_publish_on_staged_mismatch() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello world".encode("utf-8"))
    vocab_lens = [1, 2, 3, 5]

    def _bad_staged(*args, **kwargs):
        out_req, out_bytes, out_next = args[-3], args[-2], args[-1]
        out_req[0] = 7
        out_bytes[0] = 28
        out_next[0] = 4
        return TOKENIZER_BPE_OK

    req = [1]
    req_bytes = [2]
    next_cursor = [3]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        payload,
        len(payload),
        cursor,
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
        len(payload),
        req,
        req_bytes,
        next_cursor,
        staged_fn=_bad_staged,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert req == [1]
    assert req_bytes == [2]
    assert next_cursor == [3]


def test_no_partial_publish_on_canonical_mismatch() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello world".encode("utf-8"))
    vocab_lens = [1, 2, 3, 5]

    def _bad_canonical(*args, **kwargs):
        out_req, out_bytes, out_next = args[-3], args[-2], args[-1]
        out_req[0] = len(payload)
        out_bytes[0] = len(payload) * 4
        out_next[0] = len(payload) - 1
        return TOKENIZER_BPE_OK

    req = [11]
    req_bytes = [22]
    next_cursor = [33]
    cursor = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        payload,
        len(payload),
        cursor,
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
        len(payload),
        req,
        req_bytes,
        next_cursor,
        canonical_fn=_bad_canonical,
    )

    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert req == [11]
    assert req_bytes == [22]
    assert next_cursor == [33]


def test_randomized_adversarial_vectors() -> None:
    rng = random.Random(20260422_1085)
    left, right, ranks, merged = _build_rank_tables()
    vocab_lens = [1, 2, 3, 4, 5, 8]

    for _ in range(320):
        n = rng.randint(1, 64)
        payload = [rng.randrange(32, 127) for _ in range(n)]
        prompt_nbytes = rng.randint(0, n)
        cursor0 = rng.randint(0, n - prompt_nbytes)
        out_cap = max(1, n)

        req = [0]
        req_bytes = [0]
        next_cursor = [0]
        cursor = [cursor0]

        err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            payload,
            len(payload),
            cursor,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            vocab_lens,
            len(vocab_lens),
            len(vocab_lens),
            out_cap,
            req,
            req_bytes,
            next_cursor,
        )

        assert err == TOKENIZER_BPE_OK
        assert req[0] >= 0
        assert req_bytes[0] == req[0] * 4
        assert next_cursor[0] == cursor0 + prompt_nbytes
        assert cursor[0] == cursor0


if __name__ == "__main__":
    test_source_contains_signature_and_composition_chain()
    test_known_vector_success()
    test_no_partial_publish_on_staged_mismatch()
    test_no_partial_publish_on_canonical_mismatch()
    test_randomized_adversarial_vectors()
    print("ok")
