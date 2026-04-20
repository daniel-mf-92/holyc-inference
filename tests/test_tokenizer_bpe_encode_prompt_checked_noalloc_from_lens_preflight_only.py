#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromLensPreflightOnly."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_preflight_only(
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
    out_max_piece_len: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
        or out_required_token_capacity is None
        or out_max_piece_len is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    if out_token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_required = [0]
    staged_max_piece_len = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
        staged_required,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required[0] > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
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
    out_token_capacity: int,
    out_required_token_capacity: list[int] | None,
    out_max_piece_len: list[int] | None,
) -> int:
    staged_required = [0]
    staged_max_piece_len = [0]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
        staged_required,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required[0] > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = staged_required[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    return TOKENIZER_BPE_OK


def test_source_contains_preflight_only_wrapper_signature_and_atomic_publish() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensPreflightOnly("
    assert sig in source

    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityPreflightOnly(",
        1,
    )[0]
    assert "if (out_token_capacity > 0x7FFFFFFFFFFFFFFF)" in body
    assert (
        "TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityPreflightOnly("
        in body
    )
    assert "if (staged_required_token_capacity > out_token_capacity)" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_max_piece_len = staged_max_piece_len;" in body


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


def test_success_fixture_and_atomic_outputs() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    out_req_a = [0xAAAA]
    out_req_b = [0xBBBB]
    out_max_a = [0xCCCC]
    out_max_b = [0xDDDD]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_preflight_only(
        payload,
        len(payload),
        [0],
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
        out_req_a,
        out_max_a,
    )
    err_b = explicit_checked_composition(
        payload,
        len(payload),
        [0],
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
        out_req_b,
        out_max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert out_req_a == out_req_b == [len(payload)]
    assert out_max_a == out_max_b == [max(vocab_lens)]


def test_capacity_error_is_no_write() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abc".encode("utf-8"))
    vocab_lens = [1, 2, 3]

    out_req = [0x1111]
    out_max = [0x2222]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_preflight_only(
        payload,
        len(payload),
        [0],
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
        len(payload) - 1,
        out_req,
        out_max,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_req == [0x1111]
    assert out_max == [0x2222]


def test_fuzz_parity_against_explicit_checked_composition() -> None:
    random.seed(694)

    left, right, ranks, merged = _build_rank_tables()

    for _ in range(300):
        nbytes = random.randint(0, 96)
        payload = [random.randint(0, 255) for _ in range(nbytes)]

        if random.random() < 0.25:
            cursor = random.randint(nbytes + 1, nbytes + 6)
        else:
            cursor = random.randint(0, nbytes)

        if cursor <= nbytes:
            max_prompt = nbytes - cursor
            if random.random() < 0.25:
                prompt_nbytes = max_prompt + random.randint(1, 5)
            else:
                prompt_nbytes = random.randint(0, max_prompt)
        else:
            prompt_nbytes = random.randint(0, nbytes + 4)

        vocab_count = random.randint(0, 16)
        vocab_capacity = vocab_count + random.randint(0, 2)
        if random.random() < 0.1:
            vocab_capacity = max(0, vocab_count - 1)

        vocab_lens = [random.randint(0, 12) for _ in range(vocab_count)]

        if random.random() < 0.1:
            rank_table_count = len(ranks) + random.randint(1, 3)
            rank_table_capacity = len(ranks)
        else:
            rank_table_count = len(ranks)
            rank_table_capacity = len(ranks)

        out_cap = random.randint(0, nbytes + 8)

        req_a = [0x3333]
        req_b = [0x4444]
        max_a = [0x5555]
        max_b = [0x6666]
        cur_a = [cursor]
        cur_b = [cursor]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_preflight_only(
            payload,
            nbytes,
            cur_a,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_table_count,
            rank_table_capacity,
            vocab_lens,
            vocab_count,
            vocab_capacity,
            out_cap,
            req_a,
            max_a,
        )
        err_b = explicit_checked_composition(
            payload,
            nbytes,
            cur_b,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_table_count,
            rank_table_capacity,
            vocab_lens,
            vocab_count,
            vocab_capacity,
            out_cap,
            req_b,
            max_b,
        )

        assert err_a == err_b
        assert cur_a == cur_b == [cursor]
        if err_a == TOKENIZER_BPE_OK:
            assert req_a == req_b == [prompt_nbytes]
            assert max_a == max_b == [max(vocab_lens, default=0)]
        else:
            assert req_a == [0x3333]
            assert max_a == [0x5555]
            assert req_b == [0x4444]
            assert max_b == [0x6666]
