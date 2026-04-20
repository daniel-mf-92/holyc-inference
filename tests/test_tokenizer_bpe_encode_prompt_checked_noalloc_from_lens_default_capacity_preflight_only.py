#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromLensDefaultCapacityPreflightOnly."""

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


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
) -> int:
    if (
        data is None
        or io_cursor is None
        or vocab_piece_lens is None
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

    max_piece_len = 0
    for i in range(vocab_piece_count):
        if vocab_piece_lens[i] > I64_MAX:
            return TOKENIZER_BPE_ERR_OVERFLOW
        if vocab_piece_lens[i] > max_piece_len:
            max_piece_len = vocab_piece_lens[i]

    required_token_capacity = prompt_nbytes
    if prompt_nbytes and max_piece_len == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    out_required_token_capacity[0] = required_token_capacity
    out_max_piece_len[0] = max_piece_len
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
    out_required_token_capacity: list[int] | None,
    out_max_piece_len: list[int] | None,
) -> int:
    return tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
    )


def test_source_contains_preflight_helper_signature_and_staged_commits() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEDecodeTokenSpanChecked", 1
    )[0]
    assert "if (rank_table_count > rank_table_capacity)" in body
    assert "if (vocab_piece_count > vocab_piece_capacity)" in body
    assert "if (cursor > byte_len)" in body
    assert "if (prompt_nbytes > byte_len - cursor)" in body
    assert "while (piece_index < vocab_piece_count)" in body
    assert "if (prompt_nbytes && !staged_max_piece_len)" in body
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


def test_multilingual_fixture_preflight_parity() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    req_a = [0xAAAA]
    req_b = [0xBBBB]
    max_a = [0xCCCC]
    max_b = [0xDDDD]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
        req_a,
        max_a,
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
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert req_a[0] == req_b[0] == len(payload)
    assert max_a[0] == max_b[0] == max(vocab_lens)


def test_adversarial_errors_preserve_output_sentinels() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abc".encode("utf-8"))
    vocab_lens = [1, 2, 3]

    assert (
        tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
        )
        == TOKENIZER_BPE_ERR_NULL_PTR
    )

    out_req = [0xA1A1]
    out_max = [0xB2B2]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
        payload,
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
        out_req,
        out_max,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert out_req[0] == 0xA1A1
    assert out_max[0] == 0xB2B2

    out_req = [0xC3C3]
    out_max = [0xD4D4]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
        payload,
        len(payload),
        [4],
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
        out_req,
        out_max,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_req[0] == 0xC3C3
    assert out_max[0] == 0xD4D4

    out_req = [0xE5E5]
    out_max = [0xF6F6]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
        payload,
        len(payload),
        [0],
        4,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_req,
        out_max,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert out_req[0] == 0xE5E5
    assert out_max[0] == 0xF6F6

    out_req = [0x1111]
    out_max = [0x2222]
    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
        payload,
        len(payload),
        [0],
        1,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        [0, 0, 0],
        3,
        3,
        out_req,
        out_max,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert out_req[0] == 0x1111
    assert out_max[0] == 0x2222


def test_fuzz_parity_against_explicit_checked_composition() -> None:
    random.seed(693)

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

        req_a = [0x3333]
        req_b = [0x4444]
        max_a = [0x5555]
        max_b = [0x6666]
        cur_a = [cursor]
        cur_b = [cursor]

        err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_preflight_only(
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
