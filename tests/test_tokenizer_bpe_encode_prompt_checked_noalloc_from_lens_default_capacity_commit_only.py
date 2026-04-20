#!/usr/bin/env python3
"""Parity harness for ...NoAllocFromLensDefaultCapacityCommitOnly."""

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
from test_tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity import (
    tokenizer_bpe_encode_prompt_checked_noalloc_from_lens,
)


def tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only(
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

    derived_out_token_capacity = prompt_nbytes

    staged_cursor = [cursor]
    staged_token_count = [out_token_count[0]]
    staged_required_token_capacity = [out_required_token_capacity[0]]
    staged_max_piece_len = [out_max_piece_len[0]]

    stage_capacity = max(1, derived_out_token_capacity)
    if stage_capacity > I64_MAX // 4:
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_tokens = [0] * stage_capacity

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        staged_tokens,
        derived_out_token_capacity,
        staged_token_count,
        staged_required_token_capacity,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required_token_capacity[0] > derived_out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if staged_token_count[0] > derived_out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_token_count[0]):
        out_token_ids[i] = staged_tokens[i]

    out_token_count[0] = staged_token_count[0]
    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
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

    derived_out_token_capacity = prompt_nbytes

    staged_cursor = [cursor]
    staged_token_count = [out_token_count[0]]
    staged_required_token_capacity = [out_required_token_capacity[0]]
    staged_max_piece_len = [out_max_piece_len[0]]
    stage_capacity = max(1, derived_out_token_capacity)
    staged_tokens = [0] * stage_capacity

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens(
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
        vocab_piece_lens,
        vocab_piece_count,
        vocab_piece_capacity,
        staged_tokens,
        derived_out_token_capacity,
        staged_token_count,
        staged_required_token_capacity,
        staged_max_piece_len,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_required_token_capacity[0] > derived_out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if staged_token_count[0] > derived_out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_token_count[0]):
        out_token_ids[i] = staged_tokens[i]

    out_token_count[0] = staged_token_count[0]
    out_required_token_capacity[0] = staged_required_token_capacity[0]
    out_max_piece_len[0] = staged_max_piece_len[0]
    io_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def test_source_contains_default_capacity_commit_only_signature_and_staged_commit() -> None:
    source = Path("src/tokenizer/bpe.HC").read_text(encoding="utf-8")
    sig = "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 TokenizerBPEEncodePromptCheckedNoAllocFromLensDefaultCapacityPreflightOnly(",
        1,
    )[0]
    assert "derived_out_token_capacity = prompt_nbytes;" in body
    assert "staged_cursor = cursor;" in body
    assert "staged_token_count = *out_token_count;" in body
    assert "staged_required_token_capacity = *out_required_token_capacity;" in body
    assert "staged_max_piece_len = *out_max_piece_len;" in body
    assert "TokenizerBPEEncodePromptCheckedNoAllocFromLens(" in body
    assert "for (i = 0; i < staged_token_count; i++)" in body
    assert "*out_token_count = staged_token_count;" in body
    assert "*out_required_token_capacity = staged_required_token_capacity;" in body
    assert "*out_max_piece_len = staged_max_piece_len;" in body
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


def test_success_fixture_parity_and_commit() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("hello Καλημέρα 世界\n🙂 xyz".encode("utf-8"))
    vocab_lens = [1, 2, 4, 3, 7, 5, 1, 9]

    cursor_a = [0]
    cursor_b = [0]
    out_a = [0x5A5A] * 512
    out_b = [0x6B6B] * 512
    count_a = [0xAAAA]
    count_b = [0xBBBB]
    req_a = [0xCCCC]
    req_b = [0xDDDD]
    max_a = [0xEEEE]
    max_b = [0xFFFF]

    err_a = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only(
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
        count_b,
        req_b,
        max_b,
    )

    assert err_a == err_b == TOKENIZER_BPE_OK
    assert cursor_a == cursor_b == [len(payload)]
    assert count_a == count_b
    assert req_a == req_b == [len(payload)]
    assert max_a == max_b == [max(vocab_lens)]
    assert out_a[: count_a[0]] == out_b[: count_b[0]]


def test_error_no_partial_writes() -> None:
    left, right, ranks, merged = _build_rank_tables()
    payload = list("abc".encode("utf-8"))
    vocab_lens = [1, 2, 3]

    cursor = [2]
    out_ids = [0x1111] * 16
    out_count = [0x2222]
    out_req = [0x3333]
    out_max = [0x4444]

    err = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only(
        payload,
        len(payload),
        cursor,
        2,
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        vocab_lens,
        len(vocab_lens),
        len(vocab_lens),
        out_ids,
        out_count,
        out_req,
        out_max,
    )

    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor == [2]
    assert out_ids == [0x1111] * 16
    assert out_count == [0x2222]
    assert out_req == [0x3333]
    assert out_max == [0x4444]


def test_fuzz_parity_against_explicit_checked_composition() -> None:
    random.seed(695)
    left, right, ranks, merged = _build_rank_tables()

    for _ in range(300):
        nbytes = random.randint(0, 96)
        payload = [random.randint(0, 255) for _ in range(nbytes)]

        if random.random() < 0.2:
            cursor_seed = random.randint(nbytes + 1, nbytes + 5)
        else:
            cursor_seed = random.randint(0, nbytes)

        if cursor_seed <= nbytes:
            max_prompt = nbytes - cursor_seed
            if random.random() < 0.2:
                prompt_nbytes = max_prompt + random.randint(1, 6)
            else:
                prompt_nbytes = random.randint(0, max_prompt)
        else:
            prompt_nbytes = random.randint(0, nbytes + 6)

        vocab_piece_count = random.randint(0, 12)
        vocab_piece_capacity = vocab_piece_count + random.randint(0, 3)
        if random.random() < 0.15 and vocab_piece_count:
            vocab_piece_capacity = vocab_piece_count - 1
        vocab_lens = [random.randint(0, 16) for _ in range(vocab_piece_capacity)]

        rank_table_count = len(ranks)
        rank_table_capacity = rank_table_count
        if random.random() < 0.08 and rank_table_count:
            rank_table_capacity = rank_table_count - 1

        c1 = [cursor_seed]
        c2 = [cursor_seed]

        out1 = [0xA1A1] * 256
        out2 = [0xB2B2] * 256
        count1 = [0xC3C3]
        count2 = [0xD4D4]
        req1 = [0xE5E5]
        req2 = [0xF6F6]
        max1 = [0x1717]
        max2 = [0x2828]

        err1 = tokenizer_bpe_encode_prompt_checked_noalloc_from_lens_default_capacity_commit_only(
            payload,
            nbytes,
            c1,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_table_count,
            rank_table_capacity,
            vocab_lens,
            vocab_piece_count,
            vocab_piece_capacity,
            out1,
            count1,
            req1,
            max1,
        )
        err2 = explicit_checked_composition(
            payload,
            nbytes,
            c2,
            prompt_nbytes,
            left,
            right,
            ranks,
            merged,
            rank_table_count,
            rank_table_capacity,
            vocab_lens,
            vocab_piece_count,
            vocab_piece_capacity,
            out2,
            count2,
            req2,
            max2,
        )

        assert err1 == err2
        assert c1 == c2
        if err1 == TOKENIZER_BPE_OK:
            assert count1 == count2
            assert req1 == req2
            assert max1 == max2
            assert out1[: count1[0]] == out2[: count2[0]]
        else:
            assert count1 == [0xC3C3]
            assert count2 == [0xD4D4]
            assert req1 == [0xE5E5]
            assert req2 == [0xF6F6]
            assert max1 == [0x1717]
            assert max2 == [0x2828]
            assert out1 == [0xA1A1] * 256
            assert out2 == [0xB2B2] * 256


if __name__ == "__main__":
    test_source_contains_default_capacity_commit_only_signature_and_staged_commit()
    test_success_fixture_parity_and_commit()
    test_error_no_partial_writes()
    test_fuzz_parity_against_explicit_checked_composition()
    print("ok")
