#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptCheckedViaSpanTable."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_prompt_checked import (
    I32_BYTES,
    I64_MAX,
    TOKENIZER_BPE_ASCII_CLASS_DIGIT,
    TOKENIZER_BPE_ASCII_CLASS_PUNCT,
    TOKENIZER_BPE_ASCII_CLASS_WHITESPACE,
    TOKENIZER_BPE_ASCII_CLASS_WORD,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
    tokenizer_bpe_encode_prompt_checked,
    tokenizer_bpe_encode_span_checked,
)
from test_tokenizer_bpe_prompt_span_scan_checked import (
    TOKENIZER_BPE_ASCII_CLASS_NON_ASCII,
    explicit_prompt_spans_with_classes,
    tokenizer_bpe_prompt_span_scan_checked,
)


def tokenizer_bpe_encode_prompt_checked_via_span_table(
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
    out_token_ids: list[int] | None,
    out_token_capacity: int,
    out_token_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if (
        byte_len > I64_MAX
        or prompt_nbytes > I64_MAX
        or rank_table_capacity > I64_MAX
        or out_token_capacity > I64_MAX
    ):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if out_token_capacity != 0 and out_token_capacity > (I64_MAX // I32_BYTES):
        return TOKENIZER_BPE_ERR_OVERFLOW

    if rank_table_count > rank_table_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if rank_table_count > 0 and (
        rank_left_tokens is None
        or rank_right_tokens is None
        or rank_values is None
        or rank_merged_tokens is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    prompt_end = cursor + prompt_nbytes
    if prompt_end < cursor or prompt_end > byte_len:
        return TOKENIZER_BPE_ERR_OVERFLOW

    span_capacity = prompt_nbytes
    span_starts = [0] * max(1, span_capacity)
    span_lengths = [0] * max(1, span_capacity)
    span_classes = [0] * max(1, span_capacity)
    span_count = [0]
    scan_cursor = [cursor]

    err = tokenizer_bpe_prompt_span_scan_checked(
        data,
        byte_len,
        scan_cursor,
        prompt_nbytes,
        span_starts,
        span_lengths,
        span_classes,
        span_capacity,
        span_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if scan_cursor[0] != prompt_end or span_count[0] > span_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_tokens = [0] * max(1, out_token_capacity)
    staged_token_count = 0
    expected_span_start = cursor

    for span_idx in range(span_count[0]):
        span_start = span_starts[span_idx]
        span_nbytes = span_lengths[span_idx]
        span_class = span_classes[span_idx]

        if span_class not in (
            TOKENIZER_BPE_ASCII_CLASS_WORD,
            TOKENIZER_BPE_ASCII_CLASS_DIGIT,
            TOKENIZER_BPE_ASCII_CLASS_WHITESPACE,
            TOKENIZER_BPE_ASCII_CLASS_PUNCT,
            TOKENIZER_BPE_ASCII_CLASS_NON_ASCII,
        ):
            return TOKENIZER_BPE_ERR_BAD_PARAM

        if span_nbytes == 0 or span_start != expected_span_start or span_start >= prompt_end:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        span_end = span_start + span_nbytes
        if span_end < span_start or span_end > prompt_end:
            return TOKENIZER_BPE_ERR_OVERFLOW

        if staged_token_count > out_token_capacity:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        remaining_capacity = out_token_capacity - staged_token_count
        span_cursor = [span_start]
        span_token_count = [0]
        span_out = [0] * max(1, remaining_capacity)

        err = tokenizer_bpe_encode_span_checked(
            data,
            byte_len,
            span_cursor,
            span_nbytes,
            rank_left_tokens,
            rank_right_tokens,
            rank_values,
            rank_merged_tokens,
            rank_table_count,
            rank_table_capacity,
            span_out,
            remaining_capacity,
            span_token_count,
        )
        if err != TOKENIZER_BPE_OK:
            return err

        if span_cursor[0] != span_end or span_token_count[0] > remaining_capacity:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        for idx in range(span_token_count[0]):
            staged_tokens[staged_token_count + idx] = span_out[idx]

        staged_token_count += span_token_count[0]
        if staged_token_count < span_token_count[0] or staged_token_count > out_token_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

        expected_span_start = span_end

    if expected_span_start != prompt_end:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(staged_token_count):
        out_token_ids[i] = staged_tokens[i]

    out_token_count[0] = staged_token_count
    io_cursor[0] = prompt_end
    return TOKENIZER_BPE_OK


def test_known_llamacpp_style_prompt_fixture() -> None:
    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    rank_entries = sorted(
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

    left = [e[0] for e in rank_entries]
    right = [e[1] for e in rank_entries]
    ranks = [e[2] for e in rank_entries]
    merged = [e[3] for e in rank_entries]

    cursor = [0]
    out_tokens = [0x7E7E] * 256
    out_count = [0xAAAA]

    err = tokenizer_bpe_encode_prompt_checked_via_span_table(
        payload,
        len(payload),
        cursor,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(rank_entries),
        len(rank_entries),
        out_tokens,
        len(out_tokens),
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == len(payload)
    assert out_tokens[: out_count[0]] == [500, ord(","), ord(" "), 501, ord(" "), 502, ord("\t"), 503, ord("!"), ord(" "), *list("Καλημέρα 世界".encode("utf-8"))]


def test_matches_core_prompt_encoder_on_multilingual_and_malformed_inputs() -> None:
    rng = random.Random(20260418_405)
    corpus = [
        "alpha BETA_12",
        "Русский текст",
        "漢字かな交じり",
        "🙂🙃",
        "TempleOS 1996!",
        "a_b c-d e+f",
        "Καλημέρα κόσμε",
    ]

    for _ in range(1200):
        if rng.random() < 0.2:
            payload = [0xE2, 0x82] if rng.random() < 0.5 else [0xF0, 0x9F, 0x99]
        else:
            text = rng.choice(corpus)
            if rng.randint(0, 1):
                text += chr(rng.randint(32, 126))
            payload = list(text.encode("utf-8"))

        rank_entries: list[tuple[int, int, int, int]] = []
        for _ in range(rng.randint(0, 26)):
            left = rng.randint(32, 2200)
            right = rng.randint(32, 2200)
            rank = rng.randint(0, 200)
            merged = rng.randint(256, 12000)
            rank_entries.append((left, right, rank, merged))
        rank_entries.sort(key=lambda item: (item[0], item[1]))

        lefts = [e[0] for e in rank_entries]
        rights = [e[1] for e in rank_entries]
        ranks = [e[2] for e in rank_entries]
        merged = [e[3] for e in rank_entries]

        out_cap = max(1, len(payload) + 8)

        core_cursor = [0]
        core_out = [0x1111] * out_cap
        core_count = [0x2222]
        err_core = tokenizer_bpe_encode_prompt_checked(
            payload,
            len(payload),
            core_cursor,
            len(payload),
            lefts,
            rights,
            ranks,
            merged,
            len(rank_entries),
            len(rank_entries),
            core_out,
            out_cap,
            core_count,
        )

        via_cursor = [0]
        via_out = [0x1111] * out_cap
        via_count = [0x2222]
        err_via = tokenizer_bpe_encode_prompt_checked_via_span_table(
            payload,
            len(payload),
            via_cursor,
            len(payload),
            lefts,
            rights,
            ranks,
            merged,
            len(rank_entries),
            len(rank_entries),
            via_out,
            out_cap,
            via_count,
        )

        assert err_via == err_core
        assert via_cursor[0] == core_cursor[0]
        assert via_count[0] == core_count[0]
        assert via_out == core_out


def test_span_table_partition_matches_prompt_payload_partition() -> None:
    payload = list("alpha_12  +\tβeta🙂漢字!!".encode("utf-8"))

    span_starts = [0] * len(payload)
    span_lengths = [0] * len(payload)
    span_classes = [0] * len(payload)
    span_count = [0]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        span_starts,
        span_lengths,
        span_classes,
        len(payload),
        span_count,
    )
    assert err == TOKENIZER_BPE_OK

    expected = explicit_prompt_spans_with_classes(payload)
    assert span_count[0] == len(expected)
    for idx, (start, length, cls) in enumerate(expected):
        assert span_starts[idx] == start
        assert span_lengths[idx] == length
        assert span_classes[idx] == cls


if __name__ == "__main__":
    test_known_llamacpp_style_prompt_fixture()
    test_matches_core_prompt_encoder_on_multilingual_and_malformed_inputs()
    test_span_table_partition_matches_prompt_payload_partition()
    print("tokenizer_bpe_encode_prompt_checked_via_span_table_reference_checks=ok")
