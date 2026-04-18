#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodePromptChecked semantics."""

from __future__ import annotations

import random

TOKENIZER_UTF8_OK = 0
TOKENIZER_UTF8_ERR_NULL_PTR = 1
TOKENIZER_UTF8_ERR_BAD_PARAM = 2
TOKENIZER_UTF8_ERR_OVERFLOW = 3
TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS = 4
TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE = 5
TOKENIZER_UTF8_ERR_BAD_CONTINUATION = 6
TOKENIZER_UTF8_ERR_BAD_CODEPOINT = 7
TOKENIZER_UTF8_ERR_TRUNCATED = 8

TOKENIZER_BPE_OK = 0
TOKENIZER_BPE_ERR_NULL_PTR = 101
TOKENIZER_BPE_ERR_BAD_PARAM = 102
TOKENIZER_BPE_ERR_OVERFLOW = 103

I64_MAX = (1 << 63) - 1
I32_BYTES = 4

TOKENIZER_BPE_ASCII_CLASS_WORD = 1
TOKENIZER_BPE_ASCII_CLASS_DIGIT = 2
TOKENIZER_BPE_ASCII_CLASS_WHITESPACE = 3
TOKENIZER_BPE_ASCII_CLASS_PUNCT = 4


def tokenizer_bpe_ascii_class(byte: int) -> int:
    if (ord("a") <= byte <= ord("z")) or (ord("A") <= byte <= ord("Z")) or byte == ord("_"):
        return TOKENIZER_BPE_ASCII_CLASS_WORD
    if ord("0") <= byte <= ord("9"):
        return TOKENIZER_BPE_ASCII_CLASS_DIGIT
    if byte in (ord(" "), ord("\t"), ord("\n"), ord("\r"), ord("\f"), ord("\v")):
        return TOKENIZER_BPE_ASCII_CLASS_WHITESPACE
    return TOKENIZER_BPE_ASCII_CLASS_PUNCT


def tokenizer_utf8_is_continuation_byte(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def tokenizer_utf8_expected_length(lead: int, out_need: list[int] | None) -> int:
    if out_need is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR
    if lead <= 0x7F:
        out_need[0] = 1
        return TOKENIZER_UTF8_OK
    if 0xC2 <= lead <= 0xDF:
        out_need[0] = 2
        return TOKENIZER_UTF8_OK
    if 0xE0 <= lead <= 0xEF:
        out_need[0] = 3
        return TOKENIZER_UTF8_OK
    if 0xF0 <= lead <= 0xF4:
        out_need[0] = 4
        return TOKENIZER_UTF8_OK
    return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE


def tokenizer_utf8_validate_scalar(codepoint: int, out_cp: list[int] | None) -> int:
    if out_cp is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR
    if 0xD800 <= codepoint <= 0xDFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
    if codepoint > 0x10FFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
    out_cp[0] = codepoint
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_next_codepoint_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_codepoint: list[int] | None,
    out_bytes_consumed: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_codepoint is None or out_bytes_consumed is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR
    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM
    if cursor == byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    lead = data[cursor]
    need = [0]
    err = tokenizer_utf8_expected_length(lead, need)
    if err != TOKENIZER_UTF8_OK:
        return err

    width = need[0]
    if byte_len - cursor < width:
        return TOKENIZER_UTF8_ERR_TRUNCATED

    if width == 1:
        codepoint = lead
    elif width == 2:
        codepoint = lead & 0x1F
    elif width == 3:
        codepoint = lead & 0x0F
    else:
        codepoint = lead & 0x07

    if width > 1:
        b = data[cursor + 1]
        if not tokenizer_utf8_is_continuation_byte(b):
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
        codepoint = (codepoint << 6) | (b & 0x3F)

    if width > 2:
        b = data[cursor + 2]
        if not tokenizer_utf8_is_continuation_byte(b):
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
        codepoint = (codepoint << 6) | (b & 0x3F)

    if width > 3:
        b = data[cursor + 3]
        if not tokenizer_utf8_is_continuation_byte(b):
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
        codepoint = (codepoint << 6) | (b & 0x3F)

    if width == 2 and codepoint < 0x80:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
    if width == 3 and codepoint < 0x800:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
    if width == 4 and codepoint < 0x10000:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    checked = [0]
    err = tokenizer_utf8_validate_scalar(codepoint, checked)
    if err != TOKENIZER_UTF8_OK:
        return err

    io_cursor[0] = cursor + width
    out_codepoint[0] = codepoint
    out_bytes_consumed[0] = width
    return TOKENIZER_UTF8_OK


def bpe_merge_pair_token_lookup_checked(
    left_token: int,
    right_token: int,
    rank_left_tokens: list[int] | None,
    rank_right_tokens: list[int] | None,
    rank_values: list[int] | None,
    rank_merged_tokens: list[int] | None,
    rank_table_count: int,
    rank_table_capacity: int,
    out_merged_token: list[int] | None,
    out_rank: list[int] | None,
    out_found: list[bool] | None,
) -> int:
    if out_merged_token is None or out_rank is None or out_found is None:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if rank_table_capacity > I64_MAX:
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

    found = False
    best_rank = 0
    best_merged = 0
    for i in range(rank_table_count):
        if rank_left_tokens[i] != left_token or rank_right_tokens[i] != right_token:
            continue
        if not found or rank_values[i] < best_rank:
            found = True
            best_rank = rank_values[i]
            best_merged = rank_merged_tokens[i]

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_found[0] = True
    out_rank[0] = best_rank
    out_merged_token[0] = best_merged
    return TOKENIZER_BPE_OK


def bpe_merge_apply_at_index_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    left_index: int,
    merged_token: int,
    out_next_count: list[int] | None,
) -> int:
    if token_ids is None or out_next_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if token_count > token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if token_count < 2 or left_index >= token_count - 1:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    token_ids[left_index] = merged_token
    write = left_index + 1
    read = left_index + 2
    while read < token_count:
        token_ids[write] = token_ids[read]
        write += 1
        read += 1

    out_next_count[0] = token_count - 1
    return TOKENIZER_BPE_OK


def tokenizer_bpe_encode_span_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    span_nbytes: int,
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
    if byte_len > I64_MAX or out_token_capacity > I64_MAX or rank_table_capacity > I64_MAX:
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
    if span_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    span_end = cursor + span_nbytes
    if span_end < cursor or span_end > byte_len:
        return TOKENIZER_BPE_ERR_OVERFLOW

    scan = cursor
    while scan < span_end:
        cp = [0]
        used = [0]
        err = tokenizer_utf8_next_codepoint_checked(data, span_end, [scan], cp, used)
        if err != TOKENIZER_UTF8_OK:
            return err
        scan += used[0]

    if span_nbytes > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    working = [data[cursor + i] for i in range(span_nbytes)]
    working_count = span_nbytes

    while working_count >= 2:
        best_found = False
        best_left = 0
        best_rank = 0
        best_merged = 0

        for i in range(working_count - 1):
            merged = [0]
            rank = [0]
            found = [False]
            err = bpe_merge_pair_token_lookup_checked(
                working[i],
                working[i + 1],
                rank_left_tokens,
                rank_right_tokens,
                rank_values,
                rank_merged_tokens,
                rank_table_count,
                rank_table_capacity,
                merged,
                rank,
                found,
            )
            if err != TOKENIZER_BPE_OK:
                return err
            if not found[0]:
                continue
            if not best_found or rank[0] < best_rank:
                best_found = True
                best_left = i
                best_rank = rank[0]
                best_merged = merged[0]

        if not best_found:
            break

        next_count = [0]
        err = bpe_merge_apply_at_index_checked(
            working,
            working_count,
            len(working),
            best_left,
            best_merged,
            next_count,
        )
        if err != TOKENIZER_BPE_OK:
            return err
        working_count = next_count[0]

    if working_count > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(working_count):
        out_token_ids[i] = working[i]

    out_token_count[0] = working_count
    io_cursor[0] = span_end
    return TOKENIZER_BPE_OK


def tokenizer_bpe_encode_prompt_checked(
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

    staged = [0] * max(out_token_capacity, 1)
    staged_count = 0
    scan = cursor

    while scan < prompt_end:
        span_start = scan
        lead = data[scan]

        if lead <= 0x7F:
            span_class = tokenizer_bpe_ascii_class(lead)
            scan += 1
            while scan < prompt_end and data[scan] <= 0x7F and tokenizer_bpe_ascii_class(data[scan]) == span_class:
                scan += 1
        else:
            cp = [0]
            used = [0]
            err = tokenizer_utf8_next_codepoint_checked(data, prompt_end, [scan], cp, used)
            if err != TOKENIZER_UTF8_OK:
                return err
            scan += used[0]
            while scan < prompt_end and data[scan] > 0x7F:
                err = tokenizer_utf8_next_codepoint_checked(data, prompt_end, [scan], cp, used)
                if err != TOKENIZER_UTF8_OK:
                    return err
                scan += used[0]

        span_nbytes = scan - span_start
        if span_nbytes == 0:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        if staged_count > out_token_capacity:
            return TOKENIZER_BPE_ERR_BAD_PARAM
        remaining = out_token_capacity - staged_count

        span_cursor = [span_start]
        span_count = [0]
        span_out = [0] * max(remaining, 1)
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
            remaining,
            span_count,
        )
        if err != TOKENIZER_BPE_OK:
            return err
        if span_cursor[0] != scan or span_count[0] > remaining:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        for i in range(span_count[0]):
            staged[staged_count + i] = span_out[i]
        staged_count += span_count[0]
        if staged_count > out_token_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

    for i in range(staged_count):
        out_token_ids[i] = staged[i]
    out_token_count[0] = staged_count
    io_cursor[0] = prompt_end
    return TOKENIZER_BPE_OK


def explicit_prompt_spans(payload: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    scan = 0
    while scan < len(payload):
        start = scan
        lead = payload[scan]
        if lead <= 0x7F:
            span_class = tokenizer_bpe_ascii_class(lead)
            scan += 1
            while scan < len(payload) and payload[scan] <= 0x7F and tokenizer_bpe_ascii_class(payload[scan]) == span_class:
                scan += 1
            spans.append((start, scan - start))
            continue

        cp = [0]
        used = [0]
        tmp = [scan]
        err = tokenizer_utf8_next_codepoint_checked(payload, len(payload), tmp, cp, used)
        assert err == TOKENIZER_UTF8_OK
        scan = tmp[0]

        while scan < len(payload) and payload[scan] > 0x7F:
            tmp = [scan]
            err = tokenizer_utf8_next_codepoint_checked(payload, len(payload), tmp, cp, used)
            assert err == TOKENIZER_UTF8_OK
            scan = tmp[0]

        spans.append((start, scan - start))

    return spans


def run_prompt(payload: bytes, rank_entries: list[tuple[int, int, int, int]], out_cap: int = 128) -> tuple[int, int, list[int], int]:
    entries = sorted(rank_entries, key=lambda item: (item[0], item[1]))
    left = [e[0] for e in entries]
    right = [e[1] for e in entries]
    ranks = [e[2] for e in entries]
    merged = [e[3] for e in entries]

    data = list(payload)
    cursor = [0]
    out = [0xBEEF] * max(out_cap, 1)
    out_count = [0xBAD]

    err = tokenizer_bpe_encode_prompt_checked(
        data,
        len(data),
        cursor,
        len(data),
        left,
        right,
        ranks,
        merged,
        len(entries),
        len(entries),
        out,
        out_cap,
        out_count,
    )

    return err, cursor[0], out, out_count[0]


def test_known_llamacpp_style_prompt_fixture() -> None:
    payload = b"hello, world 123\tgo!"
    err, cursor, out, out_count = run_prompt(
        payload,
        [
            (108, 108, 1, 300),
            (200, 300, 2, 400),
            (104, 101, 3, 200),
            (400, 111, 0, 500),  # hello -> 500
            (119, 111, 1, 210),
            (210, 114, 2, 220),
            (220, 108, 3, 230),
            (230, 100, 0, 501),  # world -> 501
            (49, 50, 1, 310),
            (310, 51, 0, 502),  # 123 -> 502
            (103, 111, 0, 503),  # go -> 503
        ],
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor == len(payload)
    assert out_count == 9
    assert out[:out_count] == [500, ord(","), ord(" "), 501, ord(" "), 502, ord("\t"), 503, ord("!")]


def test_multilingual_prompt_and_ascii_tail_merges() -> None:
    payload = "Русский🙂test".encode("utf-8")

    # Keep only ASCII word merges for the tail "test".
    err, cursor, out, out_count = run_prompt(
        payload,
        [
            (116, 101, 2, 710),
            (710, 115, 1, 711),
            (711, 116, 0, 712),
        ],
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor == len(payload)

    prefix = list("Русский🙂".encode("utf-8"))
    assert out[: out_count - 1] == prefix
    assert out[out_count - 1] == 712


def test_malformed_utf8_no_partial_writes() -> None:
    bad = [0xE2, 0x82]
    cursor = [0]
    out = [1234, 1234, 1234]
    out_count = [777]

    err = tokenizer_bpe_encode_prompt_checked(
        bad,
        len(bad),
        cursor,
        len(bad),
        [],
        [],
        [],
        [],
        0,
        0,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_TRUNCATED
    assert cursor[0] == 0
    assert out_count[0] == 777
    assert out == [1234, 1234, 1234]


def test_capacity_adversarial_no_partial_writes() -> None:
    payload = list(b"abc def")
    cursor = [0]
    out = [4321, 4321, 4321]
    out_count = [888]

    err = tokenizer_bpe_encode_prompt_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        [],
        [],
        [],
        [],
        0,
        0,
        out,
        2,
        out_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_count[0] == 888
    assert out == [4321, 4321, 4321]


def test_randomized_prompt_equals_explicit_span_composition() -> None:
    rng = random.Random(20260418_372)

    corpus = [
        "alpha BETA_12",
        "Русский текст",
        "漢字かな交じり",
        "🙂🙃",
        "TempleOS 1996!",
        "a_b c-d e+f",
    ]

    for _ in range(1400):
        text = rng.choice(corpus)
        if rng.randint(0, 1):
            text += chr(rng.randint(32, 126))

        payload = list(text.encode("utf-8"))

        entries: list[tuple[int, int, int, int]] = []
        for _ in range(rng.randint(0, 24)):
            left = rng.randint(32, 2000)
            right = rng.randint(32, 2000)
            rank = rng.randint(0, 200)
            merged = rng.randint(256, 9000)
            entries.append((left, right, rank, merged))
        entries.sort(key=lambda item: (item[0], item[1]))

        lefts = [e[0] for e in entries]
        rights = [e[1] for e in entries]
        ranks = [e[2] for e in entries]
        merged = [e[3] for e in entries]

        # Prompt path.
        cursor = [0]
        out_prompt = [0x5555] * max(len(payload) + 8, 1)
        out_prompt_count = [0]
        err = tokenizer_bpe_encode_prompt_checked(
            payload,
            len(payload),
            cursor,
            len(payload),
            lefts,
            rights,
            ranks,
            merged,
            len(entries),
            len(entries),
            out_prompt,
            len(out_prompt),
            out_prompt_count,
        )
        assert err == TOKENIZER_BPE_OK
        assert cursor[0] == len(payload)

        # Explicit per-span composition path.
        explicit_tokens: list[int] = []
        for span_start, span_len in explicit_prompt_spans(payload):
            span_cursor = [span_start]
            span_out = [0] * (len(payload) + 4)
            span_count = [0]
            err = tokenizer_bpe_encode_span_checked(
                payload,
                len(payload),
                span_cursor,
                span_len,
                lefts,
                rights,
                ranks,
                merged,
                len(entries),
                len(entries),
                span_out,
                len(span_out),
                span_count,
            )
            assert err == TOKENIZER_BPE_OK
            assert span_cursor[0] == span_start + span_len
            explicit_tokens.extend(span_out[: span_count[0]])

        assert out_prompt[: out_prompt_count[0]] == explicit_tokens


if __name__ == "__main__":
    test_known_llamacpp_style_prompt_fixture()
    test_multilingual_prompt_and_ascii_tail_merges()
    test_malformed_utf8_no_partial_writes()
    test_capacity_adversarial_no_partial_writes()
    test_randomized_prompt_equals_explicit_span_composition()
    print("tokenizer_bpe_encode_prompt_checked_reference_checks=ok")
