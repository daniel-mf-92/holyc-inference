#!/usr/bin/env python3
"""Parity harness for TokenizerBPEEncodeSpanChecked semantics."""

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


def is_continuation_byte(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def utf8_expected_length(lead: int, out_len: list[int] | None) -> int:
    if out_len is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR
    if lead <= 0x7F:
        out_len[0] = 1
        return TOKENIZER_UTF8_OK
    if 0xC2 <= lead <= 0xDF:
        out_len[0] = 2
        return TOKENIZER_UTF8_OK
    if 0xE0 <= lead <= 0xEF:
        out_len[0] = 3
        return TOKENIZER_UTF8_OK
    if 0xF0 <= lead <= 0xF4:
        out_len[0] = 4
        return TOKENIZER_UTF8_OK
    return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE


def utf8_validate_scalar(codepoint: int, out_codepoint: list[int] | None) -> int:
    if out_codepoint is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR
    if 0xD800 <= codepoint <= 0xDFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
    if codepoint > 0x10FFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
    out_codepoint[0] = codepoint
    return TOKENIZER_UTF8_OK


def utf8_sequence_span_valid_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    span_nbytes: int,
    out_is_valid: list[bool] | None,
) -> int:
    if data is None or io_cursor is None or out_is_valid is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR
    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM
    if span_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    span_end = cursor + span_nbytes
    if span_end < cursor or span_end > byte_len:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    scan = cursor
    while scan < span_end:
        lead = data[scan]
        need = [0]
        err = utf8_expected_length(lead, need)
        if err != TOKENIZER_UTF8_OK:
            return err

        width = need[0]
        if span_end - scan < width:
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
            b = data[scan + 1]
            if not is_continuation_byte(b):
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            codepoint = (codepoint << 6) | (b & 0x3F)

        if width > 2:
            b = data[scan + 2]
            if not is_continuation_byte(b):
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            codepoint = (codepoint << 6) | (b & 0x3F)

        if width > 3:
            b = data[scan + 3]
            if not is_continuation_byte(b):
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            codepoint = (codepoint << 6) | (b & 0x3F)

        if width == 2 and codepoint < 0x80:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
        if width == 3 and codepoint < 0x800:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
        if width == 4 and codepoint < 0x10000:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

        checked = [0]
        err = utf8_validate_scalar(codepoint, checked)
        if err != TOKENIZER_UTF8_OK:
            return err

        scan += width

    out_is_valid[0] = True
    io_cursor[0] = span_end
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
    if rank_table_count == 0:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    lo = 0
    hi = rank_table_count
    while lo < hi:
        mid = lo + ((hi - lo) >> 1)
        mid_left = rank_left_tokens[mid]
        mid_right = rank_right_tokens[mid]
        if mid_left < left_token or (mid_left == left_token and mid_right < right_token):
            lo = mid + 1
        else:
            hi = mid

    first_match = lo
    if first_match >= rank_table_count:
        out_found[0] = False
        return TOKENIZER_BPE_OK
    if (
        rank_left_tokens[first_match] != left_token
        or rank_right_tokens[first_match] != right_token
    ):
        out_found[0] = False
        return TOKENIZER_BPE_OK

    scan = first_match
    found = False
    best_rank = 0
    best_merged = 0
    while scan < rank_table_count:
        if rank_left_tokens[scan] != left_token or rank_right_tokens[scan] != right_token:
            break
        if not found or rank_values[scan] < best_rank:
            best_rank = rank_values[scan]
            best_merged = rank_merged_tokens[scan]
            found = True
        scan += 1

    if not found:
        out_found[0] = False
        return TOKENIZER_BPE_OK

    out_merged_token[0] = best_merged
    out_rank[0] = best_rank
    out_found[0] = True
    return TOKENIZER_BPE_OK


def bpe_merge_apply_at_index_checked(
    token_ids: list[int] | None,
    token_count: int,
    token_capacity: int,
    left_index: int,
    merged_token: int,
    out_token_count: list[int] | None,
) -> int:
    if token_ids is None or out_token_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR
    if token_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if token_count > token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if token_count < 2:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    if left_index >= token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM
    right_index = left_index + 1
    if right_index <= left_index:
        return TOKENIZER_BPE_ERR_OVERFLOW
    if right_index >= token_count:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    token_ids[left_index] = merged_token
    src_index = right_index + 1
    while src_index < token_count:
        dst_index = src_index - 1
        if dst_index >= token_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW
        token_ids[dst_index] = token_ids[src_index]
        src_index += 1

    out_token_count[0] = token_count - 1
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
    if byte_len > I64_MAX or rank_table_capacity > I64_MAX or out_token_capacity > I64_MAX:
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

    valid_cursor = [cursor]
    span_is_valid = [False]
    err = utf8_sequence_span_valid_checked(data, byte_len, valid_cursor, span_nbytes, span_is_valid)
    if err != TOKENIZER_UTF8_OK:
        return err
    if not span_is_valid[0] or valid_cursor[0] != span_end:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if span_nbytes > out_token_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if span_nbytes == 0:
        out_token_count[0] = 0
        io_cursor[0] = span_end
        return TOKENIZER_BPE_OK

    working = [data[cursor + i] for i in range(span_nbytes)]
    working_count = span_nbytes

    while working_count >= 2:
        best_found = False
        best_left_index = 0
        best_rank = 0
        best_merged = 0

        for i in range(working_count - 1):
            pair_merged = [0]
            pair_rank = [0]
            pair_found = [False]

            err = bpe_merge_pair_token_lookup_checked(
                working[i],
                working[i + 1],
                rank_left_tokens,
                rank_right_tokens,
                rank_values,
                rank_merged_tokens,
                rank_table_count,
                rank_table_capacity,
                pair_merged,
                pair_rank,
                pair_found,
            )
            if err != TOKENIZER_BPE_OK:
                return err

            if not pair_found[0]:
                continue

            if not best_found or pair_rank[0] < best_rank:
                best_rank = pair_rank[0]
                best_left_index = i
                best_merged = pair_merged[0]
                best_found = True

        if not best_found:
            break

        next_count = [0]
        err = bpe_merge_apply_at_index_checked(
            working,
            working_count,
            len(working),
            best_left_index,
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


def run_ok_case(
    data: bytes,
    rank_entries: list[tuple[int, int, int, int]],
    expected_tokens: list[int],
) -> None:
    # Binary-search lookup expects sorted rank key order.
    rank_entries_sorted = sorted(rank_entries, key=lambda item: (item[0], item[1]))
    left = [item[0] for item in rank_entries_sorted]
    right = [item[1] for item in rank_entries_sorted]
    ranks = [item[2] for item in rank_entries_sorted]
    merged = [item[3] for item in rank_entries_sorted]

    payload = list(data)
    cursor = [0]
    out_tokens = [0x6BADBEEF] * 64
    out_count = [0x1234]

    err = tokenizer_bpe_encode_span_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(rank_entries_sorted),
        len(rank_entries_sorted),
        out_tokens,
        len(out_tokens),
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == len(payload)
    assert out_count[0] == len(expected_tokens)
    assert out_tokens[: out_count[0]] == expected_tokens


def test_known_llamacpp_style_merge_fixtures() -> None:
    # "hello" byte-level fixture with deterministic multi-pass greedy merges.
    run_ok_case(
        b"hello",
        [
            (108, 108, 1, 300),
            (200, 300, 2, 400),
            (104, 101, 3, 200),
            (300, 111, 4, 401),
            (400, 111, 0, 500),
        ],
        [500],
    )

    # "banana" fixture with duplicate keys: lowest rank for same pair must win.
    run_ok_case(
        b"banana",
        [
            (97, 110, 7, 610),
            (97, 110, 2, 611),
            (98, 97, 3, 620),
            (611, 611, 1, 700),
            (700, 97, 0, 701),
            (620, 701, 4, 750),
        ],
        [750],
    )


def test_utf8_and_malformed_span_bounds() -> None:
    rank_entries = [(0xD0, 0xA0, 1, 4000)]
    left = [rank_entries[0][0]]
    right = [rank_entries[0][1]]
    ranks = [rank_entries[0][2]]
    merged = [rank_entries[0][3]]

    payload = list("Рус".encode("utf-8"))
    cursor = [0]
    out = [77] * 32
    out_count = [55]
    err = tokenizer_bpe_encode_span_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        left,
        right,
        ranks,
        merged,
        1,
        1,
        out,
        len(out),
        out_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == len(payload)
    assert out_count[0] >= 1

    bad = [0xE2, 0x82]
    bad_cursor = [0]
    bad_out = [99] * 8
    bad_count = [17]
    err = tokenizer_bpe_encode_span_checked(
        bad,
        len(bad),
        bad_cursor,
        len(bad),
        left,
        right,
        ranks,
        merged,
        1,
        1,
        bad_out,
        len(bad_out),
        bad_count,
    )
    assert err == TOKENIZER_UTF8_ERR_TRUNCATED
    assert bad_cursor[0] == 0
    assert bad_count[0] == 17
    assert bad_out == [99] * 8


def test_error_and_no_partial_write_contracts() -> None:
    data = list(b"abc")
    left = [97]
    right = [98]
    ranks = [1]
    merged = [700]

    sentinel_tokens = [12345, 12345, 12345]
    sentinel_count = [777]
    cursor = [0]

    err = tokenizer_bpe_encode_span_checked(
        data,
        len(data),
        cursor,
        len(data),
        left,
        right,
        ranks,
        merged,
        1,
        1,
        sentinel_tokens,
        2,
        sentinel_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert sentinel_count[0] == 777
    assert sentinel_tokens == [12345, 12345, 12345]

    null_count = tokenizer_bpe_encode_span_checked(
        None,
        0,
        [0],
        0,
        left,
        right,
        ranks,
        merged,
        1,
        1,
        [0],
        1,
        [0],
    )
    assert null_count == TOKENIZER_BPE_ERR_NULL_PTR

    overflow = tokenizer_bpe_encode_span_checked(
        data,
        len(data),
        [0],
        1,
        left,
        right,
        ranks,
        merged,
        1,
        I64_MAX + 1,
        [0] * 4,
        4,
        [0],
    )
    assert overflow == TOKENIZER_BPE_ERR_OVERFLOW


def test_randomized_deterministic_merge_fixpoint() -> None:
    rng = random.Random(20260418_357)

    for _ in range(2500):
        length = rng.randint(1, 10)
        payload = [rng.randint(32, 126) for _ in range(length)]

        # Build a sorted rank table over random token pairs.
        entries = []
        for _ in range(rng.randint(0, 16)):
            left = rng.randint(32, 900)
            right = rng.randint(32, 900)
            rank = rng.randint(0, 100)
            merged = rng.randint(256, 4000)
            entries.append((left, right, rank, merged))
        entries.sort(key=lambda item: (item[0], item[1]))

        lefts = [item[0] for item in entries]
        rights = [item[1] for item in entries]
        ranks = [item[2] for item in entries]
        merged = [item[3] for item in entries]

        cursor = [0]
        out = [0xDEAD] * 32
        out_count = [0]

        err = tokenizer_bpe_encode_span_checked(
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
            out,
            len(out),
            out_count,
        )

        assert err == TOKENIZER_BPE_OK
        assert 0 <= out_count[0] <= len(payload)
        assert cursor[0] == len(payload)

