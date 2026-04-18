#!/usr/bin/env python3
"""Parity harness for TokenizerBPEPromptSpanScanChecked semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_encode_prompt_checked import (
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
    TOKENIZER_UTF8_OK,
    explicit_prompt_spans,
    tokenizer_bpe_ascii_class,
    tokenizer_bpe_encode_prompt_checked,
    tokenizer_bpe_encode_span_checked,
    tokenizer_utf8_next_codepoint_checked,
)

TOKENIZER_BPE_ASCII_CLASS_NON_ASCII = 5


def tokenizer_bpe_prompt_span_scan_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    prompt_nbytes: int,
    out_span_starts: list[int] | None,
    out_span_lengths: list[int] | None,
    out_span_classes: list[int] | None,
    out_span_capacity: int,
    out_span_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_span_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if byte_len > I64_MAX or prompt_nbytes > I64_MAX or out_span_capacity > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    if out_span_capacity > 0 and (
        out_span_starts is None or out_span_lengths is None or out_span_classes is None
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

    if prompt_nbytes == 0:
        out_span_count[0] = 0
        io_cursor[0] = cursor
        return TOKENIZER_BPE_OK

    if out_span_capacity == 0:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    staged_starts: list[int] = []
    staged_lengths: list[int] = []
    staged_classes: list[int] = []

    scan = cursor
    while scan < prompt_end:
        span_start = scan
        lead = data[scan]

        if lead <= 0x7F:
            span_class = tokenizer_bpe_ascii_class(lead)
            scan += 1
            while scan < prompt_end:
                if data[scan] > 0x7F:
                    break
                if tokenizer_bpe_ascii_class(data[scan]) != span_class:
                    break
                scan += 1
        else:
            span_class = TOKENIZER_BPE_ASCII_CLASS_NON_ASCII
            cp = [0]
            used = [0]
            tmp = [scan]
            err = tokenizer_utf8_next_codepoint_checked(data, prompt_end, tmp, cp, used)
            if err != TOKENIZER_UTF8_OK:
                return err
            scan = tmp[0]

            while scan < prompt_end and data[scan] > 0x7F:
                tmp = [scan]
                err = tokenizer_utf8_next_codepoint_checked(data, prompt_end, tmp, cp, used)
                if err != TOKENIZER_UTF8_OK:
                    return err
                scan = tmp[0]

        span_nbytes = scan - span_start
        if span_nbytes == 0:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        if len(staged_starts) >= out_span_capacity:
            return TOKENIZER_BPE_ERR_BAD_PARAM

        staged_starts.append(span_start)
        staged_lengths.append(span_nbytes)
        staged_classes.append(span_class)

        if len(staged_starts) > out_span_capacity:
            return TOKENIZER_BPE_ERR_OVERFLOW

    if scan != prompt_end:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for i in range(len(staged_starts)):
        out_span_starts[i] = staged_starts[i]
        out_span_lengths[i] = staged_lengths[i]
        out_span_classes[i] = staged_classes[i]

    out_span_count[0] = len(staged_starts)
    io_cursor[0] = prompt_end
    return TOKENIZER_BPE_OK


def explicit_prompt_spans_with_classes(payload: list[int]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for start, length in explicit_prompt_spans(payload):
        lead = payload[start]
        if lead <= 0x7F:
            lane = tokenizer_bpe_ascii_class(lead)
        else:
            lane = TOKENIZER_BPE_ASCII_CLASS_NON_ASCII
        out.append((start, length, lane))
    return out


def test_known_multilingual_span_fixture() -> None:
    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    cursor = [0]
    starts = [0xDEAD] * len(payload)
    lengths = [0xBEEF] * len(payload)
    classes = [0xAA] * len(payload)
    span_count = [0x7777]

    err = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        starts,
        lengths,
        classes,
        len(payload),
        span_count,
    )
    assert err == TOKENIZER_BPE_OK

    expected = explicit_prompt_spans_with_classes(payload)
    assert span_count[0] == len(expected)
    assert cursor[0] == len(payload)

    for i, (start, length, cls) in enumerate(expected):
        assert starts[i] == start
        assert lengths[i] == length
        assert classes[i] == cls


def test_ascii_class_partitioning() -> None:
    payload = list(b"alpha_42  +\t\\nbeta99!!")
    cursor = [0]
    starts = [0] * len(payload)
    lengths = [0] * len(payload)
    classes = [0] * len(payload)
    span_count = [0]

    err = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        starts,
        lengths,
        classes,
        len(payload),
        span_count,
    )
    assert err == TOKENIZER_BPE_OK

    observed = classes[: span_count[0]]
    assert TOKENIZER_BPE_ASCII_CLASS_WORD in observed
    assert TOKENIZER_BPE_ASCII_CLASS_DIGIT in observed
    assert TOKENIZER_BPE_ASCII_CLASS_WHITESPACE in observed
    assert TOKENIZER_BPE_ASCII_CLASS_PUNCT in observed


def test_capacity_underflow_rejected_no_partial_outputs() -> None:
    payload = list("ab cd".encode("utf-8"))
    cursor = [0]
    starts = [0x1111] * 2
    lengths = [0x2222] * 2
    classes = [0x33] * 2
    span_count = [0x4444]

    err = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        starts,
        lengths,
        classes,
        1,
        span_count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert span_count[0] == 0x4444
    assert starts == [0x1111, 0x1111]
    assert lengths == [0x2222, 0x2222]
    assert classes == [0x33, 0x33]


def test_empty_prompt_commits_empty_span_table() -> None:
    payload = list(b"abc")
    cursor = [2]
    starts = [0x77] * 4
    lengths = [0x88] * 4
    classes = [0x99] * 4
    span_count = [0xABCD]

    err = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor,
        0,
        starts,
        lengths,
        classes,
        0,
        span_count,
    )
    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == 2
    assert span_count[0] == 0
    assert starts == [0x77] * 4
    assert lengths == [0x88] * 4
    assert classes == [0x99] * 4


def test_malformed_utf8_tail_rejected_no_partial() -> None:
    payload = list(b"A\xE2\x82")
    cursor = [0]
    starts = [0x1] * 8
    lengths = [0x2] * 8
    classes = [0x3] * 8
    span_count = [0x4]

    err = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        starts,
        lengths,
        classes,
        len(payload),
        span_count,
    )
    assert err != TOKENIZER_BPE_OK
    assert cursor[0] == 0
    assert span_count[0] == 0x4
    assert starts == [0x1] * 8
    assert lengths == [0x2] * 8
    assert classes == [0x3] * 8


def test_prompt_encode_internal_boundary_parity() -> None:
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
    left = [item[0] for item in rank_entries]
    right = [item[1] for item in rank_entries]
    ranks = [item[2] for item in rank_entries]
    merged = [item[3] for item in rank_entries]

    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))

    scan_cursor = [0]
    starts = [0] * len(payload)
    lengths = [0] * len(payload)
    classes = [0] * len(payload)
    span_count = [0]
    err_scan = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        scan_cursor,
        len(payload),
        starts,
        lengths,
        classes,
        len(payload),
        span_count,
    )
    assert err_scan == TOKENIZER_BPE_OK

    prompt_cursor = [0]
    prompt_out = [0xCAFE] * (len(payload) + 8)
    prompt_count = [0]
    err_prompt = tokenizer_bpe_encode_prompt_checked(
        payload,
        len(payload),
        prompt_cursor,
        len(payload),
        left,
        right,
        ranks,
        merged,
        len(ranks),
        len(ranks),
        prompt_out,
        len(prompt_out),
        prompt_count,
    )
    assert err_prompt == TOKENIZER_BPE_OK

    composed: list[int] = []
    for i in range(span_count[0]):
        span_cursor = [starts[i]]
        span_out = [0] * (lengths[i] + 2)
        span_count_out = [0]
        err = tokenizer_bpe_encode_span_checked(
            payload,
            len(payload),
            span_cursor,
            lengths[i],
            left,
            right,
            ranks,
            merged,
            len(ranks),
            len(ranks),
            span_out,
            len(span_out),
            span_count_out,
        )
        assert err == TOKENIZER_BPE_OK
        assert span_cursor[0] == starts[i] + lengths[i]
        composed.extend(span_out[: span_count_out[0]])

    assert composed == prompt_out[: prompt_count[0]]


def test_random_valid_and_malformed_utf8_parity() -> None:
    rng = random.Random(390)

    valid_fragments = [
        b"alpha",
        b" 42 ",
        "Καλημέρα".encode("utf-8"),
        "世界".encode("utf-8"),
        b"_x!",
        "🙂".encode("utf-8"),
        b"\t\n",
    ]

    for _ in range(300):
        payload = bytearray()
        for _ in range(rng.randint(0, 8)):
            payload.extend(rng.choice(valid_fragments))

        if payload and rng.random() < 0.25:
            malformed_kind = rng.choice(("trunc3", "trunc4", "bad_cont"))
            if malformed_kind == "trunc3":
                payload.extend(b"\xE2\x82")
            elif malformed_kind == "trunc4":
                payload.extend(b"\xF0\x9F\x92")
            else:
                payload.extend(b"\xE2A")

        data = list(payload)
        cursor = [0]
        starts = [0xAA] * max(len(data), 1)
        lengths = [0xBB] * max(len(data), 1)
        classes = [0xCC] * max(len(data), 1)
        span_count = [0xDD]

        err = tokenizer_bpe_prompt_span_scan_checked(
            data,
            len(data),
            cursor,
            len(data),
            starts,
            lengths,
            classes,
            len(data),
            span_count,
        )

        if err == TOKENIZER_BPE_OK:
            expected = explicit_prompt_spans_with_classes(data)
            assert span_count[0] == len(expected)
            assert cursor[0] == len(data)
            for i, (s, n, cls) in enumerate(expected):
                assert starts[i] == s
                assert lengths[i] == n
                assert classes[i] == cls
        else:
            assert cursor[0] == 0
            assert span_count[0] == 0xDD
            assert starts == [0xAA] * max(len(data), 1)
            assert lengths == [0xBB] * max(len(data), 1)
            assert classes == [0xCC] * max(len(data), 1)


def test_null_pointer_and_overflow_guards() -> None:
    cursor = [0]
    span_count = [7]
    starts = [0] * 4
    lengths = [0] * 4
    classes = [0] * 4

    err = tokenizer_bpe_prompt_span_scan_checked(
        None,
        0,
        cursor,
        0,
        starts,
        lengths,
        classes,
        4,
        span_count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert cursor[0] == 0
    assert span_count[0] == 7

    err = tokenizer_bpe_prompt_span_scan_checked(
        [],
        0,
        cursor,
        I64_MAX + 1,
        starts,
        lengths,
        classes,
        4,
        span_count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert span_count[0] == 7


if __name__ == "__main__":
    test_known_multilingual_span_fixture()
    test_ascii_class_partitioning()
    test_capacity_underflow_rejected_no_partial_outputs()
    test_empty_prompt_commits_empty_span_table()
    test_malformed_utf8_tail_rejected_no_partial()
    test_prompt_encode_internal_boundary_parity()
    test_random_valid_and_malformed_utf8_parity()
    test_null_pointer_and_overflow_guards()
    print("test_tokenizer_bpe_prompt_span_scan_checked: ok")
