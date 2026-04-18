#!/usr/bin/env python3
"""Parity harness for TokenizerBPEPromptSpanScanCheckedDefaultValidateCursor."""

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
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
)
from test_tokenizer_bpe_prompt_span_scan_checked import (
    tokenizer_bpe_prompt_span_scan_checked,
)


def tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    prompt_nbytes: int,
    out_span_starts: list[int] | None,
    out_span_lengths: list[int] | None,
    out_span_classes: list[int] | None,
    out_span_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_span_count is None:
        return TOKENIZER_BPE_ERR_NULL_PTR

    if byte_len > I64_MAX or prompt_nbytes > I64_MAX:
        return TOKENIZER_BPE_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    if prompt_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    derived_capacity = prompt_nbytes
    return tokenizer_bpe_prompt_span_scan_checked(
        data,
        byte_len,
        io_cursor,
        prompt_nbytes,
        out_span_starts,
        out_span_lengths,
        out_span_classes,
        derived_capacity,
        out_span_count,
    )


def run_parity_case(payload: list[int], cursor_start: int, prompt_nbytes: int) -> None:
    cap = max(prompt_nbytes, 1)

    starts_core = [0x1111] * cap
    lengths_core = [0x2222] * cap
    classes_core = [0x33] * cap
    count_core = [0x4444]
    cursor_core = [cursor_start]

    starts_wrapped = starts_core.copy()
    lengths_wrapped = lengths_core.copy()
    classes_wrapped = classes_core.copy()
    count_wrapped = count_core.copy()
    cursor_wrapped = cursor_core.copy()

    err_core = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        len(payload),
        cursor_core,
        prompt_nbytes,
        starts_core,
        lengths_core,
        classes_core,
        prompt_nbytes,
        count_core,
    )
    err_wrapped = tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor(
        payload,
        len(payload),
        cursor_wrapped,
        prompt_nbytes,
        starts_wrapped,
        lengths_wrapped,
        classes_wrapped,
        count_wrapped,
    )

    assert err_wrapped == err_core
    assert cursor_wrapped[0] == cursor_core[0]
    assert count_wrapped[0] == count_core[0]
    assert starts_wrapped == starts_core
    assert lengths_wrapped == lengths_core
    assert classes_wrapped == classes_core


def test_multilingual_and_malformed_utf8_parity() -> None:
    good = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    run_parity_case(good, 0, len(good))

    malformed = list("alpha β".encode("utf-8")) + [0xE2, 0x82]
    run_parity_case(malformed, 0, len(malformed))


def test_zero_length_prompt_parity() -> None:
    payload = list(b"abc")
    run_parity_case(payload, 2, 0)


def test_validate_cursor_guards_reject_overflow_and_oob() -> None:
    starts = [0xAAAA] * 8
    lengths = [0xBBBB] * 8
    classes = [0xCC] * 8
    count = [0xDDDD]

    cursor = [4]
    err = tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor(
        [ord("a"), ord("b"), ord("c")],
        3,
        cursor,
        0,
        starts,
        lengths,
        classes,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 4
    assert count[0] == 0xDDDD

    cursor = [2]
    err = tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor(
        [ord("a"), ord("b"), ord("c")],
        3,
        cursor,
        2,
        starts,
        lengths,
        classes,
        count,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 2
    assert count[0] == 0xDDDD

    cursor = [0]
    err = tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor(
        None,
        0,
        cursor,
        0,
        starts,
        lengths,
        classes,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR

    err = tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor(
        [],
        0,
        cursor,
        I64_MAX + 1,
        starts,
        lengths,
        classes,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW


def test_randomized_parity_against_explicit_capacity_core() -> None:
    rng = random.Random(20260418_418)

    valid_fragments = [
        b"alpha",
        b" 42 ",
        "Καλημέρα".encode("utf-8"),
        "世界".encode("utf-8"),
        b"_x!",
        "🙂".encode("utf-8"),
        b"\t\n",
    ]

    for _ in range(2500):
        payload = bytearray()
        for _ in range(rng.randint(0, 10)):
            payload.extend(rng.choice(valid_fragments))

        if payload and rng.random() < 0.30:
            malformed_kind = rng.choice(("trunc3", "trunc4", "bad_cont"))
            if malformed_kind == "trunc3":
                payload.extend(b"\xE2\x82")
            elif malformed_kind == "trunc4":
                payload.extend(b"\xF0\x9F\x92")
            else:
                payload.extend(b"\xE2A")

        data = list(payload)
        cursor_start = rng.randint(0, len(data))
        prompt_nbytes = rng.randint(0, len(data) - cursor_start)
        run_parity_case(data, cursor_start, prompt_nbytes)


if __name__ == "__main__":
    test_multilingual_and_malformed_utf8_parity()
    test_zero_length_prompt_parity()
    test_validate_cursor_guards_reject_overflow_and_oob()
    test_randomized_parity_against_explicit_capacity_core()
    print("test_tokenizer_bpe_prompt_span_scan_checked_default_validate_cursor: ok")
