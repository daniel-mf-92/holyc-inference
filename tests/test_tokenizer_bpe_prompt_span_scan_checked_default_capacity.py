#!/usr/bin/env python3
"""Parity harness for TokenizerBPEPromptSpanScanCheckedDefaultCapacity."""

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
from test_tokenizer_bpe_prompt_span_scan_checked import (
    tokenizer_bpe_prompt_span_scan_checked,
)


def tokenizer_bpe_prompt_span_scan_checked_default_capacity(
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


def run_parity_case(payload: list[int] | None, byte_len: int, cursor_start: int, prompt_nbytes: int) -> None:
    cap = max(prompt_nbytes, 1)

    starts_core = [0x1111] * cap
    lengths_core = [0x2222] * cap
    classes_core = [0x33] * cap
    count_core = [0x4444]
    cursor_core = [cursor_start]

    starts_def = starts_core.copy()
    lengths_def = lengths_core.copy()
    classes_def = classes_core.copy()
    count_def = count_core.copy()
    cursor_def = cursor_core.copy()

    err_core = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        byte_len,
        cursor_core,
        prompt_nbytes,
        starts_core,
        lengths_core,
        classes_core,
        prompt_nbytes,
        count_core,
    )
    err_def = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
        payload,
        byte_len,
        cursor_def,
        prompt_nbytes,
        starts_def,
        lengths_def,
        classes_def,
        count_def,
    )

    assert err_def == err_core
    assert cursor_def[0] == cursor_core[0]
    assert count_def[0] == count_core[0]
    assert starts_def == starts_core
    assert lengths_def == lengths_core
    assert classes_def == classes_core


def test_multilingual_fixture_parity() -> None:
    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    run_parity_case(payload, len(payload), 0, len(payload))


def test_malformed_utf8_prompt_tail_parity() -> None:
    payload = list("alpha β".encode("utf-8")) + [0xE2, 0x82]
    run_parity_case(payload, len(payload), 0, len(payload))


def test_zero_length_prompt_parity() -> None:
    payload = list(b"abc")
    run_parity_case(payload, len(payload), 1, 0)


def test_default_wrapper_error_contract() -> None:
    starts = [0xAAAA] * 8
    lengths = [0xBBBB] * 8
    classes = [0xCC] * 8
    count = [0xDDDD]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
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
    assert cursor[0] == 0
    assert count[0] == 0xDDDD
    assert starts == [0xAAAA] * 8
    assert lengths == [0xBBBB] * 8
    assert classes == [0xCC] * 8


def test_nonzero_prompt_requires_output_span_tables() -> None:
    payload = list("ab".encode("utf-8"))

    for starts_none, lengths_none, classes_none in (
        (None, [0x22] * 4, [0x33] * 4),
        ([0x11] * 4, None, [0x33] * 4),
        ([0x11] * 4, [0x22] * 4, None),
    ):
        cursor = [0]
        count = [0x4444]
        starts_before = None if starts_none is None else starts_none.copy()
        lengths_before = None if lengths_none is None else lengths_none.copy()
        classes_before = None if classes_none is None else classes_none.copy()

        err = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
            payload,
            len(payload),
            cursor,
            len(payload),
            starts_none,
            lengths_none,
            classes_none,
            count,
        )

        assert err == TOKENIZER_BPE_ERR_NULL_PTR
        assert cursor[0] == 0
        assert count[0] == 0x4444
        if starts_before is not None:
            assert starts_none == starts_before
        if lengths_before is not None:
            assert lengths_none == lengths_before
        if classes_before is not None:
            assert classes_none == classes_before


def test_zero_prompt_allows_null_output_tables() -> None:
    payload = list("abc".encode("utf-8"))
    cursor = [1]
    count = [0x5555]

    err = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
        payload,
        len(payload),
        cursor,
        0,
        None,
        None,
        None,
        count,
    )

    assert err == TOKENIZER_BPE_OK
    assert cursor[0] == 1
    assert count[0] == 0


def test_out_of_bounds_and_bad_cursor_guards() -> None:
    starts = [0x1010] * 8
    lengths = [0x2020] * 8
    classes = [0x30] * 8
    count = [0x4040]

    cursor = [4]
    err = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
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
    assert count[0] == 0x4040

    cursor = [2]
    err = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
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
    assert count[0] == 0x4040


def test_randomized_parity_against_explicit_capacity_core() -> None:
    rng = random.Random(20260418_392)

    valid_fragments = [
        b"alpha",
        b" 42 ",
        "Καλημέρα".encode("utf-8"),
        "世界".encode("utf-8"),
        b"_x!",
        "🙂".encode("utf-8"),
        b"\t\n",
    ]

    for _ in range(3000):
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
        run_parity_case(data, len(data), cursor_start, prompt_nbytes)


def test_overflow_guard_parity() -> None:
    starts = [0] * 4
    lengths = [0] * 4
    classes = [0] * 4
    count = [7]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked_default_capacity(
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
    assert cursor[0] == 0
    assert count[0] == 7


if __name__ == "__main__":
    test_multilingual_fixture_parity()
    test_malformed_utf8_prompt_tail_parity()
    test_zero_length_prompt_parity()
    test_default_wrapper_error_contract()
    test_nonzero_prompt_requires_output_span_tables()
    test_zero_prompt_allows_null_output_tables()
    test_out_of_bounds_and_bad_cursor_guards()
    test_randomized_parity_against_explicit_capacity_core()
    test_overflow_guard_parity()
    print("test_tokenizer_bpe_prompt_span_scan_checked_default_capacity: ok")
