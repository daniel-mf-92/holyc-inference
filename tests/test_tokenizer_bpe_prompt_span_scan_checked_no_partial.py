#!/usr/bin/env python3
"""Parity harness for TokenizerBPEPromptSpanScanCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_bpe_prompt_span_scan_checked import (
    I64_MAX,
    TOKENIZER_BPE_ERR_BAD_PARAM,
    TOKENIZER_BPE_ERR_NULL_PTR,
    TOKENIZER_BPE_ERR_OVERFLOW,
    TOKENIZER_BPE_OK,
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
    tokenizer_bpe_prompt_span_scan_checked,
)


def tokenizer_bpe_prompt_span_scan_checked_no_partial(
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

    staged_cursor = [cursor]
    staged_span_count = [out_span_count[0]]

    staged_capacity = out_span_capacity
    if staged_capacity == 0:
        staged_capacity = 1

    if staged_capacity > (I64_MAX // 8):
        return TOKENIZER_BPE_ERR_OVERFLOW

    staged_starts = [0] * staged_capacity
    staged_lengths = [0] * staged_capacity
    staged_classes = [0] * staged_capacity

    err = tokenizer_bpe_prompt_span_scan_checked(
        data,
        byte_len,
        staged_cursor,
        prompt_nbytes,
        staged_starts,
        staged_lengths,
        staged_classes,
        out_span_capacity,
        staged_span_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_span_count[0] > out_span_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_span_count[0]):
        out_span_starts[idx] = staged_starts[idx]
        out_span_lengths[idx] = staged_lengths[idx]
        out_span_classes[idx] = staged_classes[idx]

    out_span_count[0] = staged_span_count[0]
    io_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_parity_case(
    payload: list[int] | None,
    byte_len: int,
    cursor_start: int,
    prompt_nbytes: int,
    out_capacity: int,
) -> None:
    starts_expected = [0x1010] * max(out_capacity, 1)
    lengths_expected = [0x2020] * max(out_capacity, 1)
    classes_expected = [0x33] * max(out_capacity, 1)
    count_expected = [0x4444]
    cursor_expected = [cursor_start]

    starts_wrapper = [0x1010] * max(out_capacity, 1)
    lengths_wrapper = [0x2020] * max(out_capacity, 1)
    classes_wrapper = [0x33] * max(out_capacity, 1)
    count_wrapper = [0x4444]
    cursor_wrapper = [cursor_start]

    staged_expected_cursor = [cursor_expected[0]]
    staged_expected_count = [count_expected[0]]
    staged_expected_starts = [0] * max(out_capacity, 1)
    staged_expected_lengths = [0] * max(out_capacity, 1)
    staged_expected_classes = [0] * max(out_capacity, 1)

    err_expected = tokenizer_bpe_prompt_span_scan_checked(
        payload,
        byte_len,
        staged_expected_cursor,
        prompt_nbytes,
        staged_expected_starts,
        staged_expected_lengths,
        staged_expected_classes,
        out_capacity,
        staged_expected_count,
    )
    if err_expected == TOKENIZER_BPE_OK:
        for idx in range(staged_expected_count[0]):
            starts_expected[idx] = staged_expected_starts[idx]
            lengths_expected[idx] = staged_expected_lengths[idx]
            classes_expected[idx] = staged_expected_classes[idx]
        count_expected[0] = staged_expected_count[0]
        cursor_expected[0] = staged_expected_cursor[0]

    err_wrapper = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        payload,
        byte_len,
        cursor_wrapper,
        prompt_nbytes,
        starts_wrapper,
        lengths_wrapper,
        classes_wrapper,
        out_capacity,
        count_wrapper,
    )

    assert err_wrapper == err_expected
    assert cursor_wrapper[0] == cursor_expected[0]
    assert count_wrapper[0] == count_expected[0]
    assert starts_wrapper == starts_expected
    assert lengths_wrapper == lengths_expected
    assert classes_wrapper == classes_expected


def test_multilingual_prompt_fixture_parity_vs_explicit_staged_composition() -> None:
    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    run_parity_case(payload, len(payload), 0, len(payload), len(payload))


def test_malformed_utf8_prompt_tail_no_partial() -> None:
    payload = list("alpha β".encode("utf-8")) + [0xE2, 0x82]
    run_parity_case(payload, len(payload), 0, len(payload), len(payload))


def test_exact_no_partial_write_on_error_paths() -> None:
    starts = [0xAAAA] * 8
    lengths = [0xBBBB] * 8
    classes = [0xCC] * 8
    count = [0xDDDD]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        None,
        0,
        cursor,
        0,
        starts,
        lengths,
        classes,
        8,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_NULL_PTR
    assert cursor[0] == 0 and count[0] == 0xDDDD
    assert starts == [0xAAAA] * 8
    assert lengths == [0xBBBB] * 8
    assert classes == [0xCC] * 8

    starts = [0x1111] * 6
    lengths = [0x2222] * 6
    classes = [0x33] * 6
    count = [0x4444]
    cursor = [5]
    err = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        [ord("a"), ord("b")],
        2,
        cursor,
        0,
        starts,
        lengths,
        classes,
        6,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_BAD_PARAM
    assert cursor[0] == 5 and count[0] == 0x4444
    assert starts == [0x1111] * 6
    assert lengths == [0x2222] * 6
    assert classes == [0x33] * 6


def test_out_of_bounds_span_no_partial() -> None:
    starts = [0x5555] * 6
    lengths = [0x6666] * 6
    classes = [0x77] * 6
    count = [0x8888]
    cursor = [3]

    err = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        [ord("a"), ord("b"), ord("c")],
        3,
        cursor,
        1,
        starts,
        lengths,
        classes,
        6,
        count,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 3 and count[0] == 0x8888
    assert starts == [0x5555] * 6
    assert lengths == [0x6666] * 6
    assert classes == [0x77] * 6


def test_randomized_valid_and_malformed_utf8_parity() -> None:
    rng = random.Random(391)

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

        if rng.random() < 0.12:
            out_capacity = rng.randint(0, max(0, prompt_nbytes // 2 + 1))
        else:
            out_capacity = max(prompt_nbytes, 1)

        run_parity_case(data, len(data), cursor_start, prompt_nbytes, out_capacity)


def test_null_pointer_and_overflow_guards() -> None:
    starts = [0] * 4
    lengths = [0] * 4
    classes = [0] * 4
    count = [7]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        [],
        0,
        cursor,
        I64_MAX + 1,
        starts,
        lengths,
        classes,
        4,
        count,
    )
    assert err == TOKENIZER_BPE_ERR_OVERFLOW
    assert cursor[0] == 0 and count[0] == 7


if __name__ == "__main__":
    test_multilingual_prompt_fixture_parity_vs_explicit_staged_composition()
    test_malformed_utf8_prompt_tail_no_partial()
    test_exact_no_partial_write_on_error_paths()
    test_out_of_bounds_span_no_partial()
    test_randomized_valid_and_malformed_utf8_parity()
    test_null_pointer_and_overflow_guards()
    print("test_tokenizer_bpe_prompt_span_scan_checked_no_partial: ok")
