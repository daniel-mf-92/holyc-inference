#!/usr/bin/env python3
"""Parity harness for TokenizerBPEPromptSpanScanCheckedDefaultNoPartial."""

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
from test_tokenizer_bpe_prompt_span_scan_checked_no_partial import (
    tokenizer_bpe_prompt_span_scan_checked_no_partial,
)


def tokenizer_bpe_prompt_span_scan_checked_default_no_partial(
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
    if derived_capacity > 0 and (
        out_span_starts is None or out_span_lengths is None or out_span_classes is None
    ):
        return TOKENIZER_BPE_ERR_NULL_PTR

    staged_cursor = [cursor]
    staged_count = [out_span_count[0]]

    staged_capacity = max(derived_capacity, 1)
    staged_starts = [0] * staged_capacity
    staged_lengths = [0] * staged_capacity
    staged_classes = [0] * staged_capacity

    err = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        data,
        byte_len,
        staged_cursor,
        prompt_nbytes,
        staged_starts,
        staged_lengths,
        staged_classes,
        derived_capacity,
        staged_count,
    )
    if err != TOKENIZER_BPE_OK:
        return err

    if staged_count[0] > derived_capacity:
        return TOKENIZER_BPE_ERR_BAD_PARAM

    for idx in range(staged_count[0]):
        out_span_starts[idx] = staged_starts[idx]  # type: ignore[index]
        out_span_lengths[idx] = staged_lengths[idx]  # type: ignore[index]
        out_span_classes[idx] = staged_classes[idx]  # type: ignore[index]

    out_span_count[0] = staged_count[0]
    io_cursor[0] = staged_cursor[0]
    return TOKENIZER_BPE_OK


def run_case(payload: list[int] | None, byte_len: int, cursor_start: int, prompt_nbytes: int) -> None:
    cap = max(prompt_nbytes, 1)

    starts_expected = [0x1111] * cap
    lengths_expected = [0x2222] * cap
    classes_expected = [0x33] * cap
    count_expected = [0x4444]
    cursor_expected = [cursor_start]

    starts_wrapper = starts_expected.copy()
    lengths_wrapper = lengths_expected.copy()
    classes_wrapper = classes_expected.copy()
    count_wrapper = count_expected.copy()
    cursor_wrapper = cursor_expected.copy()

    staged_expected_cursor = [cursor_expected[0]]
    staged_expected_count = [count_expected[0]]
    staged_expected_starts = [0] * cap
    staged_expected_lengths = [0] * cap
    staged_expected_classes = [0] * cap

    err_expected = tokenizer_bpe_prompt_span_scan_checked_no_partial(
        payload,
        byte_len,
        staged_expected_cursor,
        prompt_nbytes,
        staged_expected_starts,
        staged_expected_lengths,
        staged_expected_classes,
        prompt_nbytes,
        staged_expected_count,
    )
    if err_expected == TOKENIZER_BPE_OK:
        for idx in range(staged_expected_count[0]):
            starts_expected[idx] = staged_expected_starts[idx]
            lengths_expected[idx] = staged_expected_lengths[idx]
            classes_expected[idx] = staged_expected_classes[idx]
        count_expected[0] = staged_expected_count[0]
        cursor_expected[0] = staged_expected_cursor[0]

    err_wrapper = tokenizer_bpe_prompt_span_scan_checked_default_no_partial(
        payload,
        byte_len,
        cursor_wrapper,
        prompt_nbytes,
        starts_wrapper,
        lengths_wrapper,
        classes_wrapper,
        count_wrapper,
    )

    assert err_wrapper == err_expected
    assert cursor_wrapper[0] == cursor_expected[0]
    assert count_wrapper[0] == count_expected[0]
    assert starts_wrapper == starts_expected
    assert lengths_wrapper == lengths_expected
    assert classes_wrapper == classes_expected


def test_multilingual_fixture_parity_vs_explicit_staged_composition() -> None:
    payload = list("hello, world 123\tgo! Καλημέρα 世界".encode("utf-8"))
    run_case(payload, len(payload), 0, len(payload))


def test_malformed_utf8_prompt_tail_parity() -> None:
    payload = list("alpha β".encode("utf-8")) + [0xE2, 0x82]
    run_case(payload, len(payload), 0, len(payload))


def test_zero_length_prompt_commits_no_partial() -> None:
    payload = list(b"abc")
    run_case(payload, len(payload), 2, 0)


def test_error_paths_keep_outputs_unchanged() -> None:
    starts = [0xAAAA] * 8
    lengths = [0xBBBB] * 8
    classes = [0xCC] * 8
    count = [0xDDDD]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked_default_no_partial(
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
    assert cursor[0] == 0 and count[0] == 0xDDDD
    assert starts == [0xAAAA] * 8
    assert lengths == [0xBBBB] * 8
    assert classes == [0xCC] * 8

    cursor = [4]
    err = tokenizer_bpe_prompt_span_scan_checked_default_no_partial(
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
    assert cursor[0] == 4 and count[0] == 0xDDDD

    cursor = [2]
    err = tokenizer_bpe_prompt_span_scan_checked_default_no_partial(
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
    assert cursor[0] == 2 and count[0] == 0xDDDD


def test_randomized_parity_vs_explicit_staged_composition() -> None:
    rng = random.Random(20260418_417)

    valid_fragments = [
        b"alpha",
        b" 42 ",
        "Καλημέρα".encode("utf-8"),
        "世界".encode("utf-8"),
        b"_x!",
        "🙂".encode("utf-8"),
        b"\t\n",
    ]

    for _ in range(3500):
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
        run_case(data, len(data), cursor_start, prompt_nbytes)


def test_overflow_guard_parity() -> None:
    starts = [0] * 4
    lengths = [0] * 4
    classes = [0] * 4
    count = [7]
    cursor = [0]

    err = tokenizer_bpe_prompt_span_scan_checked_default_no_partial(
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
    test_multilingual_fixture_parity_vs_explicit_staged_composition()
    test_malformed_utf8_prompt_tail_parity()
    test_zero_length_prompt_commits_no_partial()
    test_error_paths_keep_outputs_unchanged()
    test_randomized_parity_vs_explicit_staged_composition()
    test_overflow_guard_parity()
    print("test_tokenizer_bpe_prompt_span_scan_checked_default_no_partial: ok")
