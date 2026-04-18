#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ValidateContinuationSpanCheckedNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_utf8_validate_continuation_span_checked import (
    I64_MAX,
    TOKENIZER_UTF8_ERR_BAD_CONTINUATION,
    TOKENIZER_UTF8_ERR_BAD_PARAM,
    TOKENIZER_UTF8_ERR_NULL_PTR,
    TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS,
    TOKENIZER_UTF8_ERR_OVERFLOW,
    TOKENIZER_UTF8_OK,
    tokenizer_utf8_validate_continuation_span_checked,
)


def tokenizer_utf8_validate_continuation_span_checked_no_partial(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    continuation_count: int,
    out_bytes_consumed: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_bytes_consumed is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    staged_cursor = [io_cursor[0]]
    staged_consumed = [out_bytes_consumed[0]]

    err = tokenizer_utf8_validate_continuation_span_checked(
        data,
        byte_len,
        staged_cursor,
        continuation_count,
        staged_consumed,
    )
    if err != TOKENIZER_UTF8_OK:
        return err

    io_cursor[0] = staged_cursor[0]
    out_bytes_consumed[0] = staged_consumed[0]
    return TOKENIZER_UTF8_OK


def compare_no_partial_to_checked(
    data: list[int],
    cursor0: int,
    continuation_count: int,
) -> None:
    cursor_core = [cursor0]
    consumed_core = [0x1122]
    err_core = tokenizer_utf8_validate_continuation_span_checked(
        data,
        len(data),
        cursor_core,
        continuation_count,
        consumed_core,
    )

    cursor_np = [cursor0]
    consumed_np = [0x1122]
    err_np = tokenizer_utf8_validate_continuation_span_checked_no_partial(
        data,
        len(data),
        cursor_np,
        continuation_count,
        consumed_np,
    )

    assert err_np == err_core
    assert cursor_np[0] == cursor_core[0]
    assert consumed_np[0] == consumed_core[0]


def test_success_and_error_parity_against_checked_core() -> None:
    vectors = [
        ([0x80], 0, 1),
        ([0x80, 0x81, 0xBF], 0, 3),
        ([0x11, 0x80, 0x99, 0xAF], 1, 3),
        ([0x80, 0x20, 0x80], 0, 2),
        ([0x80, 0x81], 1, 2),
        ([0x80, 0x81], 2, 0),
    ]

    for data, cursor0, count in vectors:
        compare_no_partial_to_checked(data, cursor0, count)


def test_no_partial_on_error_paths() -> None:
    cursor = [0]
    consumed = [0x77]
    err = tokenizer_utf8_validate_continuation_span_checked_no_partial(
        [0x80, 0x20, 0x80],
        3,
        cursor,
        2,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_CONTINUATION
    assert cursor[0] == 0
    assert consumed[0] == 0x77

    cursor = [1]
    consumed = [0x88]
    err = tokenizer_utf8_validate_continuation_span_checked_no_partial(
        [0x80, 0x81],
        2,
        cursor,
        2,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 1
    assert consumed[0] == 0x88


def test_contracts_match_checked_core() -> None:
    assert (
        tokenizer_utf8_validate_continuation_span_checked_no_partial(None, 0, [0], 0, [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked_no_partial([], 0, None, 0, [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked_no_partial([], 0, [0], 0, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor = [0]
    consumed = [0x99]
    err = tokenizer_utf8_validate_continuation_span_checked_no_partial(
        [0x80],
        I64_MAX + 1,
        cursor,
        1,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_OVERFLOW
    assert cursor[0] == 0
    assert consumed[0] == 0x99

    cursor = [3]
    consumed = [0xAB]
    err = tokenizer_utf8_validate_continuation_span_checked_no_partial(
        [0x80, 0x81],
        2,
        cursor,
        1,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor[0] == 3
    assert consumed[0] == 0xAB


def test_randomized_parity() -> None:
    rng = random.Random(20260418_3771)

    for _ in range(20000):
        n = rng.randint(0, 48)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = 0 if n == 0 else rng.randint(0, n)

        max_count = n - cursor0
        if max_count < 0:
            max_count = 0
        continuation_count = rng.randint(0, max_count + 4)

        compare_no_partial_to_checked(data, cursor0, continuation_count)


if __name__ == "__main__":
    test_success_and_error_parity_against_checked_core()
    test_no_partial_on_error_paths()
    test_contracts_match_checked_core()
    test_randomized_parity()
    print("tokenizer_utf8_validate_continuation_span_checked_no_partial_reference_checks=ok")
