#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ValidateContinuationSpanCheckedDefaultNoPartial."""

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
    TOKENIZER_UTF8_ERR_OVERFLOW,
    TOKENIZER_UTF8_OK,
    tokenizer_utf8_validate_continuation_span_checked,
)


def tokenizer_utf8_validate_continuation_span_checked_default_no_partial(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_bytes_consumed: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_bytes_consumed is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    continuation_count = byte_len - cursor

    staged_cursor = [cursor]
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


def explicit_staged_default_composition(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_bytes_consumed: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_bytes_consumed is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    continuation_count = byte_len - cursor

    staged_cursor = [cursor]
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


def compare_wrapper_vs_explicit(data: list[int], cursor0: int) -> None:
    cursor_w = [cursor0]
    consumed_w = [0x55667788]
    err_w = tokenizer_utf8_validate_continuation_span_checked_default_no_partial(
        data,
        len(data),
        cursor_w,
        consumed_w,
    )

    cursor_e = [cursor0]
    consumed_e = [0x55667788]
    err_e = explicit_staged_default_composition(
        data,
        len(data),
        cursor_e,
        consumed_e,
    )

    assert err_w == err_e
    assert cursor_w[0] == cursor_e[0]
    assert consumed_w[0] == consumed_e[0]


def test_valid_suffixes_parity() -> None:
    vectors = [
        [0x80],
        [0x80, 0x81, 0xBF],
        [0x31, 0x80, 0x90],
        [0x80, 0xBF, 0xAF, 0x9A, 0x80],
    ]

    for data in vectors:
        for cursor in range(len(data) + 1):
            compare_wrapper_vs_explicit(data, cursor)


def test_malformed_suffix_corpora_no_partial() -> None:
    vectors = [
        [0x41],
        [0x80, 0x41],
        [0x41, 0x80, 0x9F],
        [0x80, 0x81, 0x20, 0xBF],
        [0xC2, 0x80],
    ]

    for data in vectors:
        compare_wrapper_vs_explicit(data, 0)

    cursor = [0]
    consumed = [0xA5A5]
    err = tokenizer_utf8_validate_continuation_span_checked_default_no_partial(
        [0x80, 0x81, 0x20, 0xBF],
        4,
        cursor,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_CONTINUATION
    assert cursor[0] == 0
    assert consumed[0] == 0xA5A5


def test_contracts_and_overflow() -> None:
    assert (
        tokenizer_utf8_validate_continuation_span_checked_default_no_partial(None, 0, [0], [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked_default_no_partial([], 0, None, [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked_default_no_partial([], 0, [0], None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor = [0]
    consumed = [0x1234]
    err = tokenizer_utf8_validate_continuation_span_checked_default_no_partial(
        [0x80],
        I64_MAX + 1,
        cursor,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_OVERFLOW
    assert cursor[0] == 0
    assert consumed[0] == 0x1234

    cursor_bad = [2]
    consumed_bad = [0x2345]
    err = tokenizer_utf8_validate_continuation_span_checked_default_no_partial(
        [0x80],
        1,
        cursor_bad,
        consumed_bad,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor_bad[0] == 2
    assert consumed_bad[0] == 0x2345


def test_randomized_parity() -> None:
    rng = random.Random(20260418_378)

    for _ in range(12000):
        n = rng.randint(0, 48)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor = 0 if n == 0 else rng.randint(0, n)
        compare_wrapper_vs_explicit(data, cursor)


if __name__ == "__main__":
    test_valid_suffixes_parity()
    test_malformed_suffix_corpora_no_partial()
    test_contracts_and_overflow()
    test_randomized_parity()
    print("tokenizer_utf8_validate_continuation_span_checked_default_no_partial_parity=ok")
