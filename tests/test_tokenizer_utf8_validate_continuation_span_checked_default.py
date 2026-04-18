#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ValidateContinuationSpanCheckedDefault."""

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

I64_MAX = (1 << 63) - 1


def tokenizer_utf8_is_continuation_byte(byte: int) -> bool:
    return (byte & 0xC0) == 0x80


def tokenizer_utf8_validate_continuation_byte_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_continuation_byte: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_continuation_byte is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM
    if cursor == byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    continuation = data[cursor]
    if not tokenizer_utf8_is_continuation_byte(continuation):
        return TOKENIZER_UTF8_ERR_BAD_CONTINUATION

    next_cursor = cursor + 1
    if next_cursor <= cursor or next_cursor > byte_len:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    out_continuation_byte[0] = continuation
    io_cursor[0] = next_cursor
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_validate_continuation_span_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    continuation_count: int,
    out_bytes_consumed: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_bytes_consumed is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    if continuation_count > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    staged_cursor = cursor
    consumed = 0

    while consumed < continuation_count:
        cont = [0]
        err = tokenizer_utf8_validate_continuation_byte_checked(
            data,
            byte_len,
            [staged_cursor],
            cont,
        )
        if err != TOKENIZER_UTF8_OK:
            return err
        staged_cursor += 1

        next_consumed = consumed + 1
        if next_consumed <= consumed:
            return TOKENIZER_UTF8_ERR_OVERFLOW
        consumed = next_consumed

    out_bytes_consumed[0] = consumed
    io_cursor[0] = staged_cursor
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_validate_continuation_span_checked_default(
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
    return tokenizer_utf8_validate_continuation_span_checked(
        data,
        byte_len,
        io_cursor,
        continuation_count,
        out_bytes_consumed,
    )


def compare_default_vs_explicit(data: list[int], cursor0: int) -> None:
    cursor_d = [cursor0]
    consumed_d = [0xAA55AA55AA55AA55]
    err_d = tokenizer_utf8_validate_continuation_span_checked_default(
        data,
        len(data),
        cursor_d,
        consumed_d,
    )

    cursor_e = [cursor0]
    consumed_e = [0xAA55AA55AA55AA55]
    err_e = tokenizer_utf8_validate_continuation_span_checked(
        data,
        len(data),
        cursor_e,
        len(data) - cursor0,
        consumed_e,
    )

    assert err_d == err_e
    assert cursor_d[0] == cursor_e[0]
    assert consumed_d[0] == consumed_e[0]


def test_valid_continuation_suffixes() -> None:
    vectors = [
        [0x80],
        [0x80, 0x81],
        [0x80, 0xBF, 0x8A, 0x9F],
        [0x11, 0x80, 0x99, 0xAF],
        [0x41, 0x42, 0x80, 0x81, 0x82],
    ]

    for data in vectors:
        for cursor in range(len(data) + 1):
            compare_default_vs_explicit(data, cursor)


def test_invalid_suffixes_and_no_partial() -> None:
    vectors = [
        [0x41],
        [0x41, 0x80],
        [0x80, 0x41],
        [0xC2, 0x80],
    ]

    for data in vectors:
        compare_default_vs_explicit(data, 0)

    data = [0x80, 0x41, 0x80]
    cursor = [0]
    consumed = [0x77]
    err = tokenizer_utf8_validate_continuation_span_checked_default(
        data,
        len(data),
        cursor,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_CONTINUATION
    assert cursor[0] == 0
    assert consumed[0] == 0x77


def test_contracts_and_overflow() -> None:
    assert (
        tokenizer_utf8_validate_continuation_span_checked_default(None, 0, [0], [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked_default([], 0, None, [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked_default([], 0, [0], None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor = [0]
    consumed = [0x88]
    err = tokenizer_utf8_validate_continuation_span_checked_default(
        [0x80],
        I64_MAX + 1,
        cursor,
        consumed,
    )
    assert err == TOKENIZER_UTF8_ERR_OVERFLOW
    assert cursor[0] == 0
    assert consumed[0] == 0x88

    cursor_bad = [2]
    consumed_bad = [0x89]
    err = tokenizer_utf8_validate_continuation_span_checked_default(
        [0x80],
        1,
        cursor_bad,
        consumed_bad,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor_bad[0] == 2
    assert consumed_bad[0] == 0x89


def test_randomized_parity() -> None:
    rng = random.Random(20260418_3761)

    for _ in range(12000):
        n = rng.randint(0, 32)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor = 0 if n == 0 else rng.randint(0, n)
        compare_default_vs_explicit(data, cursor)


if __name__ == "__main__":
    test_valid_continuation_suffixes()
    test_invalid_suffixes_and_no_partial()
    test_contracts_and_overflow()
    test_randomized_parity()
    print("tokenizer_utf8_validate_continuation_span_checked_default_reference_checks=ok")
