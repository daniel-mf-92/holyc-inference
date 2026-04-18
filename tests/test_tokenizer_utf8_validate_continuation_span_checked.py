#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ValidateContinuationSpanChecked."""

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


def explicit_reference_composition(
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

    staged = cursor
    consumed = 0
    cont = [0x44]

    for _ in range(continuation_count):
        err = tokenizer_utf8_validate_continuation_byte_checked(data, byte_len, [staged], cont)
        if err != TOKENIZER_UTF8_OK:
            return err
        staged += 1
        consumed += 1

    out_bytes_consumed[0] = consumed
    io_cursor[0] = staged
    return TOKENIZER_UTF8_OK


def compare_against_reference(data: list[int], cursor0: int, continuation_count: int) -> None:
    cursor_a = [cursor0]
    consumed_a = [0xA1]
    err_a = tokenizer_utf8_validate_continuation_span_checked(
        data,
        len(data),
        cursor_a,
        continuation_count,
        consumed_a,
    )

    cursor_b = [cursor0]
    consumed_b = [0xA1]
    err_b = explicit_reference_composition(
        data,
        len(data),
        cursor_b,
        continuation_count,
        consumed_b,
    )

    assert err_a == err_b
    assert cursor_a[0] == cursor_b[0]
    assert consumed_a[0] == consumed_b[0]


def test_null_ptr_contract() -> None:
    cursor = [0]
    consumed = [0x55]

    assert (
        tokenizer_utf8_validate_continuation_span_checked(None, 0, cursor, 0, consumed)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked([0x80], 1, None, 1, consumed)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_span_checked([0x80], 1, cursor, 1, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )


def test_param_contracts_and_no_partial() -> None:
    cursor = [0]
    consumed = [0x66]

    assert (
        tokenizer_utf8_validate_continuation_span_checked(
            [0x80], I64_MAX + 1, cursor, 1, consumed
        )
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )
    assert cursor[0] == 0
    assert consumed[0] == 0x66

    cursor_bad = [3]
    consumed_bad = [0x67]
    assert (
        tokenizer_utf8_validate_continuation_span_checked(
            [0x80, 0x81], 2, cursor_bad, 1, consumed_bad
        )
        == TOKENIZER_UTF8_ERR_BAD_PARAM
    )
    assert cursor_bad[0] == 3
    assert consumed_bad[0] == 0x67

    cursor_oob = [1]
    consumed_oob = [0x68]
    assert (
        tokenizer_utf8_validate_continuation_span_checked(
            [0x80, 0x81], 2, cursor_oob, 2, consumed_oob
        )
        == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    )
    assert cursor_oob[0] == 1
    assert consumed_oob[0] == 0x68


def test_zero_count_is_noop_success() -> None:
    data = [0x41, 0x42, 0x43]
    cursor = [2]
    consumed = [0x99]

    err = tokenizer_utf8_validate_continuation_span_checked(data, len(data), cursor, 0, consumed)
    assert err == TOKENIZER_UTF8_OK
    assert cursor[0] == 2
    assert consumed[0] == 0


def test_frontier_counts_against_continuation_run() -> None:
    data = [0x80, 0x81, 0xBF, 0xA0, 0x41]

    for start in range(len(data)):
        max_count = len(data) - start
        for count in range(max_count + 1):
            compare_against_reference(data, start, count)


def test_malformed_utf8_fixtures_no_partial() -> None:
    fixtures = [
        ([0x80, 0x20, 0x80], 0, 2, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
        ([0x9F, 0xC2, 0x80], 0, 2, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
        ([0xA1, 0xBF, 0x7F], 1, 2, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
        ([0x80, 0x81, 0xF0], 0, 3, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
    ]

    for data, cursor0, count, expected_err in fixtures:
        cursor = [cursor0]
        consumed = [0x22]
        err = tokenizer_utf8_validate_continuation_span_checked(
            data, len(data), cursor, count, consumed
        )
        assert err == expected_err
        assert cursor[0] == cursor0
        assert consumed[0] == 0x22


def test_randomized_reference_parity() -> None:
    rng = random.Random(20260418_363)

    for _ in range(20000):
        n = rng.randint(1, 48)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        max_count = n - cursor0
        count = rng.randint(0, max_count)

        compare_against_reference(data, cursor0, count)


def test_full_continuation_sequences() -> None:
    # Valid continuation bytes from multibyte UTF-8 sequences only.
    data = [
        0x41,
        0xC2,
        0xA2,
        0xE2,
        0x82,
        0xAC,
        0xF0,
        0x9F,
        0x99,
        0x82,
        0x42,
    ]

    vectors = [
        (2, 1),  # A2
        (4, 2),  # 82 AC
        (7, 3),  # 9F 99 82
    ]

    for cursor0, count in vectors:
        cursor = [cursor0]
        consumed = [0]
        err = tokenizer_utf8_validate_continuation_span_checked(
            data, len(data), cursor, count, consumed
        )
        assert err == TOKENIZER_UTF8_OK
        assert consumed[0] == count
        assert cursor[0] == cursor0 + count


if __name__ == "__main__":
    test_null_ptr_contract()
    test_param_contracts_and_no_partial()
    test_zero_count_is_noop_success()
    test_frontier_counts_against_continuation_run()
    test_malformed_utf8_fixtures_no_partial()
    test_randomized_reference_parity()
    test_full_continuation_sequences()
    print("tokenizer_utf8_validate_continuation_span_checked_reference_checks=ok")
