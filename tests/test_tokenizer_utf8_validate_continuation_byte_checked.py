#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ValidateContinuationByteChecked."""

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


def test_null_ptr_contract() -> None:
    cursor = [0]
    out = [0xAB]

    assert (
        tokenizer_utf8_validate_continuation_byte_checked(None, 0, cursor, out)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_byte_checked([0x80], 1, None, out)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_validate_continuation_byte_checked([0x80], 1, cursor, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )


def test_parameter_and_bounds_contract() -> None:
    cursor = [0]
    out = [0xCD]

    assert (
        tokenizer_utf8_validate_continuation_byte_checked([0x80], I64_MAX + 1, cursor, out)
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )
    assert cursor[0] == 0
    assert out[0] == 0xCD

    cursor_bad = [2]
    out_bad = [0xEE]
    assert (
        tokenizer_utf8_validate_continuation_byte_checked([0x80], 1, cursor_bad, out_bad)
        == TOKENIZER_UTF8_ERR_BAD_PARAM
    )
    assert cursor_bad[0] == 2
    assert out_bad[0] == 0xEE

    cursor_end = [1]
    out_end = [0xFA]
    assert (
        tokenizer_utf8_validate_continuation_byte_checked([0x80], 1, cursor_end, out_end)
        == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    )
    assert cursor_end[0] == 1
    assert out_end[0] == 0xFA


def test_exhaustive_byte_domain() -> None:
    for byte in range(256):
        data = [byte]
        cursor = [0]
        out = [0xA5]

        err = tokenizer_utf8_validate_continuation_byte_checked(data, 1, cursor, out)
        expected_ok = 0x80 <= byte <= 0xBF

        if expected_ok:
            assert err == TOKENIZER_UTF8_OK, f"byte=0x{byte:02X}"
            assert out[0] == byte, f"byte=0x{byte:02X}"
            assert cursor[0] == 1, f"byte=0x{byte:02X}"
        else:
            assert err == TOKENIZER_UTF8_ERR_BAD_CONTINUATION, f"byte=0x{byte:02X}"
            assert out[0] == 0xA5, f"byte=0x{byte:02X}"
            assert cursor[0] == 0, f"byte=0x{byte:02X}"


def test_malformed_sequence_fixtures_no_partial() -> None:
    fixtures = [
        ([0xC2, 0x20], 1, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
        ([0xE2, 0x82, 0x28], 2, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
        ([0xF0, 0x9F, 0x20, 0x82], 2, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
        ([0x80, 0x41], 1, TOKENIZER_UTF8_ERR_BAD_CONTINUATION),
    ]

    for data, cursor0, expected_err in fixtures:
        cursor = [cursor0]
        out = [0x77]
        err = tokenizer_utf8_validate_continuation_byte_checked(data, len(data), cursor, out)
        assert err == expected_err
        assert cursor[0] == cursor0
        assert out[0] == 0x77


def test_valid_continuation_in_utf8_sequences() -> None:
    # C2 A2  => ¢
    # E2 82 AC => €
    # F0 9F 99 82 => 🙂
    payload = [0xC2, 0xA2, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x99, 0x82]

    expected_cursor_and_byte = [
        (1, 0xA2),
        (3, 0x82),
        (4, 0xAC),
        (6, 0x9F),
        (7, 0x99),
        (8, 0x82),
    ]

    for cursor0, expected_byte in expected_cursor_and_byte:
        cursor = [cursor0]
        out = [0x00]
        err = tokenizer_utf8_validate_continuation_byte_checked(payload, len(payload), cursor, out)
        assert err == TOKENIZER_UTF8_OK
        assert out[0] == expected_byte
        assert cursor[0] == cursor0 + 1


def test_randomized_domain_parity() -> None:
    rng = random.Random(20260418_3481)

    for _ in range(20000):
        n = rng.randint(1, 20)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n - 1)

        cursor = [cursor0]
        out = [0x5A]
        err = tokenizer_utf8_validate_continuation_byte_checked(data, n, cursor, out)

        byte = data[cursor0]
        if 0x80 <= byte <= 0xBF:
            assert err == TOKENIZER_UTF8_OK
            assert out[0] == byte
            assert cursor[0] == cursor0 + 1
        else:
            assert err == TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            assert out[0] == 0x5A
            assert cursor[0] == cursor0


if __name__ == "__main__":
    test_null_ptr_contract()
    test_parameter_and_bounds_contract()
    test_exhaustive_byte_domain()
    test_malformed_sequence_fixtures_no_partial()
    test_valid_continuation_in_utf8_sequences()
    test_randomized_domain_parity()
    print("tokenizer_utf8_validate_continuation_byte_checked_reference_checks=ok")
