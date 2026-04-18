#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8NextCodepointChecked semantics."""

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


def tokenizer_utf8_next_codepoint_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_codepoint: list[int] | None,
    out_bytes_consumed: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_codepoint is None
        or out_bytes_consumed is None
    ):
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM
    if cursor == byte_len:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    lead = data[cursor]

    if lead <= 0x7F:
        need = 1
        codepoint = lead
    elif 0xC2 <= lead <= 0xDF:
        need = 2
        codepoint = lead & 0x1F
    elif 0xE0 <= lead <= 0xEF:
        need = 3
        codepoint = lead & 0x0F
    elif 0xF0 <= lead <= 0xF4:
        need = 4
        codepoint = lead & 0x07
    else:
        return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE

    if byte_len - cursor < need:
        return TOKENIZER_UTF8_ERR_TRUNCATED

    if need > 1:
        cont = data[cursor + 1]
        if not tokenizer_utf8_is_continuation_byte(cont):
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
        codepoint = (codepoint << 6) | (cont & 0x3F)

    if need > 2:
        cont = data[cursor + 2]
        if not tokenizer_utf8_is_continuation_byte(cont):
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
        codepoint = (codepoint << 6) | (cont & 0x3F)

    if need > 3:
        cont = data[cursor + 3]
        if not tokenizer_utf8_is_continuation_byte(cont):
            return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
        codepoint = (codepoint << 6) | (cont & 0x3F)

    if need == 2 and codepoint < 0x80:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    if need == 3:
        if codepoint < 0x800:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
        if 0xD800 <= codepoint <= 0xDFFF:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    if need == 4:
        if codepoint < 0x10000:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT
        if codepoint > 0x10FFFF:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    next_cursor = cursor + need
    if next_cursor < cursor or next_cursor > byte_len:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    out_codepoint[0] = codepoint
    out_bytes_consumed[0] = need
    io_cursor[0] = next_cursor
    return TOKENIZER_UTF8_OK


def run_case(
    data: list[int],
    cursor0: int,
    expected_err: int,
    expected_cursor: int,
    expected_cp: int,
    expected_used: int,
) -> None:
    cursor = [cursor0]
    cp = [0xAAAA]
    used = [0xBB]
    err = tokenizer_utf8_next_codepoint_checked(data, len(data), cursor, cp, used)
    assert err == expected_err
    assert cursor[0] == expected_cursor
    assert cp[0] == expected_cp
    assert used[0] == expected_used


def test_success_corpus() -> None:
    run_case([0x41], 0, TOKENIZER_UTF8_OK, 1, 0x41, 1)
    run_case([0xC2, 0xA2], 0, TOKENIZER_UTF8_OK, 2, 0x00A2, 2)
    run_case([0xE2, 0x82, 0xAC], 0, TOKENIZER_UTF8_OK, 3, 0x20AC, 3)
    run_case([0xF0, 0x9F, 0x92, 0xA9], 0, TOKENIZER_UTF8_OK, 4, 0x1F4A9, 4)
    run_case([0x24, 0xC2, 0xA2], 1, TOKENIZER_UTF8_OK, 3, 0x00A2, 2)


def test_failures_no_partial() -> None:
    # Invalid lead bytes.
    run_case([0x80], 0, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0, 0xAAAA, 0xBB)
    run_case([0xC1, 0xBF], 0, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0, 0xAAAA, 0xBB)
    run_case([0xF5, 0x80, 0x80, 0x80], 0, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0, 0xAAAA, 0xBB)

    # Truncated forms.
    run_case([0xE2, 0x82], 0, TOKENIZER_UTF8_ERR_TRUNCATED, 0, 0xAAAA, 0xBB)
    run_case([0xF0, 0x9F, 0x92], 0, TOKENIZER_UTF8_ERR_TRUNCATED, 0, 0xAAAA, 0xBB)

    # Bad continuation.
    run_case([0xC2, 0x20], 0, TOKENIZER_UTF8_ERR_BAD_CONTINUATION, 0, 0xAAAA, 0xBB)
    run_case([0xE2, 0x82, 0x2F], 0, TOKENIZER_UTF8_ERR_BAD_CONTINUATION, 0, 0xAAAA, 0xBB)

    # Surrogate and out-of-range checks.
    run_case([0xED, 0xA0, 0x80], 0, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0, 0xAAAA, 0xBB)
    run_case([0xF4, 0x90, 0x80, 0x80], 0, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0, 0xAAAA, 0xBB)


def test_parameter_contracts() -> None:
    cursor = [0]
    cp = [111]
    used = [7]

    assert tokenizer_utf8_next_codepoint_checked(None, 0, cursor, cp, used) == TOKENIZER_UTF8_ERR_NULL_PTR
    assert tokenizer_utf8_next_codepoint_checked([], 0, None, cp, used) == TOKENIZER_UTF8_ERR_NULL_PTR
    assert tokenizer_utf8_next_codepoint_checked([], 0, cursor, None, used) == TOKENIZER_UTF8_ERR_NULL_PTR
    assert tokenizer_utf8_next_codepoint_checked([], 0, cursor, cp, None) == TOKENIZER_UTF8_ERR_NULL_PTR

    assert tokenizer_utf8_next_codepoint_checked([0x41], I64_MAX + 1, cursor, cp, used) == TOKENIZER_UTF8_ERR_OVERFLOW

    cursor2 = [3]
    cp2 = [555]
    used2 = [9]
    assert tokenizer_utf8_next_codepoint_checked([0x41, 0x42], 2, cursor2, cp2, used2) == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor2[0] == 3 and cp2[0] == 555 and used2[0] == 9

    cursor3 = [2]
    cp3 = [666]
    used3 = [8]
    assert tokenizer_utf8_next_codepoint_checked([0x41, 0x42], 2, cursor3, cp3, used3) == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor3[0] == 2 and cp3[0] == 666 and used3[0] == 8


def test_randomized_ascii_and_utf8_roundtrip() -> None:
    rng = random.Random(20260418_337)

    corpus = [
        "TempleOS",
        "λ",
        "Русский",
        "हिन्दी",
        "漢字かな交じり文",
        "🙂🙃🧠",
    ]

    for _ in range(4000):
        text = rng.choice(corpus)
        if rng.randint(0, 1):
            text += chr(rng.randint(0, 0x7F))
        payload = list(text.encode("utf-8"))

        cursor = [0]
        codepoints: list[int] = []
        while cursor[0] < len(payload):
            cp = [0]
            used = [0]
            err = tokenizer_utf8_next_codepoint_checked(payload, len(payload), cursor, cp, used)
            assert err == TOKENIZER_UTF8_OK
            assert 1 <= used[0] <= 4
            codepoints.append(cp[0])

        assert "".join(chr(v) for v in codepoints) == text


def test_randomized_mutated_invalid_sequences_no_partial() -> None:
    rng = random.Random(20260418_338)

    for _ in range(6000):
        n = rng.randint(1, 12)
        payload = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n - 1)

        cursor = [cursor0]
        cp = [0x1111]
        used = [0x22]

        err = tokenizer_utf8_next_codepoint_checked(payload, n, cursor, cp, used)
        if err != TOKENIZER_UTF8_OK:
            assert cursor[0] == cursor0
            assert cp[0] == 0x1111
            assert used[0] == 0x22
        else:
            assert cursor[0] > cursor0
            assert 1 <= used[0] <= 4


if __name__ == "__main__":
    test_success_corpus()
    test_failures_no_partial()
    test_parameter_contracts()
    test_randomized_ascii_and_utf8_roundtrip()
    test_randomized_mutated_invalid_sequences_no_partial()
    print("tokenizer_utf8_next_codepoint_checked_reference_checks=ok")
