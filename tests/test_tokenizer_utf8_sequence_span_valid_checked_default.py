#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8SequenceSpanValidCheckedDefault."""

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


def tokenizer_utf8_expected_length_from_lead_byte_checked(
    lead: int,
    out_expected_length: list[int] | None,
) -> int:
    if out_expected_length is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if lead <= 0x7F:
        need = 1
    elif 0xC2 <= lead <= 0xDF:
        need = 2
    elif 0xE0 <= lead <= 0xEF:
        need = 3
    elif 0xF0 <= lead <= 0xF4:
        need = 4
    else:
        return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE

    out_expected_length[0] = need
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_validate_codepoint_scalar_checked(
    codepoint: int,
    out_codepoint: list[int] | None,
) -> int:
    if out_codepoint is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if 0xD800 <= codepoint <= 0xDFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    if codepoint > 0x10FFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    out_codepoint[0] = codepoint
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_sequence_span_valid_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    span_nbytes: int,
    out_is_valid: list[bool] | None,
) -> int:
    if data is None or io_cursor is None or out_is_valid is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    if span_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    span_end = cursor + span_nbytes
    if span_end < cursor or span_end > byte_len:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    scan = cursor
    while scan < span_end:
        lead = data[scan]

        need = [0]
        err = tokenizer_utf8_expected_length_from_lead_byte_checked(lead, need)
        if err != TOKENIZER_UTF8_OK:
            return err

        width = need[0]
        if span_end - scan < width:
            return TOKENIZER_UTF8_ERR_TRUNCATED

        if width == 1:
            codepoint = lead
        elif width == 2:
            codepoint = lead & 0x1F
        elif width == 3:
            codepoint = lead & 0x0F
        else:
            codepoint = lead & 0x07

        if width > 1:
            cont = data[scan + 1]
            if not tokenizer_utf8_is_continuation_byte(cont):
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            codepoint = (codepoint << 6) | (cont & 0x3F)

        if width > 2:
            cont = data[scan + 2]
            if not tokenizer_utf8_is_continuation_byte(cont):
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            codepoint = (codepoint << 6) | (cont & 0x3F)

        if width > 3:
            cont = data[scan + 3]
            if not tokenizer_utf8_is_continuation_byte(cont):
                return TOKENIZER_UTF8_ERR_BAD_CONTINUATION
            codepoint = (codepoint << 6) | (cont & 0x3F)

        if width == 2 and codepoint < 0x80:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

        if width == 3 and codepoint < 0x800:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

        if width == 4 and codepoint < 0x10000:
            return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

        scalar_checked = [0]
        err = tokenizer_utf8_validate_codepoint_scalar_checked(codepoint, scalar_checked)
        if err != TOKENIZER_UTF8_OK:
            return err

        next_scan = scan + width
        if next_scan < scan or next_scan > span_end:
            return TOKENIZER_UTF8_ERR_OVERFLOW

        scan = next_scan

    out_is_valid[0] = True
    io_cursor[0] = span_end
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_sequence_span_valid_checked_default(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_is_valid: list[bool] | None,
) -> int:
    if data is None or io_cursor is None or out_is_valid is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    span_nbytes = byte_len - cursor
    return tokenizer_utf8_sequence_span_valid_checked(
        data,
        byte_len,
        io_cursor,
        span_nbytes,
        out_is_valid,
    )


def compare_default_vs_explicit(data: list[int], cursor0: int) -> None:
    cursor_d = [cursor0]
    valid_d = [False]
    err_d = tokenizer_utf8_sequence_span_valid_checked_default(
        data,
        len(data),
        cursor_d,
        valid_d,
    )

    cursor_e = [cursor0]
    valid_e = [False]
    err_e = tokenizer_utf8_sequence_span_valid_checked(
        data,
        len(data),
        cursor_e,
        len(data) - cursor0,
        valid_e,
    )

    assert err_d == err_e
    assert cursor_d[0] == cursor_e[0]
    assert valid_d[0] is valid_e[0]


def test_multilingual_valid_vectors() -> None:
    vectors = [
        "TempleOS",
        "λ",
        "Русский",
        "हिन्दी",
        "漢字かな交じり文",
        "🙂🙃🧠",
        "Aλ🙂Z",
    ]
    for text in vectors:
        data = list(text.encode("utf-8"))
        for cursor in range(len(data) + 1):
            compare_default_vs_explicit(data, cursor)


def test_invalid_utf8_vectors_and_no_partial() -> None:
    bad_vectors = [
        [0x80],
        [0xC2, 0x20],
        [0xE2, 0x82],
        [0xED, 0xA0, 0x80],
        [0xF4, 0x90, 0x80, 0x80],
        [0xC0, 0x80],
    ]

    for data in bad_vectors:
        compare_default_vs_explicit(data, 0)

    data = [0x41, 0x80, 0x42]
    cursor = [1]
    valid = [False]
    err = tokenizer_utf8_sequence_span_valid_checked_default(data, len(data), cursor, valid)
    assert err != TOKENIZER_UTF8_OK
    assert cursor[0] == 1
    assert valid[0] is False


def test_parameter_contracts() -> None:
    cursor = [0]
    valid = [False]

    assert (
        tokenizer_utf8_sequence_span_valid_checked_default(None, 0, cursor, valid)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_sequence_span_valid_checked_default([], 0, None, valid)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_sequence_span_valid_checked_default([], 0, cursor, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    assert (
        tokenizer_utf8_sequence_span_valid_checked_default([0x41], I64_MAX + 1, [0], [False])
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )

    cursor_bad = [2]
    out_bad = [False]
    assert (
        tokenizer_utf8_sequence_span_valid_checked_default([0x41], 1, cursor_bad, out_bad)
        == TOKENIZER_UTF8_ERR_BAD_PARAM
    )
    assert cursor_bad[0] == 2
    assert out_bad[0] is False


def test_randomized_default_explicit_parity() -> None:
    rng = random.Random(20260418_3611)

    for _ in range(10000):
        n = rng.randint(0, 24)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = 0 if n == 0 else rng.randint(0, n)
        compare_default_vs_explicit(data, cursor0)


def test_randomized_valid_multilingual_suffixes() -> None:
    rng = random.Random(20260418_3612)
    corpus = [
        "TempleOS loves integer math",
        "Русский текст",
        "हिन्दी वाक्य",
        "漢字かな交じり文",
        "🙂🙃🧠⚙️",
    ]

    for _ in range(5000):
        text = rng.choice(corpus)
        data = list(text.encode("utf-8"))
        if not data:
            continue

        # pick random codepoint-aligned suffix start
        cps = list(text)
        start_cp = rng.randint(0, len(cps))
        prefix = "".join(cps[:start_cp]).encode("utf-8")
        cursor = len(prefix)
        compare_default_vs_explicit(data, cursor)


if __name__ == "__main__":
    test_multilingual_valid_vectors()
    test_invalid_utf8_vectors_and_no_partial()
    test_parameter_contracts()
    test_randomized_default_explicit_parity()
    test_randomized_valid_multilingual_suffixes()
    print("tokenizer_utf8_sequence_span_valid_checked_default_reference_checks=ok")
