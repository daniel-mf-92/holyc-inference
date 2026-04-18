#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8CountCodepointsCheckedDefaultSpan."""

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


def tokenizer_utf8_count_codepoints_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    in_nbytes: int,
    out_codepoint_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_codepoint_count is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX or in_nbytes > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    if in_nbytes > byte_len - cursor:
        return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS

    span_end = cursor + in_nbytes
    if span_end < cursor or span_end > byte_len:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    scan = cursor
    decoded_count = 0
    while scan < span_end:
        cp = [0]
        used = [0]
        local_cursor = [scan]
        err = tokenizer_utf8_next_codepoint_checked(data, span_end, local_cursor, cp, used)
        if err != TOKENIZER_UTF8_OK:
            return err

        scan = local_cursor[0]
        decoded_count += 1
        if decoded_count == 0:
            return TOKENIZER_UTF8_ERR_OVERFLOW

    if scan != span_end:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    out_codepoint_count[0] = decoded_count
    io_cursor[0] = span_end
    return TOKENIZER_UTF8_OK


def tokenizer_utf8_count_codepoints_checked_default_span(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    out_codepoint_count: list[int] | None,
) -> int:
    if data is None or io_cursor is None or out_codepoint_count is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    cursor = io_cursor[0]
    if cursor > byte_len:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    in_nbytes = byte_len - cursor

    return tokenizer_utf8_count_codepoints_checked(
        data,
        byte_len,
        io_cursor,
        in_nbytes,
        out_codepoint_count,
    )


def compare_default_vs_explicit(data: list[int], cursor0: int) -> None:
    cursor_default = [cursor0]
    out_default = [0xAAAAAAAAAAAAAAAA]
    err_default = tokenizer_utf8_count_codepoints_checked_default_span(
        data,
        len(data),
        cursor_default,
        out_default,
    )

    cursor_explicit = [cursor0]
    out_explicit = [0xAAAAAAAAAAAAAAAA]
    err_explicit = tokenizer_utf8_count_codepoints_checked(
        data,
        len(data),
        cursor_explicit,
        len(data) - cursor0,
        out_explicit,
    )

    assert err_default == err_explicit
    assert cursor_default[0] == cursor_explicit[0]
    assert out_default[0] == out_explicit[0]


def test_multilingual_valid_corpora_parity() -> None:
    vectors = [
        "TempleOS",
        "λ",
        "Русский",
        "हिन्दी",
        "漢字かな交じり文",
        "🙂🙃🧠",
        "Aλ🙂Z",
        "مرحبا Temple",
    ]

    for text in vectors:
        payload = list(text.encode("utf-8"))
        compare_default_vs_explicit(payload, 0)
        if payload:
            compare_default_vs_explicit(payload, 1)


def test_malformed_sequences_parity_and_no_partial() -> None:
    vectors = [
        [0x80],
        [0xC2, 0x20],
        [0xE2, 0x82],
        [0xF0, 0x9F, 0x80],
        [0xED, 0xA0, 0x80],
        [0xF4, 0x90, 0x80, 0x80],
    ]

    for payload in vectors:
        cursor0 = 0
        before_cursor = [cursor0]
        before_count = [0xBAD0C0DE]
        err = tokenizer_utf8_count_codepoints_checked_default_span(
            payload,
            len(payload),
            before_cursor,
            before_count,
        )
        assert err != TOKENIZER_UTF8_OK
        assert before_cursor[0] == cursor0
        assert before_count[0] == 0xBAD0C0DE
        compare_default_vs_explicit(payload, cursor0)


def test_contracts_and_overflow_parity() -> None:
    payload = list("ok".encode("utf-8"))

    assert (
        tokenizer_utf8_count_codepoints_checked_default_span(None, len(payload), [0], [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_count_codepoints_checked_default_span(payload, len(payload), None, [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_count_codepoints_checked_default_span(payload, len(payload), [0], None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor = [len(payload) + 1]
    out_count = [7]
    err = tokenizer_utf8_count_codepoints_checked_default_span(
        payload,
        len(payload),
        cursor,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor[0] == len(payload) + 1
    assert out_count[0] == 7

    cursor = [0]
    out_count = [11]
    err = tokenizer_utf8_count_codepoints_checked_default_span(
        payload,
        I64_MAX + 1,
        cursor,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out_count[0] == 11


def test_randomized_valid_utf8_tail_spans() -> None:
    random.seed(0xC0DE373)

    alphabet = ["A", " ", "λ", "Ж", "ह", "🙂", "🧠", "漢", "字", "!"]

    for _ in range(250):
        length = random.randint(0, 72)
        text = "".join(random.choice(alphabet) for _ in range(length))
        data = list(text.encode("utf-8"))
        cursor0 = 0 if not data else random.randint(0, len(data))

        while cursor0 < len(data):
            try:
                bytes(data[cursor0:]).decode("utf-8")
                break
            except UnicodeDecodeError:
                cursor0 += 1

        compare_default_vs_explicit(data, cursor0)


if __name__ == "__main__":
    test_multilingual_valid_corpora_parity()
    test_malformed_sequences_parity_and_no_partial()
    test_contracts_and_overflow_parity()
    test_randomized_valid_utf8_tail_spans()
    print("ok")
