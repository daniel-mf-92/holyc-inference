#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8DecodeSpanToCodepointsChecked semantics."""

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


def tokenizer_utf8_decode_span_to_codepoints_checked(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    in_nbytes: int,
    out_codepoints: list[int] | None,
    out_codepoint_capacity: int,
    out_codepoint_count: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_codepoints is None
        or out_codepoint_count is None
    ):
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if byte_len > I64_MAX or out_codepoint_capacity > I64_MAX:
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
        err = tokenizer_utf8_next_codepoint_checked(data, span_end, [scan], cp, used)
        if err != TOKENIZER_UTF8_OK:
            return err
        scan += used[0]
        if decoded_count == out_codepoint_capacity:
            return TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
        decoded_count += 1
        if decoded_count == 0:
            return TOKENIZER_UTF8_ERR_OVERFLOW

    if scan != span_end:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    scan = cursor
    write_i = 0
    while scan < span_end:
        cp = [0]
        used = [0]
        local_cursor = [scan]
        err = tokenizer_utf8_next_codepoint_checked(data, span_end, local_cursor, cp, used)
        if err != TOKENIZER_UTF8_OK:
            return err
        scan = local_cursor[0]
        if write_i >= decoded_count:
            return TOKENIZER_UTF8_ERR_OVERFLOW
        out_codepoints[write_i] = cp[0]
        write_i += 1

    if write_i != decoded_count:
        return TOKENIZER_UTF8_ERR_BAD_PARAM

    out_codepoint_count[0] = decoded_count
    io_cursor[0] = span_end
    return TOKENIZER_UTF8_OK


def test_success_multilingual_vectors() -> None:
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
        payload = list(text.encode("utf-8"))
        cursor = [0]
        out = [0xAAAAAAAA for _ in range(max(1, len(payload)))]
        out_count = [0xBBBBBBBB]

        err = tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            len(payload),
            cursor,
            len(payload),
            out,
            len(out),
            out_count,
        )
        assert err == TOKENIZER_UTF8_OK
        assert cursor[0] == len(payload)
        expected = [ord(ch) for ch in text]
        assert out_count[0] == len(expected)
        assert out[: out_count[0]] == expected


def test_no_partial_on_capacity_and_invalid_utf8() -> None:
    text = "A🙂B"
    payload = list(text.encode("utf-8"))

    cursor = [0]
    out = [0x1111, 0x2222]
    out_count = [0x3333]
    err = tokenizer_utf8_decode_span_to_codepoints_checked(
        payload,
        len(payload),
        cursor,
        len(payload),
        out,
        2,
        out_count,
    )
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 0
    assert out == [0x1111, 0x2222]
    assert out_count[0] == 0x3333

    bad_payload = [0xE2, 0x82]  # truncated 3-byte sequence
    cursor2 = [0]
    out2 = [0x4444, 0x5555, 0x6666]
    out_count2 = [0x7777]
    err2 = tokenizer_utf8_decode_span_to_codepoints_checked(
        bad_payload,
        len(bad_payload),
        cursor2,
        len(bad_payload),
        out2,
        len(out2),
        out_count2,
    )
    assert err2 == TOKENIZER_UTF8_ERR_TRUNCATED
    assert cursor2[0] == 0
    assert out2 == [0x4444, 0x5555, 0x6666]
    assert out_count2[0] == 0x7777


def test_parameter_contracts() -> None:
    payload = [0x41]
    cursor = [0]
    out = [0xABCD]
    out_count = [0x1234]

    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked(
            None,
            0,
            cursor,
            0,
            out,
            1,
            out_count,
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            1,
            None,
            1,
            out,
            1,
            out_count,
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            1,
            cursor,
            1,
            None,
            1,
            out_count,
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            1,
            cursor,
            1,
            out,
            1,
            None,
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            I64_MAX + 1,
            cursor,
            1,
            out,
            1,
            out_count,
        )
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )

    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            1,
            cursor,
            1,
            out,
            I64_MAX + 1,
            out_count,
        )
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )

    cursor2 = [2]
    out2 = [0xDEAD]
    out_count2 = [0xBEEF]
    err = tokenizer_utf8_decode_span_to_codepoints_checked(
        payload,
        1,
        cursor2,
        0,
        out2,
        1,
        out_count2,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor2[0] == 2
    assert out2 == [0xDEAD]
    assert out_count2[0] == 0xBEEF


def test_randomized_spans_against_python_decode() -> None:
    rng = random.Random(20260418_352)

    corpus = [
        "TempleOS",
        "λ",
        "Русский",
        "हिन्दी",
        "漢字かな交じり文",
        "🙂🙃🧠",
    ]

    for _ in range(5000):
        text = rng.choice(corpus)
        if rng.randint(0, 1):
            text += chr(rng.randint(0, 0x7F))

        payload = list(text.encode("utf-8"))
        n = len(payload)
        lo = rng.randint(0, n)
        hi = rng.randint(lo, n)
        span = payload[lo:hi]

        cursor = [0]
        out = [0x99999999 for _ in range(max(1, len(span)))]
        out_count = [0x88888888]

        err = tokenizer_utf8_decode_span_to_codepoints_checked(
            span,
            len(span),
            cursor,
            len(span),
            out,
            len(out),
            out_count,
        )

        try:
            expected = [ord(ch) for ch in bytes(span).decode("utf-8")]
            assert err == TOKENIZER_UTF8_OK
            assert out_count[0] == len(expected)
            assert out[: out_count[0]] == expected
            assert cursor[0] == len(span)
        except UnicodeDecodeError:
            assert err != TOKENIZER_UTF8_OK
            assert cursor[0] == 0
            assert out == [0x99999999 for _ in range(max(1, len(span)))]
            assert out_count[0] == 0x88888888


def test_randomized_mutated_invalid_no_partial() -> None:
    rng = random.Random(20260418_353)

    for _ in range(7000):
        n = rng.randint(1, 14)
        payload = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)
        in_nbytes = rng.randint(0, n - cursor0)

        cap = max(1, rng.randint(1, n))
        cursor = [cursor0]
        out = [0x12345678 for _ in range(cap)]
        out_count = [0x87654321]

        err = tokenizer_utf8_decode_span_to_codepoints_checked(
            payload,
            n,
            cursor,
            in_nbytes,
            out,
            cap,
            out_count,
        )

        if err != TOKENIZER_UTF8_OK:
            assert cursor[0] == cursor0
            assert out == [0x12345678 for _ in range(cap)]
            assert out_count[0] == 0x87654321
        else:
            assert cursor[0] == cursor0 + in_nbytes
            assert 0 <= out_count[0] <= cap


if __name__ == "__main__":
    test_success_multilingual_vectors()
    test_no_partial_on_capacity_and_invalid_utf8()
    test_parameter_contracts()
    test_randomized_spans_against_python_decode()
    test_randomized_mutated_invalid_no_partial()
    print("tokenizer_utf8_decode_span_to_codepoints_checked_reference_checks=ok")
