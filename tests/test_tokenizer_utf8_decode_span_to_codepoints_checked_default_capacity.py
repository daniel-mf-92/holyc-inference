#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8DecodeSpanToCodepointsCheckedDefaultCapacity."""

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
        local_cursor = [scan]
        err = tokenizer_utf8_next_codepoint_checked(data, span_end, local_cursor, cp, used)
        if err != TOKENIZER_UTF8_OK:
            return err
        scan = local_cursor[0]
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


def tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
    data: list[int] | None,
    byte_len: int,
    io_cursor: list[int] | None,
    in_nbytes: int,
    out_codepoints: list[int] | None,
    out_codepoint_count: list[int] | None,
) -> int:
    if (
        data is None
        or io_cursor is None
        or out_codepoints is None
        or out_codepoint_count is None
    ):
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if in_nbytes > I64_MAX:
        return TOKENIZER_UTF8_ERR_OVERFLOW

    return tokenizer_utf8_decode_span_to_codepoints_checked(
        data,
        byte_len,
        io_cursor,
        in_nbytes,
        out_codepoints,
        in_nbytes,
        out_codepoint_count,
    )


def compare_default_vs_explicit(data: list[int], cursor0: int, in_nbytes: int) -> None:
    out_len = max(1, len(data))
    seed = [0xAAAAAAAA for _ in range(out_len)]

    cursor_default = [cursor0]
    out_default = seed.copy()
    count_default = [0xBBBBBBBB]
    err_default = tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
        data,
        len(data),
        cursor_default,
        in_nbytes,
        out_default,
        count_default,
    )

    cursor_explicit = [cursor0]
    out_explicit = seed.copy()
    count_explicit = [0xBBBBBBBB]
    err_explicit = tokenizer_utf8_decode_span_to_codepoints_checked(
        data,
        len(data),
        cursor_explicit,
        in_nbytes,
        out_explicit,
        in_nbytes,
        count_explicit,
    )

    assert err_default == err_explicit
    assert cursor_default[0] == cursor_explicit[0]
    assert count_default[0] == count_explicit[0]
    assert out_default == out_explicit


def test_multilingual_success_vectors() -> None:
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
        compare_default_vs_explicit(payload, 0, len(payload))


def test_malformed_sequences_no_partial_state() -> None:
    vectors = [
        [0x80],
        [0xC2, 0x20],
        [0xE2, 0x82],
        [0xF0, 0x9F, 0x80],
        [0xED, 0xA0, 0x80],
    ]

    for payload in vectors:
        out = [0xDEADBEEF] * max(1, len(payload))
        out_before = out.copy()
        cursor = [0]
        count = [0xBAD0C0DE]
        err = tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload,
            len(payload),
            cursor,
            len(payload),
            out,
            count,
        )
        assert err != TOKENIZER_UTF8_OK
        assert cursor[0] == 0
        assert count[0] == 0xBAD0C0DE
        assert out == out_before


def test_parameter_contracts() -> None:
    payload = [0x41, 0x42]
    cursor = [0]
    count = [123]
    out = [999, 999]

    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            None, 0, cursor, 0, out, count
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload, len(payload), None, 0, out, count
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload, len(payload), cursor, 0, None, count
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload, len(payload), cursor, 0, out, None
        )
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor_bad = [3]
    out_bad = [777, 777]
    count_bad = [888]
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload, len(payload), cursor_bad, 1, out_bad, count_bad
        )
        == TOKENIZER_UTF8_ERR_BAD_PARAM
    )
    assert cursor_bad[0] == 3
    assert count_bad[0] == 888
    assert out_bad == [777, 777]

    cursor_oob = [1]
    out_oob = [111, 111]
    count_oob = [222]
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload, len(payload), cursor_oob, 2, out_oob, count_oob
        )
        == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    )
    assert cursor_oob[0] == 1
    assert count_oob[0] == 222
    assert out_oob == [111, 111]

    cursor_of = [0]
    out_of = [333]
    count_of = [444]
    assert (
        tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            payload, len(payload), cursor_of, I64_MAX + 1, out_of, count_of
        )
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )
    assert cursor_of[0] == 0
    assert count_of[0] == 444
    assert out_of == [333]


def test_randomized_default_vs_explicit_parity() -> None:
    rng = random.Random(20260418_353)
    corpus = [
        "TempleOS loves integer math",
        "Русский текст",
        "हिन्दी वाक्य",
        "漢字かな交じり文",
        "🙂🙃🧠⚙️",
    ]

    for _ in range(4000):
        text = rng.choice(corpus)
        payload = list(text.encode("utf-8"))

        start = rng.randint(0, len(payload))
        end = rng.randint(start, len(payload))
        frag = bytes(payload[start:end])
        try:
            frag_text = frag.decode("utf-8")
        except UnicodeDecodeError:
            continue

        data = payload[:start] + list(frag_text.encode("utf-8")) + payload[end:]
        in_nbytes = len(frag_text.encode("utf-8"))
        compare_default_vs_explicit(data, start, in_nbytes)

    for _ in range(800):
        n = rng.randint(1, 8)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor = [0]
        out = [0xCAFEBABE] * n
        out_before = out.copy()
        count = [0xFACE]

        err = tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity(
            data,
            len(data),
            cursor,
            len(data),
            out,
            count,
        )

        if err != TOKENIZER_UTF8_OK:
            assert cursor[0] == 0
            assert count[0] == 0xFACE
            assert out == out_before


def main() -> None:
    test_multilingual_success_vectors()
    test_malformed_sequences_no_partial_state()
    test_parameter_contracts()
    test_randomized_default_vs_explicit_parity()
    print("tokenizer_utf8_decode_span_to_codepoints_checked_default_capacity: ok")


if __name__ == "__main__":
    main()
