#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8CountCodepointsChecked semantics."""

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


def count_reference_via_python_utf8(data: list[int], cursor: int, nbytes: int) -> tuple[int, int]:
    span = bytes(data[cursor : cursor + nbytes])
    text = span.decode("utf-8")
    return len(text), cursor + nbytes


def compare_with_python_reference(data: list[int], cursor0: int, nbytes: int) -> None:
    cursor = [cursor0]
    out_count = [0xAAAAAAAAAAAAAAAA]

    err = tokenizer_utf8_count_codepoints_checked(data, len(data), cursor, nbytes, out_count)
    assert err == TOKENIZER_UTF8_OK

    expected_count, expected_cursor = count_reference_via_python_utf8(data, cursor0, nbytes)
    assert out_count[0] == expected_count
    assert cursor[0] == expected_cursor


def test_success_multilingual_corpora() -> None:
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
        compare_with_python_reference(payload, 0, len(payload))


def test_success_subspan_counts() -> None:
    text = "pre λ🙂漢 post"
    data = list(text.encode("utf-8"))

    full = text.encode("utf-8")
    start = full.index("λ".encode("utf-8"))
    end = full.index(" ".encode("utf-8"), start)
    compare_with_python_reference(data, start, end - start)

    start2 = full.index("🙂".encode("utf-8"))
    end2 = full.index(" ".encode("utf-8"), start2)
    compare_with_python_reference(data, start2, end2 - start2)


def test_zero_length_span_commits_cursor() -> None:
    payload = list("abc".encode("utf-8"))
    cursor = [1]
    out_count = [0xDEADBEEF]

    err = tokenizer_utf8_count_codepoints_checked(payload, len(payload), cursor, 0, out_count)
    assert err == TOKENIZER_UTF8_OK
    assert out_count[0] == 0
    assert cursor[0] == 1


def test_error_no_partial_commit_on_truncated() -> None:
    payload = [0xE2, 0x82]
    cursor = [0]
    out_count = [123456]

    err = tokenizer_utf8_count_codepoints_checked(payload, len(payload), cursor, len(payload), out_count)
    assert err == TOKENIZER_UTF8_ERR_TRUNCATED
    assert cursor[0] == 0
    assert out_count[0] == 123456


def test_error_no_partial_commit_on_bad_continuation() -> None:
    payload = [0xC2, 0x41]
    cursor = [0]
    out_count = [777]

    err = tokenizer_utf8_count_codepoints_checked(payload, len(payload), cursor, len(payload), out_count)
    assert err == TOKENIZER_UTF8_ERR_BAD_CONTINUATION
    assert cursor[0] == 0
    assert out_count[0] == 777


def test_error_no_partial_commit_on_bad_lead_byte() -> None:
    payload = [0x80]
    cursor = [0]
    out_count = [321]

    err = tokenizer_utf8_count_codepoints_checked(payload, len(payload), cursor, len(payload), out_count)
    assert err == TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE
    assert cursor[0] == 0
    assert out_count[0] == 321


def test_error_contracts_null_and_bounds() -> None:
    payload = list("ok".encode("utf-8"))

    assert (
        tokenizer_utf8_count_codepoints_checked(None, len(payload), [0], len(payload), [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_count_codepoints_checked(payload, len(payload), None, len(payload), [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_count_codepoints_checked(payload, len(payload), [0], len(payload), None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor = [len(payload) + 1]
    out_count = [0]
    err = tokenizer_utf8_count_codepoints_checked(payload, len(payload), cursor, 0, out_count)
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor[0] == len(payload) + 1
    assert out_count[0] == 0

    cursor = [1]
    out_count = [11]
    err = tokenizer_utf8_count_codepoints_checked(payload, len(payload), cursor, 2, out_count)
    assert err == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 1
    assert out_count[0] == 11


def test_randomized_valid_utf8_spans() -> None:
    random.seed(0xC0DEC0DE)

    alphabet = [
        "A",
        "z",
        " ",
        "λ",
        "Ж",
        "ह",
        "🙂",
        "🧠",
        "漢",
        "字",
        "!",
    ]

    for _ in range(250):
        length = random.randint(0, 64)
        text = "".join(random.choice(alphabet) for _ in range(length))
        data = list(text.encode("utf-8"))

        if not data:
            cursor0 = 0
            nbytes = 0
        else:
            cursor0 = random.randint(0, len(data))
            max_n = len(data) - cursor0
            nbytes = random.randint(0, max_n)
            # Snap to valid UTF-8 boundary by reducing until decodable.
            while nbytes > 0:
                try:
                    bytes(data[cursor0 : cursor0 + nbytes]).decode("utf-8")
                    break
                except UnicodeDecodeError:
                    nbytes -= 1

        compare_with_python_reference(data, cursor0, nbytes)


def test_randomized_malformed_sequences_no_partial_state() -> None:
    random.seed(0xBAD5EED)

    malformed_starts = [
        [0x80],
        [0xC2],
        [0xE0, 0x80],
        [0xF4, 0x90, 0x80, 0x80],
        [0xC2, 0x20],
        [0xE2, 0x28, 0xA1],
    ]

    for prefix in malformed_starts:
        for _ in range(25):
            suffix_len = random.randint(0, 8)
            suffix = [random.randint(0, 255) for _ in range(suffix_len)]
            payload = prefix + suffix

            cursor0 = random.randint(0, min(1, len(payload)))
            nbytes = len(payload) - cursor0
            cursor = [cursor0]
            out_count = [0x1234]

            err = tokenizer_utf8_count_codepoints_checked(
                payload,
                len(payload),
                cursor,
                nbytes,
                out_count,
            )

            if err == TOKENIZER_UTF8_OK:
                # Some random suffixes can make the whole span valid; in that case
                # the helper must still exactly match Python decoding count.
                expected_count, expected_cursor = count_reference_via_python_utf8(
                    payload, cursor0, nbytes
                )
                assert out_count[0] == expected_count
                assert cursor[0] == expected_cursor
            else:
                assert err in {
                    TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE,
                    TOKENIZER_UTF8_ERR_BAD_CONTINUATION,
                    TOKENIZER_UTF8_ERR_BAD_CODEPOINT,
                    TOKENIZER_UTF8_ERR_TRUNCATED,
                }
                assert cursor[0] == cursor0
                assert out_count[0] == 0x1234


if __name__ == "__main__":
    test_success_multilingual_corpora()
    test_success_subspan_counts()
    test_zero_length_span_commits_cursor()
    test_error_no_partial_commit_on_truncated()
    test_error_no_partial_commit_on_bad_continuation()
    test_error_no_partial_commit_on_bad_lead_byte()
    test_error_contracts_null_and_bounds()
    test_randomized_valid_utf8_spans()
    test_randomized_malformed_sequences_no_partial_state()
    print("ok")
