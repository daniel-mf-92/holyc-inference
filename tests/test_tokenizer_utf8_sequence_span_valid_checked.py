#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8SequenceSpanValidChecked."""

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

        checked = [0]
        err = tokenizer_utf8_validate_codepoint_scalar_checked(codepoint, checked)
        if err != TOKENIZER_UTF8_OK:
            return err

        next_scan = scan + width
        if next_scan < scan or next_scan > span_end:
            return TOKENIZER_UTF8_ERR_OVERFLOW

        scan = next_scan

    out_is_valid[0] = True
    io_cursor[0] = span_end
    return TOKENIZER_UTF8_OK


def run_case(
    data: list[int],
    cursor0: int,
    span_nbytes: int,
    expected_err: int,
    expected_cursor: int,
    expected_valid: bool,
) -> None:
    cursor = [cursor0]
    out_valid = [False]
    err = tokenizer_utf8_sequence_span_valid_checked(
        data,
        len(data),
        cursor,
        span_nbytes,
        out_valid,
    )
    assert err == expected_err
    assert cursor[0] == expected_cursor
    assert out_valid[0] is expected_valid


def test_success_spans() -> None:
    payload = list("A¢€🙂".encode("utf-8"))
    run_case(payload, 0, len(payload), TOKENIZER_UTF8_OK, len(payload), True)

    payload2 = list("xРусскийy".encode("utf-8"))
    inner = list("Русский".encode("utf-8"))
    start = 1
    run_case(payload2, start, len(inner), TOKENIZER_UTF8_OK, start + len(inner), True)

    payload3 = list("abc".encode("utf-8"))
    run_case(payload3, 2, 0, TOKENIZER_UTF8_OK, 2, True)


def test_failures_no_partial_state() -> None:
    run_case([0x80, 0x41], 0, 1, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0, False)
    run_case([0xC2, 0x20], 0, 2, TOKENIZER_UTF8_ERR_BAD_CONTINUATION, 0, False)
    run_case([0xE2, 0x82], 0, 2, TOKENIZER_UTF8_ERR_TRUNCATED, 0, False)
    run_case([0xED, 0xA0, 0x80], 0, 3, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0, False)


def test_parameter_contracts() -> None:
    cursor = [0]
    out_valid = [False]

    assert (
        tokenizer_utf8_sequence_span_valid_checked(None, 0, cursor, 0, out_valid)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_sequence_span_valid_checked([], 0, None, 0, out_valid)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_sequence_span_valid_checked([], 0, cursor, 0, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    assert (
        tokenizer_utf8_sequence_span_valid_checked([0x41], I64_MAX + 1, cursor, 1, out_valid)
        == TOKENIZER_UTF8_ERR_OVERFLOW
    )

    cursor_bad = [3]
    out_bad = [False]
    assert (
        tokenizer_utf8_sequence_span_valid_checked([0x41, 0x42], 2, cursor_bad, 0, out_bad)
        == TOKENIZER_UTF8_ERR_BAD_PARAM
    )
    assert cursor_bad[0] == 3
    assert out_bad[0] is False

    cursor_oob = [1]
    out_oob = [False]
    assert (
        tokenizer_utf8_sequence_span_valid_checked([0x41, 0x42], 2, cursor_oob, 2, out_oob)
        == TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS
    )
    assert cursor_oob[0] == 1
    assert out_oob[0] is False


def test_randomized_valid_multilingual_windows() -> None:
    rng = random.Random(20260418_3461)
    corpus = [
        "TempleOS loves integer math",
        "Русский текст",
        "हिन्दी वाक्य",
        "漢字かな交じり文",
        "🙂🙃🧠⚙️",
    ]

    for _ in range(3000):
        text = rng.choice(corpus)
        data = list(text.encode("utf-8"))
        if not data:
            continue

        start = rng.randint(0, len(data))
        if start == len(data):
            span = 0
        else:
            end = rng.randint(start, len(data))
            span = end - start
            # Align to codepoint boundaries by re-encoding substring when needed.
            fragment = bytes(data[start : start + span])
            try:
                frag_text = fragment.decode("utf-8")
            except UnicodeDecodeError:
                continue
            span = len(frag_text.encode("utf-8"))

        cursor = [start]
        out_valid = [False]
        err = tokenizer_utf8_sequence_span_valid_checked(data, len(data), cursor, span, out_valid)
        assert err == TOKENIZER_UTF8_OK
        assert cursor[0] == start + span
        assert out_valid[0] is True


def test_randomized_mutated_no_partial_on_failures() -> None:
    rng = random.Random(20260418_3462)

    for _ in range(7000):
        n = rng.randint(1, 16)
        data = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n - 1)
        span = rng.randint(1, n - cursor0)

        cursor = [cursor0]
        out_valid = [False]
        err = tokenizer_utf8_sequence_span_valid_checked(data, n, cursor, span, out_valid)

        if err != TOKENIZER_UTF8_OK:
            assert cursor[0] == cursor0
            assert out_valid[0] is False
        else:
            assert cursor[0] == cursor0 + span
            assert out_valid[0] is True


if __name__ == "__main__":
    test_success_spans()
    test_failures_no_partial_state()
    test_parameter_contracts()
    test_randomized_valid_multilingual_windows()
    test_randomized_mutated_no_partial_on_failures()
    print("tokenizer_utf8_sequence_span_valid_checked_reference_checks=ok")
