#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8CountCodepointsCheckedDefaultSpanNoPartial."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_tokenizer_utf8_count_codepoints_checked import (
    I64_MAX,
    TOKENIZER_UTF8_ERR_BAD_PARAM,
    TOKENIZER_UTF8_ERR_NULL_PTR,
    TOKENIZER_UTF8_ERR_OVERFLOW,
    TOKENIZER_UTF8_OK,
    tokenizer_utf8_count_codepoints_checked,
)


def tokenizer_utf8_count_codepoints_checked_default_span_no_partial(
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

    staged_cursor = [cursor]
    staged_count = [out_codepoint_count[0]]
    err = tokenizer_utf8_count_codepoints_checked(
        data,
        byte_len,
        staged_cursor,
        in_nbytes,
        staged_count,
    )
    if err != TOKENIZER_UTF8_OK:
        return err

    io_cursor[0] = staged_cursor[0]
    out_codepoint_count[0] = staged_count[0]
    return TOKENIZER_UTF8_OK


def explicit_staged_default_composition(
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

    staged_cursor = [cursor]
    staged_count = [out_codepoint_count[0]]
    err = tokenizer_utf8_count_codepoints_checked(
        data,
        byte_len,
        staged_cursor,
        in_nbytes,
        staged_count,
    )
    if err != TOKENIZER_UTF8_OK:
        return err

    io_cursor[0] = staged_cursor[0]
    out_codepoint_count[0] = staged_count[0]
    return TOKENIZER_UTF8_OK


def compare_wrapper_vs_explicit(data: list[int], cursor0: int) -> None:
    cursor_w = [cursor0]
    count_w = [0xA1A2A3A4]
    err_w = tokenizer_utf8_count_codepoints_checked_default_span_no_partial(
        data,
        len(data),
        cursor_w,
        count_w,
    )

    cursor_e = [cursor0]
    count_e = [0xA1A2A3A4]
    err_e = explicit_staged_default_composition(
        data,
        len(data),
        cursor_e,
        count_e,
    )

    assert err_w == err_e
    assert cursor_w[0] == cursor_e[0]
    assert count_w[0] == count_e[0]


def test_multilingual_valid_tail_parity() -> None:
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
        data = list(text.encode("utf-8"))
        for cursor in range(len(data) + 1):
            compare_wrapper_vs_explicit(data, cursor)


def test_malformed_utf8_tail_no_partial() -> None:
    vectors = [
        [0x80],
        [0xC2, 0x20],
        [0xE2, 0x82],
        [0xF0, 0x9F, 0x80],
        [0xED, 0xA0, 0x80],
        [0xF4, 0x90, 0x80, 0x80],
    ]

    for data in vectors:
        compare_wrapper_vs_explicit(data, 0)

    cursor = [0]
    count = [0xBEEFBEEF]
    err = tokenizer_utf8_count_codepoints_checked_default_span_no_partial(
        [0xE2, 0x82],
        2,
        cursor,
        count,
    )
    assert err != TOKENIZER_UTF8_OK
    assert cursor[0] == 0
    assert count[0] == 0xBEEFBEEF


def test_contracts_and_overflow() -> None:
    assert (
        tokenizer_utf8_count_codepoints_checked_default_span_no_partial(None, 0, [0], [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_count_codepoints_checked_default_span_no_partial([], 0, None, [0])
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )
    assert (
        tokenizer_utf8_count_codepoints_checked_default_span_no_partial([], 0, [0], None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )

    cursor = [0]
    count = [0x3333]
    err = tokenizer_utf8_count_codepoints_checked_default_span_no_partial(
        [0x41],
        I64_MAX + 1,
        cursor,
        count,
    )
    assert err == TOKENIZER_UTF8_ERR_OVERFLOW
    assert cursor[0] == 0
    assert count[0] == 0x3333

    cursor_bad = [2]
    count_bad = [0x4444]
    err = tokenizer_utf8_count_codepoints_checked_default_span_no_partial(
        [0x41],
        1,
        cursor_bad,
        count_bad,
    )
    assert err == TOKENIZER_UTF8_ERR_BAD_PARAM
    assert cursor_bad[0] == 2
    assert count_bad[0] == 0x4444


def test_randomized_parity_against_explicit_staged() -> None:
    rng = random.Random(20260418_389)

    corpora = [
        "TempleOS 1996",
        "mañana déjà vu",
        "Русский текст",
        "漢字かな交じり",
        "🙂🙃",
        "γειά σου κόσμε",
    ]

    for _ in range(12000):
        text = rng.choice(corpora)
        if rng.randint(0, 1):
            text += chr(rng.randint(32, 126))
        data = list(text.encode("utf-8"))

        if rng.randint(0, 5) == 0:
            # Inject malformed tail bytes in some cases.
            data = data[:]
            if data:
                data[-1] = rng.choice([0x80, 0xC2, 0xE2, 0xF0])
            else:
                data = [0xE2, 0x82]

        cursor = 0 if not data else rng.randint(0, len(data))
        compare_wrapper_vs_explicit(data, cursor)


if __name__ == "__main__":
    test_multilingual_valid_tail_parity()
    test_malformed_utf8_tail_no_partial()
    test_contracts_and_overflow()
    test_randomized_parity_against_explicit_staged()
    print("tokenizer_utf8_count_codepoints_checked_default_span_no_partial_parity=ok")
