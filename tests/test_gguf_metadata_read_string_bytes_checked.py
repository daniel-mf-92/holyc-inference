#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadStringBytesChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1


def gguf_metadata_cursor_can_advance_checked(
    cursor: int,
    need: int,
    table_end: int,
) -> tuple[int, int | None]:
    if cursor > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if need > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if cursor > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM, None

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None

    return GGUF_META_TABLE_OK, next_cursor


def gguf_metadata_read_string_bytes_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    str_len: int,
    out_bytes_ref: list[int] | None,
    out_nbytes: int,
) -> int:
    if buf is None or cursor_ref is None or out_bytes_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if out_nbytes < str_len:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    cursor0 = cursor_ref[0]
    out0 = list(out_bytes_ref)

    err, next_cursor = gguf_metadata_cursor_can_advance_checked(cursor0, str_len, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert next_cursor is not None
    if next_cursor > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    for i in range(str_len):
        out_bytes_ref[i] = buf[cursor0 + i]

    cursor_ref[0] = next_cursor

    # The caller passes a preallocated buffer; bytes beyond str_len remain untouched.
    for i in range(str_len, len(out_bytes_ref)):
        assert out_bytes_ref[i] == out0[i]

    return GGUF_META_TABLE_OK


def test_null_ptr_and_basic_param_rejection() -> None:
    cursor = [0]
    out = [0xAA] * 8

    assert (
        gguf_metadata_read_string_bytes_checked(None, 8, cursor, 8, 2, out, len(out))
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert (
        gguf_metadata_read_string_bytes_checked([1, 2], 2, None, 2, 2, out, len(out))
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert (
        gguf_metadata_read_string_bytes_checked([1, 2], 2, cursor, 2, 2, None, 0)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )

    cursor = [0]
    out = [0x11] * 4
    err = gguf_metadata_read_string_bytes_checked([9, 8, 7], 3, cursor, 3, 3, out, 2)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out == [0x11] * 4


def test_bounds_failures_do_not_mutate() -> None:
    cursor = [2]
    out = [0xCC] * 6
    buf = [1, 2, 3, 4]

    err = gguf_metadata_read_string_bytes_checked(buf, len(buf), cursor, 3, 2, out, len(out))
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 2
    assert out == [0xCC] * 6

    err = gguf_metadata_read_string_bytes_checked(buf, 3, cursor, 4, 2, out, len(out))
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 2
    assert out == [0xCC] * 6


def test_success_copies_exact_bytes_and_advances_once() -> None:
    text = b"TempleOS"
    buf = [0xFF, 0xEE] + list(text) + [0xDD]

    cursor = [2]
    out = [0x44] * 16

    err = gguf_metadata_read_string_bytes_checked(
        buf,
        len(buf),
        cursor,
        len(buf) - 1,
        len(text),
        out,
        len(out),
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 2 + len(text)
    assert bytes(out[: len(text)]) == text
    assert out[len(text) :] == [0x44] * (len(out) - len(text))


def test_zero_length_is_noop_but_valid() -> None:
    buf = [7, 8, 9]
    cursor = [1]
    out = [0x5A, 0x5A, 0x5A]

    err = gguf_metadata_read_string_bytes_checked(buf, len(buf), cursor, len(buf), 0, out, len(out))
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 1
    assert out == [0x5A, 0x5A, 0x5A]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_201)

    for _ in range(6000):
        n = rng.randint(0, 160)
        buf = [rng.randint(0, 255) for _ in range(n)]

        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)
        str_len = rng.randint(0, 96)
        out_len = rng.randint(0, 128)

        cursor = [cursor0]
        out = [0xA7] * max(1, out_len)
        out_snapshot = list(out)

        err = gguf_metadata_read_string_bytes_checked(
            buf,
            n,
            cursor,
            table_end,
            str_len,
            out,
            out_len,
        )

        if out_len < str_len:
            assert err == GGUF_META_TABLE_ERR_BAD_PARAM
            assert cursor[0] == cursor0
            assert out == out_snapshot
            continue

        if cursor0 + str_len > table_end or cursor0 > table_end:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert cursor[0] == cursor0
            assert out == out_snapshot
            continue

        if cursor0 + str_len > n:
            assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            assert cursor[0] == cursor0
            assert out == out_snapshot
            continue

        assert err == GGUF_META_TABLE_OK
        assert cursor[0] == cursor0 + str_len
        assert out[:str_len] == buf[cursor0 : cursor0 + str_len]
        assert out[str_len:] == out_snapshot[str_len:]


def run() -> None:
    test_null_ptr_and_basic_param_rejection()
    test_bounds_failures_do_not_mutate()
    test_success_copies_exact_bytes_and_advances_once()
    test_zero_length_is_noop_but_valid()
    test_randomized_parity()
    print("gguf_metadata_read_string_bytes_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
