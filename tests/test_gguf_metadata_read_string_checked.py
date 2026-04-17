#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadStringChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

GGUF_MAX_STRING_BYTES = 1 << 20
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
    if cursor + need > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None
    return GGUF_META_TABLE_OK, cursor + need


def gguf_metadata_read_u8_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cur = cursor_ref[0]
    err, next_cursor = gguf_metadata_cursor_can_advance_checked(cur, 1, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert next_cursor is not None
    if next_cursor > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    out_value_ref[0] = buf[cur]
    cursor_ref[0] = next_cursor
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    b = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= b[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_u64_header_and_bounds_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_len_ref: list[int] | None,
    out_payload_end_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_len_ref is None
        or out_payload_end_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    str_len = [0]

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, str_len)
    if err != GGUF_META_TABLE_OK:
        return err

    err, payload_end = gguf_metadata_cursor_can_advance_checked(cur[0], str_len[0], table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert payload_end is not None
    if payload_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    if str_len[0] > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_len_ref[0] = str_len[0]
    out_payload_end_ref[0] = payload_end
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_len_u64_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_len_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_len_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    str_len = [0]
    payload_end = [0]

    err = gguf_metadata_read_string_u64_header_and_bounds_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        str_len,
        payload_end,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_len_ref[0] = str_len[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_bytes_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    str_len: int,
    out_bytes: list[int] | None,
    out_nbytes: int,
) -> int:
    if buf is None or cursor_ref is None or out_bytes is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if str_len > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if out_nbytes < str_len:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    cur = cursor_ref[0]
    err, next_cursor = gguf_metadata_cursor_can_advance_checked(cur, str_len, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert next_cursor is not None
    if next_cursor > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    for i in range(str_len):
        out_bytes[i] = buf[cur + i]

    cursor_ref[0] = next_cursor
    return GGUF_META_TABLE_OK


def gguf_metadata_read_string_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_bytes: list[int] | None,
    out_nbytes: int,
    out_len_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_bytes is None or out_len_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    str_len = [0]

    err = gguf_metadata_read_string_len_u64_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        str_len,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if out_nbytes < str_len[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    err = gguf_metadata_read_string_bytes_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        str_len[0],
        out_bytes,
        out_nbytes,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_len_ref[0] = str_len[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def encode_u64le(x: int) -> list[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out = [0xDD] * 8
    out_before = out.copy()
    out_len = [77]

    assert (
        gguf_metadata_read_string_checked(None, 64, cursor, 64, out, len(out), out_len)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 5
    assert out == out_before
    assert out_len[0] == 77

    assert (
        gguf_metadata_read_string_checked([0] * 64, 64, None, 64, out, len(out), out_len)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out == out_before

    assert (
        gguf_metadata_read_string_checked([0] * 64, 64, cursor, 64, None, len(out), out_len)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 5

    assert (
        gguf_metadata_read_string_checked([0] * 64, 64, cursor, 64, out, len(out), None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 5


def test_header_failure_does_not_mutate_outputs() -> None:
    out = [0xAA] * 16
    out_before = out.copy()
    out_len = [123]

    for table_end in range(0, 8):
        cursor = [0]
        err = gguf_metadata_read_string_checked(
            [0xBB] * 64,
            64,
            cursor,
            table_end,
            out,
            len(out),
            out_len,
        )
        assert err in (GGUF_META_TABLE_ERR_BAD_PARAM, GGUF_META_TABLE_ERR_OUT_OF_BOUNDS)
        assert cursor[0] == 0
        assert out_len[0] == 123
        assert out == out_before


def test_output_too_small_rejects_without_copy() -> None:
    text = b"holy"
    buf = encode_u64le(len(text)) + list(text)
    out = [0xFE] * 3
    out_before = out.copy()
    out_len = [0xABCD]
    cursor = [0]

    err = gguf_metadata_read_string_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out,
        len(out),
        out_len,
    )

    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_len[0] == 0xABCD
    assert out == out_before


def test_success_reads_len_and_payload_atomically() -> None:
    text = b"templeos"
    suffix = [0x99, 0x88]
    buf = encode_u64le(len(text)) + list(text) + suffix

    out = [0] * 32
    out_len = [0]
    cursor = [0]

    err = gguf_metadata_read_string_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out,
        len(out),
        out_len,
    )

    assert err == GGUF_META_TABLE_OK
    assert out_len[0] == len(text)
    assert cursor[0] == 8 + len(text)
    assert out[: len(text)] == list(text)


def test_randomized_parity() -> None:
    rng = random.Random(20260417_214)

    for _ in range(3500):
        n = rng.randint(16, 400)
        buf = [rng.randint(0, 255) for _ in range(n)]

        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end) if table_end > 0 else 0

        out_nbytes = rng.randint(0, 80)
        out = [rng.randint(0, 255) for _ in range(out_nbytes)]
        out_before = out.copy()

        out_len = [rng.randint(0, 1 << 20)]
        out_len_before = out_len[0]
        cursor = [cursor0]

        err = gguf_metadata_read_string_checked(
            buf,
            n,
            cursor,
            table_end,
            out,
            out_nbytes,
            out_len,
        )

        if err == GGUF_META_TABLE_OK:
            str_len = out_len[0]
            assert cursor[0] == cursor0 + 8 + str_len
            assert str_len <= out_nbytes
            assert out[:str_len] == buf[cursor0 + 8 : cursor0 + 8 + str_len]
        else:
            assert cursor[0] == cursor0
            assert out_len[0] == out_len_before
            assert out == out_before


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_header_failure_does_not_mutate_outputs()
    test_output_too_small_rejects_without_copy()
    test_success_reads_len_and_payload_atomically()
    test_randomized_parity()
    print("gguf_metadata_read_string_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
