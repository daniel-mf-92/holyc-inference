#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI64LECheckedNoPartial semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U64_MASK = (1 << 64) - 1


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


def gguf_metadata_read_u64le_checked_no_partial(
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


def reinterpret_u64_as_i64(value: int) -> int:
    value &= U64_MASK
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def gguf_metadata_read_i64le_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw_u64 = [0]

    err = gguf_metadata_read_u64le_checked_no_partial(
        buf,
        buf_nbytes,
        cur,
        table_end,
        raw_u64,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u64_as_i64(raw_u64[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i64le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i64le_checked_no_partial(
        buf,
        buf_nbytes,
        cursor_ref,
        table_end,
        out_value_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [6]
    out = [12345]

    assert (
        gguf_metadata_read_i64le_checked_no_partial(None, 32, cursor, 16, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 6
    assert out[0] == 12345

    assert (
        gguf_metadata_read_i64le_checked_no_partial([0] * 32, 32, None, 16, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out[0] == 12345

    assert (
        gguf_metadata_read_i64le_checked_no_partial([0] * 32, 32, cursor, 16, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 6


def test_short_reads_do_not_advance_or_write() -> None:
    buf = [0xAB] * 32
    out = [-101]

    for table_end in range(0, 8):
        cursor = [0]
        err = gguf_metadata_read_i64le_checked_no_partial(
            buf,
            len(buf),
            cursor,
            table_end,
            out,
        )
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out[0] == -101


def test_success_signed_interpretation_and_cursor_advance() -> None:
    buf = [
        0x00,
        0x00,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0x7F,
        0x00,
    ]

    cursor = [2]
    out = [0]

    err = gguf_metadata_read_i64le_checked_no_partial(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == -1
    assert cursor[0] == 10


def test_wrapper_delegates_identically() -> None:
    rng = random.Random(20260417_212)

    for _ in range(3000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor_core = [cursor0]
        out_core = [-(1 << 40)]
        err_core = gguf_metadata_read_i64le_checked_no_partial(
            buf,
            n,
            cursor_core,
            table_end,
            out_core,
        )

        cursor_wrap = [cursor0]
        out_wrap = [-(1 << 40)]
        err_wrap = gguf_metadata_read_i64le_checked(
            buf,
            n,
            cursor_wrap,
            table_end,
            out_wrap,
        )

        assert err_wrap == err_core
        assert cursor_wrap[0] == cursor_core[0]
        assert out_wrap[0] == out_core[0]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_213)

    for _ in range(5000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out = [99]
        err = gguf_metadata_read_i64le_checked_no_partial(
            buf,
            n,
            cursor,
            table_end,
            out,
        )

        if cursor0 + 8 <= table_end and cursor0 + 8 <= n:
            raw = 0
            for i in range(8):
                raw |= buf[cursor0 + i] << (8 * i)
            assert err == GGUF_META_TABLE_OK
            assert out[0] == reinterpret_u64_as_i64(raw)
            assert cursor[0] == cursor0 + 8
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out[0] == 99
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_reads_do_not_advance_or_write()
    test_success_signed_interpretation_and_cursor_advance()
    test_wrapper_delegates_identically()
    test_randomized_parity()
    print("gguf_metadata_read_i64le_checked_no_partial_reference_checks=ok")


if __name__ == "__main__":
    run()
