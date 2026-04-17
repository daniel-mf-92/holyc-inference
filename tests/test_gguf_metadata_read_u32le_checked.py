#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU32LEChecked semantics."""

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


def gguf_metadata_read_u32le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    b0 = [0]
    b1 = [0]
    b2 = [0]
    b3 = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b0)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b1)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b2)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b3)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = b0[0] | (b1[0] << 8) | (b2[0] << 16) | (b3[0] << 24)
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out = [0x11111111]

    assert (
        gguf_metadata_read_u32le_checked(None, 5, cursor, 5, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [2]
    assert out == [0x11111111]

    assert (
        gguf_metadata_read_u32le_checked([1, 2, 3, 4], 4, None, 4, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out == [0x11111111]

    assert (
        gguf_metadata_read_u32le_checked([1, 2, 3, 4], 4, cursor, 4, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [2]


def test_short_reads_do_not_advance_cursor() -> None:
    buf = [0xAA, 0xBB, 0xCC, 0xDD]

    cursor = [2]
    out = [0xDEADBEEF]
    assert (
        gguf_metadata_read_u32le_checked(buf, len(buf), cursor, len(buf), out)
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [2]
    assert out == [0xDEADBEEF]

    cursor = [1]
    out = [123]
    assert (
        gguf_metadata_read_u32le_checked(buf, 3, cursor, 4, out)
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [1]
    assert out == [123]


def test_success_decodes_little_endian_and_advances() -> None:
    buf = [0xEF, 0xBE, 0xAD, 0xDE, 0x99]
    cursor = [0]
    out = [0]

    err = gguf_metadata_read_u32le_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == 0xDEADBEEF
    assert cursor[0] == 4


def test_randomized_parity() -> None:
    rng = random.Random(20260417_194)

    for _ in range(4000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out = [0xAAAAAAAA]

        err = gguf_metadata_read_u32le_checked(buf, n, cursor, table_end, out)

        if cursor0 + 4 <= table_end and cursor0 + 4 <= n:
            expect = (
                buf[cursor0]
                | (buf[cursor0 + 1] << 8)
                | (buf[cursor0 + 2] << 16)
                | (buf[cursor0 + 3] << 24)
            )
            assert err == GGUF_META_TABLE_OK
            assert out[0] == expect
            assert cursor[0] == cursor0 + 4
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out[0] == 0xAAAAAAAA
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_reads_do_not_advance_cursor()
    test_success_decodes_little_endian_and_advances()
    test_randomized_parity()
    print("gguf_metadata_read_u32le_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
