#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU64LEChecked cursor/bounds semantics."""

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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out = [0xA5A5A5A5A5A5A5A5]

    assert (
        gguf_metadata_read_u64le_checked(None, 16, cursor, 16, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3
    assert out[0] == 0xA5A5A5A5A5A5A5A5

    assert (
        gguf_metadata_read_u64le_checked([0] * 16, 16, None, 16, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out[0] == 0xA5A5A5A5A5A5A5A5

    assert (
        gguf_metadata_read_u64le_checked([0] * 16, 16, cursor, 16, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3


def test_short_reads_do_not_advance_cursor_or_write_output() -> None:
    buf = [0xFF] * 16
    out = [0x1122334455667788]

    for table_end in range(0, 8):
        cursor = [0]
        err = gguf_metadata_read_u64le_checked(buf, len(buf), cursor, table_end, out)
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out[0] == 0x1122334455667788


def test_success_decodes_little_endian_and_advances() -> None:
    buf = [
        0xAA,
        0xBB,
        0x01,
        0x23,
        0x45,
        0x67,
        0x89,
        0xAB,
        0xCD,
        0xEF,
        0x11,
        0x22,
    ]

    cursor = [2]
    out = [0]
    err = gguf_metadata_read_u64le_checked(buf, len(buf), cursor, len(buf), out)

    assert err == GGUF_META_TABLE_OK
    assert out[0] == 0xEFCDAB8967452301
    assert cursor[0] == 10


def test_randomized_parity() -> None:
    rng = random.Random(20260417_195)

    for _ in range(4000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out = [0xDEADBEEFDEADBEEF]
        err = gguf_metadata_read_u64le_checked(buf, n, cursor, table_end, out)

        if cursor0 + 8 <= table_end and cursor0 + 8 <= n:
            expect = 0
            for i in range(8):
                expect |= buf[cursor0 + i] << (8 * i)
            assert err == GGUF_META_TABLE_OK
            assert out[0] == expect
            assert cursor[0] == cursor0 + 8
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out[0] == 0xDEADBEEFDEADBEEF
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_reads_do_not_advance_cursor_or_write_output()
    test_success_decodes_little_endian_and_advances()
    test_randomized_parity()
    print("gguf_metadata_read_u64le_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
