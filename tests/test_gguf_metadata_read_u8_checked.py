#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU8Checked cursor/bounds semantics."""

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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out = [77]

    assert gguf_metadata_read_u8_checked(None, 5, cursor, 5, out) == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [3]
    assert out == [77]

    assert gguf_metadata_read_u8_checked([1, 2, 3], 3, None, 3, out) == GGUF_META_TABLE_ERR_NULL_PTR
    assert out == [77]

    assert (
        gguf_metadata_read_u8_checked([1, 2, 3], 3, cursor, 3, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [3]


def test_overflow_and_bounds_contracts() -> None:
    cursor = [0]
    out = [0]

    assert (
        gguf_metadata_read_u8_checked([9, 8, 7], I64_MAX + 1, cursor, 3, out)
        == GGUF_META_TABLE_ERR_OVERFLOW
    )
    assert cursor == [0]
    assert out == [0]

    cursor = [4]
    out = [5]
    assert (
        gguf_metadata_read_u8_checked([1, 2, 3, 4], 4, cursor, 3, out)
        == GGUF_META_TABLE_ERR_BAD_PARAM
    )
    assert cursor == [4]
    assert out == [5]

    cursor = [I64_MAX + 1]
    out = [6]
    assert (
        gguf_metadata_read_u8_checked([1] * 8, 8, cursor, I64_MAX, out)
        == GGUF_META_TABLE_ERR_OVERFLOW
    )
    assert cursor == [I64_MAX + 1]
    assert out == [6]

    cursor = [2]
    out = [7]
    assert (
        gguf_metadata_read_u8_checked([1, 2, 3], 2, cursor, 3, out)
        == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    )
    assert cursor == [2]
    assert out == [7]


def test_success_reads_and_advances() -> None:
    buf = [10, 20, 30, 40]
    cursor = [2]
    out = [0]

    err = gguf_metadata_read_u8_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == 30
    assert cursor[0] == 3


def test_randomized_linear_scan() -> None:
    rng = random.Random(20260417_193)

    for _ in range(2000):
        n = rng.randint(1, 128)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]

        cursor = [rng.randint(0, table_end)]
        cursor0 = cursor[0]
        out = [999]

        err = gguf_metadata_read_u8_checked(buf, n, cursor, table_end, out)

        if cursor0 > table_end:  # unreachable by construction; keep explicit
            assert err == GGUF_META_TABLE_ERR_BAD_PARAM
            continue

        if cursor0 == table_end:
            assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            assert out[0] == 999
            continue

        if cursor0 + 1 > n:
            assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            assert out[0] == 999
            continue

        # Successful path: replay from original cursor.
        assert err == GGUF_META_TABLE_OK
        assert out[0] == buf[cursor0]
        assert cursor[0] == cursor0 + 1


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_overflow_and_bounds_contracts()
    test_success_reads_and_advances()
    test_randomized_linear_scan()
    print("gguf_metadata_read_u8_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
