#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadBoolChecked semantics."""

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


def gguf_metadata_read_bool_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[bool] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    if raw[0] > 1:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_value_ref[0] = raw[0] != 0
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out = [True]

    assert (
        gguf_metadata_read_bool_checked(None, 8, cursor, 8, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 2
    assert out[0] is True

    assert (
        gguf_metadata_read_bool_checked([0, 1], 2, None, 2, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out[0] is True

    assert (
        gguf_metadata_read_bool_checked([0, 1], 2, cursor, 2, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 2


def test_bounds_and_short_read_do_not_mutate() -> None:
    out = [False]

    for table_end in range(0, 1):
        cursor = [0]
        err = gguf_metadata_read_bool_checked([1], 1, cursor, table_end, out)
        assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
        assert cursor[0] == 0
        assert out[0] is False

    cursor = [1]
    out = [True]
    err = gguf_metadata_read_bool_checked([0], 1, cursor, 1, out)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 1
    assert out[0] is True


def test_rejects_non_canonical_bool_values() -> None:
    for raw in (2, 3, 127, 255):
        cursor = [0]
        out = [False]
        err = gguf_metadata_read_bool_checked([raw], 1, cursor, 1, out)
        assert err == GGUF_META_TABLE_ERR_BAD_PARAM
        assert cursor[0] == 0
        assert out[0] is False


def test_reads_canonical_bool_values() -> None:
    buf = [9, 0, 1, 4]
    cursor = [1]
    out = [False]

    err = gguf_metadata_read_bool_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] is False
    assert cursor[0] == 2

    err = gguf_metadata_read_bool_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] is True
    assert cursor[0] == 3


def test_randomized_parity() -> None:
    rng = random.Random(20260417_199)

    for _ in range(5000):
        n = rng.randint(1, 128)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out = [True]
        err = gguf_metadata_read_bool_checked(buf, n, cursor, table_end, out)

        if cursor0 + 1 <= table_end and cursor0 + 1 <= n:
            raw = buf[cursor0]
            if raw <= 1:
                assert err == GGUF_META_TABLE_OK
                assert out[0] == (raw == 1)
                assert cursor[0] == cursor0 + 1
            else:
                assert err == GGUF_META_TABLE_ERR_BAD_PARAM
                assert out[0] is True
                assert cursor[0] == cursor0
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out[0] is True
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_bounds_and_short_read_do_not_mutate()
    test_rejects_non_canonical_bool_values()
    test_reads_canonical_bool_values()
    test_randomized_parity()
    print("gguf_metadata_read_bool_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
