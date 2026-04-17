#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadBoolPairChecked semantics."""

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


def gguf_metadata_read_bool_pair_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[bool] | None,
    out_second_ref: list[bool] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [False]
    second = [False]

    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_bool_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [1]
    out_first = [True]
    out_second = [False]

    assert (
        gguf_metadata_read_bool_pair_checked(None, 3, cursor, 3, out_first, out_second)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [1]
    assert out_first == [True]
    assert out_second == [False]

    assert (
        gguf_metadata_read_bool_pair_checked([0, 1, 0], 3, None, 3, out_first, out_second)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_first == [True]
    assert out_second == [False]

    assert (
        gguf_metadata_read_bool_pair_checked([0, 1, 0], 3, cursor, 3, None, out_second)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [1]
    assert out_second == [False]

    assert (
        gguf_metadata_read_bool_pair_checked([0, 1, 0], 3, cursor, 3, out_first, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [1]
    assert out_first == [True]


def test_short_read_and_noncanonical_second_do_not_commit() -> None:
    cursor = [1]
    out_first = [False]
    out_second = [True]

    err = gguf_metadata_read_bool_pair_checked(
        [0, 1],
        2,
        cursor,
        2,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [1]
    assert out_first == [False]
    assert out_second == [True]

    cursor = [0]
    out_first = [False]
    out_second = [False]
    err = gguf_metadata_read_bool_pair_checked(
        [1, 2, 0],
        3,
        cursor,
        3,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [False]


def test_success_reads_two_bools_and_advances() -> None:
    buf = [9, 0, 1, 0]
    cursor = [1]
    out_first = [True]
    out_second = [True]

    err = gguf_metadata_read_bool_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] is False
    assert out_second[0] is True
    assert cursor[0] == 3


def test_overflow_passthrough() -> None:
    buf = [0, 1, 0, 1]

    cursor = [0]
    out_first = [False]
    out_second = [False]
    err = gguf_metadata_read_bool_pair_checked(
        buf,
        I64_MAX + 1,
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_first == [False]
    assert out_second == [False]

    cursor = [I64_MAX + 1]
    out_first = [True]
    out_second = [True]
    err = gguf_metadata_read_bool_pair_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [I64_MAX + 1]
    assert out_first == [True]
    assert out_second == [True]


def test_randomized_parity() -> None:
    rng = random.Random(20260417_216)

    for _ in range(5000):
        n = rng.randint(1, 256)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [True]
        out_second = [False]

        err = gguf_metadata_read_bool_pair_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
        )

        if cursor0 + 2 <= table_end and cursor0 + 2 <= n:
            raw_first = buf[cursor0]
            raw_second = buf[cursor0 + 1]
            if raw_first <= 1 and raw_second <= 1:
                assert err == GGUF_META_TABLE_OK
                assert out_first[0] == (raw_first == 1)
                assert out_second[0] == (raw_second == 1)
                assert cursor[0] == cursor0 + 2
            else:
                assert err == GGUF_META_TABLE_ERR_BAD_PARAM
                assert out_first == [True]
                assert out_second == [False]
                assert cursor[0] == cursor0
        else:
            assert err in (
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
                GGUF_META_TABLE_ERR_BAD_PARAM,
                GGUF_META_TABLE_ERR_OVERFLOW,
            )
            assert out_first == [True]
            assert out_second == [False]
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_short_read_and_noncanonical_second_do_not_commit()
    test_success_reads_two_bools_and_advances()
    test_overflow_passthrough()
    test_randomized_parity()
    print("gguf_metadata_read_bool_pair_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
