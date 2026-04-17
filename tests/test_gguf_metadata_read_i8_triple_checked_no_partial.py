#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI8TripleCheckedNoPartial semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U8_MASK = (1 << 8) - 1


def reinterpret_u8_as_i8(value: int) -> int:
    value &= U8_MASK
    if value >= (1 << 7):
        return value - (1 << 8)
    return value


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


def gguf_metadata_read_i8_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw_u8 = [0]

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, raw_u8)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = reinterpret_u8_as_i8(raw_u8[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i8_triple_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]

    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i8_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i8_triple_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]

    err = gguf_metadata_read_i8_triple_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
        third,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [11]
    out_first = [111]
    out_second = [222]
    out_third = [333]

    assert (
        gguf_metadata_read_i8_triple_checked_no_partial(
            None,
            32,
            cursor,
            32,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [11]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]

    assert (
        gguf_metadata_read_i8_triple_checked_no_partial(
            [0] * 32,
            32,
            None,
            32,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]


def test_third_scalar_fail_does_not_commit_anything() -> None:
    buf = [0x7F, 0x80]
    cursor = [0]
    out_first = [17]
    out_second = [23]
    out_third = [99]

    err = gguf_metadata_read_i8_triple_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [17]
    assert out_second == [23]
    assert out_third == [99]


def test_success_reads_three_i8_and_advances() -> None:
    buf = [0x10, 0xFF, 0x80, 0x7F, 0x22]
    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_i8_triple_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
    )

    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == -1
    assert out_second[0] == -128
    assert out_third[0] == 127
    assert cursor[0] == 4


def test_wrapper_matches_core_decoder_randomized() -> None:
    rng = random.Random(20260417_275)

    for _ in range(7000):
        n = rng.randint(1, 320)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor_core = [cursor0]
        core_first = [101]
        core_second = [202]
        core_third = [77]
        err_core = gguf_metadata_read_i8_triple_checked(
            buf,
            n,
            cursor_core,
            table_end,
            core_first,
            core_second,
            core_third,
        )

        cursor_wrap = [cursor0]
        wrap_first = [101]
        wrap_second = [202]
        wrap_third = [77]
        err_wrap = gguf_metadata_read_i8_triple_checked_no_partial(
            buf,
            n,
            cursor_wrap,
            table_end,
            wrap_first,
            wrap_second,
            wrap_third,
        )

        assert err_wrap == err_core
        assert cursor_wrap[0] == cursor_core[0]
        assert wrap_first[0] == core_first[0]
        assert wrap_second[0] == core_second[0]
        assert wrap_third[0] == core_third[0]


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_third_scalar_fail_does_not_commit_anything()
    test_success_reads_three_i8_and_advances()
    test_wrapper_matches_core_decoder_randomized()
    print("gguf_metadata_read_i8_triple_checked_no_partial_reference_checks=ok")
