#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadI64QuadCheckedNoPartial semantics."""

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
    byte_lane = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, byte_lane)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= byte_lane[0] << (8 * i)

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


def gguf_metadata_read_i64_quad_checked_no_partial(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
        or out_third_ref is None
        or out_fourth_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]
    third = [0]
    fourth = [0]

    err = gguf_metadata_read_i64le_checked_no_partial(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i64le_checked_no_partial(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i64le_checked_no_partial(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i64le_checked_no_partial(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def _le_u64(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(8)]


def _i64_to_u64_bits(value: int) -> int:
    return value & U64_MASK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out_first = [111]
    out_second = [222]
    out_third = [333]
    out_fourth = [444]

    assert (
        gguf_metadata_read_i64_quad_checked_no_partial(
            None,
            64,
            cursor,
            64,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [4]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]
    assert out_fourth == [444]


def test_fourth_scalar_fail_does_not_commit_anything() -> None:
    first = -1
    second = 123456789
    third = -(1 << 62)
    buf = (
        _le_u64(_i64_to_u64_bits(first))
        + _le_u64(_i64_to_u64_bits(second))
        + _le_u64(_i64_to_u64_bits(third))
        + [0xAB] * 7
    )

    cursor = [0]
    out_first = [0x111]
    out_second = [0x222]
    out_third = [0x333]
    out_fourth = [0x444]

    err = gguf_metadata_read_i64_quad_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x111]
    assert out_second == [0x222]
    assert out_third == [0x333]
    assert out_fourth == [0x444]


def test_success_reads_four_i64_and_advances() -> None:
    first = -(1 << 63)
    second = -1
    third = (1 << 63) - 1
    fourth = -1234567890123456789

    buf = (
        [0xEE]
        + _le_u64(_i64_to_u64_bits(first))
        + _le_u64(_i64_to_u64_bits(second))
        + _le_u64(_i64_to_u64_bits(third))
        + _le_u64(_i64_to_u64_bits(fourth))
        + [0x77]
    )

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_i64_quad_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first[0] == first
    assert out_second[0] == second
    assert out_third[0] == third
    assert out_fourth[0] == fourth
    assert cursor[0] == 33


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0x44] * 96
    out_first_seed = 0x0BADF00D
    out_second_seed = 0xDEADC0DE
    out_third_seed = 0xCAFEBABE
    out_fourth_seed = 0x0D15EA5E

    cursor = [0]
    out_first = [out_first_seed]
    out_second = [out_second_seed]
    out_third = [out_third_seed]
    out_fourth = [out_fourth_seed]

    err = gguf_metadata_read_i64_quad_checked_no_partial(
        buf,
        I64_MAX + 1,
        cursor,
        len(buf),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_first[0] == out_first_seed
    assert out_second[0] == out_second_seed
    assert out_third[0] == out_third_seed
    assert out_fourth[0] == out_fourth_seed


def test_randomized_roundtrip_and_cursor_progression() -> None:
    random.seed(0x1A64_4DAD)

    for _ in range(200):
        values = [random.randrange(-(1 << 63), 1 << 63) for _ in range(4)]
        prefix = [random.randrange(0, 256) for _ in range(random.randrange(0, 7))]
        suffix = [random.randrange(0, 256) for _ in range(random.randrange(0, 7))]

        payload: list[int] = []
        for value in values:
            payload.extend(_le_u64(_i64_to_u64_bits(value)))

        buf = prefix + payload + suffix
        cursor = [len(prefix)]
        out_first = [0]
        out_second = [0]
        out_third = [0]
        out_fourth = [0]

        err = gguf_metadata_read_i64_quad_checked_no_partial(
            buf,
            len(buf),
            cursor,
            len(buf),
            out_first,
            out_second,
            out_third,
            out_fourth,
        )

        assert err == GGUF_META_TABLE_OK
        assert [out_first[0], out_second[0], out_third[0], out_fourth[0]] == values
        assert cursor[0] == len(prefix) + 32


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_scalar_fail_does_not_commit_anything()
    test_success_reads_four_i64_and_advances()
    test_overflow_passthrough_and_no_commit()
    test_randomized_roundtrip_and_cursor_progression()
    print("ok")
