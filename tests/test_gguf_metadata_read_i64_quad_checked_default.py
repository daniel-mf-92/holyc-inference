#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI64QuadCheckedDefault."""

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
    if cursor > U64_MASK - need:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

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


def gguf_metadata_read_i64_quad_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i64_quad_checked_no_partial(
        buf,
        buf_nbytes,
        cursor_ref,
        table_end,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def gguf_metadata_read_i64_quad_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i64_quad_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def _le_u64(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(8)]


def _i64_to_u64_bits(value: int) -> int:
    return value & U64_MASK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [8]
    out_first = [111]
    out_second = [222]
    out_third = [333]
    out_fourth = [444]

    err = gguf_metadata_read_i64_quad_checked_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [8]
    assert out_first == [111]
    assert out_second == [222]
    assert out_third == [333]
    assert out_fourth == [444]


def test_default_table_end_matches_checked_core() -> None:
    first = -(1 << 63)
    second = -1
    third = 42
    fourth = (1 << 63) - 1

    buf = [0xAB] + _le_u64(_i64_to_u64_bits(first)) + _le_u64(_i64_to_u64_bits(second)) + _le_u64(_i64_to_u64_bits(third)) + _le_u64(_i64_to_u64_bits(fourth)) + [0xCD]

    cur_default = [1]
    cur_checked = [1]
    out_default = [[0], [0], [0], [0]]
    out_checked = [[0], [0], [0], [0]]

    err_default = gguf_metadata_read_i64_quad_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_checked = gguf_metadata_read_i64_quad_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
        out_checked[2],
        out_checked[3],
    )

    assert err_default == GGUF_META_TABLE_OK
    assert err_checked == GGUF_META_TABLE_OK
    assert cur_default == cur_checked == [33]
    assert out_default == out_checked == [[first], [second], [third], [fourth]]


def test_default_end_truncation_fails_without_partial_commit() -> None:
    first = -5
    second = 11
    third = -23

    buf = [0xFA] + _le_u64(_i64_to_u64_bits(first)) + _le_u64(_i64_to_u64_bits(second)) + _le_u64(_i64_to_u64_bits(third)) + [0x99] * 7

    cursor = [1]
    out_first = [0x111]
    out_second = [0x222]
    out_third = [0x333]
    out_fourth = [0x444]

    err = gguf_metadata_read_i64_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )

    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [1]
    assert out_first == [0x111]
    assert out_second == [0x222]
    assert out_third == [0x333]
    assert out_fourth == [0x444]


def test_randomized_default_vs_checked_parity() -> None:
    random.seed(0x1BADB002)

    for _ in range(200):
        prefix = random.randint(0, 4)
        suffix = random.randint(0, 4)
        cursor0 = prefix

        values = [random.randint(-(1 << 63), (1 << 63) - 1) for _ in range(4)]
        payload = []
        for value in values:
            payload.extend(_le_u64(_i64_to_u64_bits(value)))

        buf = [random.randint(0, 255) for _ in range(prefix)] + payload + [random.randint(0, 255) for _ in range(suffix)]

        cur_default = [cursor0]
        cur_checked = [cursor0]
        out_default = [[-1], [-2], [-3], [-4]]
        out_checked = [[-1], [-2], [-3], [-4]]

        err_default = gguf_metadata_read_i64_quad_checked_default(
            buf,
            len(buf),
            cur_default,
            out_default[0],
            out_default[1],
            out_default[2],
            out_default[3],
        )
        err_checked = gguf_metadata_read_i64_quad_checked(
            buf,
            len(buf),
            cur_checked,
            len(buf),
            out_checked[0],
            out_checked[1],
            out_checked[2],
            out_checked[3],
        )

        assert err_default == err_checked == GGUF_META_TABLE_OK
        assert cur_default == cur_checked
        assert out_default == out_checked


def main() -> None:
    test_null_ptr_and_no_partial_write()
    test_default_table_end_matches_checked_core()
    test_default_end_truncation_fails_without_partial_commit()
    test_randomized_default_vs_checked_parity()
    print("ok")


if __name__ == "__main__":
    main()
