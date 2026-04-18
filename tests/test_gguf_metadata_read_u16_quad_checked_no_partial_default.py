#!/usr/bin/env python3
"""Parity harness for GGUFMetadataReadU16QuadCheckedNoPartialDefault contract."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


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
    if cursor > U64_MAX - need:
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


def gguf_metadata_read_u16le_checked(
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

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b0)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b1)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = b0[0] | (b1[0] << 8)
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u16_quad_checked(
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

    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u16le_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u16_quad_checked_no_partial(
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

    err = gguf_metadata_read_u16_quad_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
        third,
        fourth,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u16_quad_checked_no_partial_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u16_quad_checked_no_partial(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [0x1111]
    out_second = [0x2222]
    out_third = [0x3333]
    out_fourth = [0x4444]

    err = gguf_metadata_read_u16_quad_checked_no_partial_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [5]
    assert out_first == [0x1111]
    assert out_second == [0x2222]
    assert out_third == [0x3333]
    assert out_fourth == [0x4444]


def test_success_matches_no_partial_checked_with_default_end() -> None:
    values = [0xA1B2, 0xC3D4, 0xE5F6, 0x0A0B]
    buf = [0x77]
    for value in values:
        buf.extend(_le_u16(value))
    buf.append(0x88)

    cur_default = [1]
    cur_checked = [1]
    out_default = [[0], [0], [0], [0]]
    out_checked = [[0], [0], [0], [0]]

    err_default = gguf_metadata_read_u16_quad_checked_no_partial_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_checked = gguf_metadata_read_u16_quad_checked_no_partial(
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
    assert cur_default == cur_checked == [9]
    assert out_default == out_checked
    assert out_default == [[values[0]], [values[1]], [values[2]], [values[3]]]


def test_short_payload_no_partial_commit_and_parity() -> None:
    buf = _le_u16(0x1111) + _le_u16(0x2222) + _le_u16(0x3333) + [0x7F]

    cur_default = [0]
    cur_checked = [0]
    out_default = [[0xAAAA], [0xBBBB], [0xCCCC], [0xDDDD]]
    out_checked = [[0xAAAA], [0xBBBB], [0xCCCC], [0xDDDD]]

    err_default = gguf_metadata_read_u16_quad_checked_no_partial_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
        out_default[3],
    )
    err_checked = gguf_metadata_read_u16_quad_checked_no_partial(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
        out_checked[2],
        out_checked[3],
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cur_default == cur_checked == [0]
    assert out_default == out_checked == [[0xAAAA], [0xBBBB], [0xCCCC], [0xDDDD]]


def test_randomized_parity_against_no_partial_checked() -> None:
    rng = random.Random(20260418_320)

    for _ in range(20000):
        n = rng.randint(1, 320)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        cur_default = [cursor0]
        cur_checked = [cursor0]
        out_default = [[0x1001], [0x2002], [0x3003], [0x4004]]
        out_checked = [[0x1001], [0x2002], [0x3003], [0x4004]]

        err_default = gguf_metadata_read_u16_quad_checked_no_partial_default(
            buf,
            n,
            cur_default,
            out_default[0],
            out_default[1],
            out_default[2],
            out_default[3],
        )
        err_checked = gguf_metadata_read_u16_quad_checked_no_partial(
            buf,
            n,
            cur_checked,
            n,
            out_checked[0],
            out_checked[1],
            out_checked[2],
            out_checked[3],
        )

        assert err_default == err_checked
        assert cur_default == cur_checked
        assert out_default == out_checked


def main() -> None:
    test_null_ptr_and_no_partial_write()
    test_success_matches_no_partial_checked_with_default_end()
    test_short_payload_no_partial_commit_and_parity()
    test_randomized_parity_against_no_partial_checked()
    print("ok")


if __name__ == "__main__":
    main()
