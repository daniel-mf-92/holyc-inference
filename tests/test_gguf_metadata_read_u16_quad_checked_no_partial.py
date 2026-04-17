#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU16QuadCheckedNoPartial semantics."""

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
    if cursor > ((1 << 64) - 1) - need:
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


def _le_u16(value: int) -> list[int]:
    return [value & 0xFF, (value >> 8) & 0xFF]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out_first = [0x1111]
    out_second = [0x2222]
    out_third = [0x3333]
    out_fourth = [0x4444]

    assert (
        gguf_metadata_read_u16_quad_checked_no_partial(
            None,
            16,
            cursor,
            16,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [4]
    assert out_first == [0x1111]
    assert out_second == [0x2222]
    assert out_third == [0x3333]
    assert out_fourth == [0x4444]


def test_fourth_scalar_fail_does_not_commit_prior_lanes_or_cursor() -> None:
    first = 0x1111
    second = 0x2222
    third = 0x3333
    buf = _le_u16(first) + _le_u16(second) + _le_u16(third) + [0xAB]

    cursor = [0]
    out_first = [0xAAAA]
    out_second = [0xBBBB]
    out_third = [0xCCCC]
    out_fourth = [0xDDDD]

    err = gguf_metadata_read_u16_quad_checked_no_partial(
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
    assert out_first == [0xAAAA]
    assert out_second == [0xBBBB]
    assert out_third == [0xCCCC]
    assert out_fourth == [0xDDDD]


def test_success_reads_four_u16_and_advances() -> None:
    first = 0xBEEF
    second = 0x1234
    third = 0xCAFE
    fourth = 0x0F0F
    buf = (
        [0x77]
        + _le_u16(first)
        + _le_u16(second)
        + _le_u16(third)
        + _le_u16(fourth)
        + [0x88]
    )

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u16_quad_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(buf) - 1,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor == [9]
    assert out_first == [first]
    assert out_second == [second]
    assert out_third == [third]
    assert out_fourth == [fourth]


def test_buffer_bound_precedes_table_bound() -> None:
    first = 0xAAAA
    second = 0xBBBB
    third = 0xCCCC
    fourth = 0xDDDD
    full = _le_u16(first) + _le_u16(second) + _le_u16(third) + _le_u16(fourth)
    buf = full[:-1]

    cursor = [0]
    out_first = [0x1]
    out_second = [0x2]
    out_third = [0x3]
    out_fourth = [0x4]

    err = gguf_metadata_read_u16_quad_checked_no_partial(
        buf,
        len(buf),
        cursor,
        len(full),
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0x1]
    assert out_second == [0x2]
    assert out_third == [0x3]
    assert out_fourth == [0x4]


def test_randomized_parity_vs_staged_reference() -> None:
    rng = random.Random(261)

    for _ in range(300):
        prefix_len = rng.randint(0, 7)
        suffix_len = rng.randint(0, 7)
        start = prefix_len

        lanes = [rng.randint(0, 0xFFFF) for _ in range(4)]
        payload = []
        for lane in lanes:
            payload.extend(_le_u16(lane))

        buf = [rng.randint(0, 255) for _ in range(prefix_len)]
        buf.extend(payload)
        buf.extend(rng.randint(0, 255) for _ in range(suffix_len))

        cursor_a = [start]
        cursor_b = [start]
        a1 = [0xA001]
        a2 = [0xA002]
        a3 = [0xA003]
        a4 = [0xA004]

        err_a = gguf_metadata_read_u16_quad_checked_no_partial(
            buf,
            len(buf),
            cursor_a,
            len(buf),
            a1,
            a2,
            a3,
            a4,
        )

        o1 = [0xB001]
        o2 = [0xB002]
        o3 = [0xB003]
        o4 = [0xB004]
        err_b = gguf_metadata_read_u16_quad_checked(
            buf,
            len(buf),
            cursor_b,
            len(buf),
            o1,
            o2,
            o3,
            o4,
        )

        assert err_a == err_b == GGUF_META_TABLE_OK
        assert cursor_a == cursor_b
        assert [a1[0], a2[0], a3[0], a4[0]] == lanes
        assert [o1[0], o2[0], o3[0], o4[0]] == lanes


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_scalar_fail_does_not_commit_prior_lanes_or_cursor()
    test_success_reads_four_u16_and_advances()
    test_buffer_bound_precedes_table_bound()
    test_randomized_parity_vs_staged_reference()
    print("ok")
