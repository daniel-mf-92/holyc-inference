#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI32PairCheckedNoPartialDefault."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1


def _to_i32(value: int) -> int:
    value &= 0xFFFFFFFF
    if value & 0x80000000:
        return value - 0x100000000
    return value


def _le_i32(value: int) -> list[int]:
    raw = value & 0xFFFFFFFF
    return [
        raw & 0xFF,
        (raw >> 8) & 0xFF,
        (raw >> 16) & 0xFF,
        (raw >> 24) & 0xFF,
    ]


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


def gguf_metadata_read_i32le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw_u32 = [0]

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, raw_u32)
    if err != GGUF_META_TABLE_OK:
        return err

    out_value_ref[0] = _to_i32(raw_u32[0])
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_pair_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_first_ref is None
        or out_second_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    first = [0]
    second = [0]

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i32_pair_checked_no_partial_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i32_pair_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [13]
    out_first = [111]
    out_second = [222]

    err = gguf_metadata_read_i32_pair_checked_no_partial_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [13]
    assert out_first == [111]
    assert out_second == [222]


def test_success_matches_explicit_checked_core() -> None:
    first = -2147483648
    second = 2147483647
    prefix = [0x8A, 0x8B, 0x8C]
    buf = prefix + _le_i32(first) + _le_i32(second) + [0x9A]

    cur_default = [3]
    cur_checked = [3]
    out_default = [[0], [0]]
    out_checked = [[0], [0]]

    err_default = gguf_metadata_read_i32_pair_checked_no_partial_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_i32_pair_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
    )

    assert err_default == err_checked == GGUF_META_TABLE_OK
    assert cur_default == cur_checked == [11]
    assert out_default == out_checked == [[first], [second]]


def test_truncation_has_no_cursor_or_output_commit() -> None:
    buf = _le_i32(-42) + [0xEF, 0xBE, 0xAD]

    cur_default = [0]
    cur_checked = [0]
    out_default = [[0x1111], [0x2222]]
    out_checked = [[0x1111], [0x2222]]

    err_default = gguf_metadata_read_i32_pair_checked_no_partial_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_i32_pair_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cur_default == cur_checked == [0]
    assert out_default == out_checked == [[0x1111], [0x2222]]


def test_bad_param_and_overflow_parity() -> None:
    buf = [0xAA] * 32

    cur_default = [9]
    cur_checked = [9]
    out_default = [[10], [20]]
    out_checked = [[10], [20]]

    err_default = gguf_metadata_read_i32_pair_checked_no_partial_default(
        buf,
        8,
        cur_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_i32_pair_checked(
        buf,
        8,
        cur_checked,
        8,
        out_checked[0],
        out_checked[1],
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cur_default == cur_checked == [9]
    assert out_default == out_checked == [[10], [20]]

    cur_default = [0]
    cur_checked = [0]
    out_default = [[30], [40]]
    out_checked = [[30], [40]]

    err_default = gguf_metadata_read_i32_pair_checked_no_partial_default(
        buf,
        I64_MAX + 1,
        cur_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_i32_pair_checked(
        buf,
        I64_MAX + 1,
        cur_checked,
        I64_MAX + 1,
        out_checked[0],
        out_checked[1],
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OVERFLOW
    assert cur_default == cur_checked == [0]
    assert out_default == out_checked == [[30], [40]]


def test_randomized_parity_vs_explicit_checked_core() -> None:
    random.seed(0x49333234)

    for _ in range(400):
        buf_len = random.randrange(0, 48)
        buf = [random.randrange(0, 256) for _ in range(buf_len)]
        cursor_seed = random.randrange(0, 52)

        out_default = [random.randrange(-(1 << 31), 1 << 31)]
        out_default_2 = [random.randrange(-(1 << 31), 1 << 31)]
        out_checked = [out_default[0]]
        out_checked_2 = [out_default_2[0]]

        cur_default = [cursor_seed]
        cur_checked = [cursor_seed]

        err_default = gguf_metadata_read_i32_pair_checked_no_partial_default(
            buf,
            buf_len,
            cur_default,
            out_default,
            out_default_2,
        )
        err_checked = gguf_metadata_read_i32_pair_checked(
            buf,
            buf_len,
            cur_checked,
            buf_len,
            out_checked,
            out_checked_2,
        )

        assert err_default == err_checked
        assert cur_default == cur_checked
        assert out_default == out_checked
        assert out_default_2 == out_checked_2


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_matches_explicit_checked_core()
    test_truncation_has_no_cursor_or_output_commit()
    test_bad_param_and_overflow_parity()
    test_randomized_parity_vs_explicit_checked_core()
    print("ok")
