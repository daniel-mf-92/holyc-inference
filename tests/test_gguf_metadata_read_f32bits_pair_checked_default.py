#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadF32BitsPairCheckedDefault."""

from __future__ import annotations

import random
import struct

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


def gguf_metadata_read_f32bitsle_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_bits_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_bits_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    bits = [0]
    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, bits)
    if err != GGUF_META_TABLE_OK:
        return err

    out_bits_ref[0] = bits[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f32bits_pair_checked(
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

    err = gguf_metadata_read_f32bitsle_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f32bitsle_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f32bits_pair_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_f32bits_pair_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
    )


def _le_u32(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(4)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [9]
    out_first = [0x11111111]
    out_second = [0x22222222]

    err = gguf_metadata_read_f32bits_pair_checked_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [9]
    assert out_first == [0x11111111]
    assert out_second == [0x22222222]


def test_success_matches_checked_with_default_end() -> None:
    first = struct.unpack("<I", struct.pack("<f", 1.0))[0]
    second = struct.unpack("<I", struct.pack("<f", -2.5))[0]
    buf = [0xAA] + _le_u32(first) + _le_u32(second) + [0x55]

    cur_default = [1]
    cur_checked = [1]
    out_default = [[0], [0]]
    out_checked = [[0], [0]]

    err_default = gguf_metadata_read_f32bits_pair_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_f32bits_pair_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
    )

    assert err_default == GGUF_META_TABLE_OK
    assert err_checked == GGUF_META_TABLE_OK
    assert cur_default == cur_checked == [9]
    assert out_default == out_checked


def test_truncation_matches_checked_and_no_commit() -> None:
    first = struct.unpack("<I", struct.pack("<f", 0.5))[0]
    buf = _le_u32(first) + [0x10, 0x20, 0x30]

    cur_default = [0]
    cur_checked = [0]
    out_default = [[0xAAAA0001], [0xBBBB0002]]
    out_checked = [[0xAAAA0001], [0xBBBB0002]]

    err_default = gguf_metadata_read_f32bits_pair_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_f32bits_pair_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
    )

    assert err_default == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert err_checked == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cur_default == cur_checked == [0]
    assert out_default == out_checked == [[0xAAAA0001], [0xBBBB0002]]


def test_cursor_past_end_bad_param_propagates() -> None:
    buf = [0x42] * 20
    cursor_default = [21]
    cursor_checked = [21]
    out_default = [[0], [0]]
    out_checked = [[0], [0]]

    err_default = gguf_metadata_read_f32bits_pair_checked_default(
        buf,
        len(buf),
        cursor_default,
        out_default[0],
        out_default[1],
    )
    err_checked = gguf_metadata_read_f32bits_pair_checked(
        buf,
        len(buf),
        cursor_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
    )

    assert err_default == GGUF_META_TABLE_ERR_BAD_PARAM
    assert err_checked == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor_default == cursor_checked == [21]
    assert out_default == out_checked == [[0], [0]]


def test_randomized_default_wrapper_parity() -> None:
    rng = random.Random(20260418_312)

    for _ in range(6000):
        n = rng.randint(1, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor_start = rng.randint(0, n + 2)

        cur_default = [cursor_start]
        cur_checked = [cursor_start]
        out_default = [[0x10000000], [0x20000000]]
        out_checked = [[0x10000000], [0x20000000]]

        err_default = gguf_metadata_read_f32bits_pair_checked_default(
            buf,
            len(buf),
            cur_default,
            out_default[0],
            out_default[1],
        )
        err_checked = gguf_metadata_read_f32bits_pair_checked(
            buf,
            len(buf),
            cur_checked,
            len(buf),
            out_checked[0],
            out_checked[1],
        )

        assert err_default == err_checked
        assert cur_default == cur_checked
        assert out_default == out_checked


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_matches_checked_with_default_end()
    test_truncation_matches_checked_and_no_commit()
    test_cursor_past_end_bad_param_propagates()
    test_randomized_default_wrapper_parity()
    print("ok")
