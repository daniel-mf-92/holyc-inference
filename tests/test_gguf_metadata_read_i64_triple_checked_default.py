#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadI64TripleCheckedDefault."""

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


def gguf_metadata_read_u64le_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    lane = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, lane)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= lane[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def reinterpret_u64_as_i64(value: int) -> int:
    value &= U64_MASK
    if value >= (1 << 63):
        return value - (1 << 64)
    return value


def gguf_metadata_read_i64le_checked(
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

    err = gguf_metadata_read_u64le_checked(
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


def gguf_metadata_read_i64_triple_checked(
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

    err = gguf_metadata_read_i64le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i64le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_i64le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_i64_triple_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_i64_triple_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
    )


def _le_u64(value: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(8)]


def _i64_to_u64_bits(value: int) -> int:
    return value & U64_MASK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [12]
    out_first = [0x101]
    out_second = [0x202]
    out_third = [0x303]

    err = gguf_metadata_read_i64_triple_checked_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [12]
    assert out_first == [0x101]
    assert out_second == [0x202]
    assert out_third == [0x303]


def test_success_matches_checked_core_exactly() -> None:
    first = -(1 << 63)
    second = -1234567890
    third = (1 << 63) - 1
    buf = [0xAA] + _le_u64(_i64_to_u64_bits(first)) + _le_u64(_i64_to_u64_bits(second)) + _le_u64(_i64_to_u64_bits(third)) + [0xBB]

    cur_default = [1]
    cur_checked = [1]
    out_default = [[0], [0], [0]]
    out_checked = [[0], [0], [0]]

    err_default = gguf_metadata_read_i64_triple_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
    )
    err_checked = gguf_metadata_read_i64_triple_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
        out_checked[2],
    )

    assert err_default == GGUF_META_TABLE_OK
    assert err_checked == GGUF_META_TABLE_OK
    assert cur_default == cur_checked == [25]
    assert out_default == out_checked
    assert out_default[0] == [first]
    assert out_default[1] == [second]
    assert out_default[2] == [third]


def test_default_end_truncation_matches_checked_no_commit() -> None:
    first = -1
    second = 42
    buf = _le_u64(_i64_to_u64_bits(first)) + _le_u64(_i64_to_u64_bits(second)) + [0xCC] * 7

    cur_default = [0]
    cur_checked = [0]
    out_default = [[11], [22], [33]]
    out_checked = [[11], [22], [33]]

    err_default = gguf_metadata_read_i64_triple_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
    )
    err_checked = gguf_metadata_read_i64_triple_checked(
        buf,
        len(buf),
        cur_checked,
        len(buf),
        out_checked[0],
        out_checked[1],
        out_checked[2],
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cur_default == cur_checked == [0]
    assert out_default == out_checked == [[11], [22], [33]]


def test_randomized_parity_with_checked_core() -> None:
    rng = random.Random(0x16400306)

    for _ in range(300):
        values = [
            rng.randint(-(1 << 63), (1 << 63) - 1),
            rng.randint(-(1 << 63), (1 << 63) - 1),
            rng.randint(-(1 << 63), (1 << 63) - 1),
        ]

        prefix_len = rng.randint(0, 4)
        suffix_len = rng.randint(0, 4)
        prefix = [rng.randint(0, 255) for _ in range(prefix_len)]
        suffix = [rng.randint(0, 255) for _ in range(suffix_len)]

        payload = []
        for value in values:
            payload.extend(_le_u64(_i64_to_u64_bits(value)))

        trim = rng.randint(0, 24)
        if trim > 0:
            payload = payload[: len(payload) - trim]

        buf = prefix + payload + suffix
        buf_nbytes = len(buf)

        start = prefix_len if rng.random() < 0.7 else rng.randint(0, buf_nbytes)

        cur_default = [start]
        cur_checked = [start]
        out_default = [[-11], [-22], [-33]]
        out_checked = [[-11], [-22], [-33]]

        err_default = gguf_metadata_read_i64_triple_checked_default(
            buf,
            buf_nbytes,
            cur_default,
            out_default[0],
            out_default[1],
            out_default[2],
        )
        err_checked = gguf_metadata_read_i64_triple_checked(
            buf,
            buf_nbytes,
            cur_checked,
            buf_nbytes,
            out_checked[0],
            out_checked[1],
            out_checked[2],
        )

        assert err_default == err_checked
        assert cur_default == cur_checked
        assert out_default == out_checked


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_matches_checked_core_exactly()
    test_default_end_truncation_matches_checked_no_commit()
    test_randomized_parity_with_checked_core()
    print("ok")
