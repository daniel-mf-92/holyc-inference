#!/usr/bin/env python3
"""Parity harness for GGUFMetadataReadU64TripleCheckedDefault contract."""

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
    out = 0
    lane = [0]

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, lane)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= lane[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64_triple_checked(
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

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64_triple_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u64_triple_checked(
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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [7]
    out_first = [0x1111111111111111]
    out_second = [0x2222222222222222]
    out_third = [0x3333333333333333]

    err = gguf_metadata_read_u64_triple_checked_default(
        None,
        64,
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert cursor == [7]
    assert out_first == [0x1111111111111111]
    assert out_second == [0x2222222222222222]
    assert out_third == [0x3333333333333333]


def test_success_uses_default_end_and_matches_checked() -> None:
    first = 0x0123456789ABCDEF
    second = 0x0F1E2D3C4B5A6978
    third = 0x8899AABBCCDDEEFF
    buf = [0xCC] + _le_u64(first) + _le_u64(second) + _le_u64(third) + [0x11]

    cur_default = [1]
    cur_checked = [1]
    out_default = [[0], [0], [0]]
    out_checked = [[0], [0], [0]]

    err_default = gguf_metadata_read_u64_triple_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
    )
    err_checked = gguf_metadata_read_u64_triple_checked(
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


def test_truncation_matches_checked_and_no_commit() -> None:
    first = 0xAAAAAAAAAAAAAAAA
    second = 0xBBBBBBBBBBBBBBBB
    buf = _le_u64(first) + _le_u64(second) + [0x44] * 6

    cur_default = [0]
    cur_checked = [0]
    out_default = [[9], [8], [7]]
    out_checked = [[9], [8], [7]]

    err_default = gguf_metadata_read_u64_triple_checked_default(
        buf,
        len(buf),
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
    )
    err_checked = gguf_metadata_read_u64_triple_checked(
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
    assert out_default == out_checked == [[9], [8], [7]]


def test_overflow_passthrough_and_no_commit() -> None:
    buf = [0x55] * 64

    cur_default = [2]
    cur_checked = [2]
    out_default = [[111], [222], [333]]
    out_checked = [[111], [222], [333]]

    err_default = gguf_metadata_read_u64_triple_checked_default(
        buf,
        I64_MAX + 1,
        cur_default,
        out_default[0],
        out_default[1],
        out_default[2],
    )
    err_checked = gguf_metadata_read_u64_triple_checked(
        buf,
        I64_MAX + 1,
        cur_checked,
        I64_MAX + 1,
        out_checked[0],
        out_checked[1],
        out_checked[2],
    )

    assert err_default == err_checked == GGUF_META_TABLE_ERR_OVERFLOW
    assert cur_default == cur_checked == [2]
    assert out_default == out_checked == [[111], [222], [333]]


def test_randomized_default_vs_checked_table_end_buf_nbytes() -> None:
    rng = random.Random(20260418_295)

    for _ in range(6000):
        n = rng.randint(0, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]

        cursor0 = rng.randint(0, max(0, n + 10))
        out0 = [rng.getrandbits(64)]
        out1 = [rng.getrandbits(64)]
        out2 = [rng.getrandbits(64)]

        d_cursor = [cursor0]
        c_cursor = [cursor0]
        d_out0 = out0.copy()
        d_out1 = out1.copy()
        d_out2 = out2.copy()
        c_out0 = out0.copy()
        c_out1 = out1.copy()
        c_out2 = out2.copy()

        err_default = gguf_metadata_read_u64_triple_checked_default(
            buf,
            n,
            d_cursor,
            d_out0,
            d_out1,
            d_out2,
        )
        err_checked = gguf_metadata_read_u64_triple_checked(
            buf,
            n,
            c_cursor,
            n,
            c_out0,
            c_out1,
            c_out2,
        )

        assert err_default == err_checked
        assert d_cursor == c_cursor
        assert d_out0 == c_out0
        assert d_out1 == c_out1
        assert d_out2 == c_out2


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_success_uses_default_end_and_matches_checked()
    test_truncation_matches_checked_and_no_commit()
    test_overflow_passthrough_and_no_commit()
    test_randomized_default_vs_checked_table_end_buf_nbytes()
    print("ok")
