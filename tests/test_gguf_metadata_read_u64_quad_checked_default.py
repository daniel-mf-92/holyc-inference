#!/usr/bin/env python3
"""Parity harness for GGUFMetadataReadU64QuadCheckedDefault contract."""

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
    byte = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, byte)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= byte[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64_quad_checked(
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

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u64_quad_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
    out_fourth_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u64_quad_checked(
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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [7]
    out_first = [0x101]
    out_second = [0x202]
    out_third = [0x303]
    out_fourth = [0x404]

    assert (
        gguf_metadata_read_u64_quad_checked_default(
            None,
            64,
            cursor,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [7]
    assert out_first == [0x101]
    assert out_second == [0x202]
    assert out_third == [0x303]
    assert out_fourth == [0x404]


def test_default_end_short_payload_fails_without_commit() -> None:
    first = 0x1111222233334444
    second = 0x5555666677778888
    third = 0x9999AAAABBBBCCCC
    buf = _le_u64(first) + _le_u64(second) + _le_u64(third) + [0xEF] * 7

    cursor = [0]
    out_first = [11]
    out_second = [22]
    out_third = [33]
    out_fourth = [44]

    err = gguf_metadata_read_u64_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [11]
    assert out_second == [22]
    assert out_third == [33]
    assert out_fourth == [44]


def test_success_reads_four_u64_and_advances() -> None:
    first = 0x0123456789ABCDEF
    second = 0x0F1E2D3C4B5A6978
    third = 0x0011223344556677
    fourth = 0x8899AABBCCDDEEFF
    buf = [0x42] + _le_u64(first) + _le_u64(second) + _le_u64(third) + _le_u64(fourth) + [0x99]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u64_quad_checked_default(
        buf,
        len(buf),
        cursor,
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


def test_matches_checked_core_when_table_end_equals_buf_nbytes() -> None:
    rng = random.Random(20260418_292)

    for _ in range(6000):
        n = rng.randint(1, 512)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n)

        c_core = [cursor0]
        c_def = [cursor0]

        core_first = [0xA1]
        core_second = [0xB2]
        core_third = [0xC3]
        core_fourth = [0xD4]

        def_first = [0xA1]
        def_second = [0xB2]
        def_third = [0xC3]
        def_fourth = [0xD4]

        err_core = gguf_metadata_read_u64_quad_checked(
            buf,
            n,
            c_core,
            n,
            core_first,
            core_second,
            core_third,
            core_fourth,
        )
        err_def = gguf_metadata_read_u64_quad_checked_default(
            buf,
            n,
            c_def,
            def_first,
            def_second,
            def_third,
            def_fourth,
        )

        assert err_def == err_core
        assert c_def == c_core
        assert def_first == core_first
        assert def_second == core_second
        assert def_third == core_third
        assert def_fourth == core_fourth


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_default_end_short_payload_fails_without_commit()
    test_success_reads_four_u64_and_advances()
    test_matches_checked_core_when_table_end_equals_buf_nbytes()
    print("ok")
