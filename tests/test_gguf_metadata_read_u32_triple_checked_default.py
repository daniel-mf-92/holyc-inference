#!/usr/bin/env python3
"""Parity checks for GGUFMetadataReadU32TripleCheckedDefault."""

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


def gguf_metadata_read_u32_triple_checked(
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

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u32_triple_checked_default(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    out_first_ref: list[int] | None,
    out_second_ref: list[int] | None,
    out_third_ref: list[int] | None,
) -> int:
    return gguf_metadata_read_u32_triple_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
    )


def _le_u32(value: int) -> list[int]:
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [9]
    out_first = [0x11111111]
    out_second = [0x22222222]
    out_third = [0x33333333]

    assert (
        gguf_metadata_read_u32_triple_checked_default(
            None,
            32,
            cursor,
            out_first,
            out_second,
            out_third,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [9]
    assert out_first == [0x11111111]
    assert out_second == [0x22222222]
    assert out_third == [0x33333333]


def test_uses_default_end_and_no_commit_on_short_payload() -> None:
    first = 0x89ABCDEF
    second = 0x01234567
    buf = _le_u32(first) + _le_u32(second) + [0x11, 0x22, 0x33]

    cursor = [0]
    out_first = [0xAAAAAAAA]
    out_second = [0xBBBBBBBB]
    out_third = [0xCCCCCCCC]

    err = gguf_metadata_read_u32_triple_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_first == [0xAAAAAAAA]
    assert out_second == [0xBBBBBBBB]
    assert out_third == [0xCCCCCCCC]


def test_success_from_mid_cursor() -> None:
    first = 0x0BADF00D
    second = 0xFEEDFACE
    third = 0x10203040
    buf = [0xAA, 0xBB] + _le_u32(first) + _le_u32(second) + _le_u32(third) + [0xCC]

    cursor = [2]
    out_first = [0]
    out_second = [0]
    out_third = [0]

    err = gguf_metadata_read_u32_triple_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [first]
    assert out_second == [second]
    assert out_third == [third]
    assert cursor == [14]


def test_exact_cursor_and_output_parity_with_checked_core() -> None:
    rng = random.Random(0xC0FFEE32)

    for _ in range(400):
        buf_len = rng.randint(0, 80)
        buf = [rng.randrange(0, 256) for _ in range(buf_len)]

        cursor0 = rng.randint(0, 96)
        out_a0 = [rng.randrange(0, 1 << 32)]
        out_b0 = [rng.randrange(0, 1 << 32)]
        out_c0 = [rng.randrange(0, 1 << 32)]

        cursor1 = [cursor0]
        out_a1 = [out_a0[0]]
        out_b1 = [out_b0[0]]
        out_c1 = [out_c0[0]]

        cursor2 = [cursor0]
        out_a2 = [out_a0[0]]
        out_b2 = [out_b0[0]]
        out_c2 = [out_c0[0]]

        err_default = gguf_metadata_read_u32_triple_checked_default(
            buf,
            buf_len,
            cursor1,
            out_a1,
            out_b1,
            out_c1,
        )
        err_core = gguf_metadata_read_u32_triple_checked(
            buf,
            buf_len,
            cursor2,
            buf_len,
            out_a2,
            out_b2,
            out_c2,
        )

        assert err_default == err_core
        assert cursor1 == cursor2
        assert out_a1 == out_a2
        assert out_b1 == out_b2
        assert out_c1 == out_c2


def test_i64_overflow_guard_matches_checked_core() -> None:
    buf = [0] * 32
    cursor_default = [0]
    out_default = [0xDEADBEEF]
    out2_default = [0xCAFEBABE]
    out3_default = [0x12345678]

    cursor_core = [0]
    out_core = [out_default[0]]
    out2_core = [out2_default[0]]
    out3_core = [out3_default[0]]

    huge = I64_MAX + 1
    err_default = gguf_metadata_read_u32_triple_checked_default(
        buf,
        huge,
        cursor_default,
        out_default,
        out2_default,
        out3_default,
    )
    err_core = gguf_metadata_read_u32_triple_checked(
        buf,
        huge,
        cursor_core,
        huge,
        out_core,
        out2_core,
        out3_core,
    )

    assert err_default == GGUF_META_TABLE_ERR_OVERFLOW
    assert err_default == err_core
    assert cursor_default == cursor_core == [0]
    assert out_default == out_core == [0xDEADBEEF]
    assert out2_default == out2_core == [0xCAFEBABE]
    assert out3_default == out3_core == [0x12345678]


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_uses_default_end_and_no_commit_on_short_payload()
    test_success_from_mid_cursor()
    test_exact_cursor_and_output_parity_with_checked_core()
    test_i64_overflow_guard_matches_checked_core()
    print("gguf_metadata_read_u32_triple_checked_default: ok")
