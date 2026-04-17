#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadF64BitsQuadChecked semantics."""

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
    b = [0]
    out = 0

    for i in range(8):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= b[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bitsle_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_bits_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_bits_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    raw = [0]
    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, raw)
    if err != GGUF_META_TABLE_OK:
        return err

    out_bits_ref[0] = raw[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bits_pair_checked(
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

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bits_triple_checked(
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

    err = gguf_metadata_read_f64bits_pair_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_f64bits_quad_checked(
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

    err = gguf_metadata_read_f64bits_triple_checked(
        buf,
        buf_nbytes,
        cur,
        table_end,
        first,
        second,
        third,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    err = gguf_metadata_read_f64bitsle_checked(buf, buf_nbytes, cur, table_end, fourth)
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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [4]
    out_first = [0x1111222233334444]
    out_second = [0xAAAABBBBCCCCDDDD]
    out_third = [0x0123456789ABCDEF]
    out_fourth = [0x0FEDCBA987654321]

    assert (
        gguf_metadata_read_f64bits_quad_checked(
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
    assert out_first == [0x1111222233334444]
    assert out_second == [0xAAAABBBBCCCCDDDD]
    assert out_third == [0x0123456789ABCDEF]
    assert out_fourth == [0x0FEDCBA987654321]


def test_fourth_lane_fail_does_not_commit_outputs_or_cursor() -> None:
    first = 0x0123456789ABCDEF
    second = 0x0FEDCBA987654321
    third = 0x1122334455667788
    buf = _le_u64(first) + _le_u64(second) + _le_u64(third) + [0xAA] * 7

    cursor = [0]
    out_first = [0x1111111111111111]
    out_second = [0x2222222222222222]
    out_third = [0x3333333333333333]
    out_fourth = [0x4444444444444444]

    err = gguf_metadata_read_f64bits_quad_checked(
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
    assert out_first == [0x1111111111111111]
    assert out_second == [0x2222222222222222]
    assert out_third == [0x3333333333333333]
    assert out_fourth == [0x4444444444444444]


def test_triple_stage_fail_does_not_commit_outputs_or_cursor() -> None:
    first = 0x0123456789ABCDEF
    second = 0x0FEDCBA987654321
    buf = _le_u64(first) + _le_u64(second) + [0xAA] * 7

    cursor = [0]
    out_first = [0x1111111111111111]
    out_second = [0x2222222222222222]
    out_third = [0x3333333333333333]
    out_fourth = [0x4444444444444444]

    err = gguf_metadata_read_f64bits_quad_checked(
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
    assert out_first == [0x1111111111111111]
    assert out_second == [0x2222222222222222]
    assert out_third == [0x3333333333333333]
    assert out_fourth == [0x4444444444444444]


def test_success_reads_four_f64_raw_bit_patterns_and_advances() -> None:
    first = 0x0123456789ABCDEF
    second = 0x0FEDCBA987654321
    third = 0x1122334455667788
    fourth = 0x8899AABBCCDDEEFF

    prefix = [0xFE, 0xED]
    buf = prefix + _le_u64(first) + _le_u64(second) + _le_u64(third) + _le_u64(fourth)
    cursor = [2]

    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_f64bits_quad_checked(
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
    assert cursor[0] == 34


def test_randomized_parity() -> None:
    rng = random.Random(20260417_258)

    for _ in range(4000):
        n = rng.randint(1, 512)
        table_end = rng.randint(0, n)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_first = [0xAAAAAAAAAAAAAAAA]
        out_second = [0xBBBBBBBBBBBBBBBB]
        out_third = [0xCCCCCCCCCCCCCCCC]
        out_fourth = [0xDDDDDDDDDDDDDDDD]

        err = gguf_metadata_read_f64bits_quad_checked(
            buf,
            n,
            cursor,
            table_end,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )

        if cursor0 + 32 <= table_end and cursor0 + 32 <= n:
            expected = []
            for lane in range(4):
                lane_bits = 0
                base = cursor0 + lane * 8
                for i in range(8):
                    lane_bits |= buf[base + i] << (8 * i)
                expected.append(lane_bits)

            assert err == GGUF_META_TABLE_OK
            assert out_first[0] == expected[0]
            assert out_second[0] == expected[1]
            assert out_third[0] == expected[2]
            assert out_fourth[0] == expected[3]
            assert cursor[0] == cursor0 + 32
        else:
            assert err in (
                GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
                GGUF_META_TABLE_ERR_BAD_PARAM,
                GGUF_META_TABLE_ERR_OVERFLOW,
            )
            assert out_first[0] == 0xAAAAAAAAAAAAAAAA
            assert out_second[0] == 0xBBBBBBBBBBBBBBBB
            assert out_third[0] == 0xCCCCCCCCCCCCCCCC
            assert out_fourth[0] == 0xDDDDDDDDDDDDDDDD
            assert cursor[0] == cursor0


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_triple_stage_fail_does_not_commit_outputs_or_cursor()
    test_fourth_lane_fail_does_not_commit_outputs_or_cursor()
    test_success_reads_four_f64_raw_bit_patterns_and_advances()
    test_randomized_parity()
    print("gguf_metadata_read_f64bits_quad_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
