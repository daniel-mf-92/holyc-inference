#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadU8QuadCheckedNoPartial semantics."""

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


def gguf_metadata_read_u8_triple_checked(
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

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, first)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, second)
    if err != GGUF_META_TABLE_OK:
        return err
    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, third)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u8_quad_checked(
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

    err = gguf_metadata_read_u8_triple_checked(
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

    err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, fourth)
    if err != GGUF_META_TABLE_OK:
        return err

    out_first_ref[0] = first[0]
    out_second_ref[0] = second[0]
    out_third_ref[0] = third[0]
    out_fourth_ref[0] = fourth[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_read_u8_quad_checked_no_partial(
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

    err = gguf_metadata_read_u8_quad_checked(
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


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out_first = [11]
    out_second = [22]
    out_third = [33]
    out_fourth = [44]

    assert (
        gguf_metadata_read_u8_quad_checked_no_partial(
            None,
            8,
            cursor,
            8,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [2]
    assert out_first == [11]
    assert out_second == [22]
    assert out_third == [33]
    assert out_fourth == [44]


def test_fourth_lane_fail_keeps_outputs_and_cursor() -> None:
    buf = [0xA1, 0xA2, 0xA3]
    cursor = [0]
    out_first = [0x11]
    out_second = [0x22]
    out_third = [0x33]
    out_fourth = [0x44]

    err = gguf_metadata_read_u8_quad_checked_no_partial(
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
    assert out_first == [0x11]
    assert out_second == [0x22]
    assert out_third == [0x33]
    assert out_fourth == [0x44]


def test_success_reads_four_u8_and_advances() -> None:
    buf = [7, 8, 9, 10, 11, 12]
    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u8_quad_checked_no_partial(
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
    assert cursor == [5]
    assert out_first == [8]
    assert out_second == [9]
    assert out_third == [10]
    assert out_fourth == [11]


def test_randomized_parity_with_checked_core() -> None:
    rng = random.Random(0x8BADF00D)

    for _ in range(2000):
        n = rng.randint(0, 24)
        buf = [rng.randint(0, 255) for _ in range(n)]
        cursor0 = rng.randint(0, n + 2)
        table_end = rng.randint(0, n + 2)

        cur_core = [cursor0]
        out_core_0 = [rng.randint(0, 255)]
        out_core_1 = [rng.randint(0, 255)]
        out_core_2 = [rng.randint(0, 255)]
        out_core_3 = [rng.randint(0, 255)]

        cur_np = [cursor0]
        out_np_0 = [out_core_0[0]]
        out_np_1 = [out_core_1[0]]
        out_np_2 = [out_core_2[0]]
        out_np_3 = [out_core_3[0]]

        err_core = gguf_metadata_read_u8_quad_checked(
            buf,
            n,
            cur_core,
            table_end,
            out_core_0,
            out_core_1,
            out_core_2,
            out_core_3,
        )
        err_np = gguf_metadata_read_u8_quad_checked_no_partial(
            buf,
            n,
            cur_np,
            table_end,
            out_np_0,
            out_np_1,
            out_np_2,
            out_np_3,
        )

        assert err_np == err_core
        assert cur_np == cur_core
        assert out_np_0 == out_core_0
        assert out_np_1 == out_core_1
        assert out_np_2 == out_core_2
        assert out_np_3 == out_core_3


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_fourth_lane_fail_keeps_outputs_and_cursor()
    test_success_reads_four_u8_and_advances()
    test_randomized_parity_with_checked_core()
    print("gguf_metadata_read_u8_quad_checked_no_partial_reference_checks=ok")
