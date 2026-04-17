#!/usr/bin/env python3
"""Parity harness for GGUFMetadataReadU8QuadCheckedDefault contract."""

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1


def gguf_metadata_cursor_can_advance_checked(cursor: int, need: int, table_end: int):
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


def gguf_metadata_read_u8_checked(buf, buf_nbytes, cursor_ref, table_end, out_value_ref):
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
    buf, buf_nbytes, cursor_ref, table_end, out_first_ref, out_second_ref, out_third_ref
):
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
    buf,
    buf_nbytes,
    cursor_ref,
    table_end,
    out_first_ref,
    out_second_ref,
    out_third_ref,
    out_fourth_ref,
):
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


def gguf_metadata_read_u8_quad_checked_default(
    buf,
    buf_nbytes,
    cursor_ref,
    out_first_ref,
    out_second_ref,
    out_third_ref,
    out_fourth_ref,
):
    return gguf_metadata_read_u8_quad_checked(
        buf,
        buf_nbytes,
        cursor_ref,
        buf_nbytes,
        out_first_ref,
        out_second_ref,
        out_third_ref,
        out_fourth_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [5]
    out_first = [0x11]
    out_second = [0x22]
    out_third = [0x33]
    out_fourth = [0x44]

    assert (
        gguf_metadata_read_u8_quad_checked_default(
            None,
            16,
            cursor,
            out_first,
            out_second,
            out_third,
            out_fourth,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor == [5]
    assert out_first == [0x11]
    assert out_second == [0x22]
    assert out_third == [0x33]
    assert out_fourth == [0x44]


def test_uses_default_end_and_no_commit_on_short_payload() -> None:
    buf = [0xDE, 0xAD, 0xBE]

    cursor = [0]
    out_first = [0xAA]
    out_second = [0xBB]
    out_third = [0xCC]
    out_fourth = [0xDD]

    err = gguf_metadata_read_u8_quad_checked_default(
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
    assert out_first == [0xAA]
    assert out_second == [0xBB]
    assert out_third == [0xCC]
    assert out_fourth == [0xDD]


def test_success_reads_four_u8_and_advances() -> None:
    buf = [0x55, 0x10, 0x20, 0x30, 0x40, 0x99]

    cursor = [1]
    out_first = [0]
    out_second = [0]
    out_third = [0]
    out_fourth = [0]

    err = gguf_metadata_read_u8_quad_checked_default(
        buf,
        len(buf),
        cursor,
        out_first,
        out_second,
        out_third,
        out_fourth,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_first == [0x10]
    assert out_second == [0x20]
    assert out_third == [0x30]
    assert out_fourth == [0x40]
    assert cursor == [5]
