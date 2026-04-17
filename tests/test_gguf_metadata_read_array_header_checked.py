#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadArrayHeaderChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

GGUF_MAX_ARRAY_ELEMS = 1 << 24
I64_MAX = (1 << 63) - 1

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

SUPPORTED_TYPES = {
    GGUF_TYPE_UINT8,
    GGUF_TYPE_INT8,
    GGUF_TYPE_UINT16,
    GGUF_TYPE_INT16,
    GGUF_TYPE_UINT32,
    GGUF_TYPE_INT32,
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_BOOL,
    GGUF_TYPE_STRING,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_UINT64,
    GGUF_TYPE_INT64,
    GGUF_TYPE_FLOAT64,
}


def gguf_meta_type_supported(gguf_type: int) -> bool:
    return gguf_type in SUPPORTED_TYPES


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
    if cursor > (0xFFFFFFFFFFFFFFFF - need):
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
    b = [0]
    out = 0

    for i in range(4):
        err = gguf_metadata_read_u8_checked(buf, buf_nbytes, cur, table_end, b)
        if err != GGUF_META_TABLE_OK:
            return err
        out |= b[0] << (8 * i)

    out_value_ref[0] = out
    cursor_ref[0] = cur[0]
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


def gguf_metadata_read_array_header_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_elem_type_ref: list[int] | None,
    out_elem_count_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_elem_type_ref is None
        or out_elem_count_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    cur = [cursor_ref[0]]
    elem_type = [0]
    elem_count = [0]

    err = gguf_metadata_read_u32le_checked(buf, buf_nbytes, cur, table_end, elem_type)
    if err != GGUF_META_TABLE_OK:
        return err

    if elem_type[0] == GGUF_TYPE_ARRAY:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if not gguf_meta_type_supported(elem_type[0]):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    err = gguf_metadata_read_u64le_checked(buf, buf_nbytes, cur, table_end, elem_count)
    if err != GGUF_META_TABLE_OK:
        return err

    if elem_count[0] > GGUF_MAX_ARRAY_ELEMS:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_elem_type_ref[0] = elem_type[0]
    out_elem_count_ref[0] = elem_count[0]
    cursor_ref[0] = cur[0]
    return GGUF_META_TABLE_OK


def encode_u32le(x: int) -> list[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(4)]


def encode_u64le(x: int) -> list[int]:
    return [(x >> (8 * i)) & 0xFF for i in range(8)]


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [3]
    out_elem_type = [77]
    out_elem_count = [88]

    assert (
        gguf_metadata_read_array_header_checked(
            None,
            64,
            cursor,
            64,
            out_elem_type,
            out_elem_count,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 3
    assert out_elem_type[0] == 77
    assert out_elem_count[0] == 88

    assert (
        gguf_metadata_read_array_header_checked(
            [0] * 64,
            64,
            None,
            64,
            out_elem_type,
            out_elem_count,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_elem_type[0] == 77
    assert out_elem_count[0] == 88


def test_truncated_header_does_not_mutate() -> None:
    buf = [0x11] * 32
    out_elem_type = [31]
    out_elem_count = [63]

    for table_end in range(0, 11):
        cursor = [0]
        err = gguf_metadata_read_array_header_checked(
            buf,
            len(buf),
            cursor,
            table_end,
            out_elem_type,
            out_elem_count,
        )
        assert err in (
            GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
            GGUF_META_TABLE_ERR_BAD_PARAM,
            GGUF_META_TABLE_ERR_OVERFLOW,
        )
        assert cursor[0] == 0
        assert out_elem_type[0] == 31
        assert out_elem_count[0] == 63


def test_rejects_nested_array_and_unsupported_elem_type() -> None:
    out_elem_type = [5]
    out_elem_count = [7]

    buf_nested = encode_u32le(GGUF_TYPE_ARRAY) + encode_u64le(1)
    cursor = [0]
    err = gguf_metadata_read_array_header_checked(
        buf_nested,
        len(buf_nested),
        cursor,
        len(buf_nested),
        out_elem_type,
        out_elem_count,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_elem_type[0] == 5
    assert out_elem_count[0] == 7

    bad_type = 0xDEADBEEF
    buf_bad_type = encode_u32le(bad_type) + encode_u64le(1)
    cursor = [0]
    err = gguf_metadata_read_array_header_checked(
        buf_bad_type,
        len(buf_bad_type),
        cursor,
        len(buf_bad_type),
        out_elem_type,
        out_elem_count,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0


def test_rejects_count_above_global_cap() -> None:
    out_elem_type = [0]
    out_elem_count = [0]

    buf = encode_u32le(GGUF_TYPE_UINT8) + encode_u64le(GGUF_MAX_ARRAY_ELEMS + 1)
    cursor = [0]

    err = gguf_metadata_read_array_header_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_elem_type,
        out_elem_count,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert out_elem_type[0] == 0
    assert out_elem_count[0] == 0


def test_success_reads_elem_type_and_count() -> None:
    elem_type = GGUF_TYPE_STRING
    elem_count = 123456
    buf = encode_u32le(elem_type) + encode_u64le(elem_count) + [0xAA, 0xBB, 0xCC]

    cursor = [0]
    out_elem_type = [0]
    out_elem_count = [0]

    err = gguf_metadata_read_array_header_checked(
        buf,
        len(buf),
        cursor,
        len(buf),
        out_elem_type,
        out_elem_count,
    )

    assert err == GGUF_META_TABLE_OK
    assert out_elem_type[0] == elem_type
    assert out_elem_count[0] == elem_count
    assert cursor[0] == 12


def test_randomized_parity() -> None:
    rng = random.Random(20260417_203)

    for _ in range(5000):
        n = rng.randint(12, 256)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out_elem_type = [0xABCD]
        out_elem_count = [0x1234]
        err = gguf_metadata_read_array_header_checked(
            buf,
            n,
            cursor,
            table_end,
            out_elem_type,
            out_elem_count,
        )

        header_fits = cursor0 + 12 <= table_end and cursor0 + 12 <= n
        if not header_fits:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert cursor[0] == cursor0
            assert out_elem_type[0] == 0xABCD
            assert out_elem_count[0] == 0x1234
            continue

        elem_type = sum(buf[cursor0 + i] << (8 * i) for i in range(4))
        elem_count = sum(buf[cursor0 + 4 + i] << (8 * i) for i in range(8))

        if elem_type == GGUF_TYPE_ARRAY or not gguf_meta_type_supported(elem_type):
            assert err == GGUF_META_TABLE_ERR_BAD_PARAM
            assert cursor[0] == cursor0
            assert out_elem_type[0] == 0xABCD
            assert out_elem_count[0] == 0x1234
            continue

        if elem_count > GGUF_MAX_ARRAY_ELEMS:
            assert err == GGUF_META_TABLE_ERR_BAD_PARAM
            assert cursor[0] == cursor0
            assert out_elem_type[0] == 0xABCD
            assert out_elem_count[0] == 0x1234
            continue

        assert err == GGUF_META_TABLE_OK
        assert cursor[0] == cursor0 + 12
        assert out_elem_type[0] == elem_type
        assert out_elem_count[0] == elem_count


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_truncated_header_does_not_mutate()
    test_rejects_nested_array_and_unsupported_elem_type()
    test_rejects_count_above_global_cap()
    test_success_reads_elem_type_and_count()
    test_randomized_parity()
    print("gguf_metadata_read_array_header_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
