#!/usr/bin/env python3
"""Parity checks for GGUFMetadataArrayPayloadSpanFromFixedTypeAtCursorChecked."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

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

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


def gguf_metadata_array_fixed_elem_width_bytes_checked(
    elem_type: int,
    out_elem_width_bytes_ref: list[int] | None,
) -> int:
    if out_elem_width_bytes_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if elem_type in (GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL):
        out_elem_width_bytes_ref[0] = 1
        return GGUF_META_TABLE_OK
    if elem_type in (GGUF_TYPE_UINT16, GGUF_TYPE_INT16):
        out_elem_width_bytes_ref[0] = 2
        return GGUF_META_TABLE_OK
    if elem_type in (GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32):
        out_elem_width_bytes_ref[0] = 4
        return GGUF_META_TABLE_OK
    if elem_type in (GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64):
        out_elem_width_bytes_ref[0] = 8
        return GGUF_META_TABLE_OK

    return GGUF_META_TABLE_ERR_BAD_PARAM


def gguf_metadata_mul_u64_checked(
    lhs: int,
    rhs: int,
    out_product_ref: list[int] | None,
) -> int:
    if out_product_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if lhs > I64_MAX or rhs > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if rhs and lhs > (U64_MAX // rhs):
        return GGUF_META_TABLE_ERR_OVERFLOW

    product = lhs * rhs
    if product > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    out_product_ref[0] = product
    return GGUF_META_TABLE_OK


def gguf_metadata_array_fixed_payload_bytes_checked(
    elem_count: int,
    elem_width_bytes: int,
    out_payload_bytes_ref: list[int] | None,
) -> int:
    return gguf_metadata_mul_u64_checked(
        elem_count,
        elem_width_bytes,
        out_payload_bytes_ref,
    )


def gguf_metadata_cursor_can_advance_checked(
    cursor: int,
    need: int,
    table_end: int,
    out_next_cursor_ref: list[int] | None,
) -> int:
    if out_next_cursor_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if cursor > I64_MAX or need > I64_MAX or table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if cursor > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if cursor > (U64_MAX - need):
        return GGUF_META_TABLE_ERR_OVERFLOW

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    out_next_cursor_ref[0] = next_cursor
    return GGUF_META_TABLE_OK


def gguf_metadata_array_payload_span_validate_checked(
    payload_start: int,
    payload_bytes: int,
    table_end: int,
    buf_nbytes: int,
    out_payload_end_ref: list[int] | None,
) -> int:
    if out_payload_end_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR
    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    payload_end_ref = [0]
    err = gguf_metadata_cursor_can_advance_checked(
        payload_start,
        payload_bytes,
        table_end,
        payload_end_ref,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    payload_end = payload_end_ref[0]
    if payload_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    out_payload_end_ref[0] = payload_end
    return GGUF_META_TABLE_OK


def gguf_metadata_array_payload_span_from_fixed_type_checked(
    elem_type: int,
    elem_count: int,
    payload_start: int,
    table_end: int,
    buf_nbytes: int,
    out_payload_bytes_ref: list[int] | None,
    out_payload_end_ref: list[int] | None,
) -> int:
    if out_payload_bytes_ref is None or out_payload_end_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    elem_width_ref = [0]
    err = gguf_metadata_array_fixed_elem_width_bytes_checked(elem_type, elem_width_ref)
    if err != GGUF_META_TABLE_OK:
        return err

    payload_bytes_ref = [0]
    err = gguf_metadata_array_fixed_payload_bytes_checked(
        elem_count,
        elem_width_ref[0],
        payload_bytes_ref,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    payload_end_ref = [0]
    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start,
        payload_bytes_ref[0],
        table_end,
        buf_nbytes,
        payload_end_ref,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    out_payload_bytes_ref[0] = payload_bytes_ref[0]
    out_payload_end_ref[0] = payload_end_ref[0]
    return GGUF_META_TABLE_OK


def gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
    elem_type: int,
    elem_count: int,
    cursor_ref: list[int] | None,
    table_end: int,
    buf_nbytes: int,
    out_payload_bytes_ref: list[int] | None,
    out_payload_end_ref: list[int] | None,
) -> int:
    if (
        cursor_ref is None
        or out_payload_bytes_ref is None
        or out_payload_end_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    payload_start = cursor_ref[0]
    return gguf_metadata_array_payload_span_from_fixed_type_checked(
        elem_type,
        elem_count,
        payload_start,
        table_end,
        buf_nbytes,
        out_payload_bytes_ref,
        out_payload_end_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [33]
    out_bytes = [0xAAAA]
    out_end = [0xBBBB]

    assert (
        gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
            GGUF_TYPE_UINT8,
            3,
            None,
            128,
            128,
            out_bytes,
            out_end,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_bytes[0] == 0xAAAA
    assert out_end[0] == 0xBBBB

    assert (
        gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
            GGUF_TYPE_UINT8,
            3,
            cursor,
            128,
            128,
            None,
            out_end,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 33
    assert out_end[0] == 0xBBBB

    assert (
        gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
            GGUF_TYPE_UINT8,
            3,
            cursor,
            128,
            128,
            out_bytes,
            None,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 33
    assert out_bytes[0] == 0xAAAA


def test_cursor_unchanged_and_success() -> None:
    cursor = [19]
    out_bytes = [0]
    out_end = [0]

    err = gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
        GGUF_TYPE_UINT16,
        11,
        cursor,
        256,
        256,
        out_bytes,
        out_end,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_bytes[0] == 22
    assert out_end[0] == 41
    assert cursor[0] == 19


def test_error_passthrough_and_no_partial_writes() -> None:
    out_bytes = [1234]
    out_end = [5678]

    cursor = [10]
    err = gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
        GGUF_TYPE_STRING,
        1,
        cursor,
        256,
        256,
        out_bytes,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert out_bytes[0] == 1234
    assert out_end[0] == 5678
    assert cursor[0] == 10

    cursor = [0]
    err = gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
        GGUF_TYPE_UINT64,
        I64_MAX,
        cursor,
        I64_MAX,
        I64_MAX,
        out_bytes,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_bytes[0] == 1234
    assert out_end[0] == 5678
    assert cursor[0] == 0


def test_fuzz_matches_direct_from_fixed_type() -> None:
    rng = random.Random(20260417_235)

    elem_types = [
        GGUF_TYPE_UINT8,
        GGUF_TYPE_INT8,
        GGUF_TYPE_UINT16,
        GGUF_TYPE_INT16,
        GGUF_TYPE_UINT32,
        GGUF_TYPE_INT32,
        GGUF_TYPE_FLOAT32,
        GGUF_TYPE_BOOL,
        GGUF_TYPE_UINT64,
        GGUF_TYPE_INT64,
        GGUF_TYPE_FLOAT64,
        GGUF_TYPE_STRING,
        GGUF_TYPE_ARRAY,
        13,
        0xFFFFFFFF,
    ]

    for _ in range(7000):
        table_end = rng.randint(0, 1 << 18)
        cursor_val = rng.randint(0, table_end) if table_end else 0
        elem_count = rng.randint(0, 1 << 18)
        buf_nbytes = rng.randint(0, 1 << 18)
        elem_type = rng.choice(elem_types)

        cursor = [cursor_val]
        out_from_cursor_bytes = [0xCAFE]
        out_from_cursor_end = [0xBABE]
        out_direct_bytes = [0x1111]
        out_direct_end = [0x2222]

        err_from_cursor = gguf_metadata_array_payload_span_from_fixed_type_at_cursor_checked(
            elem_type,
            elem_count,
            cursor,
            table_end,
            buf_nbytes,
            out_from_cursor_bytes,
            out_from_cursor_end,
        )

        err_direct = gguf_metadata_array_payload_span_from_fixed_type_checked(
            elem_type,
            elem_count,
            cursor_val,
            table_end,
            buf_nbytes,
            out_direct_bytes,
            out_direct_end,
        )

        assert err_from_cursor == err_direct
        assert cursor[0] == cursor_val

        if err_from_cursor == GGUF_META_TABLE_OK:
            assert out_from_cursor_bytes[0] == out_direct_bytes[0]
            assert out_from_cursor_end[0] == out_direct_end[0]
        else:
            assert out_from_cursor_bytes[0] == 0xCAFE
            assert out_from_cursor_end[0] == 0xBABE


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_cursor_unchanged_and_success()
    test_error_passthrough_and_no_partial_writes()
    test_fuzz_matches_direct_from_fixed_type()
    print("ok")
