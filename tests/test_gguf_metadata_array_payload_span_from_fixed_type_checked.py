#!/usr/bin/env python3
"""Parity checks for GGUFMetadataArrayPayloadSpanFromFixedTypeChecked semantics."""

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
    if rhs != 0 and lhs > (U64_MAX // rhs):
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


def test_null_out_ptrs_no_partial_write() -> None:
    out_bytes = [111]
    out_end = [222]

    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_UINT8,
        1,
        0,
        10,
        10,
        None,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert out_end[0] == 222

    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_UINT8,
        1,
        0,
        10,
        10,
        out_bytes,
        None,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR
    assert out_bytes[0] == 111


def test_reject_variable_width_types() -> None:
    out_bytes = [333]
    out_end = [444]

    for elem_type in (GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, 13, 0xFFFF):
        out_bytes[0] = 333
        out_end[0] = 444
        err = gguf_metadata_array_payload_span_from_fixed_type_checked(
            elem_type,
            8,
            16,
            4096,
            4096,
            out_bytes,
            out_end,
        )
        assert err == GGUF_META_TABLE_ERR_BAD_PARAM
        assert out_bytes[0] == 333
        assert out_end[0] == 444


def test_mul_overflow_passthrough() -> None:
    out_bytes = [555]
    out_end = [666]

    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_UINT64,
        I64_MAX,
        0,
        I64_MAX,
        I64_MAX,
        out_bytes,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_bytes[0] == 555
    assert out_end[0] == 666


def test_span_validate_passthrough_no_partial_write() -> None:
    out_bytes = [777]
    out_end = [888]

    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_UINT32,
        8,
        100,
        120,
        1024,
        out_bytes,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert out_bytes[0] == 777
    assert out_end[0] == 888


def test_success_cases_and_fuzz() -> None:
    out_bytes = [0]
    out_end = [0]

    err = gguf_metadata_array_payload_span_from_fixed_type_checked(
        GGUF_TYPE_INT16,
        5,
        40,
        100,
        100,
        out_bytes,
        out_end,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_bytes[0] == 10
    assert out_end[0] == 50

    rng = random.Random(20260417_231)
    fixed_types = [
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
    ]
    widths = {
        GGUF_TYPE_UINT8: 1,
        GGUF_TYPE_INT8: 1,
        GGUF_TYPE_BOOL: 1,
        GGUF_TYPE_UINT16: 2,
        GGUF_TYPE_INT16: 2,
        GGUF_TYPE_UINT32: 4,
        GGUF_TYPE_INT32: 4,
        GGUF_TYPE_FLOAT32: 4,
        GGUF_TYPE_UINT64: 8,
        GGUF_TYPE_INT64: 8,
        GGUF_TYPE_FLOAT64: 8,
    }

    for _ in range(5000):
        elem_type = rng.choice(fixed_types)
        width = widths[elem_type]
        table_end = rng.randint(0, 20_000)
        payload_start = rng.randint(0, table_end)
        elem_count = rng.randint(0, 5000)
        buf_nbytes = rng.randint(0, 20_000)

        out_bytes_ref = [0xBEEF]
        out_end_ref = [0xCAFE]
        err = gguf_metadata_array_payload_span_from_fixed_type_checked(
            elem_type,
            elem_count,
            payload_start,
            table_end,
            buf_nbytes,
            out_bytes_ref,
            out_end_ref,
        )

        payload_bytes = elem_count * width
        if payload_bytes > I64_MAX:
            assert err == GGUF_META_TABLE_ERR_OVERFLOW
            assert out_bytes_ref[0] == 0xBEEF
            assert out_end_ref[0] == 0xCAFE
            continue

        payload_end = payload_start + payload_bytes
        if payload_end > table_end or payload_end > buf_nbytes:
            assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
            assert out_bytes_ref[0] == 0xBEEF
            assert out_end_ref[0] == 0xCAFE
        else:
            assert err == GGUF_META_TABLE_OK
            assert out_bytes_ref[0] == payload_bytes
            assert out_end_ref[0] == payload_end


if __name__ == "__main__":
    test_null_out_ptrs_no_partial_write()
    test_reject_variable_width_types()
    test_mul_overflow_passthrough()
    test_span_validate_passthrough_no_partial_write()
    test_success_cases_and_fuzz()
    print("ok")
