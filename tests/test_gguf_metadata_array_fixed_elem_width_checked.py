#!/usr/bin/env python3
"""Reference checks for GGUFMetadataArrayFixedElemWidthBytesChecked semantics."""

from __future__ import annotations

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2

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


def test_null_out_ptr() -> None:
    assert (
        gguf_metadata_array_fixed_elem_width_bytes_checked(
            GGUF_TYPE_UINT8,
            None,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )


def test_all_fixed_width_types() -> None:
    out_width = [999]

    cases = [
        (GGUF_TYPE_UINT8, 1),
        (GGUF_TYPE_INT8, 1),
        (GGUF_TYPE_BOOL, 1),
        (GGUF_TYPE_UINT16, 2),
        (GGUF_TYPE_INT16, 2),
        (GGUF_TYPE_UINT32, 4),
        (GGUF_TYPE_INT32, 4),
        (GGUF_TYPE_FLOAT32, 4),
        (GGUF_TYPE_UINT64, 8),
        (GGUF_TYPE_INT64, 8),
        (GGUF_TYPE_FLOAT64, 8),
    ]

    for elem_type, expected in cases:
        out_width[0] = 999
        err = gguf_metadata_array_fixed_elem_width_bytes_checked(elem_type, out_width)
        assert err == GGUF_META_TABLE_OK
        assert out_width[0] == expected


def test_variable_and_unknown_types_rejected_no_partial_write() -> None:
    out_width = [12345]

    for elem_type in (GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, 13, 77, 0xFFFFFFFF):
        out_width[0] = 12345
        err = gguf_metadata_array_fixed_elem_width_bytes_checked(elem_type, out_width)
        assert err == GGUF_META_TABLE_ERR_BAD_PARAM
        assert out_width[0] == 12345


if __name__ == "__main__":
    test_null_out_ptr()
    test_all_fixed_width_types()
    test_variable_and_unknown_types_rejected_no_partial_write()
    print("ok")
