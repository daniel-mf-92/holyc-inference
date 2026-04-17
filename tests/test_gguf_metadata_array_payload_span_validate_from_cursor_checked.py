#!/usr/bin/env python3
"""Parity checks for GGUFMetadataArrayPayloadSpanValidateFromCursorChecked semantics."""

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
    if cursor > (U64_MAX - need):
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None

    return GGUF_META_TABLE_OK, next_cursor


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

    err, payload_end = gguf_metadata_cursor_can_advance_checked(
        payload_start,
        payload_bytes,
        table_end,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    assert payload_end is not None
    if payload_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    out_payload_end_ref[0] = payload_end
    return GGUF_META_TABLE_OK


def gguf_metadata_array_payload_span_validate_from_cursor_checked(
    cursor_ref: list[int] | None,
    payload_bytes: int,
    table_end: int,
    buf_nbytes: int,
    out_payload_end_ref: list[int] | None,
) -> int:
    if cursor_ref is None or out_payload_end_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    payload_start = cursor_ref[0]
    return gguf_metadata_array_payload_span_validate_checked(
        payload_start,
        payload_bytes,
        table_end,
        buf_nbytes,
        out_payload_end_ref,
    )


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [17]
    out_end = [0xCAFE]

    assert (
        gguf_metadata_array_payload_span_validate_from_cursor_checked(
            None,
            4,
            128,
            128,
            out_end,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out_end[0] == 0xCAFE

    assert (
        gguf_metadata_array_payload_span_validate_from_cursor_checked(
            cursor,
            4,
            128,
            128,
            None,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 17


def test_does_not_mutate_cursor_on_success_or_error() -> None:
    cursor = [41]
    out_end = [0]

    err = gguf_metadata_array_payload_span_validate_from_cursor_checked(
        cursor,
        9,
        100,
        100,
        out_end,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_end[0] == 50
    assert cursor[0] == 41

    out_end[0] = 1234
    err = gguf_metadata_array_payload_span_validate_from_cursor_checked(
        cursor,
        1000,
        100,
        100,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert out_end[0] == 1234
    assert cursor[0] == 41


def test_passthrough_of_core_validation_errors() -> None:
    out_end = [888]

    cursor = [50]
    err = gguf_metadata_array_payload_span_validate_from_cursor_checked(
        cursor,
        1,
        40,
        100,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert out_end[0] == 888
    assert cursor[0] == 50

    cursor = [I64_MAX + 1]
    err = gguf_metadata_array_payload_span_validate_from_cursor_checked(
        cursor,
        0,
        I64_MAX,
        I64_MAX,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_end[0] == 888
    assert cursor[0] == I64_MAX + 1

    cursor = [0]
    err = gguf_metadata_array_payload_span_validate_from_cursor_checked(
        cursor,
        0,
        0,
        I64_MAX + 1,
        out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_end[0] == 888
    assert cursor[0] == 0


def test_fuzz_matches_direct_span_validator() -> None:
    rng = random.Random(20260417_233)

    for _ in range(6000):
        table_end = rng.randint(0, 1 << 18)
        cursor_val = rng.randint(0, table_end) if table_end else 0
        payload_bytes = rng.randint(0, 1 << 18)
        buf_nbytes = rng.randint(0, 1 << 18)

        cursor = [cursor_val]
        out_from_cursor = [0xAAAA]
        out_direct = [0xBBBB]

        err_from_cursor = gguf_metadata_array_payload_span_validate_from_cursor_checked(
            cursor,
            payload_bytes,
            table_end,
            buf_nbytes,
            out_from_cursor,
        )
        err_direct = gguf_metadata_array_payload_span_validate_checked(
            cursor_val,
            payload_bytes,
            table_end,
            buf_nbytes,
            out_direct,
        )

        assert err_from_cursor == err_direct
        if err_from_cursor == GGUF_META_TABLE_OK:
            assert out_from_cursor[0] == out_direct[0]
        else:
            assert out_from_cursor[0] == 0xAAAA

        assert cursor[0] == cursor_val


if __name__ == "__main__":
    test_null_ptr_and_no_partial_write()
    test_does_not_mutate_cursor_on_success_or_error()
    test_passthrough_of_core_validation_errors()
    test_fuzz_matches_direct_span_validator()
    print("ok")
