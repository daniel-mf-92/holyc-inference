#!/usr/bin/env python3
"""Reference checks for GGUFMetadataArrayPayloadSpanValidateChecked semantics."""

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


def test_null_out_and_no_partial_write() -> None:
    assert (
        gguf_metadata_array_payload_span_validate_checked(0, 0, 0, 0, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )


def test_buf_size_overflow_guard() -> None:
    out_end = [123]
    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=0,
        payload_bytes=0,
        table_end=0,
        buf_nbytes=I64_MAX + 1,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_end[0] == 123


def test_zero_length_window_is_legal() -> None:
    out_end = [0xDEAD]
    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=17,
        payload_bytes=0,
        table_end=17,
        buf_nbytes=17,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_end[0] == 17


def test_cursor_domain_errors_passthrough() -> None:
    out_end = [55]

    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=19,
        payload_bytes=1,
        table_end=18,
        buf_nbytes=64,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert out_end[0] == 55

    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=I64_MAX + 1,
        payload_bytes=0,
        table_end=I64_MAX,
        buf_nbytes=I64_MAX,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_end[0] == 55


def test_unsigned_add_overflow_guard() -> None:
    out_end = [9001]
    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=U64_MAX - 5,
        payload_bytes=10,
        table_end=U64_MAX,
        buf_nbytes=I64_MAX,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out_end[0] == 9001


def test_window_must_fit_table_and_buffer() -> None:
    out_end = [111]

    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=40,
        payload_bytes=25,
        table_end=60,
        buf_nbytes=100,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert out_end[0] == 111

    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=20,
        payload_bytes=30,
        table_end=60,
        buf_nbytes=45,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert out_end[0] == 111


def test_success_and_fuzz() -> None:
    out_end = [0]
    err = gguf_metadata_array_payload_span_validate_checked(
        payload_start=31,
        payload_bytes=19,
        table_end=88,
        buf_nbytes=50,
        out_payload_end_ref=out_end,
    )
    assert err == GGUF_META_TABLE_OK
    assert out_end[0] == 50

    rng = random.Random(20260417_218)
    for _ in range(5000):
        table_end = rng.randint(0, 5000)
        payload_start = rng.randint(0, table_end)
        payload_bytes = rng.randint(0, 5000)
        buf_nbytes = rng.randint(0, 5000)

        out_ref = [0xFACE]
        err = gguf_metadata_array_payload_span_validate_checked(
            payload_start,
            payload_bytes,
            table_end,
            buf_nbytes,
            out_ref,
        )

        expected_end = payload_start + payload_bytes
        if expected_end <= table_end and expected_end <= buf_nbytes:
            assert err == GGUF_META_TABLE_OK
            assert out_ref[0] == expected_end
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_OVERFLOW)
            if err != GGUF_META_TABLE_OK:
                assert out_ref[0] == 0xFACE


if __name__ == "__main__":
    test_null_out_and_no_partial_write()
    test_buf_size_overflow_guard()
    test_zero_length_window_is_legal()
    test_cursor_domain_errors_passthrough()
    test_unsigned_add_overflow_guard()
    test_window_must_fit_table_and_buffer()
    test_success_and_fuzz()
    print("ok")
