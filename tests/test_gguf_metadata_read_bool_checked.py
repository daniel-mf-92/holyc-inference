#!/usr/bin/env python3
"""Reference checks for GGUFMetadataReadBoolChecked semantics."""

from __future__ import annotations

import random
from pathlib import Path

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

GGUF_TYPE_BOOL = 7

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

    next_cursor = cursor + need
    if next_cursor > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None

    return GGUF_META_TABLE_OK, next_cursor


def gguf_metadata_cursor_typed_span_bytes_checked(
    value_type: int,
    variable_payload_bytes: int,
) -> tuple[int, int | None]:
    if variable_payload_bytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    if value_type == GGUF_TYPE_BOOL:
        return GGUF_META_TABLE_OK, 1

    return GGUF_META_TABLE_ERR_BAD_PARAM, None


def gguf_metadata_cursor_advance_checked(
    cursor_ref: list[int] | None,
    table_end: int,
    value_type: int,
    variable_payload_bytes: int,
    out_span_start_ref: list[int] | None,
    out_span_bytes_ref: list[int] | None,
    out_span_end_ref: list[int] | None,
) -> int:
    if (
        cursor_ref is None
        or out_span_start_ref is None
        or out_span_bytes_ref is None
        or out_span_end_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cur = cursor_ref[0]
    if cur > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    err, span_bytes = gguf_metadata_cursor_typed_span_bytes_checked(
        value_type,
        variable_payload_bytes,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    assert span_bytes is not None
    err, span_end = gguf_metadata_cursor_can_advance_checked(cur, span_bytes, table_end)
    if err != GGUF_META_TABLE_OK:
        return err

    assert span_end is not None
    out_span_start_ref[0] = cur
    out_span_bytes_ref[0] = span_bytes
    out_span_end_ref[0] = span_end
    cursor_ref[0] = span_end
    return GGUF_META_TABLE_OK


def gguf_metadata_read_bool_checked(
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    out_value_ref: list[bool] | None,
) -> int:
    if buf is None or cursor_ref is None or out_value_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if buf_nbytes > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if table_end > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    cur = [cursor_ref[0]]
    if cur[0] > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    span_start = [0]
    span_bytes = [0]
    span_end = [0]

    err = gguf_metadata_cursor_advance_checked(
        cur,
        table_end,
        GGUF_TYPE_BOOL,
        0,
        span_start,
        span_bytes,
        span_end,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if span_start[0] > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if span_bytes[0] != 1:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if span_end[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    raw = buf[span_start[0]]
    if raw > 1:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_value_ref[0] = raw != 0
    cursor_ref[0] = span_end[0]
    return GGUF_META_TABLE_OK


def test_null_ptr_and_no_partial_write() -> None:
    cursor = [2]
    out = [True]

    assert (
        gguf_metadata_read_bool_checked(None, 8, cursor, 8, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 2
    assert out[0] is True

    assert (
        gguf_metadata_read_bool_checked([0, 1], 2, None, 2, out)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert out[0] is True

    assert (
        gguf_metadata_read_bool_checked([0, 1], 2, cursor, 2, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )
    assert cursor[0] == 2


def test_bounds_and_short_read_do_not_mutate() -> None:
    out = [False]

    cursor = [0]
    err = gguf_metadata_read_bool_checked([1], 1, cursor, 0, out)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 0
    assert out[0] is False

    cursor = [1]
    out = [True]
    err = gguf_metadata_read_bool_checked([0], 1, cursor, 1, out)
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 1
    assert out[0] is True


def test_overflow_guards_passthrough_from_typed_span_path() -> None:
    buf = [0, 1, 0, 1]

    cursor = [0]
    out = [True]
    err = gguf_metadata_read_bool_checked(buf, I64_MAX + 1, cursor, len(buf), out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out[0] is True

    cursor = [I64_MAX + 1]
    out = [False]
    err = gguf_metadata_read_bool_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == I64_MAX + 1
    assert out[0] is False

    cursor = [0]
    out = [False]
    err = gguf_metadata_read_bool_checked(buf, len(buf), cursor, I64_MAX + 1, out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor[0] == 0
    assert out[0] is False


def test_rejects_non_canonical_bool_values() -> None:
    for raw in (2, 3, 127, 255):
        cursor = [0]
        out = [False]
        err = gguf_metadata_read_bool_checked([raw], 1, cursor, 1, out)
        assert err == GGUF_META_TABLE_ERR_BAD_PARAM
        assert cursor[0] == 0
        assert out[0] is False


def test_reads_canonical_bool_values() -> None:
    buf = [9, 0, 1, 4]
    cursor = [1]
    out = [False]

    err = gguf_metadata_read_bool_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] is False
    assert cursor[0] == 2

    err = gguf_metadata_read_bool_checked(buf, len(buf), cursor, len(buf), out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] is True
    assert cursor[0] == 3


def test_cursor_greater_than_table_end_is_bad_param_and_no_partial() -> None:
    cursor = [5]
    out = [True]
    err = gguf_metadata_read_bool_checked([0] * 8, 8, cursor, 4, out)
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 5
    assert out[0] is True


def test_cursor_advances_exactly_one_on_success() -> None:
    rng = random.Random(20260423_1245)

    for _ in range(1000):
        n = rng.randint(1, 64)
        buf = [rng.randint(0, 1) for _ in range(n)]
        cursor0 = rng.randint(0, n - 1)
        cursor = [cursor0]
        out = [False]

        err = gguf_metadata_read_bool_checked(buf, n, cursor, n, out)
        assert err == GGUF_META_TABLE_OK
        assert cursor[0] == cursor0 + 1
        assert out[0] == (buf[cursor0] == 1)


def test_randomized_parity() -> None:
    rng = random.Random(20260417_199)

    for _ in range(5000):
        n = rng.randint(1, 128)
        buf = [rng.randint(0, 255) for _ in range(n)]
        table_end = rng.randint(0, n)
        cursor0 = rng.randint(0, table_end)

        cursor = [cursor0]
        out = [True]
        err = gguf_metadata_read_bool_checked(buf, n, cursor, table_end, out)

        if cursor0 + 1 <= table_end and cursor0 + 1 <= n:
            raw = buf[cursor0]
            if raw <= 1:
                assert err == GGUF_META_TABLE_OK
                assert out[0] == (raw == 1)
                assert cursor[0] == cursor0 + 1
            else:
                assert err == GGUF_META_TABLE_ERR_BAD_PARAM
                assert out[0] is True
                assert cursor[0] == cursor0
        else:
            assert err in (GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, GGUF_META_TABLE_ERR_BAD_PARAM)
            assert out[0] is True
            assert cursor[0] == cursor0


def test_holyc_uses_typed_bool_span_contract() -> None:
    source = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")

    sig = "I32 GGUFMetadataReadBoolChecked(U8 *buf,"
    start = source.rfind(sig)
    assert start >= 0

    next_sig = source.find("I32 GGUFMetadataReadBoolPairChecked(U8 *buf,", start)
    assert next_sig > start

    body = source[start:next_sig]
    assert "GGUFMetadataCursorAdvanceChecked" in body
    assert "GGUF_TYPE_BOOL" in body
    assert "span_bytes != 1" in body
    assert "raw_u8 > 1" in body


def run() -> None:
    test_null_ptr_and_no_partial_write()
    test_bounds_and_short_read_do_not_mutate()
    test_overflow_guards_passthrough_from_typed_span_path()
    test_rejects_non_canonical_bool_values()
    test_reads_canonical_bool_values()
    test_cursor_greater_than_table_end_is_bad_param_and_no_partial()
    test_cursor_advances_exactly_one_on_success()
    test_randomized_parity()
    test_holyc_uses_typed_bool_span_contract()
    print("gguf_metadata_read_bool_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
