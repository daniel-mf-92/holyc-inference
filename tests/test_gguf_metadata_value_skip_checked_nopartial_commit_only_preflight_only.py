#!/usr/bin/env python3
"""Harness for GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnly (IQ-1248)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_gguf_metadata_value_skip_checked import (
    GGUF_META_TABLE_ERR_BAD_PARAM,
    GGUF_META_TABLE_ERR_NULL_PTR,
    GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
    GGUF_META_TABLE_OK,
    GGUF_TYPE_ARRAY,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT16,
    _string_payload,
    _u32le,
    _u64le,
    gguf_metadata_value_skip_checked_reference,
)
from test_gguf_metadata_value_skip_checked_nopartial_commit_only import (
    gguf_metadata_value_skip_checked_nopartial_commit_only_reference,
)


def gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
    *,
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    value_type: int,
    out_next_cursor_ref: list[int] | None,
) -> int:
    if buf is None or cursor_ref is None or out_next_cursor_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    snapshot_buf_nbytes = buf_nbytes
    snapshot_table_end = table_end
    snapshot_cursor = cursor_ref[0]
    snapshot_value_type = value_type

    staged_commit_cursor = [snapshot_cursor]
    staged_commit_next_cursor = [0]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=staged_commit_cursor,
        table_end=table_end,
        value_type=value_type,
        out_next_cursor_ref=staged_commit_next_cursor,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if staged_commit_cursor[0] != staged_commit_next_cursor[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_commit_next_cursor[0] > buf_nbytes or staged_commit_next_cursor[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    staged_canonical_cursor = [snapshot_cursor]
    err = gguf_metadata_value_skip_checked_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=staged_canonical_cursor,
        table_end=table_end,
        value_type=value_type,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if staged_canonical_cursor[0] > buf_nbytes or staged_canonical_cursor[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    if staged_commit_next_cursor[0] != staged_canonical_cursor[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if (
        snapshot_buf_nbytes != buf_nbytes
        or snapshot_table_end != table_end
        or snapshot_value_type != value_type
        or snapshot_cursor != cursor_ref[0]
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_next_cursor_ref[0] = staged_commit_next_cursor[0]
    cursor_ref[0] = staged_commit_next_cursor[0]
    return GGUF_META_TABLE_OK


def test_source_contains_iq1248_function_and_dual_path_parity() -> None:
    source = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")

    sig = "I32 GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnly(U8 *buf,"
    sig_def = (
        "I32 GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnly(U8 *buf,\n"
        "                                                                 U64 buf_nbytes,\n"
        "                                                                 U64 *cursor,\n"
        "                                                                 U64 table_end,\n"
        "                                                                 U32 value_type,\n"
        "                                                                 U64 *out_next_cursor)\n"
        "{"
    )
    assert sig in source
    assert sig_def in source

    body = source.split(sig_def, 1)[1].split("// Backward-compatible alias retained", 1)[0]
    assert "GGUFMetadataValueSkipCheckedNoPartialCommitOnly(buf," in body
    assert "GGUFMetadataValueSkipChecked(buf," in body
    assert "snapshot_cursor = *cursor;" in body
    assert "staged_commit_next_cursor != staged_canonical_cursor" in body
    assert "*out_next_cursor = staged_commit_next_cursor;" in body
    assert "*cursor = staged_commit_next_cursor;" in body


def test_null_ptr_contracts() -> None:
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=None,
        buf_nbytes=0,
        cursor_ref=[0],
        table_end=0,
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=[0],
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=[0],
        buf_nbytes=1,
        cursor_ref=None,
        table_end=1,
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=[0],
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=[0],
        buf_nbytes=1,
        cursor_ref=[0],
        table_end=1,
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=None,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR


def test_scalar_success_and_strict_parity_publish() -> None:
    buf = list(range(128))
    cursor = [11]
    out_next = [777]

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 13
    assert out_next[0] == 13


def test_type_vector_bad_type_no_partial_publish() -> None:
    buf = [0] * 64
    cursor = [10]
    out_next = [888]

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=0xDEAD,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 10
    assert out_next[0] == 888


def test_bounds_vector_string_truncated_no_partial_publish() -> None:
    payload = _string_payload(b"temple")
    full = [0xAA] * 5 + list(payload)
    truncated = full[:-1]

    cursor = [5]
    out_next = [1234]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=truncated,
        buf_nbytes=len(truncated),
        cursor_ref=cursor,
        table_end=len(truncated),
        value_type=GGUF_TYPE_STRING,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 5
    assert out_next[0] == 1234


def test_array_span_parity_success() -> None:
    strings = [b"holy", b"c", b"os"]
    payload = _u32le(GGUF_TYPE_STRING) + _u64le(len(strings)) + b"".join(_string_payload(s) for s in strings)
    buf = [0] * 17 + list(payload) + [0] * 4

    cursor = [17]
    out_next = [0]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_ARRAY,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 17 + len(payload)
    assert out_next[0] == 17 + len(payload)


def test_array_span_bounds_error_no_partial_publish() -> None:
    strings = [b"abc", b"def"]
    payload = _u32le(GGUF_TYPE_STRING) + _u64le(len(strings)) + b"".join(_string_payload(s) for s in strings)
    buf = [0] * 13 + list(payload)
    truncated = buf[:-2]

    cursor = [13]
    out_next = [999]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=truncated,
        buf_nbytes=len(truncated),
        cursor_ref=cursor,
        table_end=len(truncated),
        value_type=GGUF_TYPE_ARRAY,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 13
    assert out_next[0] == 999


def test_allows_out_cursor_alias_with_cursor_pointer() -> None:
    buf = list(range(64))
    cursor_and_out = [4]

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor_and_out,
        table_end=len(buf),
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=cursor_and_out,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor_and_out[0] == 6
