#!/usr/bin/env python3
"""Harness for ...ParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-1326)."""

from __future__ import annotations

from pathlib import Path
import random
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
)
from test_gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity import (
    gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference,
)
from test_gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only import (
    gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference,
)


def gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
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

    staged_commit_only_cursor = [snapshot_cursor]
    staged_commit_only_next = [0]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=staged_commit_only_cursor,
        table_end=table_end,
        value_type=value_type,
        out_next_cursor_ref=staged_commit_only_next,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if staged_commit_only_cursor[0] != staged_commit_only_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_commit_only_next[0] > buf_nbytes or staged_commit_only_next[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    staged_canonical_parity_cursor = [snapshot_cursor]
    staged_canonical_parity_next = [0]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=staged_canonical_parity_cursor,
        table_end=table_end,
        value_type=value_type,
        out_next_cursor_ref=staged_canonical_parity_next,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if staged_canonical_parity_cursor[0] != staged_canonical_parity_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_canonical_parity_next[0] > buf_nbytes or staged_canonical_parity_next[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    if staged_commit_only_next[0] != staged_canonical_parity_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if (
        snapshot_buf_nbytes != buf_nbytes
        or snapshot_table_end != table_end
        or snapshot_value_type != value_type
        or snapshot_cursor != cursor_ref[0]
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_next_cursor_ref[0] = staged_commit_only_next[0]
    cursor_ref[0] = staged_commit_only_next[0]
    return GGUF_META_TABLE_OK


def test_source_contains_iq1326_function_and_strict_publish() -> None:
    source = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")

    sig = "I32 GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(U8 *buf,"
    sig_def = (
        "I32 GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(U8 *buf,\n"
        "                                                                                                                           U64 buf_nbytes,\n"
        "                                                                                                                           U64 *cursor,\n"
        "                                                                                                                           U64 table_end,\n"
        "                                                                                                                           U32 value_type,\n"
        "                                                                                                                           U64 *out_next_cursor)\n"
        "{"
    )
    assert sig in source
    assert sig_def in source

    body = source.split(sig_def, 1)[1].split("// Backward-compatible alias retained", 1)[0]
    assert "GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "GGUFMetadataValueSkipCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity(" in body
    assert "snapshot_cursor = *cursor;" in body
    assert "staged_commit_only_next_cursor != staged_canonical_parity_next_cursor" in body
    assert "*out_next_cursor = staged_commit_only_next_cursor;" in body
    assert "*cursor = staged_commit_only_next_cursor;" in body


def test_null_ptr_contracts() -> None:
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=None,
        buf_nbytes=0,
        cursor_ref=[0],
        table_end=0,
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=[0],
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=[0],
        buf_nbytes=1,
        cursor_ref=None,
        table_end=1,
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=[0],
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=[0],
        buf_nbytes=1,
        cursor_ref=[0],
        table_end=1,
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=None,
    )
    assert err == GGUF_META_TABLE_ERR_NULL_PTR


def test_scalar_success_and_publish() -> None:
    buf = list(range(64))
    cursor = [10]
    out_next = [777]

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_UINT16,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 12
    assert out_next[0] == 12


def test_bad_type_no_partial_publish() -> None:
    buf = [0] * 64
    cursor = [8]
    out_next = [0xA5A5]

    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=0xFFFF,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 8
    assert out_next[0] == 0xA5A5


def test_array_string_span_success_and_truncation() -> None:
    payload = _u32le(GGUF_TYPE_STRING) + _u64le(2) + _string_payload(b"truth") + _string_payload(b"os")
    buf = [0] * 4 + list(payload) + [0] * 5

    cursor = [4]
    out_next = [0]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        value_type=GGUF_TYPE_ARRAY,
        out_next_cursor_ref=out_next,
    )
    assert err == GGUF_META_TABLE_OK
    assert cursor[0] == 4 + len(payload)
    assert out_next[0] == 4 + len(payload)

    truncated = buf[: 4 + len(payload) - 1]
    cursor2 = [4]
    out_next2 = [0xBEEF]
    err = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
        buf=truncated,
        buf_nbytes=len(truncated),
        cursor_ref=cursor2,
        table_end=len(truncated),
        value_type=GGUF_TYPE_ARRAY,
        out_next_cursor_ref=out_next2,
    )
    assert err != GGUF_META_TABLE_OK
    assert cursor2[0] == 4
    assert out_next2[0] == 0xBEEF


def test_randomized_parity_vectors() -> None:
    rng = random.Random(20260424_1326)

    for _ in range(1200):
        kind = rng.randrange(4)

        if kind == 0:
            buf = [rng.randrange(256) for _ in range(96)]
            value_type = GGUF_TYPE_UINT16
            cursor0 = rng.randrange(0, 94)
        elif kind == 1:
            n = rng.randrange(0, 16)
            s = bytes(rng.randrange(97, 123) for _ in range(n))
            payload = _string_payload(s)
            prefix = [rng.randrange(256) for _ in range(rng.randrange(0, 12))]
            suffix = [rng.randrange(256) for _ in range(rng.randrange(0, 8))]
            buf = prefix + list(payload) + suffix
            value_type = GGUF_TYPE_STRING
            cursor0 = len(prefix)
            if rng.random() < 0.2 and len(buf) > cursor0:
                buf = buf[:-1]
        elif kind == 2:
            elem_n = rng.randrange(0, 10)
            payload = _u32le(GGUF_TYPE_UINT16) + _u64le(elem_n)
            payload += b"".join(_u64le(rng.randrange(65536))[:2] for _ in range(elem_n))
            prefix = [0] * rng.randrange(0, 9)
            buf = prefix + list(payload)
            value_type = GGUF_TYPE_ARRAY
            cursor0 = len(prefix)
            if rng.random() < 0.2 and len(buf) > cursor0:
                buf = buf[:-1]
        else:
            elem_n = rng.randrange(0, 6)
            strings = [bytes(rng.randrange(97, 123) for _ in range(rng.randrange(0, 6))) for _ in range(elem_n)]
            payload = _u32le(GGUF_TYPE_STRING) + _u64le(elem_n) + b"".join(_string_payload(s) for s in strings)
            prefix = [rng.randrange(256) for _ in range(rng.randrange(0, 5))]
            buf = prefix + list(payload)
            value_type = GGUF_TYPE_ARRAY
            cursor0 = len(prefix)
            if rng.random() < 0.2 and len(buf) > cursor0:
                buf = buf[:-1]

        cursor_a = [cursor0]
        out_a = [0xAA]
        status_a = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_reference(
            buf=buf,
            buf_nbytes=len(buf),
            cursor_ref=cursor_a,
            table_end=len(buf),
            value_type=value_type,
            out_next_cursor_ref=out_a,
        )

        cursor_b = [cursor0]
        out_b = [0xBB]
        status_b = gguf_metadata_value_skip_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_reference(
            buf=buf,
            buf_nbytes=len(buf),
            cursor_ref=cursor_b,
            table_end=len(buf),
            value_type=value_type,
            out_next_cursor_ref=out_b,
        )

        assert status_a == status_b
        if status_a == GGUF_META_TABLE_OK:
            assert cursor_a[0] == cursor_b[0]
            assert out_a[0] == out_b[0]
        else:
            assert cursor_a[0] == cursor0
            assert out_a[0] == 0xAA


if __name__ == "__main__":
    test_source_contains_iq1326_function_and_strict_publish()
    test_null_ptr_contracts()
    test_scalar_success_and_publish()
    test_bad_type_no_partial_publish()
    test_array_string_span_success_and_truncation()
    test_randomized_parity_vectors()
    print("ok")
