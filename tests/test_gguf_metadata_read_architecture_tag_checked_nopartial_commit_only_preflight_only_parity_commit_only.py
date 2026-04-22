#!/usr/bin/env python3
"""Harness for IQ-1110 architecture commit-only wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_gguf_metadata_read_architecture_tag_checked_nopartial import (
    GGUF_META_PARSE_ERR_NOT_FOUND,
    GGUF_META_TABLE_ERR_BAD_PARAM,
    GGUF_META_TABLE_ERR_NULL_PTR,
    GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
    GGUF_META_TABLE_OK,
    GGUF_TYPE_STRING,
    _encode_kv,
)
from test_gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only import (
    gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference,
)
from test_gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity import (
    gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference,
)


def gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
    *,
    buf: bytes | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    metadata_count: int,
    out_arch_tag_ref: list[int] | None,
    out_tag_offset_ref: list[int] | None,
    out_tag_len_ref: list[int] | None,
    out_next_cursor_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_arch_tag_ref is None
        or out_tag_offset_ref is None
        or out_tag_len_ref is None
        or out_next_cursor_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if (
        cursor_ref is out_arch_tag_ref
        or cursor_ref is out_tag_offset_ref
        or cursor_ref is out_tag_len_ref
        or cursor_ref is out_next_cursor_ref
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if (
        out_arch_tag_ref is out_tag_offset_ref
        or out_arch_tag_ref is out_tag_len_ref
        or out_arch_tag_ref is out_next_cursor_ref
        or out_tag_offset_ref is out_tag_len_ref
        or out_tag_offset_ref is out_next_cursor_ref
        or out_tag_len_ref is out_next_cursor_ref
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    buf_snapshot = buf
    buf_nbytes_snapshot = buf_nbytes
    table_end_snapshot = table_end
    metadata_count_snapshot = metadata_count

    cursor_snapshot = cursor_ref[0]

    cursor_ref_snapshot = cursor_ref
    out_arch_tag_ref_snapshot = out_arch_tag_ref
    out_tag_offset_ref_snapshot = out_tag_offset_ref
    out_tag_len_ref_snapshot = out_tag_len_ref
    out_next_cursor_ref_snapshot = out_next_cursor_ref

    staged_arch = [0]
    staged_off = [0]
    staged_len = [0]
    staged_next = [0]
    staged_cursor = [cursor_snapshot]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=staged_cursor,
        table_end=table_end,
        metadata_count=metadata_count,
        out_arch_tag_ref=staged_arch,
        out_tag_offset_ref=staged_off,
        out_tag_len_ref=staged_len,
        out_next_cursor_ref=staged_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc

    if staged_cursor[0] != staged_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if staged_len[0] > buf_nbytes - staged_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    staged_end = staged_off[0] + staged_len[0]
    if staged_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if staged_end != staged_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_next[0] > buf_nbytes or staged_next[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    canonical_arch = [0]
    canonical_off = [0]
    canonical_len = [0]
    canonical_next = [0]
    canonical_cursor = [cursor_snapshot]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=canonical_cursor,
        table_end=table_end,
        metadata_count=metadata_count,
        out_arch_tag=canonical_arch,
        out_tag_offset=canonical_off,
        out_tag_len=canonical_len,
        out_next_cursor=canonical_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc

    if canonical_cursor[0] != canonical_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if canonical_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if canonical_len[0] > buf_nbytes - canonical_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    canonical_end = canonical_off[0] + canonical_len[0]
    if canonical_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if canonical_end != canonical_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if canonical_next[0] > buf_nbytes or canonical_next[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    if (
        buf_snapshot is not buf
        or buf_nbytes_snapshot != buf_nbytes
        or table_end_snapshot != table_end
        or metadata_count_snapshot != metadata_count
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if (
        cursor_ref_snapshot is not cursor_ref
        or out_arch_tag_ref_snapshot is not out_arch_tag_ref
        or out_tag_offset_ref_snapshot is not out_tag_offset_ref
        or out_tag_len_ref_snapshot is not out_tag_len_ref
        or out_next_cursor_ref_snapshot is not out_next_cursor_ref
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if staged_arch[0] != canonical_arch[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_off[0] != canonical_off[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_len[0] != canonical_len[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_next[0] != canonical_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if staged_len[0] <= 0:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if staged_off[0] > staged_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_arch_tag_ref[0] = staged_arch[0]
    out_tag_offset_ref[0] = staged_off[0]
    out_tag_len_ref[0] = staged_len[0]
    out_next_cursor_ref[0] = staged_next[0]
    cursor_ref[0] = staged_next[0]
    return GGUF_META_TABLE_OK


def _make_blob(entries: list[tuple[bytes, bytes]]) -> bytes:
    blob = bytearray()
    for key, value in entries:
        blob.extend(_encode_kv(key, GGUF_TYPE_STRING, value))
    return bytes(blob)


def test_success_publishes_once() -> None:
    buf = _make_blob(
        [
            (b"general.architecture", b"llama"),
            (b"architecture", b"phi3"),
        ]
    )

    cursor = [0]
    out_arch = [0x11]
    out_off = [0x22]
    out_len = [0x33]
    out_next = [0x44]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=2,
        out_arch_tag_ref=out_arch,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=out_len,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_TABLE_OK
    assert out_arch[0] == 1
    assert bytes(buf[out_off[0] : out_off[0] + out_len[0]]) == b"llama"
    assert cursor[0] == out_next[0]


def test_missing_key_no_partial_publish() -> None:
    buf = _make_blob([(b"foo", b"bar")])

    cursor = [0]
    out_arch = [0x51]
    out_off = [0x52]
    out_len = [0x53]
    out_next = [0x54]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=1,
        out_arch_tag_ref=out_arch,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=out_len,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_PARSE_ERR_NOT_FOUND
    assert cursor == [0]
    assert out_arch == [0x51]
    assert out_off == [0x52]
    assert out_len == [0x53]
    assert out_next == [0x54]


def test_duplicate_same_priority_rejected_no_partial_publish() -> None:
    buf = _make_blob(
        [
            (b"architecture", b"llama"),
            (b"architecture", b"phi3"),
        ]
    )

    cursor = [0]
    out_arch = [9]
    out_off = [8]
    out_len = [7]
    out_next = [6]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=2,
        out_arch_tag_ref=out_arch,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=out_len,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_arch == [9]
    assert out_off == [8]
    assert out_len == [7]
    assert out_next == [6]


def test_truncated_span_rejected_no_partial_publish() -> None:
    buf = _make_blob([(b"general.architecture", b"llama")])

    cursor = [0]
    out_arch = [101]
    out_off = [102]
    out_len = [103]
    out_next = [104]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf) - 1,
        metadata_count=1,
        out_arch_tag_ref=out_arch,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=out_len,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_arch == [101]
    assert out_off == [102]
    assert out_len == [103]
    assert out_next == [104]


def test_source_contains_iq1110_signature_and_chain() -> None:
    src = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert src.count(sig) == 1
    body = src.split(sig, 1)[1].split("Bool GGUFMetaCanRead(", 1)[0]

    assert "GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (cursor_ptr_snapshot != cursor ||" in body
    assert "if (tag_end_staged > table_end)" in body
    assert "if (tag_end_canonical > table_end)" in body
    assert "if (arch_staged != arch_canonical)" in body
    assert "if (tag_len_staged <= 0)" in body


def run() -> None:
    test_success_publishes_once()
    test_missing_key_no_partial_publish()
    test_duplicate_same_priority_rejected_no_partial_publish()
    test_truncated_span_rejected_no_partial_publish()
    test_source_contains_iq1110_signature_and_chain()
    print("gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok")


if __name__ == "__main__":
    run()
