#!/usr/bin/env python3
"""Harness for IQ-1119 architecture parity-preflight parity gate."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from test_gguf_metadata_read_architecture_tag_checked_nopartial import (
    GGUF_META_TABLE_ERR_BAD_PARAM,
    GGUF_META_TABLE_ERR_NULL_PTR,
    GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
    GGUF_META_TABLE_ERR_TYPE_MISMATCH,
    GGUF_META_TABLE_OK,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT8,
    _encode_kv,
)
from test_gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only import (
    GGUF_META_TABLE_ERR_TYPE_MISMATCH as GGUF_META_TABLE_ERR_TYPE_MISMATCH_CHAIN,
    gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference,
)
from test_gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only import (
    gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference,
)


def gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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

    pre_arch = [0]
    pre_off = [0]
    pre_len = [0]
    pre_next = [0]
    pre_cursor = [cursor_snapshot]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=pre_cursor,
        table_end=table_end,
        metadata_count=metadata_count,
        out_arch_tag=pre_arch,
        out_tag_offset=pre_off,
        out_tag_len=pre_len,
        out_next_cursor=pre_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc

    if pre_cursor[0] != pre_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if pre_len[0] > buf_nbytes - pre_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    pre_end = pre_off[0] + pre_len[0]
    if pre_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if pre_end != pre_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_next[0] > buf_nbytes or pre_next[0] > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    commit_arch = [0]
    commit_off = [0]
    commit_len = [0]
    commit_next = [0]
    commit_cursor = [cursor_snapshot]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=commit_cursor,
        table_end=table_end,
        metadata_count=metadata_count,
        out_arch_tag_ref=commit_arch,
        out_tag_offset_ref=commit_off,
        out_tag_len_ref=commit_len,
        out_next_cursor_ref=commit_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc

    if commit_cursor[0] != commit_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if commit_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if commit_len[0] > buf_nbytes - commit_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

    commit_end = commit_off[0] + commit_len[0]
    if commit_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if commit_end != commit_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if commit_next[0] > buf_nbytes or commit_next[0] > table_end:
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

    if pre_arch[0] != commit_arch[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_off[0] != commit_off[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_len[0] != commit_len[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_next[0] != commit_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if pre_len[0] <= 0:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_off[0] > pre_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_arch_tag_ref[0] = pre_arch[0]
    out_tag_offset_ref[0] = pre_off[0]
    out_tag_len_ref[0] = pre_len[0]
    out_next_cursor_ref[0] = pre_next[0]
    cursor_ref[0] = pre_next[0]
    return GGUF_META_TABLE_OK


def _make_blob(entries: list[tuple[bytes, int, bytes | int]]) -> bytes:
    blob = bytearray()
    for key, value_type, value in entries:
        blob.extend(_encode_kv(key, value_type, value))
    return bytes(blob)


def test_success_publishes_from_preflight_tuple() -> None:
    buf = _make_blob(
        [
            (b"architecture", GGUF_TYPE_STRING, b"phi3"),
            (b"general.architecture", GGUF_TYPE_STRING, b"llama"),
        ]
    )

    cursor = [0]
    out_arch = [77]
    out_off = [66]
    out_len = [55]
    out_next = [44]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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


def test_duplicate_key_rejected_without_publish() -> None:
    buf = _make_blob(
        [
            (b"architecture", GGUF_TYPE_STRING, b"llama"),
            (b"architecture", GGUF_TYPE_STRING, b"phi3"),
        ]
    )

    cursor = [0]
    out_arch = [9]
    out_off = [8]
    out_len = [7]
    out_next = [6]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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


def test_type_mismatch_rejected_without_publish() -> None:
    buf = _make_blob(
        [
            (b"general.architecture", GGUF_TYPE_UINT8, 3),
        ]
    )

    cursor = [0]
    out_arch = [12]
    out_off = [13]
    out_len = [14]
    out_next = [15]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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

    assert rc == GGUF_META_TABLE_ERR_TYPE_MISMATCH_CHAIN
    assert cursor == [0]
    assert out_arch == [12]
    assert out_off == [13]
    assert out_len == [14]
    assert out_next == [15]


def test_truncated_span_rejected_without_publish() -> None:
    buf = _make_blob(
        [
            (b"general.architecture", GGUF_TYPE_STRING, b"llama"),
        ]
    )

    cursor = [0]
    out_arch = [21]
    out_off = [22]
    out_len = [23]
    out_next = [24]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only_parity_reference(
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
    assert out_arch == [21]
    assert out_off == [22]
    assert out_len == [23]
    assert out_next == [24]


def test_source_contains_iq1119_function_and_chain() -> None:
    src = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")
    sig = (
        "I32 GGUFMetadataReadArchitectureTagCheckedNoPartial"
        "CommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    )
    assert src.count(sig) >= 1

    def_start = -1
    search_from = 0
    while True:
        idx = src.find(sig, search_from)
        if idx < 0:
            break
        brace_idx = src.find("{", idx)
        semi_idx = src.find(";", idx)
        if brace_idx >= 0 and (semi_idx < 0 or brace_idx < semi_idx):
            def_start = idx
            break
        search_from = idx + len(sig)

    assert def_start >= 0
    body = src[def_start:].split("Bool GGUFMetaCanRead(", 1)[0]

    assert "GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "if (cursor_ptr_snapshot != cursor ||" in body
    assert "if (arch_preflight != arch_parity_commit)" in body
    assert "if (tag_end_preflight > table_end)" in body
    assert "if (tag_end_parity_commit > table_end)" in body


def run() -> None:
    test_success_publishes_from_preflight_tuple()
    test_duplicate_key_rejected_without_publish()
    test_type_mismatch_rejected_without_publish()
    test_truncated_span_rejected_without_publish()
    test_source_contains_iq1119_function_and_chain()
    print(
        "gguf_metadata_read_architecture_tag_checked_nopartial_"
        "commit_only_preflight_only_parity_commit_only_preflight_only_parity=ok"
    )


if __name__ == "__main__":
    run()
