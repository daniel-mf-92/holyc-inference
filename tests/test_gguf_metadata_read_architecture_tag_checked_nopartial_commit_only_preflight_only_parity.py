#!/usr/bin/env python3
"""Harness for IQ-1109 architecture preflight parity gate."""

from __future__ import annotations

from pathlib import Path

from test_gguf_metadata_read_architecture_tag_checked_nopartial import (
    GGUF_META_PARSE_ERR_NOT_FOUND,
    GGUF_META_TABLE_ERR_BAD_PARAM,
    GGUF_META_TABLE_ERR_NULL_PTR,
    GGUF_META_TABLE_OK,
    GGUF_TYPE_STRING,
    _encode_kv,
    gguf_metadata_read_architecture_tag_checked_nopartial_reference,
)
from test_gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only import (
    gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference,
)


def gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
    *,
    buf: list[int] | None,
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

    cursor_snapshot = cursor_ref[0]

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
        out_arch_tag_ref=pre_arch,
        out_tag_offset_ref=pre_off,
        out_tag_len_ref=pre_len,
        out_next_cursor_ref=pre_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc
    if pre_cursor[0] != pre_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    can_arch = [0]
    can_off = [0]
    can_len = [0]
    can_next = [0]
    can_cursor = [cursor_snapshot]
    rc = gguf_metadata_read_architecture_tag_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=buf_nbytes,
        cursor_ref=can_cursor,
        table_end=table_end,
        metadata_count=metadata_count,
        out_arch_tag_ref=can_arch,
        out_tag_offset_ref=can_off,
        out_tag_len_ref=can_len,
        out_next_cursor_ref=can_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc
    if can_cursor[0] != can_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if pre_arch[0] != can_arch[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_off[0] != can_off[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_len[0] != can_len[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_next[0] != can_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_arch_tag_ref[0] = pre_arch[0]
    out_tag_offset_ref[0] = pre_off[0]
    out_tag_len_ref[0] = pre_len[0]
    out_next_cursor_ref[0] = pre_next[0]
    cursor_ref[0] = pre_next[0]
    return GGUF_META_TABLE_OK


def _make_blob(entries: list[tuple[bytes, bytes]]) -> list[int]:
    blob = bytearray()
    for key, value in entries:
        blob.extend(_encode_kv(key, GGUF_TYPE_STRING, value))
    return list(blob)


def test_parity_success_commits_tuple_once() -> None:
    buf = _make_blob([
        (b"general.architecture", b"llama"),
        (b"architecture", b"phi3"),
    ])

    cursor = [0]
    out_arch = [99]
    out_off = [88]
    out_len = [77]
    out_next = [66]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
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


def test_missing_key_is_not_found_without_publish() -> None:
    buf = _make_blob([(b"foo", b"bar")])

    cursor = [0]
    out_arch = [5]
    out_off = [6]
    out_len = [7]
    out_next = [8]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
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
    assert out_arch == [5]
    assert out_off == [6]
    assert out_len == [7]
    assert out_next == [8]


def test_duplicate_same_priority_key_rejected_without_publish() -> None:
    buf = _make_blob([
        (b"architecture", b"llama"),
        (b"architecture", b"phi3"),
    ])

    cursor = [0]
    out_arch = [10]
    out_off = [11]
    out_len = [12]
    out_next = [13]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
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
    assert out_arch == [10]


def test_source_contains_iq1109_parity_function() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "gguf" / "metadata.HC"
    body = src.read_text(encoding="utf-8")

    assert "I32 GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartial(buf," in body


def run() -> None:
    test_parity_success_commits_tuple_once()
    test_missing_key_is_not_found_without_publish()
    test_duplicate_same_priority_key_rejected_without_publish()
    test_source_contains_iq1109_parity_function()
    print("gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
