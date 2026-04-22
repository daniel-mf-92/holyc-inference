#!/usr/bin/env python3
"""Harness for IQ-1109 architecture preflight parity gate."""

from __future__ import annotations

from pathlib import Path

from test_gguf_metadata_read_architecture_tag_checked_nopartial import (
    GGUF_META_PARSE_ERR_NOT_FOUND,
    GGUF_META_TABLE_ERR_BAD_PARAM,
    GGUF_META_TABLE_ERR_NULL_PTR,
    GGUF_META_TABLE_ERR_OUT_OF_BOUNDS,
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

    cursor_ref_snapshot = cursor_ref
    out_arch_tag_ref_snapshot = out_arch_tag_ref
    out_tag_offset_ref_snapshot = out_tag_offset_ref
    out_tag_len_ref_snapshot = out_tag_len_ref
    out_next_cursor_ref_snapshot = out_next_cursor_ref

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
        out_arch_tag=pre_arch,
        out_tag_offset=pre_off,
        out_tag_len=pre_len,
        out_next_cursor=pre_next,
    )
    if rc != GGUF_META_TABLE_OK:
        return rc
    if pre_cursor[0] != pre_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_len[0] <= 0:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if pre_off[0] > pre_next[0]:
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


def _make_blob(entries: list[tuple[bytes, bytes]]) -> bytes:
    blob = bytearray()
    for key, value in entries:
        blob.extend(_encode_kv(key, GGUF_TYPE_STRING, value))
    return bytes(blob)


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


def test_span_overflow_rejected_without_publish() -> None:
    # Table end truncates the architecture payload span; parity wrapper must
    # reject and keep all caller-visible state unchanged.
    buf = _make_blob([
        (b"general.architecture", b"llama"),
    ])

    cursor = [0]
    out_arch = [31]
    out_off = [32]
    out_len = [33]
    out_next = [34]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
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
    assert out_arch == [31]
    assert out_off == [32]
    assert out_len == [33]
    assert out_next == [34]


def test_cursor_gt_buf_nbytes_rejected_without_publish() -> None:
    buf = _make_blob([
        (b"general.architecture", b"llama"),
    ])

    cursor = [len(buf) + 1]
    out_arch = [41]
    out_off = [42]
    out_len = [43]
    out_next = [44]

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

    assert rc == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [len(buf) + 1]
    assert out_arch == [41]
    assert out_off == [42]
    assert out_len == [43]
    assert out_next == [44]


def test_source_contains_iq1109_parity_function() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "gguf" / "metadata.HC"
    body = src.read_text(encoding="utf-8")

    assert "I32 GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartial(buf," in body
    assert "if (*cursor > buf_nbytes)" in body
    assert "if ((U8 *)cursor == (U8 *)out_arch_tag" in body
    assert "buf_snapshot = buf;" in body
    assert "if (cursor_ptr_snapshot != cursor" in body
    assert "if (tag_end_preflight > table_end)" in body
    assert "if (tag_end_canonical > table_end)" in body


def test_output_aliases_are_rejected_without_publish() -> None:
    buf = _make_blob([
        (b"general.architecture", b"llama"),
    ])

    shared = [123]
    cursor = [0]
    out_off = [55]
    out_len = [66]
    out_next = [77]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=1,
        out_arch_tag_ref=shared,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=shared,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert shared == [123]
    assert out_off == [55]
    assert out_next == [77]


def run() -> None:
    test_parity_success_commits_tuple_once()
    test_missing_key_is_not_found_without_publish()
    test_duplicate_same_priority_key_rejected_without_publish()
    test_span_overflow_rejected_without_publish()
    test_cursor_gt_buf_nbytes_rejected_without_publish()
    test_source_contains_iq1109_parity_function()
    test_output_aliases_are_rejected_without_publish()
    print("gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
