#!/usr/bin/env python3
"""Harness for GGUFMetadataFindKeySpanCheckedNoPartialCommitOnlyPreflightOnly (IQ-1118)."""

from __future__ import annotations

from pathlib import Path

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4
GGUF_META_PARSE_ERR_NOT_FOUND = 9

GGUF_MAX_METADATA_COUNT = 1 << 20
GGUF_MAX_STRING_BYTES = 1 << 20
I64_MAX = (1 << 63) - 1

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_STRING = 8


def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "little", signed=False)


def _u64(x: int) -> bytes:
    return int(x).to_bytes(8, "little", signed=False)


def _kv_string(key: str, value: str) -> bytes:
    key_b = key.encode("ascii")
    value_b = value.encode("ascii")
    return _u64(len(key_b)) + key_b + _u32(GGUF_TYPE_STRING) + _u64(len(value_b)) + value_b


def _kv_u32(key: str, value: int) -> bytes:
    key_b = key.encode("ascii")
    return _u64(len(key_b)) + key_b + _u32(GGUF_TYPE_UINT32) + _u32(value)


def _type_supported(value_type: int) -> bool:
    return 0 <= value_type <= 12


def _skip_value_payload(buf: bytes, buf_nbytes: int, cur: int, table_end: int, value_type: int) -> tuple[int, int]:
    if value_type == GGUF_TYPE_UINT32:
        nxt = cur + 4
        if nxt > table_end or nxt > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, cur
        return GGUF_META_TABLE_OK, nxt

    if value_type == GGUF_TYPE_STRING:
        hdr_end = cur + 8
        if hdr_end > table_end or hdr_end > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, cur
        str_len = int.from_bytes(buf[cur:hdr_end], "little")
        if str_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM, cur
        nxt = hdr_end + str_len
        if nxt > table_end or nxt > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, cur
        return GGUF_META_TABLE_OK, nxt

    return GGUF_META_TABLE_ERR_BAD_PARAM, cur


def _base_find_key_span(
    buf: bytes | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    metadata_count: int,
    target_key_bytes: bytes | None,
    target_key_len: int,
    out_key_offset: list[int] | None,
    out_key_len: list[int] | None,
    out_value_type: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or target_key_bytes is None
        or out_key_offset is None
        or out_key_len is None
        or out_value_type is None
        or out_next_cursor is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if metadata_count > GGUF_MAX_METADATA_COUNT:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if target_key_len <= 0 or target_key_len > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    scan = cursor_ref[0]
    if scan > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    found: tuple[int, int, int, int] | None = None

    for _ in range(metadata_count):
        key_len_end = scan + 8
        if key_len_end > table_end or key_len_end > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        key_len = int.from_bytes(buf[scan:key_len_end], "little")
        if key_len <= 0 or key_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        key_off = key_len_end
        key_end = key_off + key_len
        if key_end > table_end or key_end > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

        key_match = False
        if key_len == target_key_len:
            key_match = buf[key_off:key_end] == target_key_bytes

        type_end = key_end + 4
        if type_end > table_end or type_end > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        value_type = int.from_bytes(buf[key_end:type_end], "little")
        if not _type_supported(value_type):
            return GGUF_META_TABLE_ERR_BAD_PARAM

        err, next_cursor = _skip_value_payload(buf, buf_nbytes, type_end, table_end, value_type)
        if err != GGUF_META_TABLE_OK:
            return err

        if key_match:
            if found is not None:
                return GGUF_META_TABLE_ERR_BAD_PARAM
            found = (key_off, key_len, value_type, next_cursor)

        scan = next_cursor

    if found is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND

    out_key_offset[0], out_key_len[0], out_value_type[0], out_next_cursor[0] = found
    cursor_ref[0] = found[3]
    return GGUF_META_TABLE_OK


def gguf_metadata_find_key_span_checked_nopartial_commit_only_preflight_only_reference(
    buf: bytes | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    metadata_count: int,
    target_key_bytes: bytes | None,
    target_key_len: int,
    out_key_offset: list[int] | None,
    out_key_len: list[int] | None,
    out_value_type: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or target_key_bytes is None
        or out_key_offset is None
        or out_key_len is None
        or out_value_type is None
        or out_next_cursor is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if buf_nbytes > I64_MAX or table_end > I64_MAX or metadata_count > I64_MAX or target_key_len > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if metadata_count > GGUF_MAX_METADATA_COUNT:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if target_key_len <= 0 or target_key_len > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if table_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if cursor_ref[0] > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if cursor_ref[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if cursor_ref[0] > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    snapshot_cursor = cursor_ref[0]

    stage_cursor = [snapshot_cursor]
    stage_key_off = [0]
    stage_key_len = [0]
    stage_value_type = [0]
    stage_next_cursor = [0]
    err = _base_find_key_span(
        buf,
        buf_nbytes,
        stage_cursor,
        table_end,
        metadata_count,
        target_key_bytes,
        target_key_len,
        stage_key_off,
        stage_key_len,
        stage_value_type,
        stage_next_cursor,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if stage_cursor[0] != stage_next_cursor[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if stage_key_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if stage_key_len[0] > buf_nbytes - stage_key_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if stage_key_len[0] != target_key_len:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    stage_key_end = stage_key_off[0] + stage_key_len[0]
    if stage_key_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if stage_next_cursor[0] < stage_key_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if buf[stage_key_off[0] : stage_key_off[0] + stage_key_len[0]] != target_key_bytes:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    verify_cursor = [snapshot_cursor]
    verify_key_off = [0]
    verify_key_len = [0]
    verify_value_type = [0]
    verify_next_cursor = [0]
    err = _base_find_key_span(
        buf,
        buf_nbytes,
        verify_cursor,
        table_end,
        metadata_count,
        target_key_bytes,
        target_key_len,
        verify_key_off,
        verify_key_len,
        verify_value_type,
        verify_next_cursor,
    )
    if err != GGUF_META_TABLE_OK:
        return err

    if verify_cursor[0] != verify_next_cursor[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if verify_key_len[0] != target_key_len:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if (
        stage_key_off[0] != verify_key_off[0]
        or stage_key_len[0] != verify_key_len[0]
        or stage_value_type[0] != verify_value_type[0]
        or stage_next_cursor[0] != verify_next_cursor[0]
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_key_offset[0] = stage_key_off[0]
    out_key_len[0] = stage_key_len[0]
    out_value_type[0] = stage_value_type[0]
    out_next_cursor[0] = stage_next_cursor[0]
    cursor_ref[0] = stage_next_cursor[0]
    return GGUF_META_TABLE_OK


def test_success_exact_key_tuple_published_once() -> None:
    blob = (
        _kv_string("general.name", "tiny")
        + _kv_string("general.architecture", "llama")
        + _kv_u32("llama.block_count", 32)
    )
    cursor = [0]
    key_off = [777]
    key_len = [777]
    value_type = [777]
    next_cursor = [777]

    rc = gguf_metadata_find_key_span_checked_nopartial_commit_only_preflight_only_reference(
        blob,
        len(blob),
        cursor,
        len(blob),
        3,
        b"general.architecture",
        len(b"general.architecture"),
        key_off,
        key_len,
        value_type,
        next_cursor,
    )

    assert rc == GGUF_META_TABLE_OK
    assert blob[key_off[0] : key_off[0] + key_len[0]] == b"general.architecture"
    assert value_type[0] == GGUF_TYPE_STRING
    assert cursor[0] == next_cursor[0]


def test_missing_key_preserves_outputs_and_cursor() -> None:
    blob = _kv_u32("llama.layers", 24) + _kv_string("general.name", "tiny")
    cursor = [0]
    key_off = [10]
    key_len = [11]
    value_type = [12]
    next_cursor = [13]

    rc = gguf_metadata_find_key_span_checked_nopartial_commit_only_preflight_only_reference(
        blob,
        len(blob),
        cursor,
        len(blob),
        2,
        b"general.architecture",
        len(b"general.architecture"),
        key_off,
        key_len,
        value_type,
        next_cursor,
    )

    assert rc == GGUF_META_PARSE_ERR_NOT_FOUND
    assert cursor[0] == 0
    assert key_off[0] == 10
    assert key_len[0] == 11
    assert value_type[0] == 12
    assert next_cursor[0] == 13


def test_duplicate_key_rejected_no_partial_publish() -> None:
    blob = _kv_string("general.architecture", "llama") + _kv_string("general.architecture", "mistral")
    cursor = [0]
    key_off = [1]
    key_len = [2]
    value_type = [3]
    next_cursor = [4]

    rc = gguf_metadata_find_key_span_checked_nopartial_commit_only_preflight_only_reference(
        blob,
        len(blob),
        cursor,
        len(blob),
        2,
        b"general.architecture",
        len(b"general.architecture"),
        key_off,
        key_len,
        value_type,
        next_cursor,
    )

    assert rc == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor[0] == 0
    assert key_off[0] == 1
    assert key_len[0] == 2
    assert value_type[0] == 3
    assert next_cursor[0] == 4


def test_truncated_buffer_rejected_no_partial_publish() -> None:
    full = _kv_string("general.architecture", "llama") + _kv_u32("x", 1)
    blob = full[:-2]
    cursor = [0]
    key_off = [100]
    key_len = [101]
    value_type = [102]
    next_cursor = [103]

    rc = gguf_metadata_find_key_span_checked_nopartial_commit_only_preflight_only_reference(
        blob,
        len(blob),
        cursor,
        len(blob),
        2,
        b"general.architecture",
        len(b"general.architecture"),
        key_off,
        key_len,
        value_type,
        next_cursor,
    )

    assert rc == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor[0] == 0
    assert key_off[0] == 100
    assert key_len[0] == 101
    assert value_type[0] == 102
    assert next_cursor[0] == 103


def test_source_contains_iq1118_signature_and_checks() -> None:
    src = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")
    assert "I32 GGUFMetadataFindKeySpanCheckedNoPartialCommitOnlyPreflightOnly(U8 *buf" in src
    assert "GGUFMetadataFindKeySpanCheckedNoPartial(buf," in src
    assert "if (key_len_staged != target_key_len)" in src
    assert "if (key_off_staged != key_off_canonical)" in src


if __name__ == "__main__":
    test_success_exact_key_tuple_published_once()
    test_missing_key_preserves_outputs_and_cursor()
    test_duplicate_key_rejected_no_partial_publish()
    test_truncated_buffer_rejected_no_partial_publish()
    test_source_contains_iq1118_signature_and_checks()
    print("ok")
