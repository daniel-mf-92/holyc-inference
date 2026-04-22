#!/usr/bin/env python3
"""Harness for GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnly (IQ-1117)."""

from __future__ import annotations

from pathlib import Path

GGUF_META_PARSE_ERR_NOT_FOUND = 9

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4
GGUF_META_TABLE_ERR_TYPE_MISMATCH = 10

GGUF_TYPE_STRING = 8
GGUF_TYPE_UINT32 = 4

GGUF_ARCH_TAG_LLAMA = 1
GGUF_ARCH_TAG_MISTRAL = 2
GGUF_ARCH_TAG_QWEN2 = 3
GGUF_ARCH_TAG_PHI3 = 4

GGUF_MAX_METADATA_COUNT = 1 << 20
GGUF_MAX_STRING_BYTES = 1 << 20
I64_MAX = (1 << 63) - 1


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


def _canon_arch_tag(value: bytes) -> int | None:
    if value == b"llama":
        return GGUF_ARCH_TAG_LLAMA
    if value == b"mistral":
        return GGUF_ARCH_TAG_MISTRAL
    if value in (b"qwen2", b"qwen", b"qwen2.5"):
        return GGUF_ARCH_TAG_QWEN2
    if value in (b"phi3", b"phi-3"):
        return GGUF_ARCH_TAG_PHI3
    return None


def _key_priority(key: bytes) -> int | None:
    if key == b"general.architecture":
        return 0
    if key == b"architecture":
        return 1
    if key == b"model.architecture":
        return 2
    return None


def _read_arch_base(
    buf: bytes | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    metadata_count: int,
    out_arch_tag: list[int] | None,
    out_tag_offset: list[int] | None,
    out_tag_len: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_arch_tag is None
        or out_tag_offset is None
        or out_tag_len is None
        or out_next_cursor is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if metadata_count > GGUF_MAX_METADATA_COUNT:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if table_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if cursor_ref[0] > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    scan = cursor_ref[0]
    found = False
    found_prio = 2**63 - 1
    staged_arch = 0
    staged_off = 0
    staged_len = 0
    staged_next = 0

    for _ in range(metadata_count):
        if scan + 8 > table_end:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        key_len = int.from_bytes(buf[scan : scan + 8], "little")
        scan += 8
        if key_len <= 0 or key_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM
        if scan + key_len > table_end:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        key_off = scan
        key = buf[key_off : key_off + key_len]
        scan += key_len

        if scan + 4 > table_end:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        value_type = int.from_bytes(buf[scan : scan + 4], "little")
        scan += 4

        prio = _key_priority(key)
        if prio is None:
            if value_type == GGUF_TYPE_STRING:
                if scan + 8 > table_end:
                    return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
                val_len = int.from_bytes(buf[scan : scan + 8], "little")
                scan += 8
                if val_len > GGUF_MAX_STRING_BYTES:
                    return GGUF_META_TABLE_ERR_BAD_PARAM
                if scan + val_len > table_end:
                    return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
                scan += val_len
            elif value_type == GGUF_TYPE_UINT32:
                if scan + 4 > table_end:
                    return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
                scan += 4
            else:
                return GGUF_META_TABLE_ERR_BAD_PARAM
            continue

        if value_type != GGUF_TYPE_STRING:
            return GGUF_META_TABLE_ERR_TYPE_MISMATCH

        if scan + 8 > table_end:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
        val_len = int.from_bytes(buf[scan : scan + 8], "little")
        scan += 8
        if val_len <= 0 or val_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM
        if scan + val_len > table_end:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS

        val_off = scan
        val = buf[val_off : val_off + val_len]
        scan += val_len

        arch_tag = _canon_arch_tag(val)
        if arch_tag is None:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        if found and prio == found_prio:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        if (not found) or (prio < found_prio):
            found = True
            found_prio = prio
            staged_arch = arch_tag
            staged_off = val_off
            staged_len = val_len
            staged_next = scan

    if not found:
        return GGUF_META_PARSE_ERR_NOT_FOUND

    out_arch_tag[0] = staged_arch
    out_tag_offset[0] = staged_off
    out_tag_len[0] = staged_len
    out_next_cursor[0] = staged_next
    cursor_ref[0] = staged_next
    return GGUF_META_TABLE_OK


def gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
    buf: bytes | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    metadata_count: int,
    out_arch_tag: list[int] | None,
    out_tag_offset: list[int] | None,
    out_tag_len: list[int] | None,
    out_next_cursor: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or out_arch_tag is None
        or out_tag_offset is None
        or out_tag_len is None
        or out_next_cursor is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if (
        cursor_ref is out_arch_tag
        or cursor_ref is out_tag_offset
        or cursor_ref is out_tag_len
        or cursor_ref is out_next_cursor
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if (
        out_arch_tag is out_tag_offset
        or out_arch_tag is out_tag_len
        or out_arch_tag is out_next_cursor
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if out_tag_offset is out_tag_len or out_tag_offset is out_next_cursor or out_tag_len is out_next_cursor:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if buf_nbytes > I64_MAX or table_end > I64_MAX or metadata_count > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if metadata_count > GGUF_MAX_METADATA_COUNT:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if table_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if cursor_ref[0] > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if cursor_ref[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if cursor_ref[0] > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    snapshot = cursor_ref[0]

    stage_cursor = [snapshot]
    stage_arch = [0]
    stage_off = [0]
    stage_len = [0]
    stage_next = [0]
    err = _read_arch_base(
        buf,
        buf_nbytes,
        stage_cursor,
        table_end,
        metadata_count,
        stage_arch,
        stage_off,
        stage_len,
        stage_next,
    )
    if err != GGUF_META_TABLE_OK:
        return err
    if stage_cursor[0] != stage_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if stage_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if stage_len[0] > buf_nbytes - stage_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    stage_end = stage_off[0] + stage_len[0]
    if stage_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if stage_end != stage_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    verify_cursor = [snapshot]
    verify_arch = [0]
    verify_off = [0]
    verify_len = [0]
    verify_next = [0]
    err = _read_arch_base(
        buf,
        buf_nbytes,
        verify_cursor,
        table_end,
        metadata_count,
        verify_arch,
        verify_off,
        verify_len,
        verify_next,
    )
    if err != GGUF_META_TABLE_OK:
        return err
    if verify_cursor[0] != verify_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if verify_off[0] > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if verify_len[0] > buf_nbytes - verify_off[0]:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    verify_end = verify_off[0] + verify_len[0]
    if verify_end > table_end:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if verify_end != verify_next[0]:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if (
        stage_arch[0] != verify_arch[0]
        or stage_off[0] != verify_off[0]
        or stage_len[0] != verify_len[0]
        or stage_next[0] != verify_next[0]
    ):
        return GGUF_META_TABLE_ERR_BAD_PARAM

    out_arch_tag[0] = stage_arch[0]
    out_tag_offset[0] = stage_off[0]
    out_tag_len[0] = stage_len[0]
    out_next_cursor[0] = stage_next[0]
    cursor_ref[0] = stage_next[0]
    return GGUF_META_TABLE_OK


def test_source_contains_iq_1117_symbol_and_parity_checks() -> None:
    source = Path("src/gguf/metadata.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFMetadataReadArchitectureTagCheckedNoPartialCommitOnlyPreflightOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split("Bool GGUFMetaCanRead(", 1)[0]
    assert "GGUFMetadataReadArchitectureTagCheckedNoPartial(buf" in body
    assert "cursor_snapshot = *cursor;" in body
    assert "if (tag_end_stage != next_cursor_stage)" in body
    assert "if (arch_stage != arch_verify)" in body
    assert "if ((U8 *)cursor == (U8 *)out_arch_tag" in body
    assert "if (cursor_ptr_snapshot != cursor ||" in body
    assert "*cursor = next_cursor_stage;" in body


def test_pointer_alias_rejected_no_publish() -> None:
    payload = _kv_string("general.architecture", "llama")
    cursor = [0]
    shared = [12345]
    out_len = [33]
    out_next = [44]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        1,
        shared,
        shared,
        out_len,
        out_next,
    )

    assert status == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert shared == [12345]
    assert out_len == [33]
    assert out_next == [44]


def test_success_matches_base_and_publishes_once() -> None:
    payload = _kv_string("architecture", "mistral") + _kv_string("general.architecture", "qwen2")

    cursor = [0]
    out_arch = [111]
    out_off = [222]
    out_len = [333]
    out_next = [444]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        2,
        out_arch,
        out_off,
        out_len,
        out_next,
    )

    assert status == GGUF_META_TABLE_OK
    assert out_arch == [GGUF_ARCH_TAG_QWEN2]
    assert payload[out_off[0] : out_off[0] + out_len[0]] == b"qwen2"
    assert out_next == [cursor[0]]


def test_missing_key_returns_not_found_no_publish() -> None:
    payload = _kv_string("tokenizer.ggml.model", "bpe")
    cursor = [0]
    out_arch = [1]
    out_off = [2]
    out_len = [3]
    out_next = [4]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        1,
        out_arch,
        out_off,
        out_len,
        out_next,
    )

    assert status == GGUF_META_PARSE_ERR_NOT_FOUND
    assert cursor == [0]
    assert out_arch == [1]
    assert out_off == [2]
    assert out_len == [3]
    assert out_next == [4]


def test_duplicate_priority_rejected_no_publish() -> None:
    payload = _kv_string("general.architecture", "llama") + _kv_string("general.architecture", "mistral")
    cursor = [0]
    out_arch = [51]
    out_off = [52]
    out_len = [53]
    out_next = [54]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        2,
        out_arch,
        out_off,
        out_len,
        out_next,
    )

    assert status == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_arch == [51]
    assert out_off == [52]
    assert out_len == [53]
    assert out_next == [54]


def test_type_mismatch_rejected_no_publish() -> None:
    payload = _kv_u32("general.architecture", 7)
    cursor = [0]
    out_arch = [61]
    out_off = [62]
    out_len = [63]
    out_next = [64]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        1,
        out_arch,
        out_off,
        out_len,
        out_next,
    )

    assert status == GGUF_META_TABLE_ERR_TYPE_MISMATCH
    assert cursor == [0]
    assert out_arch == [61]
    assert out_off == [62]
    assert out_len == [63]
    assert out_next == [64]


def test_truncated_value_span_rejected_no_publish() -> None:
    payload = _kv_string("general.architecture", "llama")[:-1]
    cursor = [0]
    out_arch = [71]
    out_off = [72]
    out_len = [73]
    out_next = [74]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        1,
        out_arch,
        out_off,
        out_len,
        out_next,
    )

    assert status == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_arch == [71]
    assert out_off == [72]
    assert out_len == [73]
    assert out_next == [74]


def test_overflow_guard_rejects_large_metadata_count_no_publish() -> None:
    payload = _kv_string("general.architecture", "llama")
    cursor = [0]
    out_arch = [81]
    out_off = [82]
    out_len = [83]
    out_next = [84]

    status = gguf_metadata_read_architecture_tag_checked_nopartial_commit_only_preflight_only_reference(
        payload,
        len(payload),
        cursor,
        len(payload),
        I64_MAX + 1,
        out_arch,
        out_off,
        out_len,
        out_next,
    )

    assert status == GGUF_META_TABLE_ERR_OVERFLOW
    assert cursor == [0]
    assert out_arch == [81]
    assert out_off == [82]
    assert out_len == [83]
    assert out_next == [84]


if __name__ == "__main__":
    test_source_contains_iq_1117_symbol_and_parity_checks()
    test_pointer_alias_rejected_no_publish()
    test_success_matches_base_and_publishes_once()
    test_missing_key_returns_not_found_no_publish()
    test_duplicate_priority_rejected_no_publish()
    test_type_mismatch_rejected_no_publish()
    test_truncated_value_span_rejected_no_publish()
    test_overflow_guard_rejects_large_metadata_count_no_publish()
    print("ok")
