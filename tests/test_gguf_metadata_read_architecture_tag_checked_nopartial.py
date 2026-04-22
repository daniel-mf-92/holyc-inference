#!/usr/bin/env python3
"""Harness for GGUFMetadataReadArchitectureTagCheckedNoPartial (IQ-1099)."""

from __future__ import annotations

from pathlib import Path

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4
GGUF_META_TABLE_ERR_TYPE_MISMATCH = 5
GGUF_META_PARSE_ERR_NOT_FOUND = 9

I64_MAX = (1 << 63) - 1
GGUF_MAX_METADATA_COUNT = 1 << 20
GGUF_MAX_STRING_BYTES = 1 << 20

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_STRING = 8

GGUF_ARCH_TAG_LLAMA = 1
GGUF_ARCH_TAG_MISTRAL = 2
GGUF_ARCH_TAG_QWEN2 = 3
GGUF_ARCH_TAG_PHI3 = 4


def _u32le(v: int) -> bytes:
    return int(v & 0xFFFFFFFF).to_bytes(4, "little", signed=False)


def _u64le(v: int) -> bytes:
    return int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)


def _gguf_string_bytes(s: bytes) -> bytes:
    return _u64le(len(s)) + s


def _encode_value(value_type: int, payload: int | bytes) -> bytes:
    if value_type == GGUF_TYPE_UINT8:
        return _u32le(value_type) + int(payload).to_bytes(1, "little", signed=False)
    if value_type == GGUF_TYPE_STRING:
        assert isinstance(payload, (bytes, bytearray))
        return _u32le(value_type) + _gguf_string_bytes(bytes(payload))
    raise ValueError(f"unsupported value_type={value_type}")


def _encode_kv(key: bytes, value_type: int, payload: int | bytes) -> bytes:
    return _gguf_string_bytes(key) + _encode_value(value_type, payload)


def _architecture_priority(key: bytes) -> int | None:
    if key == b"general.architecture":
        return 0
    if key == b"architecture":
        return 1
    if key == b"model.architecture":
        return 2
    return None


def _canonical_arch_tag(value: bytes) -> int | None:
    if value == b"llama":
        return GGUF_ARCH_TAG_LLAMA
    if value == b"mistral":
        return GGUF_ARCH_TAG_MISTRAL
    if value in (b"qwen2", b"qwen", b"qwen2.5"):
        return GGUF_ARCH_TAG_QWEN2
    if value in (b"phi3", b"phi-3"):
        return GGUF_ARCH_TAG_PHI3
    return None


def gguf_metadata_read_architecture_tag_checked_nopartial_reference(
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

    if (
        buf_nbytes > I64_MAX
        or table_end > I64_MAX
        or metadata_count > I64_MAX
        or cursor_ref[0] > I64_MAX
    ):
        return GGUF_META_TABLE_ERR_OVERFLOW

    if metadata_count > GGUF_MAX_METADATA_COUNT:
        return GGUF_META_TABLE_ERR_BAD_PARAM
    if table_end > buf_nbytes:
        return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    if cursor_ref[0] > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    def ensure_advance(cur: int, need: int) -> tuple[int, int | None]:
        nxt = cur + need
        if nxt > table_end:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None
        if nxt > buf_nbytes:
            return GGUF_META_TABLE_ERR_OUT_OF_BOUNDS, None
        return GGUF_META_TABLE_OK, nxt

    def read_u32(cur: int) -> tuple[int, int | None, int | None]:
        err, nxt = ensure_advance(cur, 4)
        if err != GGUF_META_TABLE_OK:
            return err, None, None
        assert nxt is not None
        value = buf[cur] | (buf[cur + 1] << 8) | (buf[cur + 2] << 16) | (buf[cur + 3] << 24)
        return GGUF_META_TABLE_OK, value, nxt

    def read_u64(cur: int) -> tuple[int, int | None, int | None]:
        err, nxt = ensure_advance(cur, 8)
        if err != GGUF_META_TABLE_OK:
            return err, None, None
        assert nxt is not None
        value = 0
        for i in range(8):
            value |= buf[cur + i] << (8 * i)
        return GGUF_META_TABLE_OK, value, nxt

    def skip_value(cur: int, value_type: int) -> tuple[int, int | None]:
        if value_type == GGUF_TYPE_UINT8:
            return ensure_advance(cur, 1)
        if value_type == GGUF_TYPE_STRING:
            err, string_len, cur2 = read_u64(cur)
            if err != GGUF_META_TABLE_OK:
                return err, None
            assert string_len is not None and cur2 is not None
            if string_len <= 0 or string_len > GGUF_MAX_STRING_BYTES:
                return GGUF_META_TABLE_ERR_BAD_PARAM, None
            return ensure_advance(cur2, string_len)
        return GGUF_META_TABLE_ERR_BAD_PARAM, None

    scan_cursor = cursor_ref[0]
    found_priority = None
    staged: tuple[int, int, int, int] | None = None

    for _ in range(metadata_count):
        err, key_len, cur2 = read_u64(scan_cursor)
        if err != GGUF_META_TABLE_OK:
            return err
        assert key_len is not None and cur2 is not None
        if key_len <= 0 or key_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        key_off = cur2
        err, key_end = ensure_advance(key_off, key_len)
        if err != GGUF_META_TABLE_OK:
            return err
        assert key_end is not None
        key_bytes = bytes(buf[key_off:key_end])
        scan_cursor = key_end

        err, value_type, cur3 = read_u32(scan_cursor)
        if err != GGUF_META_TABLE_OK:
            return err
        assert value_type is not None and cur3 is not None
        scan_cursor = cur3

        priority = _architecture_priority(key_bytes)
        if priority is None:
            err, nxt = skip_value(scan_cursor, value_type)
            if err != GGUF_META_TABLE_OK:
                return err
            assert nxt is not None
            scan_cursor = nxt
            continue

        if value_type != GGUF_TYPE_STRING:
            return GGUF_META_TABLE_ERR_TYPE_MISMATCH

        err, str_len, cur4 = read_u64(scan_cursor)
        if err != GGUF_META_TABLE_OK:
            return err
        assert str_len is not None and cur4 is not None
        if str_len <= 0 or str_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        value_off = cur4
        err, value_end = ensure_advance(value_off, str_len)
        if err != GGUF_META_TABLE_OK:
            return err
        assert value_end is not None
        scan_cursor = value_end

        tag = _canonical_arch_tag(bytes(buf[value_off:value_end]))
        if tag is None:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        if found_priority is not None and priority == found_priority:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        if found_priority is None or priority < found_priority:
            found_priority = priority
            staged = (tag, value_off, str_len, scan_cursor)

    if staged is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND

    out_arch_tag_ref[0], out_tag_offset_ref[0], out_tag_len_ref[0], out_next_cursor_ref[0] = staged
    cursor_ref[0] = staged[3]
    return GGUF_META_TABLE_OK


def _make_blob(entries: list[tuple[bytes, int, int | bytes]]) -> list[int]:
    blob = bytearray()
    for key, value_type, payload in entries:
        blob += _encode_kv(key, value_type, payload)
    return list(blob)


def test_prefers_general_architecture_and_commits_atomically() -> None:
    entries = [
        (b"architecture", GGUF_TYPE_STRING, b"mistral"),
        (b"general.architecture", GGUF_TYPE_STRING, b"llama"),
        (b"model.architecture", GGUF_TYPE_STRING, b"phi-3"),
    ]
    buf = _make_blob(entries)

    cursor = [0]
    out_arch = [0]
    out_off = [0]
    out_len = [0]
    out_next = [0]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=len(entries),
        out_arch_tag_ref=out_arch,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=out_len,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_TABLE_OK
    assert out_arch == [GGUF_ARCH_TAG_LLAMA]
    assert bytes(buf[out_off[0] : out_off[0] + out_len[0]]) == b"llama"
    assert cursor == out_next


def test_missing_architecture_key_is_not_found_without_mutation() -> None:
    entries = [(b"foo", GGUF_TYPE_UINT8, 7)]
    buf = _make_blob(entries)

    cursor = [0]
    out_arch = [123]
    out_off = [124]
    out_len = [125]
    out_next = [126]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_reference(
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
    assert out_arch == [123]
    assert out_off == [124]
    assert out_len == [125]
    assert out_next == [126]


def test_duplicate_same_priority_key_returns_bad_param_without_commit() -> None:
    entries = [
        (b"architecture", GGUF_TYPE_STRING, b"mistral"),
        (b"architecture", GGUF_TYPE_STRING, b"llama"),
    ]
    buf = _make_blob(entries)

    cursor = [0]
    out_arch = [11]
    out_off = [12]
    out_len = [13]
    out_next = [14]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_reference(
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
    assert out_arch == [11]
    assert out_off == [12]
    assert out_len == [13]
    assert out_next == [14]


def test_architecture_value_type_mismatch_is_rejected() -> None:
    entries = [(b"general.architecture", GGUF_TYPE_UINT8, 3)]
    buf = _make_blob(entries)

    cursor = [0]
    out_arch = [77]
    out_off = [78]
    out_len = [79]
    out_next = [80]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_reference(
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

    assert rc == GGUF_META_TABLE_ERR_TYPE_MISMATCH
    assert cursor == [0]
    assert out_arch == [77]


def test_truncated_architecture_string_span_is_out_of_bounds() -> None:
    full = bytes(_make_blob([(b"general.architecture", GGUF_TYPE_STRING, b"qwen2")]))
    buf = list(full[:-2])

    cursor = [0]
    out_arch = [1]
    out_off = [2]
    out_len = [3]
    out_next = [4]

    rc = gguf_metadata_read_architecture_tag_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(full),
        metadata_count=1,
        out_arch_tag_ref=out_arch,
        out_tag_offset_ref=out_off,
        out_tag_len_ref=out_len,
        out_next_cursor_ref=out_next,
    )

    assert rc == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]
    assert out_arch == [1]
    assert out_next == [4]


def test_source_contains_iq1099_function_and_alias_literals() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "gguf" / "metadata.HC"
    body = src.read_text(encoding="utf-8")

    assert "I32 GGUFMetadataReadArchitectureTagCheckedNoPartial(U8 *buf," in body
    assert '"general.architecture"' in body
    assert '"architecture"' in body
    assert '"model.architecture"' in body
    assert "GGUFMetadataCanonicalArchTagFromSpanChecked(" in body
    assert "GGUF_META_TABLE_ERR_TYPE_MISMATCH" in body


def run() -> None:
    test_prefers_general_architecture_and_commits_atomically()
    test_missing_architecture_key_is_not_found_without_mutation()
    test_duplicate_same_priority_key_returns_bad_param_without_commit()
    test_architecture_value_type_mismatch_is_rejected()
    test_truncated_architecture_string_span_is_out_of_bounds()
    test_source_contains_iq1099_function_and_alias_literals()
    print("gguf_metadata_read_architecture_tag_checked_nopartial=ok")


if __name__ == "__main__":
    run()
