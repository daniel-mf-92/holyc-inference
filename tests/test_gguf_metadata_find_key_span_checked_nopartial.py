#!/usr/bin/env python3
"""Harness for GGUFMetadataFindKeySpanCheckedNoPartial (IQ-1105)."""

from __future__ import annotations

import random
from pathlib import Path

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4
GGUF_META_PARSE_ERR_NOT_FOUND = 9

I64_MAX = (1 << 63) - 1
GGUF_MAX_METADATA_COUNT = 1 << 20
GGUF_MAX_STRING_BYTES = 1 << 20
GGUF_MAX_ARRAY_ELEMS = 1 << 24

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_FIXED_WIDTH = {
    GGUF_TYPE_UINT8: 1,
    GGUF_TYPE_INT8: 1,
    GGUF_TYPE_BOOL: 1,
    GGUF_TYPE_UINT16: 2,
    GGUF_TYPE_INT16: 2,
    GGUF_TYPE_UINT32: 4,
    GGUF_TYPE_INT32: 4,
    GGUF_TYPE_FLOAT32: 4,
    GGUF_TYPE_UINT64: 8,
    GGUF_TYPE_INT64: 8,
    GGUF_TYPE_FLOAT64: 8,
}


def _u32le(v: int) -> bytes:
    return int(v & 0xFFFFFFFF).to_bytes(4, "little", signed=False)


def _u64le(v: int) -> bytes:
    return int(v & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little", signed=False)


def _gguf_string_bytes(s: bytes) -> bytes:
    return _u64le(len(s)) + s


def _encode_scalar_payload(value_type: int, value: int | bytes) -> bytes:
    if value_type in _FIXED_WIDTH:
        return int(value).to_bytes(_FIXED_WIDTH[value_type], "little", signed=False)
    if value_type == GGUF_TYPE_STRING:
        assert isinstance(value, (bytes, bytearray))
        return _gguf_string_bytes(bytes(value))
    raise ValueError(f"unsupported scalar type {value_type}")


def _encode_value(value_type: int, value: int | bytes | tuple[int, list[int | bytes]]) -> bytes:
    if value_type != GGUF_TYPE_ARRAY:
        return _u32le(value_type) + _encode_scalar_payload(value_type, value)

    assert isinstance(value, tuple)
    elem_type, elems = value
    payload = _u32le(elem_type) + _u64le(len(elems))
    if elem_type == GGUF_TYPE_STRING:
        for item in elems:
            assert isinstance(item, (bytes, bytearray))
            payload += _gguf_string_bytes(bytes(item))
    else:
        for item in elems:
            payload += _encode_scalar_payload(elem_type, int(item))
    return _u32le(GGUF_TYPE_ARRAY) + payload


def _encode_kv(key: bytes, value_type: int, value: int | bytes | tuple[int, list[int | bytes]]) -> bytes:
    return _gguf_string_bytes(key) + _encode_value(value_type, value)


def _type_supported(value_type: int) -> bool:
    return GGUF_TYPE_UINT8 <= value_type <= GGUF_TYPE_FLOAT64


def gguf_metadata_find_key_span_checked_nopartial_reference(
    *,
    buf: list[int] | None,
    buf_nbytes: int,
    cursor_ref: list[int] | None,
    table_end: int,
    metadata_count: int,
    target_key_bytes: list[int] | None,
    target_key_len: int,
    out_key_offset_ref: list[int] | None,
    out_key_len_ref: list[int] | None,
    out_value_type_ref: list[int] | None,
    out_next_cursor_ref: list[int] | None,
) -> int:
    if (
        buf is None
        or cursor_ref is None
        or target_key_bytes is None
        or out_key_offset_ref is None
        or out_key_len_ref is None
        or out_value_type_ref is None
        or out_next_cursor_ref is None
    ):
        return GGUF_META_TABLE_ERR_NULL_PTR

    if (
        buf_nbytes > I64_MAX
        or table_end > I64_MAX
        or metadata_count > I64_MAX
        or target_key_len > I64_MAX
        or cursor_ref[0] > I64_MAX
    ):
        return GGUF_META_TABLE_ERR_OVERFLOW

    if metadata_count > GGUF_MAX_METADATA_COUNT:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    if target_key_len <= 0 or target_key_len > GGUF_MAX_STRING_BYTES:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    scan_cursor = cursor_ref[0]
    if scan_cursor > table_end:
        return GGUF_META_TABLE_ERR_BAD_PARAM

    staged = None

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
        value = (
            buf[cur]
            | (buf[cur + 1] << 8)
            | (buf[cur + 2] << 16)
            | (buf[cur + 3] << 24)
        )
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

    def skip_value_payload(cur: int, value_type: int) -> tuple[int, int | None]:
        if value_type in _FIXED_WIDTH:
            return ensure_advance(cur, _FIXED_WIDTH[value_type])

        if value_type == GGUF_TYPE_STRING:
            err, str_len, cur2 = read_u64(cur)
            if err != GGUF_META_TABLE_OK:
                return err, None
            assert str_len is not None and cur2 is not None
            if str_len > GGUF_MAX_STRING_BYTES:
                return GGUF_META_TABLE_ERR_BAD_PARAM, None
            return ensure_advance(cur2, str_len)

        if value_type == GGUF_TYPE_ARRAY:
            err, elem_type, cur2 = read_u32(cur)
            if err != GGUF_META_TABLE_OK:
                return err, None
            err, elem_count, cur3 = read_u64(cur2)
            if err != GGUF_META_TABLE_OK:
                return err, None
            assert elem_type is not None and elem_count is not None and cur3 is not None

            if elem_type == GGUF_TYPE_ARRAY or not _type_supported(elem_type):
                return GGUF_META_TABLE_ERR_BAD_PARAM, None
            if elem_count > GGUF_MAX_ARRAY_ELEMS:
                return GGUF_META_TABLE_ERR_BAD_PARAM, None

            cur4 = cur3
            if elem_type in _FIXED_WIDTH:
                if elem_count and _FIXED_WIDTH[elem_type] > I64_MAX // elem_count:
                    return GGUF_META_TABLE_ERR_OVERFLOW, None
                need = _FIXED_WIDTH[elem_type] * elem_count
                return ensure_advance(cur4, need)

            if elem_type == GGUF_TYPE_STRING:
                for _ in range(elem_count):
                    err, str_len, cur5 = read_u64(cur4)
                    if err != GGUF_META_TABLE_OK:
                        return err, None
                    assert str_len is not None and cur5 is not None
                    if str_len > GGUF_MAX_STRING_BYTES:
                        return GGUF_META_TABLE_ERR_BAD_PARAM, None
                    err, cur4 = ensure_advance(cur5, str_len)
                    if err != GGUF_META_TABLE_OK:
                        return err, None
                return GGUF_META_TABLE_OK, cur4

            return GGUF_META_TABLE_ERR_BAD_PARAM, None

        return GGUF_META_TABLE_ERR_BAD_PARAM, None

    for _ in range(metadata_count):
        err, key_len, cur = read_u64(scan_cursor)
        if err != GGUF_META_TABLE_OK:
            return err
        assert key_len is not None and cur is not None

        if key_len <= 0 or key_len > GGUF_MAX_STRING_BYTES:
            return GGUF_META_TABLE_ERR_BAD_PARAM

        key_off = cur
        err, key_payload_end = ensure_advance(cur, key_len)
        if err != GGUF_META_TABLE_OK:
            return err
        assert key_payload_end is not None

        key_match = False
        if key_len == target_key_len:
            key_match = True
            for lane in range(key_len):
                if buf[key_off + lane] != target_key_bytes[lane]:
                    key_match = False
                    break

        err, value_type, cur2 = read_u32(key_payload_end)
        if err != GGUF_META_TABLE_OK:
            return err
        assert value_type is not None and cur2 is not None

        if not _type_supported(value_type):
            return GGUF_META_TABLE_ERR_BAD_PARAM

        err, next_cursor = skip_value_payload(cur2, value_type)
        if err != GGUF_META_TABLE_OK:
            return err
        assert next_cursor is not None

        if key_match:
            if staged is not None:
                return GGUF_META_TABLE_ERR_BAD_PARAM
            staged = (key_off, key_len, value_type, next_cursor)

        scan_cursor = next_cursor

    if staged is None:
        return GGUF_META_PARSE_ERR_NOT_FOUND

    out_key_offset_ref[0], out_key_len_ref[0], out_value_type_ref[0], out_next_cursor_ref[0] = staged
    cursor_ref[0] = staged[3]
    return GGUF_META_TABLE_OK


def _make_metadata_blob(entries: list[tuple[bytes, int, int | bytes | tuple[int, list[int | bytes]]]]) -> list[int]:
    blob = bytearray()
    for key, value_type, value in entries:
        blob.extend(_encode_kv(key, value_type, value))
    return list(blob)


def test_success_scalar_key_found_and_commits_cursor() -> None:
    entries = [
        (b"general.name", GGUF_TYPE_STRING, b"tiny"),
        (b"general.architecture", GGUF_TYPE_STRING, b"llama"),
        (b"llama.block_count", GGUF_TYPE_UINT32, 32),
    ]
    buf = _make_metadata_blob(entries)

    cursor = [0]
    out_key_offset = [999]
    out_key_len = [999]
    out_value_type = [999]
    out_next_cursor = [999]

    target = list(b"general.architecture")
    rc = gguf_metadata_find_key_span_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=len(entries),
        target_key_bytes=target,
        target_key_len=len(target),
        out_key_offset_ref=out_key_offset,
        out_key_len_ref=out_key_len,
        out_value_type_ref=out_value_type,
        out_next_cursor_ref=out_next_cursor,
    )

    assert rc == GGUF_META_TABLE_OK
    key_slice = bytes(buf[out_key_offset[0] : out_key_offset[0] + out_key_len[0]])
    assert key_slice == b"general.architecture"
    assert out_value_type[0] == GGUF_TYPE_STRING
    assert out_next_cursor[0] == cursor[0]


def test_not_found_preserves_outputs_and_cursor() -> None:
    entries = [(b"a", GGUF_TYPE_UINT8, 1), (b"b", GGUF_TYPE_UINT16, 2)]
    buf = _make_metadata_blob(entries)

    cursor = [0]
    out_key_offset = [101]
    out_key_len = [102]
    out_value_type = [103]
    out_next_cursor = [104]

    rc = gguf_metadata_find_key_span_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=len(entries),
        target_key_bytes=list(b"missing"),
        target_key_len=len(b"missing"),
        out_key_offset_ref=out_key_offset,
        out_key_len_ref=out_key_len,
        out_value_type_ref=out_value_type,
        out_next_cursor_ref=out_next_cursor,
    )

    assert rc == GGUF_META_PARSE_ERR_NOT_FOUND
    assert cursor == [0]
    assert out_key_offset == [101]
    assert out_key_len == [102]
    assert out_value_type == [103]
    assert out_next_cursor == [104]


def test_duplicate_key_rejected_as_bad_param() -> None:
    entries = [
        (b"dup", GGUF_TYPE_UINT8, 1),
        (b"dup", GGUF_TYPE_UINT16, 2),
    ]
    buf = _make_metadata_blob(entries)

    cursor = [0]
    out_key_offset = [11]
    out_key_len = [12]
    out_value_type = [13]
    out_next_cursor = [14]

    rc = gguf_metadata_find_key_span_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(buf),
        metadata_count=len(entries),
        target_key_bytes=list(b"dup"),
        target_key_len=3,
        out_key_offset_ref=out_key_offset,
        out_key_len_ref=out_key_len,
        out_value_type_ref=out_value_type,
        out_next_cursor_ref=out_next_cursor,
    )

    assert rc == GGUF_META_TABLE_ERR_BAD_PARAM
    assert cursor == [0]
    assert out_key_offset == [11]
    assert out_key_len == [12]
    assert out_value_type == [13]
    assert out_next_cursor == [14]


def test_truncated_array_string_payload_is_out_of_bounds() -> None:
    entries = [
        (b"arr", GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, [b"one", b"two"])),
    ]
    full = _make_metadata_blob(entries)
    buf = full[:-2]

    cursor = [0]
    out_key_offset = [0]
    out_key_len = [0]
    out_value_type = [0]
    out_next_cursor = [0]

    rc = gguf_metadata_find_key_span_checked_nopartial_reference(
        buf=buf,
        buf_nbytes=len(buf),
        cursor_ref=cursor,
        table_end=len(full),
        metadata_count=1,
        target_key_bytes=list(b"arr"),
        target_key_len=3,
        out_key_offset_ref=out_key_offset,
        out_key_len_ref=out_key_len,
        out_value_type_ref=out_value_type,
        out_next_cursor_ref=out_next_cursor,
    )

    assert rc == GGUF_META_TABLE_ERR_OUT_OF_BOUNDS
    assert cursor == [0]


def test_randomized_unique_lookup_vectors() -> None:
    rng = random.Random(20260422_1105)

    for _ in range(250):
        count = rng.randint(1, 24)
        keys: list[bytes] = []
        entries: list[tuple[bytes, int, int | bytes | tuple[int, list[int | bytes]]]] = []

        while len(keys) < count:
            n = rng.randint(1, 12)
            candidate = bytes(rng.randint(ord("a"), ord("z")) for _ in range(n))
            if candidate in keys:
                continue
            keys.append(candidate)

        for key in keys:
            choice = rng.randint(0, 4)
            if choice == 0:
                entries.append((key, GGUF_TYPE_UINT32, rng.randint(0, 2**32 - 1)))
            elif choice == 1:
                entries.append((key, GGUF_TYPE_STRING, key[::-1]))
            elif choice == 2:
                elems = [rng.randint(0, 255) for _ in range(rng.randint(0, 16))]
                entries.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT8, elems)))
            elif choice == 3:
                elems = [rng.randint(0, 2**16 - 1) for _ in range(rng.randint(0, 12))]
                entries.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_UINT16, elems)))
            else:
                elems = [bytes([rng.randint(65, 90)]) * rng.randint(0, 4) for _ in range(rng.randint(0, 8))]
                entries.append((key, GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, elems)))

        buf = _make_metadata_blob(entries)
        pick = rng.choice(keys)

        cursor = [0]
        out_key_offset = [-1]
        out_key_len = [-1]
        out_value_type = [-1]
        out_next_cursor = [-1]

        rc = gguf_metadata_find_key_span_checked_nopartial_reference(
            buf=buf,
            buf_nbytes=len(buf),
            cursor_ref=cursor,
            table_end=len(buf),
            metadata_count=len(entries),
            target_key_bytes=list(pick),
            target_key_len=len(pick),
            out_key_offset_ref=out_key_offset,
            out_key_len_ref=out_key_len,
            out_value_type_ref=out_value_type,
            out_next_cursor_ref=out_next_cursor,
        )

        assert rc == GGUF_META_TABLE_OK
        assert bytes(buf[out_key_offset[0] : out_key_offset[0] + out_key_len[0]]) == pick
        assert _type_supported(out_value_type[0])
        assert cursor[0] == out_next_cursor[0]


def test_source_contains_iq1105_function_and_contract_calls() -> None:
    src = Path(__file__).resolve().parents[1] / "src" / "gguf" / "metadata.HC"
    body = src.read_text(encoding="utf-8")

    sig = (
        "I32 GGUFMetadataFindKeySpanCheckedNoPartial(U8 *buf,"
    )
    assert sig in body
    assert "GGUFMetadataSkipValuePayloadChecked(" in body
    assert "GGUFMetadataReadStringLenU64Checked(" in body
    assert "GGUFMetadataReadArrayHeaderChecked(" in body
    assert "return GGUF_META_PARSE_ERR_NOT_FOUND;" in body


def run() -> None:
    test_success_scalar_key_found_and_commits_cursor()
    test_not_found_preserves_outputs_and_cursor()
    test_duplicate_key_rejected_as_bad_param()
    test_truncated_array_string_payload_is_out_of_bounds()
    test_randomized_unique_lookup_vectors()
    test_source_contains_iq1105_function_and_contract_calls()
    print("gguf_metadata_find_key_span_checked_nopartial=ok")


if __name__ == "__main__":
    run()
