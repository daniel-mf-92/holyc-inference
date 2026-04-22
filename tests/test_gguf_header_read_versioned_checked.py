#!/usr/bin/env python3
"""Parity harness for GGUFHeaderReadVersionedChecked (IQ-1136)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

GGUF_MAGIC_U32 = 0x46554747
GGUF_VERSION_V2 = 2
GGUF_VERSION_V3 = 3
GGUF_HEADER_BYTES = 24
I64_MAX = (1 << 63) - 1

GGUF_HEADER_OK = 0
GGUF_HEADER_ERR_NULL_PTR = 1
GGUF_HEADER_ERR_BAD_PARAM = 2
GGUF_HEADER_ERR_OVERFLOW = 3
GGUF_HEADER_ERR_TRUNCATED = 4
GGUF_HEADER_ERR_BAD_MAGIC = 5
GGUF_HEADER_ERR_BAD_VERSION = 6


@dataclass
class GGUFHeaderView:
    magic: int
    version: int
    tensor_count: int
    metadata_kv_count: int


def header_can_read_span(total_bytes: int, offset: int, need_bytes: int) -> bool:
    if offset > total_bytes:
        return False
    if need_bytes > ((1 << 64) - 1) - offset:
        return False
    return offset + need_bytes <= total_bytes


def header_validate_and_size_checked(header: GGUFHeaderView):
    if header.magic != GGUF_MAGIC_U32:
        return GGUF_HEADER_ERR_BAD_MAGIC, None
    if header.version not in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        return GGUF_HEADER_ERR_BAD_VERSION, None
    if header.tensor_count > I64_MAX:
        return GGUF_HEADER_ERR_OVERFLOW, None
    if header.metadata_kv_count > I64_MAX:
        return GGUF_HEADER_ERR_OVERFLOW, None
    if GGUF_HEADER_BYTES > I64_MAX:
        return GGUF_HEADER_ERR_OVERFLOW, None
    return GGUF_HEADER_OK, GGUF_HEADER_BYTES


def gguf_header_read_versioned_checked(
    buf: bytes | None,
    buf_nbytes: int,
    cursor: int | None,
    out_header: GGUFHeaderView | None,
    out_header_bytes_present: bool,
) -> tuple[int, int | None, GGUFHeaderView | None, int | None]:
    if buf is None or cursor is None or out_header is None or not out_header_bytes_present:
        return GGUF_HEADER_ERR_NULL_PTR, None, None, None

    if buf_nbytes > I64_MAX:
        return GGUF_HEADER_ERR_OVERFLOW, None, None, None

    cur = cursor
    if cur > I64_MAX:
        return GGUF_HEADER_ERR_OVERFLOW, None, None, None
    if cur > buf_nbytes:
        return GGUF_HEADER_ERR_BAD_PARAM, None, None, None

    if not header_can_read_span(buf_nbytes, cur, GGUF_HEADER_BYTES):
        return GGUF_HEADER_ERR_TRUNCATED, None, None, None

    magic_u32 = struct.unpack_from("<I", buf, cur)[0]
    cur += 4
    version_u32 = struct.unpack_from("<I", buf, cur)[0]
    cur += 4
    tensor_count = struct.unpack_from("<Q", buf, cur)[0]
    cur += 8
    metadata_kv_count = struct.unpack_from("<Q", buf, cur)[0]
    next_cursor = cur + 8

    staged = GGUFHeaderView(
        magic=magic_u32,
        version=version_u32,
        tensor_count=tensor_count,
        metadata_kv_count=metadata_kv_count,
    )
    err, consumed = header_validate_and_size_checked(staged)
    if err != GGUF_HEADER_OK:
        return err, None, None, None

    if consumed != GGUF_HEADER_BYTES:
        return GGUF_HEADER_ERR_BAD_PARAM, None, None, None
    if next_cursor > buf_nbytes:
        return GGUF_HEADER_ERR_TRUNCATED, None, None, None

    return GGUF_HEADER_OK, next_cursor, staged, consumed


def build_header(
    *,
    magic: int = GGUF_MAGIC_U32,
    version: int = GGUF_VERSION_V3,
    tensor_count: int = 1,
    metadata_kv_count: int = 0,
) -> bytes:
    return struct.pack("<IIQQ", magic, version, tensor_count, metadata_kv_count)


def test_source_contains_iq1136_function() -> None:
    source = Path("src/gguf/header.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFHeaderReadVersionedChecked(U8 *buf,"
    assert sig in source
    body = source.split(sig, 1)[1].split("Bool GGUFCanRead(", 1)[0]

    assert "if (!buf || !cursor || !out_header || !out_header_bytes)" in body
    assert "if (!GGUFHeaderCanReadSpan(buf_nbytes, cur, GGUF_HEADER_BYTES))" in body
    assert "status = GGUFReadU32LEChecked(buf, buf_nbytes, cur, &magic_u32);" in body
    assert "status = GGUFReadU32LEChecked(buf, buf_nbytes, cur, &version_u32);" in body
    assert "status = GGUFReadU64LEChecked(buf, buf_nbytes, cur, &tensor_count);" in body
    assert "status = GGUFReadU64LEChecked(buf, buf_nbytes, cur, &metadata_kv_count);" in body
    assert "status = GGUFHeaderValidateAndSizeChecked(&staged_header, &header_bytes);" in body
    assert "*cursor = next_cursor;" in body


def test_success_at_nonzero_cursor_v2_v3() -> None:
    prefix = b"\x99" * 13
    suffix = b"\xFE\xED"

    for version in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        payload = prefix + build_header(version=version, tensor_count=42, metadata_kv_count=7) + suffix
        out = GGUFHeaderView(0xDEAD, 0xBEEF, 111, 222)
        initial = GGUFHeaderView(out.magic, out.version, out.tensor_count, out.metadata_kv_count)

        err, next_cursor, staged, consumed = gguf_header_read_versioned_checked(
            payload,
            len(payload),
            len(prefix),
            out,
            True,
        )

        assert err == GGUF_HEADER_OK
        assert consumed == GGUF_HEADER_BYTES
        assert next_cursor == len(prefix) + GGUF_HEADER_BYTES
        assert staged is not None
        assert staged.magic == GGUF_MAGIC_U32
        assert staged.version == version
        assert staged.tensor_count == 42
        assert staged.metadata_kv_count == 7
        assert out == initial


def test_failure_paths_reject_without_publish() -> None:
    base = build_header()

    vectors = [
        (build_header(magic=0x00000000), GGUF_HEADER_ERR_BAD_MAGIC, 0),
        (build_header(version=1), GGUF_HEADER_ERR_BAD_VERSION, 0),
        (build_header(tensor_count=I64_MAX + 1), GGUF_HEADER_ERR_OVERFLOW, 0),
        (build_header(metadata_kv_count=I64_MAX + 1), GGUF_HEADER_ERR_OVERFLOW, 0),
        (base, GGUF_HEADER_ERR_BAD_PARAM, len(base) + 1),
    ]

    for payload, expected, cursor in vectors:
        out = GGUFHeaderView(9, 8, 7, 6)
        before = GGUFHeaderView(out.magic, out.version, out.tensor_count, out.metadata_kv_count)
        err, next_cursor, staged, consumed = gguf_header_read_versioned_checked(
            payload,
            len(payload),
            cursor,
            out,
            True,
        )
        assert err == expected
        assert next_cursor is None
        assert staged is None
        assert consumed is None
        assert out == before


def test_truncation_and_null_contracts() -> None:
    payload = build_header()
    out = GGUFHeaderView(1, 2, 3, 4)

    for nbytes in range(0, GGUF_HEADER_BYTES):
        err, next_cursor, staged, consumed = gguf_header_read_versioned_checked(
            payload[:nbytes],
            nbytes,
            0,
            out,
            True,
        )
        assert err == GGUF_HEADER_ERR_TRUNCATED
        assert next_cursor is None
        assert staged is None
        assert consumed is None

    err, *_ = gguf_header_read_versioned_checked(None, len(payload), 0, out, True)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, *_ = gguf_header_read_versioned_checked(payload, len(payload), None, out, True)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, *_ = gguf_header_read_versioned_checked(payload, len(payload), 0, None, True)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, *_ = gguf_header_read_versioned_checked(payload, len(payload), 0, out, False)
    assert err == GGUF_HEADER_ERR_NULL_PTR


def run() -> None:
    test_source_contains_iq1136_function()
    test_success_at_nonzero_cursor_v2_v3()
    test_failure_paths_reject_without_publish()
    test_truncation_and_null_contracts()
    print("gguf_header_read_versioned_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
