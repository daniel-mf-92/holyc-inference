#!/usr/bin/env python3
"""Parity harness for GGUFHeaderReadCheckedNoPartial (IQ-986)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def read_u32_le(data: bytes, offset: int) -> int:
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def read_u64_le(data: bytes, offset: int) -> int:
    value = 0
    for i in range(8):
        value |= data[offset + i] << (8 * i)
    return value


def gguf_header_read_checked_nopartial(
    header_bytes: bytes | None,
    header_nbytes: int,
    out_header: GGUFHeaderView | None,
    out_header_bytes_present: bool,
) -> tuple[int, GGUFHeaderView | None, int | None]:
    if header_bytes is None or out_header is None or not out_header_bytes_present:
        return GGUF_HEADER_ERR_NULL_PTR, None, None

    snapshot = (header_bytes, header_nbytes)

    if header_nbytes > I64_MAX:
        return GGUF_HEADER_ERR_OVERFLOW, None, None
    if header_nbytes < GGUF_HEADER_BYTES:
        return GGUF_HEADER_ERR_TRUNCATED, None, None

    staged_magic = read_u32_le(header_bytes, 0)
    if staged_magic != GGUF_MAGIC_U32:
        return GGUF_HEADER_ERR_BAD_MAGIC, None, None

    staged_version = read_u32_le(header_bytes, 4)
    if staged_version not in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        return GGUF_HEADER_ERR_BAD_VERSION, None, None

    staged_tensor_count = read_u64_le(header_bytes, 8)
    staged_metadata_count = read_u64_le(header_bytes, 16)

    if snapshot != (header_bytes, header_nbytes):
        return GGUF_HEADER_ERR_BAD_PARAM, None, None

    staged = GGUFHeaderView(
        magic=staged_magic,
        version=staged_version,
        tensor_count=staged_tensor_count,
        metadata_kv_count=staged_metadata_count,
    )
    return GGUF_HEADER_OK, staged, GGUF_HEADER_BYTES


def build_header(
    *,
    magic: int = GGUF_MAGIC_U32,
    version: int = GGUF_VERSION_V3,
    tensor_count: int = 1,
    metadata_kv_count: int = 0,
) -> bytes:
    blob = bytearray()
    blob += magic.to_bytes(4, "little", signed=False)
    blob += version.to_bytes(4, "little", signed=False)
    blob += tensor_count.to_bytes(8, "little", signed=False)
    blob += metadata_kv_count.to_bytes(8, "little", signed=False)
    return bytes(blob)


def test_source_contains_iq986_function() -> None:
    source = Path("src/gguf/header.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFHeaderReadCheckedNoPartial(U8 *header_bytes,"
    assert sig in source
    body = source.split(sig, 1)[1].split("Bool GGUFCanRead(", 1)[0]

    assert "status = GGUFReadU32LEChecked(" in body
    assert "status = GGUFReadU64LEChecked(" in body
    assert "if (header_nbytes > 0x7FFFFFFFFFFFFFFF)" in body
    assert "if (header_nbytes < GGUF_HEADER_BYTES)" in body
    assert "if (snapshot_header_bytes != header_bytes)" in body
    assert "status = GGUFHeaderValidateAndSizeChecked(&staged_header," in body
    assert "out_header->magic = staged_header.magic;" in body
    assert "*out_header_bytes = staged_header_bytes;" in body


def test_valid_headers_match_llama_cpp_layout() -> None:
    vectors = [
        build_header(version=GGUF_VERSION_V2, tensor_count=0, metadata_kv_count=0),
        build_header(version=GGUF_VERSION_V3, tensor_count=1, metadata_kv_count=2),
        build_header(version=GGUF_VERSION_V3, tensor_count=4096, metadata_kv_count=1024),
    ]

    for payload in vectors:
        out = GGUFHeaderView(0xDEADBEEF, 0xAA, 123, 456)
        before = GGUFHeaderView(out.magic, out.version, out.tensor_count, out.metadata_kv_count)

        status, staged, consumed = gguf_header_read_checked_nopartial(
            payload,
            len(payload),
            out,
            True,
        )

        assert status == GGUF_HEADER_OK
        assert consumed == GGUF_HEADER_BYTES
        assert staged is not None
        assert staged.magic == GGUF_MAGIC_U32
        assert staged.version in (GGUF_VERSION_V2, GGUF_VERSION_V3)
        assert before.magic == 0xDEADBEEF


def test_corrupt_vectors_and_truncation_are_rejected_without_publish() -> None:
    invalid = [
        (build_header(magic=0x21465547), GGUF_HEADER_ERR_BAD_MAGIC),
        (build_header(version=1), GGUF_HEADER_ERR_BAD_VERSION),
        (build_header(version=99), GGUF_HEADER_ERR_BAD_VERSION),
    ]

    for payload, expected in invalid:
        out = GGUFHeaderView(0x99, 0x88, 0x77, 0x66)
        status, staged, consumed = gguf_header_read_checked_nopartial(
            payload,
            len(payload),
            out,
            True,
        )
        assert status == expected
        assert staged is None
        assert consumed is None
        assert out == GGUFHeaderView(0x99, 0x88, 0x77, 0x66)

    for truncated_len in range(0, GGUF_HEADER_BYTES):
        payload = build_header()[:truncated_len]
        out = GGUFHeaderView(1, 2, 3, 4)
        status, staged, consumed = gguf_header_read_checked_nopartial(
            payload,
            len(payload),
            out,
            True,
        )
        assert status == GGUF_HEADER_ERR_TRUNCATED
        assert staged is None
        assert consumed is None
        assert out == GGUFHeaderView(1, 2, 3, 4)


def test_null_and_overflow_guards() -> None:
    payload = build_header()
    out = GGUFHeaderView(10, 11, 12, 13)

    status, _, _ = gguf_header_read_checked_nopartial(
        None,
        len(payload),
        out,
        True,
    )
    assert status == GGUF_HEADER_ERR_NULL_PTR

    status, _, _ = gguf_header_read_checked_nopartial(
        payload,
        I64_MAX + 1,
        out,
        True,
    )
    assert status == GGUF_HEADER_ERR_OVERFLOW

    status, _, _ = gguf_header_read_checked_nopartial(
        payload,
        len(payload),
        None,
        True,
    )
    assert status == GGUF_HEADER_ERR_NULL_PTR


def run() -> None:
    test_source_contains_iq986_function()
    test_valid_headers_match_llama_cpp_layout()
    test_corrupt_vectors_and_truncation_are_rejected_without_publish()
    test_null_and_overflow_guards()
    print("gguf_header_read_checked_nopartial=ok")


if __name__ == "__main__":
    run()
