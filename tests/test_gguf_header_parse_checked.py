#!/usr/bin/env python3
"""Reference checks for GGUFHeaderParseChecked semantics."""

from __future__ import annotations

import struct

GGUF_MAGIC_U32 = 0x46554747
GGUF_VERSION_V2 = 2
GGUF_VERSION_V3 = 3

GGUF_HEADER_OK = 0
GGUF_HEADER_ERR_NULL_PTR = 1
GGUF_HEADER_ERR_BAD_PARAM = 2
GGUF_HEADER_ERR_OVERFLOW = 3
GGUF_HEADER_ERR_TRUNCATED = 4
GGUF_HEADER_ERR_BAD_MAGIC = 5
GGUF_HEADER_ERR_BAD_VERSION = 6

GGUF_HEADER_BYTES = 24
I64_MAX_VALUE = (1 << 63) - 1


def parse_header_checked(
    header_bytes: bytes | None,
    out_header_present: bool = True,
    out_header_bytes_present: bool = True,
    header_nbytes_override: int | None = None,
):
    if header_bytes is None or not out_header_present or not out_header_bytes_present:
        return GGUF_HEADER_ERR_NULL_PTR, None, None

    header_nbytes = len(header_bytes) if header_nbytes_override is None else header_nbytes_override

    if header_nbytes < 0:
        return GGUF_HEADER_ERR_BAD_PARAM, None, None
    if header_nbytes > I64_MAX_VALUE:
        return GGUF_HEADER_ERR_OVERFLOW, None, None
    if header_nbytes < GGUF_HEADER_BYTES:
        return GGUF_HEADER_ERR_TRUNCATED, None, None

    magic_u32, version_u32, tensor_count, metadata_kv_count = struct.unpack_from("<IIQQ", header_bytes, 0)

    if magic_u32 != GGUF_MAGIC_U32:
        return GGUF_HEADER_ERR_BAD_MAGIC, None, None
    if version_u32 not in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        return GGUF_HEADER_ERR_BAD_VERSION, None, None

    out_header = {
        "magic": magic_u32,
        "version": version_u32,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
    }
    return GGUF_HEADER_OK, out_header, GGUF_HEADER_BYTES


def make_header_blob(*, magic: int = GGUF_MAGIC_U32, version: int = GGUF_VERSION_V3, tensor_count: int = 3, metadata_kv_count: int = 5) -> bytes:
    return struct.pack("<IIQQ", magic, version, tensor_count, metadata_kv_count)


def test_null_ptr_contracts() -> None:
    err, _, _ = parse_header_checked(None)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, _, _ = parse_header_checked(make_header_blob(), out_header_present=False)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, _, _ = parse_header_checked(make_header_blob(), out_header_bytes_present=False)
    assert err == GGUF_HEADER_ERR_NULL_PTR


def test_overflow_and_truncation_contracts() -> None:
    blob = make_header_blob()

    err, _, _ = parse_header_checked(blob, header_nbytes_override=I64_MAX_VALUE + 1)
    assert err == GGUF_HEADER_ERR_OVERFLOW

    err, _, _ = parse_header_checked(blob[: GGUF_HEADER_BYTES - 1])
    assert err == GGUF_HEADER_ERR_TRUNCATED


def test_magic_and_version_validation() -> None:
    err, _, _ = parse_header_checked(make_header_blob(magic=0xDEADBEEF))
    assert err == GGUF_HEADER_ERR_BAD_MAGIC

    err, _, _ = parse_header_checked(make_header_blob(version=1))
    assert err == GGUF_HEADER_ERR_BAD_VERSION


def test_success_v2_and_v3() -> None:
    for version in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        blob = make_header_blob(version=version, tensor_count=42, metadata_kv_count=99)
        err, header, consumed = parse_header_checked(blob)

        assert err == GGUF_HEADER_OK
        assert consumed == GGUF_HEADER_BYTES
        assert header == {
            "magic": GGUF_MAGIC_U32,
            "version": version,
            "tensor_count": 42,
            "metadata_kv_count": 99,
        }


def run() -> None:
    test_null_ptr_contracts()
    test_overflow_and_truncation_contracts()
    test_magic_and_version_validation()
    test_success_v2_and_v3()
    print("gguf_header_parse_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
