#!/usr/bin/env python3
"""Reference checks for GGUF endian-safe integer read helpers."""

from __future__ import annotations

GGUF_HEADER_OK = 0
GGUF_HEADER_ERR_NULL_PTR = 1
GGUF_HEADER_ERR_BAD_PARAM = 2
GGUF_HEADER_ERR_OVERFLOW = 3
GGUF_HEADER_ERR_TRUNCATED = 4
GGUF_HEADER_ERR_BAD_MAGIC = 5
GGUF_HEADER_ERR_BAD_VERSION = 6


def can_read_span(total_bytes: int, offset: int, need_bytes: int) -> bool:
    if offset > total_bytes:
        return False
    if need_bytes > ((1 << 64) - 1) - offset:
        return False
    return offset + need_bytes <= total_bytes


def read_u16_le_checked(buf: bytes | None, offset: int, out_present: bool = True):
    if buf is None or not out_present:
        return GGUF_HEADER_ERR_NULL_PTR, None
    if not can_read_span(len(buf), offset, 2):
        return GGUF_HEADER_ERR_TRUNCATED, None
    return GGUF_HEADER_OK, (buf[offset] | (buf[offset + 1] << 8))


def read_u32_le_checked(buf: bytes | None, offset: int, out_present: bool = True):
    if buf is None or not out_present:
        return GGUF_HEADER_ERR_NULL_PTR, None
    if not can_read_span(len(buf), offset, 4):
        return GGUF_HEADER_ERR_TRUNCATED, None
    value = (
        buf[offset]
        | (buf[offset + 1] << 8)
        | (buf[offset + 2] << 16)
        | (buf[offset + 3] << 24)
    )
    return GGUF_HEADER_OK, value


def read_u64_le_checked(buf: bytes | None, offset: int, out_present: bool = True):
    if buf is None or not out_present:
        return GGUF_HEADER_ERR_NULL_PTR, None
    if not can_read_span(len(buf), offset, 8):
        return GGUF_HEADER_ERR_TRUNCATED, None
    value = 0
    for idx in range(8):
        value |= buf[offset + idx] << (8 * idx)
    return GGUF_HEADER_OK, value


def as_i32_from_u32(raw: int) -> int:
    if raw & 0x80000000:
        return -(((~raw) + 1) & 0xFFFFFFFF)
    return raw


def as_i64_from_u64(raw: int) -> int:
    if raw & 0x8000000000000000:
        return -(((~raw) + 1) & 0xFFFFFFFFFFFFFFFF)
    return raw


def read_i32_le_checked(buf: bytes | None, offset: int, out_present: bool = True):
    err, raw = read_u32_le_checked(buf, offset, out_present)
    if err:
        return err, None
    return GGUF_HEADER_OK, as_i32_from_u32(raw)


def read_i64_le_checked(buf: bytes | None, offset: int, out_present: bool = True):
    err, raw = read_u64_le_checked(buf, offset, out_present)
    if err:
        return err, None
    return GGUF_HEADER_OK, as_i64_from_u64(raw)


def test_null_ptr_contracts() -> None:
    err, _ = read_u16_le_checked(None, 0)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, _ = read_i32_le_checked(b"\x00\x00\x00\x00", 0, out_present=False)
    assert err == GGUF_HEADER_ERR_NULL_PTR


def test_truncation_contracts() -> None:
    err, _ = read_u16_le_checked(b"\xAA", 0)
    assert err == GGUF_HEADER_ERR_TRUNCATED

    err, _ = read_i64_le_checked(b"\x00" * 7, 0)
    assert err == GGUF_HEADER_ERR_TRUNCATED


def test_u16_u32_u64_little_endian_decode() -> None:
    blob = bytes.fromhex("34 12 EF CD AB 89 08 07 06 05 04 03 02 01")

    err, v16 = read_u16_le_checked(blob, 0)
    assert err == GGUF_HEADER_OK
    assert v16 == 0x1234

    err, v32 = read_u32_le_checked(blob, 2)
    assert err == GGUF_HEADER_OK
    assert v32 == 0x89ABCDEF

    err, v64 = read_u64_le_checked(blob, 6)
    assert err == GGUF_HEADER_OK
    assert v64 == 0x0102030405060708


def test_signed_twos_complement_decode() -> None:
    blob_i32 = bytes.fromhex("FF FF FF FF 00 00 00 80")
    blob_i64 = bytes.fromhex(
        "FF FF FF FF FF FF FF FF 00 00 00 00 00 00 00 80"
    )

    err, i32_neg1 = read_i32_le_checked(blob_i32, 0)
    assert err == GGUF_HEADER_OK
    assert i32_neg1 == -1

    err, i32_min = read_i32_le_checked(blob_i32, 4)
    assert err == GGUF_HEADER_OK
    assert i32_min == -(1 << 31)

    err, i64_neg1 = read_i64_le_checked(blob_i64, 0)
    assert err == GGUF_HEADER_OK
    assert i64_neg1 == -1

    err, i64_min = read_i64_le_checked(blob_i64, 8)
    assert err == GGUF_HEADER_OK
    assert i64_min == -(1 << 63)


def run() -> None:
    test_null_ptr_contracts()
    test_truncation_contracts()
    test_u16_u32_u64_little_endian_decode()
    test_signed_twos_complement_decode()
    print("gguf_header_read_endian_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
