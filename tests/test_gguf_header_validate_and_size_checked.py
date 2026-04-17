#!/usr/bin/env python3
"""Reference checks for GGUFHeaderValidateAndSizeChecked semantics."""

from __future__ import annotations

GGUF_MAGIC_U32 = 0x46554747
GGUF_VERSION_V2 = 2
GGUF_VERSION_V3 = 3
GGUF_HEADER_BYTES = 24

GGUF_HEADER_OK = 0
GGUF_HEADER_ERR_NULL_PTR = 1
GGUF_HEADER_ERR_BAD_PARAM = 2
GGUF_HEADER_ERR_OVERFLOW = 3
GGUF_HEADER_ERR_TRUNCATED = 4
GGUF_HEADER_ERR_BAD_MAGIC = 5
GGUF_HEADER_ERR_BAD_VERSION = 6

I64_MAX_VALUE = (1 << 63) - 1


def validate_and_size_checked(
    header: dict[str, int] | None,
    out_header_bytes_present: bool = True,
):
    if header is None or not out_header_bytes_present:
        return GGUF_HEADER_ERR_NULL_PTR, None

    if header["magic"] != GGUF_MAGIC_U32:
        return GGUF_HEADER_ERR_BAD_MAGIC, None

    if header["version"] not in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        return GGUF_HEADER_ERR_BAD_VERSION, None

    if header["tensor_count"] > I64_MAX_VALUE:
        return GGUF_HEADER_ERR_OVERFLOW, None

    if header["metadata_kv_count"] > I64_MAX_VALUE:
        return GGUF_HEADER_ERR_OVERFLOW, None

    return GGUF_HEADER_OK, GGUF_HEADER_BYTES


def make_header(
    *,
    magic: int = GGUF_MAGIC_U32,
    version: int = GGUF_VERSION_V3,
    tensor_count: int = 3,
    metadata_kv_count: int = 5,
) -> dict[str, int]:
    return {
        "magic": magic,
        "version": version,
        "tensor_count": tensor_count,
        "metadata_kv_count": metadata_kv_count,
    }


def test_null_ptr_contracts() -> None:
    err, _ = validate_and_size_checked(None)
    assert err == GGUF_HEADER_ERR_NULL_PTR

    err, _ = validate_and_size_checked(make_header(), out_header_bytes_present=False)
    assert err == GGUF_HEADER_ERR_NULL_PTR


def test_magic_and_version_validation() -> None:
    err, _ = validate_and_size_checked(make_header(magic=0xDEADBEEF))
    assert err == GGUF_HEADER_ERR_BAD_MAGIC

    err, _ = validate_and_size_checked(make_header(version=1))
    assert err == GGUF_HEADER_ERR_BAD_VERSION


def test_count_overflow_contracts() -> None:
    err, _ = validate_and_size_checked(make_header(tensor_count=I64_MAX_VALUE + 1))
    assert err == GGUF_HEADER_ERR_OVERFLOW

    err, _ = validate_and_size_checked(make_header(metadata_kv_count=I64_MAX_VALUE + 1))
    assert err == GGUF_HEADER_ERR_OVERFLOW


def test_success_v2_v3_and_empty_counts() -> None:
    for version in (GGUF_VERSION_V2, GGUF_VERSION_V3):
        err, consumed = validate_and_size_checked(
            make_header(version=version, tensor_count=0, metadata_kv_count=0)
        )
        assert err == GGUF_HEADER_OK
        assert consumed == GGUF_HEADER_BYTES


def run() -> None:
    test_null_ptr_contracts()
    test_magic_and_version_validation()
    test_count_overflow_contracts()
    test_success_v2_v3_and_empty_counts()
    print("gguf_header_validate_and_size_checked_reference_checks=ok")


if __name__ == "__main__":
    run()
