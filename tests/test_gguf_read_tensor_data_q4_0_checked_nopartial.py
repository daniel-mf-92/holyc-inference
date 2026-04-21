#!/usr/bin/env python3
"""Synthetic fixture checks for GGUFReadTensorDataQ4_0CheckedNoPartial (IQ-925)."""

from __future__ import annotations

from pathlib import Path

GGUF_READER_OK = 0
GGUF_READER_ERR_NULL_PTR = 1
GGUF_READER_ERR_BAD_PARAM = 2
GGUF_READER_ERR_OVERFLOW = 3
GGUF_READER_ERR_TRUNCATED = 4
GGUF_READER_ERR_BAD_TYPE = 5
GGUF_READER_ERR_BAD_DIM = 6
GGUF_READER_ERR_BAD_CAPACITY = 7

GGUF_READER_GGML_TYPE_Q4_0 = 2
GGUF_READER_Q4_0_BLOCK_SIZE = 32
GGUF_READER_Q4_0_BLOCK_BYTES = 18


def gguf_read_tensor_data_q4_0_checked_nopartial(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: dict | None,
    out_q4_0_blocks: bytearray | None,
    out_q4_0_blocks_capacity: int,
    out_blocks_read: list[int] | None,
    out_bytes_read: list[int] | None,
) -> int:
    if (
        gguf_bytes is None
        or tensor_info is None
        or out_q4_0_blocks is None
        or out_blocks_read is None
        or out_bytes_read is None
    ):
        return GGUF_READER_ERR_NULL_PTR

    n_dims = int(tensor_info["n_dims"])
    ggml_type = int(tensor_info["ggml_type"])
    dims = [int(x) for x in tensor_info["dims"]]
    offset = int(tensor_info["offset"])

    if n_dims <= 0 or n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM
    if ggml_type != GGUF_READER_GGML_TYPE_Q4_0:
        return GGUF_READER_ERR_BAD_TYPE

    element_count = 1
    for dim in dims[:n_dims]:
        if dim <= 0:
            return GGUF_READER_ERR_BAD_DIM
        element_count *= dim

    if element_count % GGUF_READER_Q4_0_BLOCK_SIZE != 0:
        return GGUF_READER_ERR_BAD_DIM

    block_count = element_count // GGUF_READER_Q4_0_BLOCK_SIZE
    required_bytes = block_count * GGUF_READER_Q4_0_BLOCK_BYTES

    payload_start = tensor_data_base + offset
    payload_end = payload_start + required_bytes

    if payload_start > gguf_nbytes or payload_end > gguf_nbytes:
        return GGUF_READER_ERR_TRUNCATED
    if required_bytes > out_q4_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY

    staged = gguf_bytes[payload_start:payload_end]
    if len(staged) != required_bytes:
        return GGUF_READER_ERR_TRUNCATED

    for idx, value in enumerate(staged):
        out_q4_0_blocks[idx] = value

    out_blocks_read[0] = block_count
    out_bytes_read[0] = required_bytes
    return GGUF_READER_OK


def build_synthetic_fixture(block_count: int, tensor_data_base: int, tensor_offset: int) -> tuple[bytes, dict, bytes]:
    payload = bytes(((idx * 37 + 11) & 0xFF) for idx in range(block_count * GGUF_READER_Q4_0_BLOCK_BYTES))
    prefix = bytes([0xA5] * tensor_data_base)
    gap = bytes([0x00] * tensor_offset)
    gguf_bytes = prefix + gap + payload + bytes([0xCC] * 9)
    tensor_info = {
        "n_dims": 2,
        "dims": [block_count, GGUF_READER_Q4_0_BLOCK_SIZE, 0, 0],
        "ggml_type": GGUF_READER_GGML_TYPE_Q4_0,
        "offset": tensor_offset,
    }
    return gguf_bytes, tensor_info, payload


def test_source_contains_iq925_function() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ4_0CheckedNoPartial("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "staged_ggml_type != GGUF_READER_GGML_TYPE_Q4_0" in body
    assert "staged_element_count % GGUF_READER_Q4_0_BLOCK_SIZE" in body
    assert "staged_payload_end > gguf_nbytes" in body
    assert "for (copy_idx = 0; copy_idx < staged_required_bytes; ++copy_idx)" in body
    assert "*out_blocks_read = (I64)staged_block_count;" in body
    assert "*out_bytes_read = (I64)staged_required_bytes;" in body


def test_happy_path_synthetic_fixture() -> None:
    gguf_bytes, tensor_info, expected_payload = build_synthetic_fixture(
        block_count=3,
        tensor_data_base=64,
        tensor_offset=40,
    )

    out = bytearray(len(expected_payload) + 8)
    out_blocks = [0]
    out_bytes = [0]

    err = gguf_read_tensor_data_q4_0_checked_nopartial(
        gguf_bytes,
        len(gguf_bytes),
        64,
        tensor_info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )

    assert err == GGUF_READER_OK
    assert out_blocks[0] == 3
    assert out_bytes[0] == len(expected_payload)
    assert bytes(out[: len(expected_payload)]) == expected_payload


def test_bad_type_bad_dims_and_truncation_no_partial_publish() -> None:
    gguf_bytes, tensor_info, expected_payload = build_synthetic_fixture(
        block_count=2,
        tensor_data_base=32,
        tensor_offset=16,
    )

    out = bytearray([0xEF] * len(expected_payload))
    before = bytes(out)
    out_blocks = [777]
    out_bytes = [888]

    bad_type = dict(tensor_info)
    bad_type["ggml_type"] = 8
    err = gguf_read_tensor_data_q4_0_checked_nopartial(
        gguf_bytes,
        len(gguf_bytes),
        32,
        bad_type,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_TYPE
    assert bytes(out) == before
    assert out_blocks == [777]
    assert out_bytes == [888]

    bad_dims = dict(tensor_info)
    bad_dims["dims"] = [65, 1, 0, 0]
    err = gguf_read_tensor_data_q4_0_checked_nopartial(
        gguf_bytes,
        len(gguf_bytes),
        32,
        bad_dims,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_DIM
    assert bytes(out) == before

    err = gguf_read_tensor_data_q4_0_checked_nopartial(
        gguf_bytes,
        len(gguf_bytes) - 5,
        32,
        tensor_info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert bytes(out) == before


def test_capacity_guard_and_null_ptr_guard() -> None:
    gguf_bytes, tensor_info, expected_payload = build_synthetic_fixture(
        block_count=4,
        tensor_data_base=48,
        tensor_offset=24,
    )

    out = bytearray(len(expected_payload) - 1)
    out_blocks = [0]
    out_bytes = [0]

    err = gguf_read_tensor_data_q4_0_checked_nopartial(
        gguf_bytes,
        len(gguf_bytes),
        48,
        tensor_info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY

    err = gguf_read_tensor_data_q4_0_checked_nopartial(
        None,
        len(gguf_bytes),
        48,
        tensor_info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_NULL_PTR


def main() -> None:
    test_source_contains_iq925_function()
    test_happy_path_synthetic_fixture()
    test_bad_type_bad_dims_and_truncation_no_partial_publish()
    test_capacity_guard_and_null_ptr_guard()
    print("ok")


if __name__ == "__main__":
    main()
