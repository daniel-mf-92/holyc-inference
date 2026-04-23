#!/usr/bin/env python3
"""Parity/spec harness for GGUFReaderLoadTensorSliceQ8_0Checked (IQ-1219)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


GGUF_READER_OK = 0
GGUF_READER_ERR_NULL_PTR = 1
GGUF_READER_ERR_BAD_PARAM = 2
GGUF_READER_ERR_OVERFLOW = 3
GGUF_READER_ERR_TRUNCATED = 4
GGUF_READER_ERR_BAD_TYPE = 5
GGUF_READER_ERR_BAD_DIM = 6
GGUF_READER_ERR_BAD_CAPACITY = 7

GGUF_READER_GGML_TYPE_Q8_0 = 8
GGUF_READER_Q8_0_BLOCK_SIZE = 32
GGUF_READER_Q8_0_BLOCK_BYTES = 34
U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1


@dataclass
class TensorInfoQ80:
    n_dims: int
    dims: list[int]
    ggml_type: int
    offset: int


def try_add_u64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs > U64_MAX - rhs:
        return False, 0
    return True, lhs + rhs


def try_mul_u64(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > U64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def tensor_matches_snapshot(tensor_info: TensorInfoQ80, snap: TensorInfoQ80) -> bool:
    return (
        tensor_info.n_dims == snap.n_dims
        and tensor_info.ggml_type == snap.ggml_type
        and tensor_info.offset == snap.offset
        and tensor_info.dims[:4] == snap.dims[:4]
    )


def gguf_reader_load_tensor_slice_q8_0_checked(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ80 | None,
    slice_block_start: int,
    slice_block_count: int,
    out_q8_0_blocks: bytearray | None,
    out_q8_0_blocks_capacity: int,
    out_blocks_read_present: bool,
    out_bytes_read_present: bool,
    out_slice_payload_offset_present: bool,
    mutate_before_revalidate: bool = False,
) -> tuple[int, int | None, int | None, int | None]:
    if (
        gguf_bytes is None
        or tensor_info is None
        or out_q8_0_blocks is None
        or not out_blocks_read_present
        or not out_bytes_read_present
        or not out_slice_payload_offset_present
    ):
        return GGUF_READER_ERR_NULL_PTR, None, None, None

    snap = TensorInfoQ80(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM, None, None, None
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q8_0:
        return GGUF_READER_ERR_BAD_TYPE, None, None, None

    total_elements = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM, None, None, None
        ok, total_elements = try_mul_u64(total_elements, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW, None, None, None

    if total_elements % GGUF_READER_Q8_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM, None, None, None

    total_blocks = total_elements // GGUF_READER_Q8_0_BLOCK_SIZE

    if slice_block_start > total_blocks:
        return GGUF_READER_ERR_BAD_PARAM, None, None, None
    ok, slice_end_block = try_add_u64(slice_block_start, slice_block_count)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None
    if slice_end_block > total_blocks:
        return GGUF_READER_ERR_BAD_PARAM, None, None, None

    ok, tensor_bytes = try_mul_u64(total_blocks, GGUF_READER_Q8_0_BLOCK_BYTES)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None
    ok, slice_byte_offset = try_mul_u64(slice_block_start, GGUF_READER_Q8_0_BLOCK_BYTES)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None
    ok, slice_bytes = try_mul_u64(slice_block_count, GGUF_READER_Q8_0_BLOCK_BYTES)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None

    ok, tensor_payload_start = try_add_u64(tensor_data_base, snap.offset)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None
    ok, tensor_payload_end = try_add_u64(tensor_payload_start, tensor_bytes)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None

    if tensor_payload_start > gguf_nbytes or tensor_payload_end > gguf_nbytes:
        return GGUF_READER_ERR_TRUNCATED, None, None, None

    ok, slice_payload_start = try_add_u64(tensor_payload_start, slice_byte_offset)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None
    ok, slice_payload_end = try_add_u64(slice_payload_start, slice_bytes)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None, None

    if slice_payload_start > tensor_payload_end:
        return GGUF_READER_ERR_BAD_PARAM, None, None, None
    if slice_payload_end > tensor_payload_end:
        return GGUF_READER_ERR_BAD_PARAM, None, None, None
    if slice_bytes > out_q8_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY, None, None, None

    if mutate_before_revalidate:
        tensor_info.offset += 1

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM, None, None, None

    if slice_block_count > I64_MAX or slice_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW, None, None, None

    staged = gguf_bytes[slice_payload_start:slice_payload_end]
    for idx, value in enumerate(staged):
        out_q8_0_blocks[idx] = value

    return GGUF_READER_OK, slice_block_count, slice_bytes, slice_payload_start


def test_source_contains_iq1219_function_and_slice_guards() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReaderLoadTensorSliceQ8_0Checked("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "if (slice_block_start > staged_total_blocks)" in body
    assert "if (staged_slice_end_block > staged_total_blocks)" in body
    assert "if (staged_slice_payload_end > staged_tensor_payload_end)" in body
    assert "*out_slice_payload_offset = staged_slice_payload_start;" in body


def test_null_and_basic_parameter_guards() -> None:
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 0)
    out = bytearray(68)

    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        None, 0, 0, info, 0, 1, out, len(out), True, True, True
    )
    assert err == GGUF_READER_ERR_NULL_PTR
    assert blocks is None and nbytes is None and offset is None

    bad_type = TensorInfoQ80(1, [64, 0, 0, 0], 2, 0)
    err, _, _, _ = gguf_reader_load_tensor_slice_q8_0_checked(
        b"\x00" * 96, 96, 0, bad_type, 0, 1, out, len(out), True, True, True
    )
    assert err == GGUF_READER_ERR_BAD_TYPE

    bad_dim = TensorInfoQ80(1, [33, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 0)
    err, _, _, _ = gguf_reader_load_tensor_slice_q8_0_checked(
        b"\x00" * 128, 128, 0, bad_dim, 0, 1, out, len(out), True, True, True
    )
    assert err == GGUF_READER_ERR_BAD_DIM

    err, _, _, _ = gguf_reader_load_tensor_slice_q8_0_checked(
        b"\x00" * 128, 128, 0, info, 3, 1, out, len(out), True, True, True
    )
    assert err == GGUF_READER_ERR_BAD_PARAM


def test_range_truncation_capacity_mutation_and_success() -> None:
    # 3 blocks payload, tensor starts at byte 7.
    payload = bytes((idx * 17 + 5) & 0xFF for idx in range(3 * GGUF_READER_Q8_0_BLOCK_BYTES))
    blob = bytes([0xAA] * 7) + payload + bytes([0xBB] * 9)
    info = TensorInfoQ80(1, [96, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 7)
    out = bytearray(b"\xCC" * 200)
    before = bytes(out)

    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        blob,
        len(blob),
        0,
        info,
        2,
        2,
        out,
        len(out),
        True,
        True,
        True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert blocks is None and nbytes is None and offset is None
    assert bytes(out) == before

    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        blob,
        108,
        0,
        info,
        1,
        2,
        out,
        len(out),
        True,
        True,
        True,
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert blocks is None and nbytes is None and offset is None
    assert bytes(out) == before

    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        blob,
        len(blob),
        0,
        info,
        1,
        2,
        out,
        GGUF_READER_Q8_0_BLOCK_BYTES,
        True,
        True,
        True,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert blocks is None and nbytes is None and offset is None
    assert bytes(out) == before

    info_mut = TensorInfoQ80(1, [96, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 7)
    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        blob,
        len(blob),
        0,
        info_mut,
        1,
        1,
        out,
        len(out),
        True,
        True,
        True,
        mutate_before_revalidate=True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert blocks is None and nbytes is None and offset is None
    assert bytes(out) == before

    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        blob,
        len(blob),
        0,
        info,
        1,
        2,
        out,
        len(out),
        True,
        True,
        True,
    )
    assert err == GGUF_READER_OK
    assert blocks == 2
    assert nbytes == 2 * GGUF_READER_Q8_0_BLOCK_BYTES
    assert offset == 7 + GGUF_READER_Q8_0_BLOCK_BYTES
    expected = payload[GGUF_READER_Q8_0_BLOCK_BYTES : 3 * GGUF_READER_Q8_0_BLOCK_BYTES]
    assert bytes(out[:nbytes]) == expected


def test_zero_length_slice_is_valid_and_no_write() -> None:
    payload = bytes((idx * 3 + 9) & 0xFF for idx in range(2 * GGUF_READER_Q8_0_BLOCK_BYTES))
    blob = bytes([0x11] * 4) + payload
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    out = bytearray(b"\xF0" * 40)
    before = bytes(out)

    err, blocks, nbytes, offset = gguf_reader_load_tensor_slice_q8_0_checked(
        blob,
        len(blob),
        0,
        info,
        2,
        0,
        out,
        len(out),
        True,
        True,
        True,
    )
    assert err == GGUF_READER_OK
    assert blocks == 0
    assert nbytes == 0
    assert offset == 4 + 2 * GGUF_READER_Q8_0_BLOCK_BYTES
    assert bytes(out) == before


def run() -> None:
    test_source_contains_iq1219_function_and_slice_guards()
    test_null_and_basic_parameter_guards()
    test_range_truncation_capacity_mutation_and_success()
    test_zero_length_slice_is_valid_and_no_write()
    print("gguf_reader_load_tensor_slice_q8_0_checked=ok")


if __name__ == "__main__":
    run()
