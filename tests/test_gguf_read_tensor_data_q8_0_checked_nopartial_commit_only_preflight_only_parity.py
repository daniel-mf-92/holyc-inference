#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-945)."""

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


def gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ80 | None,
    out_q8_0_blocks: bytearray | None,
    out_q8_0_blocks_capacity: int,
    out_blocks_read: list[int] | None,
    out_bytes_read: list[int] | None,
) -> int:
    if (
        gguf_bytes is None
        or tensor_info is None
        or out_q8_0_blocks is None
        or out_blocks_read is None
        or out_bytes_read is None
    ):
        return GGUF_READER_ERR_NULL_PTR

    snap = TensorInfoQ80(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q8_0:
        return GGUF_READER_ERR_BAD_TYPE

    element_count = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM
        ok, element_count = try_mul_u64(element_count, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW

    if element_count % GGUF_READER_Q8_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM

    block_count = element_count // GGUF_READER_Q8_0_BLOCK_SIZE
    ok, required_bytes = try_mul_u64(block_count, GGUF_READER_Q8_0_BLOCK_BYTES)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW

    ok, payload_start = try_add_u64(tensor_data_base, snap.offset)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW
    ok, payload_end = try_add_u64(payload_start, required_bytes)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW

    if payload_start > gguf_nbytes or payload_end > gguf_nbytes:
        return GGUF_READER_ERR_TRUNCATED
    if required_bytes > out_q8_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM

    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW

    out_blocks_read[0] = block_count
    out_bytes_read[0] = required_bytes
    return GGUF_READER_OK


def gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ80 | None,
    out_q8_0_blocks: bytearray | None,
    out_q8_0_blocks_capacity: int,
    out_blocks_read: list[int] | None,
    out_bytes_read: list[int] | None,
    mutate_between_staging_and_revalidate: bool = False,
) -> int:
    if (
        gguf_bytes is None
        or tensor_info is None
        or out_q8_0_blocks is None
        or out_blocks_read is None
        or out_bytes_read is None
    ):
        return GGUF_READER_ERR_NULL_PTR
    if out_blocks_read is out_bytes_read:
        return GGUF_READER_ERR_BAD_PARAM

    snap = TensorInfoQ80(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )
    staged_capacity = out_q8_0_blocks_capacity

    staged_blocks = [0]
    staged_bytes = [0]
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
        staged_blocks,
        staged_bytes,
    )
    if err != GGUF_READER_OK:
        return err

    if mutate_between_staging_and_revalidate:
        tensor_info.offset += 1

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q8_0:
        return GGUF_READER_ERR_BAD_TYPE

    element_count = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM
        ok, element_count = try_mul_u64(element_count, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW

    if element_count % GGUF_READER_Q8_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM

    block_count = element_count // GGUF_READER_Q8_0_BLOCK_SIZE
    ok, required_bytes = try_mul_u64(block_count, GGUF_READER_Q8_0_BLOCK_BYTES)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW

    ok, payload_start = try_add_u64(tensor_data_base, snap.offset)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW
    ok, payload_end = try_add_u64(payload_start, required_bytes)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW

    if payload_start > gguf_nbytes or payload_end > gguf_nbytes:
        return GGUF_READER_ERR_TRUNCATED
    if required_bytes > staged_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY
    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM
    if staged_capacity != out_q8_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_PARAM

    if staged_blocks[0] != block_count or staged_bytes[0] != required_bytes:
        return GGUF_READER_ERR_BAD_PARAM

    out_blocks_read[0] = staged_blocks[0]
    out_bytes_read[0] = staged_bytes[0]
    return GGUF_READER_OK


def test_source_contains_iq945_function_and_geometry_parity_logic() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (!GGUFReaderTryMulU64(recomputed_block_count," in body
    assert "if (!GGUFReaderTryAddU64(tensor_data_base," in body
    assert "if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||" in body
    assert "if (staged_out_capacity != out_q8_0_blocks_capacity)" in body
    assert "(U8 *)out_blocks_read == (U8 *)tensor_info" in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body


def test_truncation_capacity_snapshot_mismatch_and_success() -> None:
    payload = bytes((idx * 3 + 11) & 0xFF for idx in range(68))
    blob = b"\x01\x02\x03\x04" + payload + b"\x99" * 12

    out = bytearray(b"\xCC" * 256)
    out_before = bytes(out)
    out_blocks = [777]
    out_bytes = [888]

    trunc_info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 40)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob, len(blob), 0, trunc_info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert out_blocks == [777]
    assert out_bytes == [888]
    assert bytes(out) == out_before

    cap_info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob, len(blob), 0, cap_info, out, 67, out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert out_blocks == [777]
    assert out_bytes == [888]
    assert bytes(out) == out_before

    snap_info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob,
        len(blob),
        0,
        snap_info,
        out,
        len(out),
        out_blocks,
        out_bytes,
        mutate_between_staging_and_revalidate=True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert out_blocks == [777]
    assert out_bytes == [888]
    assert bytes(out) == out_before

    ok_info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob, len(blob), 0, ok_info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_OK
    assert out_blocks == [2]
    assert out_bytes == [68]
    assert bytes(out) == out_before


def run() -> None:
    test_source_contains_iq945_function_and_geometry_parity_logic()
    test_truncation_capacity_snapshot_mismatch_and_success()
    print("gguf_read_tensor_data_q8_0_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
