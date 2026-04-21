#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ4_0...PreflightOnlyParityCommitOnly (IQ-939)."""

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

GGUF_READER_GGML_TYPE_Q4_0 = 2
GGUF_READER_Q4_0_BLOCK_SIZE = 32
GGUF_READER_Q4_0_BLOCK_BYTES = 18
U64_MAX = (1 << 64) - 1
I64_MAX = (1 << 63) - 1


@dataclass
class TensorInfoQ40:
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


def tensor_matches_snapshot(tensor_info: TensorInfoQ40, snap: TensorInfoQ40) -> bool:
    return (
        tensor_info.n_dims == snap.n_dims
        and tensor_info.ggml_type == snap.ggml_type
        and tensor_info.offset == snap.offset
        and tensor_info.dims[:4] == snap.dims[:4]
    )


def q4_preflight(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ40 | None,
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

    snap = TensorInfoQ40(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q4_0:
        return GGUF_READER_ERR_BAD_TYPE

    element_count = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM
        ok, element_count = try_mul_u64(element_count, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW

    if element_count % GGUF_READER_Q4_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM

    block_count = element_count // GGUF_READER_Q4_0_BLOCK_SIZE
    ok, required_bytes = try_mul_u64(block_count, GGUF_READER_Q4_0_BLOCK_BYTES)
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
    if required_bytes > out_q4_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM

    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW

    out_blocks_read[0] = block_count
    out_bytes_read[0] = required_bytes
    return GGUF_READER_OK


def q4_checked_no_partial(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ40 | None,
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

    snap = TensorInfoQ40(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q4_0:
        return GGUF_READER_ERR_BAD_TYPE

    element_count = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM
        ok, element_count = try_mul_u64(element_count, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW

    if element_count % GGUF_READER_Q4_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM

    block_count = element_count // GGUF_READER_Q4_0_BLOCK_SIZE
    ok, required_bytes = try_mul_u64(block_count, GGUF_READER_Q4_0_BLOCK_BYTES)
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
    if required_bytes > out_q4_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM

    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW

    staged = gguf_bytes[payload_start:payload_end]
    for idx, value in enumerate(staged):
        out_q4_0_blocks[idx] = value
    out_blocks_read[0] = block_count
    out_bytes_read[0] = required_bytes
    return GGUF_READER_OK


def q4_preflight_parity(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ40 | None,
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
    if out_blocks_read is out_bytes_read:
        return GGUF_READER_ERR_BAD_PARAM

    snap = TensorInfoQ40(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    staged_blocks = [0]
    staged_bytes = [0]
    err = q4_preflight(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        staged_blocks,
        staged_bytes,
    )
    if err != GGUF_READER_OK:
        return err

    scratch = bytearray(staged_bytes[0]) if staged_bytes[0] else bytearray(1)
    recomputed_blocks = [0]
    recomputed_bytes = [0]
    err = q4_checked_no_partial(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        scratch,
        len(scratch),
        recomputed_blocks,
        recomputed_bytes,
    )
    if err != GGUF_READER_OK:
        return err

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM
    if (staged_blocks[0], staged_bytes[0]) != (recomputed_blocks[0], recomputed_bytes[0]):
        return GGUF_READER_ERR_BAD_PARAM

    out_blocks_read[0] = staged_blocks[0]
    out_bytes_read[0] = staged_bytes[0]
    return GGUF_READER_OK


def q4_preflight_parity_commit_only(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ40 | None,
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
    if out_blocks_read is out_bytes_read:
        return GGUF_READER_ERR_BAD_PARAM

    snap = TensorInfoQ40(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )
    staged_capacity = out_q4_0_blocks_capacity

    staged_blocks = [0]
    staged_bytes = [0]
    err = q4_preflight_parity(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        staged_blocks,
        staged_bytes,
    )
    if err != GGUF_READER_OK:
        return err

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM
    if staged_capacity != out_q4_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_PARAM
    if staged_blocks[0] < 0 or staged_bytes[0] < 0:
        return GGUF_READER_ERR_OVERFLOW
    if staged_bytes[0] > staged_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY

    out_blocks_read[0] = staged_blocks[0]
    out_bytes_read[0] = staged_bytes[0]
    return GGUF_READER_OK


def test_source_contains_iq939_q4_commit_only_function() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "if (staged_out_capacity != out_q4_0_blocks_capacity)" in body
    assert "if ((U64)staged_bytes_read_i64 > staged_out_capacity)" in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body
    assert "*out_bytes_read = staged_bytes_read_i64;" in body


def test_error_paths_are_no_publish_and_no_write() -> None:
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 0)
    blob = bytes(range(128))
    out = bytearray(b"\xDA" * 128)
    out_before = bytes(out)
    out_blocks = [19]
    out_bytes = [23]

    err = q4_preflight_parity_commit_only(
        None,
        0,
        0,
        info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_NULL_PTR
    assert out_blocks == [19]
    assert out_bytes == [23]
    assert bytes(out) == out_before

    err = q4_preflight_parity_commit_only(
        blob,
        len(blob),
        0,
        info,
        out,
        len(out),
        out_blocks,
        out_blocks,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert out_blocks == [19]
    assert bytes(out) == out_before

    bad_cap_info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 4)
    err = q4_preflight_parity_commit_only(
        b"\x00" * 64,
        64,
        0,
        bad_cap_info,
        out,
        8,
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert out_blocks == [19]
    assert out_bytes == [23]
    assert bytes(out) == out_before


def test_success_publishes_exact_tuple_and_preserves_output_buffer() -> None:
    payload = bytes((idx * 5 + 1) & 0xFF for idx in range(36))
    blob = b"\x01\x02\x03\x04" + payload + b"\xEF" * 7
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 4)

    out = bytearray(b"\xAA" * 96)
    out_before = bytes(out)
    out_blocks = [0]
    out_bytes = [0]

    err = q4_preflight_parity_commit_only(
        blob,
        len(blob),
        0,
        info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_OK
    assert out_blocks == [2]
    assert out_bytes == [36]
    assert bytes(out) == out_before


def run() -> None:
    test_source_contains_iq939_q4_commit_only_function()
    test_error_paths_are_no_publish_and_no_write()
    test_success_publishes_exact_tuple_and_preserves_output_buffer()
    print("gguf_read_tensor_data_q4_0_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok")


if __name__ == "__main__":
    run()
