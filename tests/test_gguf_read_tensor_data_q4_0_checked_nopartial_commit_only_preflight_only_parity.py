#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-937)."""

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


def gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only(
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


def gguf_read_tensor_data_q4_0_checked_no_partial(
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


def gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity(
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
    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only(
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
    err = gguf_read_tensor_data_q4_0_checked_no_partial(
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


def test_source_contains_iq937_q4_parity_function_and_staging_contract() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split("\nI32 ", 1)[0]

    assert "status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "status = GGUFReadTensorDataQ4_0CheckedNoPartial(" in body
    assert "if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||" in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body
    assert "*out_bytes_read = staged_bytes_read_i64;" in body


def test_bad_vectors_and_no_publish_behavior() -> None:
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 0)
    blob = bytes(range(64))
    out = bytearray(b"\xAA" * 64)
    out_before = bytes(out)
    out_blocks = [101]
    out_bytes = [202]

    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity(
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
    assert out_blocks == [101]
    assert out_bytes == [202]
    assert bytes(out) == out_before

    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity(
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
    assert out_blocks == [101]
    assert bytes(out) == out_before


def test_success_parity_and_output_immutability() -> None:
    payload = bytes((idx * 11 + 7) & 0xFF for idx in range(36))
    blob = b"\xA0\xA1\xA2\xA3" + payload + b"\xFE" * 9
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 4)

    out = bytearray(b"\xCC" * 72)
    out_before = bytes(out)
    out_blocks = [0]
    out_bytes = [0]

    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity(
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
    test_source_contains_iq937_q4_parity_function_and_staging_contract()
    test_bad_vectors_and_no_publish_behavior()
    test_success_parity_and_output_immutability()
    print("gguf_read_tensor_data_q4_0_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
