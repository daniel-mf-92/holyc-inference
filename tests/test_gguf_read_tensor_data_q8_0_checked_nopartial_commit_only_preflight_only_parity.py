#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParity (IQ-938)."""

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
    mutate_before_revalidate: bool = False,
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

    if mutate_before_revalidate:
        tensor_info.offset += 1

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM

    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW

    out_blocks_read[0] = block_count
    out_bytes_read[0] = required_bytes
    return GGUF_READER_OK


def gguf_read_tensor_data_q8_0_checked_no_partial(
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

    staged = gguf_bytes[payload_start:payload_end]
    for idx, value in enumerate(staged):
        out_q8_0_blocks[idx] = value

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
    mutate_between_calls: bool = False,
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

    if mutate_between_calls:
        tensor_info.offset += 1

    recomputed = bytearray(staged_bytes[0])
    recomputed_blocks = [0]
    recomputed_bytes = [0]
    err = gguf_read_tensor_data_q8_0_checked_no_partial(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        recomputed,
        len(recomputed),
        recomputed_blocks,
        recomputed_bytes,
    )
    if err != GGUF_READER_OK:
        return err

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM

    if staged_blocks[0] != recomputed_blocks[0] or staged_bytes[0] != recomputed_bytes[0]:
        return GGUF_READER_ERR_BAD_PARAM

    out_blocks_read[0] = staged_blocks[0]
    out_bytes_read[0] = staged_bytes[0]
    return GGUF_READER_OK


def test_source_contains_iq938_function_and_parity_core_calls() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "GGUFReadTensorDataQ8_0CheckedNoPartial(" in body
    assert "if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||" in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body
    assert "*out_bytes_read = staged_bytes_read_i64;" in body


def test_null_alias_and_bad_inputs_do_not_publish_or_mutate_output() -> None:
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    out = bytearray(b"\xD7" * 160)
    out_before = bytes(out)
    out_blocks = [99]
    out_bytes = [100]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        None, 0, 0, info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_NULL_PTR
    assert out_blocks == [99]
    assert out_bytes == [100]
    assert bytes(out) == out_before

    bad_dim_info = TensorInfoQ80(1, [65, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 0)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        b"\x00" * 200, 200, 0, bad_dim_info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_BAD_DIM
    assert out_blocks == [99]
    assert out_bytes == [100]
    assert bytes(out) == out_before

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        b"\x00" * 32, 32, 0, info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert out_blocks == [99]
    assert out_bytes == [100]
    assert bytes(out) == out_before


def test_capacity_snapshot_parity_and_success_publish_exact_counts() -> None:
    payload = bytes((idx * 5 + 7) & 0xFF for idx in range(68))
    blob = b"\xFA\xFB\xFC\xFD" + payload + b"\xAA" * 9

    out = bytearray(b"\xA1" * 256)
    out_before = bytes(out)

    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    out_blocks = [411]
    out_bytes = [412]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob,
        len(blob),
        0,
        info,
        out,
        67,
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert out_blocks == [411]
    assert out_bytes == [412]
    assert bytes(out) == out_before

    info2 = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob,
        len(blob),
        0,
        info2,
        out,
        len(out),
        out_blocks,
        out_bytes,
        mutate_between_calls=True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert out_blocks == [411]
    assert out_bytes == [412]
    assert bytes(out) == out_before

    info3 = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only_parity(
        blob,
        len(blob),
        0,
        info3,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_OK
    assert out_blocks == [2]
    assert out_bytes == [68]
    assert bytes(out) == out_before


def run() -> None:
    test_source_contains_iq938_function_and_parity_core_calls()
    test_null_alias_and_bad_inputs_do_not_publish_or_mutate_output()
    test_capacity_snapshot_parity_and_success_publish_exact_counts()
    print("gguf_read_tensor_data_q8_0_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
