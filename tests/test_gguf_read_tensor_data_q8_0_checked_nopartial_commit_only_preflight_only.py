#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly (IQ-935)."""

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


def test_source_contains_iq935_function_and_preflight_no_write_path() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "// Preflight-only: never writes out_q8_0_blocks, only diagnostics." in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body
    assert "*out_bytes_read = staged_bytes_read_i64;" in body
    assert "for (copy_idx = 0;" not in body


def test_null_and_bad_input_vectors_do_not_publish_or_mutate_output() -> None:
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 0)
    out = bytearray(b"\xA5" * 96)
    out_before = bytes(out)
    out_blocks = [111]
    out_bytes = [222]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
        None, 0, 0, info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_NULL_PTR
    assert out_blocks == [111]
    assert out_bytes == [222]
    assert bytes(out) == out_before

    bad_dim = TensorInfoQ80(1, [33, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 0)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
        b"\x00" * 128, 128, 0, bad_dim, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_BAD_DIM
    assert out_blocks == [111]
    assert out_bytes == [222]
    assert bytes(out) == out_before

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
        b"\x00" * 32, 32, 24, info, out, len(out), out_blocks, out_bytes
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert out_blocks == [111]
    assert out_bytes == [222]
    assert bytes(out) == out_before


def test_capacity_snapshot_mismatch_and_success_publish_exact_counts() -> None:
    payload = bytes((idx * 3 + 9) & 0xFF for idx in range(68))
    blob = b"\x10\x11\x12\x13" + payload + b"\xEF" * 7
    out = bytearray(b"\xCC" * 128)
    out_before = bytes(out)

    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    out_blocks = [301]
    out_bytes = [302]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
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
    assert out_blocks == [301]
    assert out_bytes == [302]
    assert bytes(out) == out_before

    info2 = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
        blob,
        len(blob),
        0,
        info2,
        out,
        len(out),
        out_blocks,
        out_bytes,
        mutate_before_revalidate=True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert out_blocks == [301]
    assert out_bytes == [302]
    assert bytes(out) == out_before

    info3 = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only_preflight_only(
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
    test_source_contains_iq935_function_and_preflight_no_write_path()
    test_null_and_bad_input_vectors_do_not_publish_or_mutate_output()
    test_capacity_snapshot_mismatch_and_success_publish_exact_counts()
    print("gguf_read_tensor_data_q8_0_checked_nopartial_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
