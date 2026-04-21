#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnly (IQ-934)."""

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


def gguf_read_tensor_data_q8_0_checked_no_partial(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ80 | None,
    out_q8_0_blocks: bytearray | None,
    out_q8_0_blocks_capacity: int,
) -> tuple[int, int | None, int | None]:
    if gguf_bytes is None or tensor_info is None or out_q8_0_blocks is None:
        return GGUF_READER_ERR_NULL_PTR, None, None

    snap = TensorInfoQ80(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM, None, None
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q8_0:
        return GGUF_READER_ERR_BAD_TYPE, None, None

    element_count = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM, None, None
        ok, element_count = try_mul_u64(element_count, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW, None, None

    if element_count % GGUF_READER_Q8_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM, None, None

    block_count = element_count // GGUF_READER_Q8_0_BLOCK_SIZE
    ok, required_bytes = try_mul_u64(block_count, GGUF_READER_Q8_0_BLOCK_BYTES)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None

    ok, payload_start = try_add_u64(tensor_data_base, snap.offset)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None
    ok, payload_end = try_add_u64(payload_start, required_bytes)
    if not ok:
        return GGUF_READER_ERR_OVERFLOW, None, None

    if payload_start > gguf_nbytes or payload_end > gguf_nbytes:
        return GGUF_READER_ERR_TRUNCATED, None, None
    if required_bytes > out_q8_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY, None, None

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM, None, None

    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW, None, None

    staged = gguf_bytes[payload_start:payload_end]
    for idx, value in enumerate(staged):
        out_q8_0_blocks[idx] = value

    return GGUF_READER_OK, block_count, required_bytes


def gguf_read_tensor_data_q8_0_checked_no_partial_commit_only(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ80 | None,
    out_q8_0_blocks: bytearray | None,
    out_q8_0_blocks_capacity: int,
    out_blocks_read: list[int] | None,
    out_bytes_read: list[int] | None,
    mutate_before_commit_snapshot_revalidate: bool = False,
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

    err, staged_blocks, staged_bytes = gguf_read_tensor_data_q8_0_checked_no_partial(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q8_0_blocks,
        out_q8_0_blocks_capacity,
    )
    if err != GGUF_READER_OK:
        return err

    if mutate_before_commit_snapshot_revalidate:
        tensor_info.dims[0] += 32

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM

    out_blocks_read[0] = staged_blocks
    out_bytes_read[0] = staged_bytes
    return GGUF_READER_OK


def test_source_contains_iq934_function_and_commit_only_staging() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ8_0CheckedNoPartialCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = GGUFReadTensorDataQ8_0CheckedNoPartial(" in body
    assert "staged_blocks_read_i64" in body
    assert "staged_bytes_read_i64" in body
    assert "if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info," in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body
    assert "*out_bytes_read = staged_bytes_read_i64;" in body


def test_null_and_truncation_do_not_publish_diagnostics() -> None:
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 0)
    out = bytearray(b"\xAA" * 68)

    out_blocks = [111]
    out_bytes = [222]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only(
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
    assert out_blocks == [111]
    assert out_bytes == [222]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only(
        b"\x10" * 16,
        16,
        16,
        info,
        out,
        len(out),
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert out_blocks == [111]
    assert out_bytes == [222]


def test_capacity_and_snapshot_mismatch_vectors() -> None:
    payload = bytes((idx * 7 + 1) & 0xFF for idx in range(68))
    blob = b"\xF0\xF1\xF2\xF3" + payload + b"\xEE" * 5
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)

    out = bytearray(b"\x55" * 96)
    out_blocks = [301]
    out_bytes = [302]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only(
        blob,
        len(blob),
        0,
        info,
        out,
        34,
        out_blocks,
        out_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert out_blocks == [301]
    assert out_bytes == [302]
    assert out == bytearray(b"\x55" * 96)

    info2 = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)
    out2 = bytearray(b"\x66" * 96)
    out_blocks2 = [401]
    out_bytes2 = [402]
    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only(
        blob,
        len(blob),
        0,
        info2,
        out2,
        len(out2),
        out_blocks2,
        out_bytes2,
        mutate_before_commit_snapshot_revalidate=True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert out_blocks2 == [401]
    assert out_bytes2 == [402]


def test_success_publishes_exact_blocks_and_bytes() -> None:
    payload = bytes((idx * 13 + 5) & 0xFF for idx in range(68))
    blob = b"\x90\x91\x92\x93" + payload + b"\xED" * 7
    info = TensorInfoQ80(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q8_0, 4)

    out = bytearray(b"\x77" * 128)
    out_blocks = [-1]
    out_bytes = [-1]

    err = gguf_read_tensor_data_q8_0_checked_no_partial_commit_only(
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
    assert out_bytes == [68]
    assert bytes(out[:68]) == payload
    assert bytes(out[68:]) == bytes([0x77] * (len(out) - 68))


def run() -> None:
    test_source_contains_iq934_function_and_commit_only_staging()
    test_null_and_truncation_do_not_publish_diagnostics()
    test_capacity_and_snapshot_mismatch_vectors()
    test_success_publishes_exact_blocks_and_bytes()
    print("gguf_read_tensor_data_q8_0_checked_nopartial_commit_only=ok")


if __name__ == "__main__":
    run()
