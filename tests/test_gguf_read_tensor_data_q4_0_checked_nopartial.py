#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ4_0CheckedNoPartial (IQ-925)."""

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


def gguf_read_tensor_data_q4_0_checked_no_partial(
    gguf_bytes: bytes | None,
    gguf_nbytes: int,
    tensor_data_base: int,
    tensor_info: TensorInfoQ40 | None,
    out_q4_0_blocks: bytearray | None,
    out_q4_0_blocks_capacity: int,
    out_blocks_read_present: bool,
    out_bytes_read_present: bool,
    mutate_before_revalidate: bool = False,
) -> tuple[int, int | None, int | None]:
    if (
        gguf_bytes is None
        or tensor_info is None
        or out_q4_0_blocks is None
        or not out_blocks_read_present
        or not out_bytes_read_present
    ):
        return GGUF_READER_ERR_NULL_PTR, None, None

    snap = TensorInfoQ40(
        n_dims=tensor_info.n_dims,
        dims=list(tensor_info.dims[:4]),
        ggml_type=tensor_info.ggml_type,
        offset=tensor_info.offset,
    )

    if snap.n_dims <= 0 or snap.n_dims > 4:
        return GGUF_READER_ERR_BAD_DIM, None, None
    if snap.ggml_type != GGUF_READER_GGML_TYPE_Q4_0:
        return GGUF_READER_ERR_BAD_TYPE, None, None

    element_count = 1
    for axis in range(snap.n_dims):
        axis_dim = snap.dims[axis]
        if axis_dim <= 0:
            return GGUF_READER_ERR_BAD_DIM, None, None
        ok, element_count = try_mul_u64(element_count, axis_dim)
        if not ok:
            return GGUF_READER_ERR_OVERFLOW, None, None

    if element_count % GGUF_READER_Q4_0_BLOCK_SIZE:
        return GGUF_READER_ERR_BAD_DIM, None, None

    block_count = element_count // GGUF_READER_Q4_0_BLOCK_SIZE
    ok, required_bytes = try_mul_u64(block_count, GGUF_READER_Q4_0_BLOCK_BYTES)
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
    if required_bytes > out_q4_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY, None, None

    if mutate_before_revalidate:
        tensor_info.offset += 1

    if not tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM, None, None

    if block_count > I64_MAX or required_bytes > I64_MAX:
        return GGUF_READER_ERR_OVERFLOW, None, None

    staged = gguf_bytes[payload_start:payload_end]
    for idx, value in enumerate(staged):
        out_q4_0_blocks[idx] = value

    return GGUF_READER_OK, block_count, required_bytes


def test_source_contains_iq925_function_and_snapshot_guard() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ4_0CheckedNoPartial("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "GGUFReaderTensorInfoQ4_0MatchesSnapshot(" in source
    assert "if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(tensor_info," in body
    assert "staged_blocks_read_i64" in body
    assert "staged_bytes_read_i64" in body


def test_null_and_dim_type_guards() -> None:
    info = TensorInfoQ40(1, [32, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 0)
    out = bytearray(18)

    err, blocks, nbytes = gguf_read_tensor_data_q4_0_checked_no_partial(
        None,
        0,
        0,
        info,
        out,
        len(out),
        True,
        True,
    )
    assert err == GGUF_READER_ERR_NULL_PTR
    assert blocks is None and nbytes is None

    bad_type = TensorInfoQ40(1, [32, 0, 0, 0], 1, 0)
    err, _, _ = gguf_read_tensor_data_q4_0_checked_no_partial(
        b"\x00" * 64,
        64,
        0,
        bad_type,
        out,
        len(out),
        True,
        True,
    )
    assert err == GGUF_READER_ERR_BAD_TYPE

    bad_dim = TensorInfoQ40(1, [33, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 0)
    err, _, _ = gguf_read_tensor_data_q4_0_checked_no_partial(
        b"\x00" * 128,
        128,
        0,
        bad_dim,
        out,
        len(out),
        True,
        True,
    )
    assert err == GGUF_READER_ERR_BAD_DIM


def test_truncated_and_capacity_preserve_output_buffer() -> None:
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 8)
    out = bytearray(b"\xAA" * 40)
    before = bytes(out)

    err, blocks, nbytes = gguf_read_tensor_data_q4_0_checked_no_partial(
        b"\x11" * 32,
        32,
        16,
        info,
        out,
        len(out),
        True,
        True,
    )
    assert err == GGUF_READER_ERR_TRUNCATED
    assert blocks is None and nbytes is None
    assert bytes(out) == before

    info_ok = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 0)
    err, blocks, nbytes = gguf_read_tensor_data_q4_0_checked_no_partial(
        bytes(range(64)),
        64,
        0,
        info_ok,
        out,
        8,
        True,
        True,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert blocks is None and nbytes is None
    assert bytes(out) == before


def test_snapshot_revalidation_and_success_fixture_copy() -> None:
    payload = bytes((idx * 7) & 0xFF for idx in range(36))
    prefix = bytes([0xF1, 0xF2, 0xF3, 0xF4])
    suffix = bytes([0xE1] * 10)
    blob = prefix + payload + suffix

    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 4)
    out = bytearray(b"\x55" * 48)

    err, blocks, nbytes = gguf_read_tensor_data_q4_0_checked_no_partial(
        blob,
        len(blob),
        0,
        info,
        out,
        len(out),
        True,
        True,
        mutate_before_revalidate=True,
    )
    assert err == GGUF_READER_ERR_BAD_PARAM
    assert blocks is None and nbytes is None
    assert out == bytearray(b"\x55" * 48)

    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 4)
    err, blocks, nbytes = gguf_read_tensor_data_q4_0_checked_no_partial(
        blob,
        len(blob),
        0,
        info,
        out,
        len(out),
        True,
        True,
    )
    assert err == GGUF_READER_OK
    assert blocks == 2
    assert nbytes == 36
    assert bytes(out[:36]) == payload
    assert bytes(out[36:]) == bytes([0x55] * 12)


def run() -> None:
    test_source_contains_iq925_function_and_snapshot_guard()
    test_null_and_dim_type_guards()
    test_truncated_and_capacity_preserve_output_buffer()
    test_snapshot_revalidation_and_success_fixture_copy()
    print("gguf_read_tensor_data_q4_0_checked_nopartial=ok")


if __name__ == "__main__":
    run()
