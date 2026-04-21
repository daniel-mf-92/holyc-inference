#!/usr/bin/env python3
"""Parity harness for GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly (IQ-941)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_PATH = Path(
    "tests/test_gguf_read_tensor_data_q4_0_checked_nopartial_commit_only_preflight_only_parity_commit_only.py"
)
_SPEC = importlib.util.spec_from_file_location("q4_commit_only_base", _BASE_PATH)
assert _SPEC and _SPEC.loader
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

GGUF_READER_OK = _base.GGUF_READER_OK
GGUF_READER_ERR_NULL_PTR = _base.GGUF_READER_ERR_NULL_PTR
GGUF_READER_ERR_BAD_PARAM = _base.GGUF_READER_ERR_BAD_PARAM
GGUF_READER_ERR_OVERFLOW = _base.GGUF_READER_ERR_OVERFLOW
GGUF_READER_ERR_BAD_CAPACITY = _base.GGUF_READER_ERR_BAD_CAPACITY
GGUF_READER_GGML_TYPE_Q4_0 = _base.GGUF_READER_GGML_TYPE_Q4_0
TensorInfoQ40 = _base.TensorInfoQ40


def gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    staged_out_capacity = out_q4_0_blocks_capacity

    staged_blocks = [0]
    staged_bytes = [0]
    err = _base.gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity_commit_only(
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

    recomputed_blocks = [0]
    recomputed_bytes = [0]
    err = _base.gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only(
        gguf_bytes,
        gguf_nbytes,
        tensor_data_base,
        tensor_info,
        out_q4_0_blocks,
        out_q4_0_blocks_capacity,
        recomputed_blocks,
        recomputed_bytes,
    )
    if err != GGUF_READER_OK:
        return err

    if not _base.tensor_matches_snapshot(tensor_info, snap):
        return GGUF_READER_ERR_BAD_PARAM
    if staged_out_capacity != out_q4_0_blocks_capacity:
        return GGUF_READER_ERR_BAD_PARAM

    if (staged_blocks[0], staged_bytes[0]) != (recomputed_blocks[0], recomputed_bytes[0]):
        return GGUF_READER_ERR_BAD_PARAM
    if staged_blocks[0] < 0 or staged_bytes[0] < 0:
        return GGUF_READER_ERR_OVERFLOW
    if staged_bytes[0] > staged_out_capacity:
        return GGUF_READER_ERR_BAD_CAPACITY

    out_blocks_read[0] = staged_blocks[0]
    out_bytes_read[0] = staged_bytes[0]
    return GGUF_READER_OK


def test_source_contains_iq941_q4_preflight_only_contract() -> None:
    source = Path("src/gguf/reader.HC").read_text(encoding="utf-8")
    sig = "I32 GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("\n\n\nI32 ", 1)[0]

    assert "status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "status = GGUFReadTensorDataQ4_0CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "if (!GGUFReaderTensorInfoQ4_0MatchesSnapshot(" in body
    assert "if (staged_out_capacity != out_q4_0_blocks_capacity)" in body
    assert "if (staged_blocks_read_i64 != recomputed_blocks_read_i64 ||" in body
    assert "staged_bytes_read_i64 != recomputed_bytes_read_i64" in body
    assert "if (staged_blocks_read_i64 < 0 || staged_bytes_read_i64 < 0)" in body
    assert "if ((U64)staged_bytes_read_i64 > staged_out_capacity)" in body
    assert "*out_blocks_read = staged_blocks_read_i64;" in body
    assert "*out_bytes_read = staged_bytes_read_i64;" in body


def test_bad_vectors_and_no_publish_behavior() -> None:
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 0)
    blob = bytes(range(64))
    out = bytearray(b"\xA5" * 80)
    out_before = bytes(out)
    out_blocks = [111]
    out_bytes = [222]

    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    assert bytes(out) == out_before

    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity_commit_only_preflight_only(
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
    assert out_blocks == [111]
    assert bytes(out) == out_before


def test_success_preflight_only_and_capacity_rejection() -> None:
    payload = bytes((idx * 13 + 5) & 0xFF for idx in range(36))
    blob = b"\x55\x66\x77\x88" + payload + b"\x00" * 16
    info = TensorInfoQ40(1, [64, 0, 0, 0], GGUF_READER_GGML_TYPE_Q4_0, 4)

    out = bytearray(b"\x3C" * 96)
    out_before = bytes(out)
    out_blocks = [0]
    out_bytes = [0]

    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity_commit_only_preflight_only(
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

    rejected_blocks = [333]
    rejected_bytes = [444]
    err = gguf_read_tensor_data_q4_0_checked_no_partial_commit_only_preflight_only_parity_commit_only_preflight_only(
        blob,
        len(blob),
        0,
        info,
        out,
        35,
        rejected_blocks,
        rejected_bytes,
    )
    assert err == GGUF_READER_ERR_BAD_CAPACITY
    assert rejected_blocks == [333]
    assert rejected_bytes == [444]
    assert bytes(out) == out_before


def run() -> None:
    test_source_contains_iq941_q4_preflight_only_contract()
    test_bad_vectors_and_no_publish_behavior()
    test_success_preflight_only_and_capacity_rejection()
    print("gguf_read_tensor_data_q4_0_checked_nopartial_commit_only_preflight_only_parity_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
