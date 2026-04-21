#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnly (IQ-947)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_PATH = Path("tests/test_q8_0_dequantize_block_q16_checked_no_partial_array.py")
_SPEC = importlib.util.spec_from_file_location("q8_commit_only_preflight_base", _BASE_PATH)
assert _SPEC and _SPEC.loader
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

Q8_0_OK = _base.Q8_0_OK
Q8_0_ERR_NULL_PTR = _base.Q8_0_ERR_NULL_PTR
Q8_0_ERR_BAD_DST_LEN = _base.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_OVERFLOW = _base.Q8_0_ERR_OVERFLOW
Q8_0_VALUES_PER_BLOCK = _base.Q8_0_VALUES_PER_BLOCK
Q8_0_BLOCK_BYTES = 34
Q8_0_I64_MAX = (1 << 63) - 1

make_q8_block = _base.make_q8_block


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
    dst_block_stride_values: int,
    out_block_count,
    out_required_src_blocks,
    out_required_dst_values,
    out_required_src_bytes,
    out_required_dst_bytes,
) -> int:
    if (
        src_blocks is None
        or dst_q16 is None
        or out_block_count is None
        or out_required_src_blocks is None
        or out_required_dst_values is None
        or out_required_src_bytes is None
        or out_required_dst_bytes is None
    ):
        return Q8_0_ERR_NULL_PTR

    outs = [
        out_block_count,
        out_required_src_blocks,
        out_required_dst_values,
        out_required_src_bytes,
        out_required_dst_bytes,
    ]
    if len({id(x) for x in outs}) != len(outs):
        return Q8_0_ERR_BAD_DST_LEN

    if src_blocks is dst_q16:
        return Q8_0_ERR_BAD_DST_LEN

    if src_block_capacity < 0 or dst_q16_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if src_block_stride <= 0 or dst_block_stride_values <= 0:
        return Q8_0_ERR_BAD_DST_LEN
    if dst_block_stride_values < Q8_0_VALUES_PER_BLOCK:
        return Q8_0_ERR_BAD_DST_LEN

    if block_count == 0:
        req_src_blocks = 0
        req_dst_values = 0
    else:
        src_last_offset = (block_count - 1) * src_block_stride
        dst_last_base = (block_count - 1) * dst_block_stride_values
        req_src_blocks = src_last_offset + 1
        req_dst_values = dst_last_base + Q8_0_VALUES_PER_BLOCK
        if req_src_blocks > Q8_0_I64_MAX or req_dst_values > Q8_0_I64_MAX:
            return Q8_0_ERR_OVERFLOW

    req_src_bytes = req_src_blocks * Q8_0_BLOCK_BYTES
    req_dst_bytes = req_dst_values * 8
    if req_src_bytes > Q8_0_I64_MAX or req_dst_bytes > Q8_0_I64_MAX:
        return Q8_0_ERR_OVERFLOW

    if req_src_blocks > src_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if req_dst_values > dst_q16_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    out_block_count[0] = block_count
    out_required_src_blocks[0] = req_src_blocks
    out_required_dst_values[0] = req_dst_values
    out_required_src_bytes[0] = req_src_bytes
    out_required_dst_bytes[0] = req_dst_bytes
    return Q8_0_OK


def test_source_contains_iq947_preflight_only_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1," in body
    assert "if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks," in body
    assert "if (staged_required_src_blocks > snapshot_src_block_capacity)" in body
    assert "if (staged_required_dst_values > snapshot_dst_q16_capacity)" in body
    assert "// Atomic publish: commit all diagnostics only after full preflight success." in body


def test_preflight_is_no_write_to_destination() -> None:
    import random

    rng = random.Random(947)
    src_blocks = [make_q8_block(rng) for _ in range(6)]
    out = [4444] * 160
    before = out[:]

    block_out = [0]
    req_src_blocks_out = [0]
    req_dst_values_out = [0]
    req_src_bytes_out = [0]
    req_dst_bytes_out = [0]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only(
        src_blocks,
        6,
        2,
        3,
        out,
        len(out),
        40,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_OK
    assert out == before
    assert block_out == [3]
    assert req_src_blocks_out == [5]
    assert req_dst_values_out == [112]
    assert req_src_bytes_out == [5 * Q8_0_BLOCK_BYTES]
    assert req_dst_bytes_out == [112 * 8]


def test_failure_keeps_publish_cells_unchanged() -> None:
    import random

    rng = random.Random(2026947)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    out = [123] * 96

    block_out = [9]
    req_src_blocks_out = [10]
    req_dst_values_out = [11]
    req_src_bytes_out = [12]
    req_dst_bytes_out = [13]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only(
        src_blocks,
        4,
        1,
        3,
        out,
        len(out),
        31,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert block_out == [9]
    assert req_src_blocks_out == [10]
    assert req_dst_values_out == [11]
    assert req_src_bytes_out == [12]
    assert req_dst_bytes_out == [13]


def run() -> None:
    test_source_contains_iq947_preflight_only_contract()
    test_preflight_is_no_write_to_destination()
    test_failure_keeps_publish_cells_unchanged()
    print("q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
