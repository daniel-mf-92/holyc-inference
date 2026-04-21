#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStrideCommitOnly (IQ-950)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path


_BASE_PATH = Path("tests/test_q8_0_dequantize_block_q16_checked_no_partial_array.py")
_SPEC = importlib.util.spec_from_file_location("q8_default_stride_commit_only_base", _BASE_PATH)
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


def q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
    src_blocks,
    src_block_capacity: int,
    src_block_stride: int,
    block_count: int,
    dst_q16,
    dst_q16_capacity: int,
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
    if (
        out_block_count is src_blocks
        or out_required_src_blocks is src_blocks
        or out_required_dst_values is src_blocks
        or out_required_src_bytes is src_blocks
        or out_required_dst_bytes is src_blocks
    ):
        return Q8_0_ERR_BAD_DST_LEN
    if (
        out_block_count is dst_q16
        or out_required_src_blocks is dst_q16
        or out_required_dst_values is dst_q16
        or out_required_src_bytes is dst_q16
        or out_required_dst_bytes is dst_q16
    ):
        return Q8_0_ERR_BAD_DST_LEN

    if src_block_capacity < 0 or dst_q16_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if src_block_stride <= 0:
        return Q8_0_ERR_BAD_DST_LEN

    dst_stride = Q8_0_VALUES_PER_BLOCK

    if block_count == 0:
        req_src_blocks = 0
        req_dst_values = 0
        req_src_bytes = 0
        req_dst_bytes = 0
    else:
        src_last_offset = (block_count - 1) * src_block_stride
        dst_last_base = (block_count - 1) * dst_stride
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

    status = _base.q8_0_dequantize_block_q16_checked_no_partial_array(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_stride,
    )
    if status != Q8_0_OK:
        return status

    out_block_count[0] = block_count
    out_required_src_blocks[0] = req_src_blocks
    out_required_dst_values[0] = req_dst_values
    out_required_src_bytes[0] = req_src_bytes
    out_required_dst_bytes[0] = req_dst_bytes
    return Q8_0_OK


def test_source_contains_iq950_default_stride_commit_only_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayDefaultStrideCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "dst_block_stride_values = Q8_0_BLOCK_SIZE;" in body
    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArray(" in body
    assert "if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks," in body
    assert "if (!Q8_0TryMulI64NonNeg(staged_required_dst_values," in body
    assert "*out_required_src_bytes = staged_required_src_bytes;" in body
    assert "*out_required_dst_bytes = staged_required_dst_bytes;" in body


def test_null_alias_and_stride_guards() -> None:
    src_blocks = [make_q8_block(random.Random(9500)) for _ in range(2)]
    dst = [777] * 64

    a = [101]
    b = [102]
    c = [103]
    d = [104]
    e = [105]

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
            None,
            0,
            1,
            0,
            dst,
            len(dst),
            a,
            b,
            c,
            d,
            e,
        )
        == Q8_0_ERR_NULL_PTR
    )

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
            src_blocks,
            2,
            1,
            1,
            dst,
            len(dst),
            a,
            a,
            c,
            d,
            e,
        )
        == Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
            src_blocks,
            2,
            1,
            1,
            dst,
            len(dst),
            dst,
            b,
            c,
            d,
            e,
        )
        == Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
            src_blocks,
            2,
            0,
            1,
            dst,
            len(dst),
            a,
            b,
            c,
            d,
            e,
        )
        == Q8_0_ERR_BAD_DST_LEN
    )



def test_success_commits_and_publishes_default_stride_geometry() -> None:
    rng = random.Random(9501)
    block_count = 3
    src_stride = 2
    src_capacity = 6

    src_blocks = [make_q8_block(rng) for _ in range(src_capacity)]
    out = [31337] * 128

    block_out = [0]
    req_src_blocks_out = [0]
    req_dst_values_out = [0]
    req_src_bytes_out = [0]
    req_dst_bytes_out = [0]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
        src_blocks,
        src_capacity,
        src_stride,
        block_count,
        out,
        len(out),
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_OK
    assert block_out == [block_count]
    assert req_src_blocks_out == [5]
    assert req_dst_values_out == [96]
    assert req_src_bytes_out == [5 * Q8_0_BLOCK_BYTES]
    assert req_dst_bytes_out == [96 * 8]



def test_capacity_failure_no_partial_and_no_publish() -> None:
    rng = random.Random(9502)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    dst = [4242] * 95
    dst_before = dst[:]

    block_out = [9]
    req_src_blocks_out = [10]
    req_dst_values_out = [11]
    req_src_bytes_out = [12]
    req_dst_bytes_out = [13]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
        src_blocks,
        src_block_capacity=4,
        src_block_stride=1,
        block_count=3,
        dst_q16=dst,
        dst_q16_capacity=len(dst),
        out_block_count=block_out,
        out_required_src_blocks=req_src_blocks_out,
        out_required_dst_values=req_dst_values_out,
        out_required_src_bytes=req_src_bytes_out,
        out_required_dst_bytes=req_dst_bytes_out,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert dst == dst_before
    assert block_out == [9]
    assert req_src_blocks_out == [10]
    assert req_dst_values_out == [11]
    assert req_src_bytes_out == [12]
    assert req_dst_bytes_out == [13]



def test_zero_block_publishes_zero_tuple_without_writes() -> None:
    rng = random.Random(9503)
    src_blocks = [make_q8_block(rng) for _ in range(3)]
    dst = [8181] * 32
    dst_before = dst[:]

    block_out = [1]
    req_src_blocks_out = [2]
    req_dst_values_out = [3]
    req_src_bytes_out = [4]
    req_dst_bytes_out = [5]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only(
        src_blocks,
        src_block_capacity=3,
        src_block_stride=2,
        block_count=0,
        dst_q16=dst,
        dst_q16_capacity=len(dst),
        out_block_count=block_out,
        out_required_src_blocks=req_src_blocks_out,
        out_required_dst_values=req_dst_values_out,
        out_required_src_bytes=req_src_bytes_out,
        out_required_dst_bytes=req_dst_bytes_out,
    )
    assert err == Q8_0_OK
    assert dst == dst_before
    assert block_out == [0]
    assert req_src_blocks_out == [0]
    assert req_dst_values_out == [0]
    assert req_src_bytes_out == [0]
    assert req_dst_bytes_out == [0]



def run() -> None:
    test_source_contains_iq950_default_stride_commit_only_contract()
    test_null_alias_and_stride_guards()
    test_success_commits_and_publishes_default_stride_geometry()
    test_capacity_failure_no_partial_and_no_publish()
    test_zero_block_publishes_zero_tuple_without_writes()
    print("q8_0_dequantize_block_q16_checked_no_partial_array_default_stride_commit_only=ok")


if __name__ == "__main__":
    run()
