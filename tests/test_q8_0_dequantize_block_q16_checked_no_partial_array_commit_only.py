#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnly (IQ-944)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_BASE_PATH = Path("tests/test_q8_0_dequantize_block_q16_checked_no_partial_array.py")
_SPEC = importlib.util.spec_from_file_location("q8_commit_only_base", _BASE_PATH)
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


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
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

    if src_block_capacity < 0 or dst_q16_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if src_block_stride <= 0 or dst_block_stride_values <= 0:
        return Q8_0_ERR_BAD_DST_LEN
    if dst_block_stride_values < Q8_0_VALUES_PER_BLOCK:
        return Q8_0_ERR_BAD_DST_LEN
    if src_blocks is dst_q16:
        return Q8_0_ERR_BAD_DST_LEN
    if (
        out_block_count is dst_q16
        or out_required_src_blocks is dst_q16
        or out_required_dst_values is dst_q16
        or out_required_src_bytes is dst_q16
        or out_required_dst_bytes is dst_q16
    ):
        return Q8_0_ERR_BAD_DST_LEN
    if (
        out_block_count is src_blocks
        or out_required_src_blocks is src_blocks
        or out_required_dst_values is src_blocks
        or out_required_src_bytes is src_blocks
        or out_required_dst_bytes is src_blocks
    ):
        return Q8_0_ERR_BAD_DST_LEN

    if block_count == 0:
        req_src_blocks = 0
        req_dst_values = 0
        req_src_bytes = 0
        req_dst_bytes = 0
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

    status = _base.q8_0_dequantize_block_q16_checked_no_partial_array(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
    )
    if status != Q8_0_OK:
        return status

    out_block_count[0] = block_count
    out_required_src_blocks[0] = req_src_blocks
    out_required_dst_values[0] = req_dst_values
    out_required_src_bytes[0] = req_src_bytes
    out_required_dst_bytes[0] = req_dst_bytes
    return Q8_0_OK


def test_source_contains_iq944_commit_only_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "if (out_block_count == out_required_src_blocks ||" in body
    assert "if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks," in body
    assert "if (!Q8_0TryMulI64NonNeg(staged_required_dst_values," in body
    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArray(" in body
    assert "*out_required_src_bytes = staged_required_src_bytes;" in body
    assert "*out_required_dst_bytes = staged_required_dst_bytes;" in body


def test_null_and_alias_guard_no_publish() -> None:
    out = [5] * 96
    a = [101]
    b = [202]
    c = [303]
    d = [404]
    e = [505]

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
            None,
            0,
            1,
            1,
            out,
            len(out),
            32,
            a,
            b,
            c,
            d,
            e,
        )
        == Q8_0_ERR_NULL_PTR
    )

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
            [],
            0,
            1,
            0,
            out,
            len(out),
            32,
            a,
            a,
            c,
            d,
            e,
        )
        == Q8_0_ERR_BAD_DST_LEN
    )
    assert a == [101]
    assert c == [303]


def test_alias_with_src_or_dst_rejected_and_no_publish() -> None:
    import random

    rng = random.Random(9441)
    src_blocks = [make_q8_block(rng) for _ in range(3)]
    dst = [1234] * 96
    a = [1]
    b = [2]
    c = [3]
    d = [4]
    e = [5]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
        src_blocks,
        3,
        1,
        1,
        dst,
        len(dst),
        32,
        dst,
        b,
        c,
        d,
        e,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert b == [2]
    assert c == [3]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
        src_blocks,
        3,
        1,
        1,
        src_blocks,
        3,
        32,
        a,
        b,
        c,
        d,
        e,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert a == [1]


def test_zero_blocks_publishes_zero_diagnostics_and_preserves_output() -> None:
    import random

    rng = random.Random(9442)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    out = [2718] * 64
    before = out[:]

    block_out = [111]
    req_src_blocks_out = [222]
    req_dst_values_out = [333]
    req_src_bytes_out = [444]
    req_dst_bytes_out = [555]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
        src_blocks,
        src_block_capacity=4,
        src_block_stride=3,
        block_count=0,
        dst_q16=out,
        dst_q16_capacity=len(out),
        dst_block_stride_values=48,
        out_block_count=block_out,
        out_required_src_blocks=req_src_blocks_out,
        out_required_dst_values=req_dst_values_out,
        out_required_src_bytes=req_src_bytes_out,
        out_required_dst_bytes=req_dst_bytes_out,
    )
    assert err == Q8_0_OK
    assert out == before
    assert block_out == [0]
    assert req_src_blocks_out == [0]
    assert req_dst_values_out == [0]
    assert req_src_bytes_out == [0]
    assert req_dst_bytes_out == [0]


def test_success_commits_and_publishes_geometry() -> None:
    import random

    rng = random.Random(944)
    block_count = 3
    src_stride = 2
    dst_stride = 40
    src_capacity = block_count * src_stride

    src_blocks = [make_q8_block(rng) for _ in range(src_capacity)]
    out = [777] * (block_count * dst_stride)

    block_out = [0]
    req_src_blocks_out = [0]
    req_dst_values_out = [0]
    req_src_bytes_out = [0]
    req_dst_bytes_out = [0]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
        src_blocks,
        src_capacity,
        src_stride,
        block_count,
        out,
        len(out),
        dst_stride,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_OK
    assert block_out == [block_count]
    assert req_src_blocks_out == [5]
    assert req_dst_values_out == [112]
    assert req_src_bytes_out == [5 * Q8_0_BLOCK_BYTES]
    assert req_dst_bytes_out == [112 * 8]


def test_capacity_failure_keeps_output_and_diagnostics_unchanged() -> None:
    import random

    rng = random.Random(2026944)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    out = [31337] * 95
    out_before = out[:]

    block_out = [9]
    req_src_blocks_out = [10]
    req_dst_values_out = [11]
    req_src_bytes_out = [12]
    req_dst_bytes_out = [13]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only(
        src_blocks,
        src_block_capacity=4,
        src_block_stride=1,
        block_count=3,
        dst_q16=out,
        dst_q16_capacity=len(out),
        dst_block_stride_values=32,
        out_block_count=block_out,
        out_required_src_blocks=req_src_blocks_out,
        out_required_dst_values=req_dst_values_out,
        out_required_src_bytes=req_src_bytes_out,
        out_required_dst_bytes=req_dst_bytes_out,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert out == out_before
    assert block_out == [9]
    assert req_src_blocks_out == [10]
    assert req_dst_values_out == [11]
    assert req_src_bytes_out == [12]
    assert req_dst_bytes_out == [13]


def run() -> None:
    test_source_contains_iq944_commit_only_contract()
    test_null_and_alias_guard_no_publish()
    test_alias_with_src_or_dst_rejected_and_no_publish()
    test_success_commits_and_publishes_geometry()
    test_zero_blocks_publishes_zero_diagnostics_and_preserves_output()
    test_capacity_failure_keeps_output_and_diagnostics_unchanged()
    print("q8_0_dequantize_block_q16_checked_no_partial_array_commit_only=ok")


if __name__ == "__main__":
    run()
