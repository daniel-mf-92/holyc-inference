#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParity (IQ-949)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path


_BASE_PATH = Path("tests/test_q8_0_dequantize_block_q16_checked_no_partial_array.py")
_SPEC = importlib.util.spec_from_file_location("q8_commit_preflight_parity_base", _BASE_PATH)
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


def _try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > Q8_0_I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def _try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    i64_min = -(1 << 63)
    if rhs > 0 and lhs > Q8_0_I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < i64_min - rhs:
        return False, 0
    return True, lhs + rhs


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
    if src_block_stride <= 0 or dst_block_stride_values <= 0:
        return Q8_0_ERR_BAD_DST_LEN
    if dst_block_stride_values < Q8_0_VALUES_PER_BLOCK:
        return Q8_0_ERR_BAD_DST_LEN

    if block_count == 0:
        req_src_blocks = 0
        req_dst_values = 0
    else:
        ok, src_last_offset = _try_mul_i64_nonneg(block_count - 1, src_block_stride)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, dst_last_base = _try_mul_i64_nonneg(block_count - 1, dst_block_stride_values)
        if not ok:
            return Q8_0_ERR_OVERFLOW

        ok, req_src_blocks = _try_add_i64(src_last_offset, 1)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, req_dst_values = _try_add_i64(dst_last_base, Q8_0_VALUES_PER_BLOCK)
        if not ok:
            return Q8_0_ERR_OVERFLOW

    ok, req_src_bytes = _try_mul_i64_nonneg(req_src_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, req_dst_bytes = _try_mul_i64_nonneg(req_dst_values, 8)
    if not ok:
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


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
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
    mutate_snapshot: bool = False,
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

    snap_src_blocks = src_blocks
    snap_dst_q16 = dst_q16
    snap_src_block_capacity = src_block_capacity
    snap_src_block_stride = src_block_stride
    snap_block_count = block_count
    snap_dst_q16_capacity = dst_q16_capacity
    snap_dst_block_stride_values = dst_block_stride_values

    staged_block_count = [0]
    staged_required_src_blocks = [0]
    staged_required_dst_values = [0]
    staged_required_src_bytes = [0]
    staged_required_dst_bytes = [0]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        staged_block_count,
        staged_required_src_blocks,
        staged_required_dst_values,
        staged_required_src_bytes,
        staged_required_dst_bytes,
    )
    if err != Q8_0_OK:
        return err

    if mutate_snapshot:
        block_count += 1

    if snap_src_block_capacity < 0 or snap_dst_q16_capacity < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if snap_block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN
    if snap_src_block_stride <= 0 or snap_dst_block_stride_values <= 0:
        return Q8_0_ERR_BAD_DST_LEN
    if snap_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK:
        return Q8_0_ERR_BAD_DST_LEN

    if snap_block_count == 0:
        recomputed_required_src_blocks = 0
        recomputed_required_dst_values = 0
    else:
        ok, src_last_offset = _try_mul_i64_nonneg(snap_block_count - 1, snap_src_block_stride)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, dst_last_base = _try_mul_i64_nonneg(snap_block_count - 1, snap_dst_block_stride_values)
        if not ok:
            return Q8_0_ERR_OVERFLOW

        ok, recomputed_required_src_blocks = _try_add_i64(src_last_offset, 1)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, recomputed_required_dst_values = _try_add_i64(dst_last_base, Q8_0_VALUES_PER_BLOCK)
        if not ok:
            return Q8_0_ERR_OVERFLOW

    ok, recomputed_required_src_bytes = _try_mul_i64_nonneg(recomputed_required_src_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, recomputed_required_dst_bytes = _try_mul_i64_nonneg(recomputed_required_dst_values, 8)
    if not ok:
        return Q8_0_ERR_OVERFLOW

    if (
        src_blocks is not snap_src_blocks
        or dst_q16 is not snap_dst_q16
        or src_block_capacity != snap_src_block_capacity
        or src_block_stride != snap_src_block_stride
        or block_count != snap_block_count
        or dst_q16_capacity != snap_dst_q16_capacity
        or dst_block_stride_values != snap_dst_block_stride_values
    ):
        return Q8_0_ERR_BAD_DST_LEN

    if recomputed_required_src_blocks > snap_src_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if recomputed_required_dst_values > snap_dst_q16_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    if (
        staged_block_count[0] != snap_block_count
        or staged_required_src_blocks[0] != recomputed_required_src_blocks
        or staged_required_dst_values[0] != recomputed_required_dst_values
        or staged_required_src_bytes[0] != recomputed_required_src_bytes
        or staged_required_dst_bytes[0] != recomputed_required_dst_bytes
    ):
        return Q8_0_ERR_BAD_DST_LEN

    out_block_count[0] = staged_block_count[0]
    out_required_src_blocks[0] = staged_required_src_blocks[0]
    out_required_dst_values[0] = staged_required_dst_values[0]
    out_required_src_bytes[0] = staged_required_src_bytes[0]
    out_required_dst_bytes[0] = staged_required_dst_bytes[0]
    return Q8_0_OK


def test_source_contains_iq949_preflight_parity_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnly(" in body
    assert "if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1," in body
    assert "if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks," in body
    assert "if (staged_block_count != snapshot_block_count ||" in body
    assert "*out_required_dst_bytes = staged_required_dst_bytes;" in body


def test_null_and_alias_guards() -> None:
    out = [0] * 160
    a = [101]
    b = [202]
    c = [303]
    d = [404]
    e = [505]

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
            None,
            0,
            1,
            0,
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
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
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


def test_success_parity_and_no_write() -> None:
    rng = random.Random(949)
    src_blocks = [make_q8_block(rng) for _ in range(8)]
    dst = [7777] * 192
    dst_before = dst[:]

    block_out = [0]
    req_src_blocks_out = [0]
    req_dst_values_out = [0]
    req_src_bytes_out = [0]
    req_dst_bytes_out = [0]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
        src_blocks,
        8,
        2,
        4,
        dst,
        len(dst),
        40,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_OK
    assert dst == dst_before
    assert block_out == [4]
    assert req_src_blocks_out == [7]
    assert req_dst_values_out == [152]
    assert req_src_bytes_out == [7 * Q8_0_BLOCK_BYTES]
    assert req_dst_bytes_out == [152 * 8]


def test_snapshot_mismatch_rejected_no_publish() -> None:
    rng = random.Random(1949)
    src_blocks = [make_q8_block(rng) for _ in range(6)]
    dst = [31415] * 192

    block_out = [11]
    req_src_blocks_out = [12]
    req_dst_values_out = [13]
    req_src_bytes_out = [14]
    req_dst_bytes_out = [15]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
        src_blocks,
        6,
        1,
        3,
        dst,
        len(dst),
        32,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
        mutate_snapshot=True,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert block_out == [11]
    assert req_src_blocks_out == [12]
    assert req_dst_values_out == [13]
    assert req_src_bytes_out == [14]
    assert req_dst_bytes_out == [15]


def test_capacity_and_overflow_paths() -> None:
    rng = random.Random(2949)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    dst = [99] * 96

    block_out = [1]
    req_src_blocks_out = [2]
    req_dst_values_out = [3]
    req_src_bytes_out = [4]
    req_dst_bytes_out = [5]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
        src_blocks,
        4,
        2,
        3,
        dst,
        len(dst),
        32,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert block_out == [1]
    assert req_src_blocks_out == [2]
    assert req_dst_values_out == [3]

    huge = (1 << 62)
    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity(
        src_blocks,
        Q8_0_I64_MAX,
        huge,
        3,
        dst,
        Q8_0_I64_MAX,
        32,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_ERR_OVERFLOW


def run() -> None:
    test_source_contains_iq949_preflight_parity_contract()
    test_null_and_alias_guards()
    test_success_parity_and_no_write()
    test_snapshot_mismatch_rejected_no_publish()
    test_capacity_and_overflow_paths()
    print("q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
