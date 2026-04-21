#!/usr/bin/env python3
"""Parity harness for Q8_0DequantizeBlockQ16...PreflightOnlyParityCommitOnly (IQ-951)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path


_BASE_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity.py"
)
_SPEC = importlib.util.spec_from_file_location("q8_commit_only_parity_base", _BASE_PATH)
assert _SPEC and _SPEC.loader
_base = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _base
_SPEC.loader.exec_module(_base)

Q8_0_OK = _base.Q8_0_OK
Q8_0_ERR_NULL_PTR = _base.Q8_0_ERR_NULL_PTR
Q8_0_ERR_BAD_DST_LEN = _base.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_OVERFLOW = _base.Q8_0_ERR_OVERFLOW
Q8_0_VALUES_PER_BLOCK = _base.Q8_0_VALUES_PER_BLOCK
Q8_0_BLOCK_BYTES = _base.Q8_0_BLOCK_BYTES
Q8_0_I64_MAX = _base.Q8_0_I64_MAX

make_q8_block = _base.make_q8_block
preflight_parity = (
    _base.q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity
)
try_mul_i64_nonneg = _base._try_mul_i64_nonneg


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
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

    err = preflight_parity(
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

    if staged_block_count[0] != snap_block_count:
        return Q8_0_ERR_BAD_DST_LEN
    if (
        staged_required_src_blocks[0] < 0
        or staged_required_dst_values[0] < 0
        or staged_required_src_bytes[0] < 0
        or staged_required_dst_bytes[0] < 0
    ):
        return Q8_0_ERR_OVERFLOW

    if staged_required_src_blocks[0] > snap_src_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if staged_required_dst_values[0] > snap_dst_q16_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    ok, expected_required_src_bytes = try_mul_i64_nonneg(
        staged_required_src_blocks[0], Q8_0_BLOCK_BYTES
    )
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, expected_required_dst_bytes = try_mul_i64_nonneg(staged_required_dst_values[0], 8)
    if not ok:
        return Q8_0_ERR_OVERFLOW

    if staged_required_src_bytes[0] != expected_required_src_bytes:
        return Q8_0_ERR_BAD_DST_LEN
    if staged_required_dst_bytes[0] != expected_required_dst_bytes:
        return Q8_0_ERR_BAD_DST_LEN

    out_block_count[0] = staged_block_count[0]
    out_required_src_blocks[0] = staged_required_src_blocks[0]
    out_required_dst_values[0] = staged_required_dst_values[0]
    out_required_src_bytes[0] = staged_required_src_bytes[0]
    out_required_dst_bytes[0] = staged_required_dst_bytes[0]
    return Q8_0_OK


def test_source_contains_iq951_commit_only_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParity(" in body
    assert "if (staged_block_count != snapshot_block_count)" in body
    assert "if (!Q8_0TryMulI64NonNeg(staged_required_src_blocks," in body
    assert "if (staged_required_src_bytes != expected_required_src_bytes)" in body
    assert "*out_required_dst_bytes = staged_required_dst_bytes;" in body


def test_null_and_alias_guards() -> None:
    out = [0] * 128
    a = [1]
    b = [2]
    c = [3]
    d = [4]
    e = [5]

    assert (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
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
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
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


def test_success_and_atomic_publish() -> None:
    rng = random.Random(951)
    src_blocks = [make_q8_block(rng) for _ in range(10)]
    dst = [111] * 320

    block_out = [71]
    req_src_blocks_out = [72]
    req_dst_values_out = [73]
    req_src_bytes_out = [74]
    req_dst_bytes_out = [75]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
        src_blocks,
        10,
        2,
        5,
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
    assert block_out == [5]
    assert req_src_blocks_out == [9]
    assert req_dst_values_out == [192]
    assert req_src_bytes_out == [9 * Q8_0_BLOCK_BYTES]
    assert req_dst_bytes_out == [192 * 8]


def test_snapshot_mismatch_no_publish() -> None:
    rng = random.Random(1951)
    src_blocks = [make_q8_block(rng) for _ in range(8)]
    dst = [333] * 256

    block_out = [91]
    req_src_blocks_out = [92]
    req_dst_values_out = [93]
    req_src_bytes_out = [94]
    req_dst_bytes_out = [95]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
        src_blocks,
        8,
        1,
        4,
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
    assert block_out == [91]
    assert req_src_blocks_out == [92]
    assert req_dst_values_out == [93]
    assert req_src_bytes_out == [94]
    assert req_dst_bytes_out == [95]


def test_overflow_and_capacity_failures() -> None:
    rng = random.Random(2951)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    dst = [777] * 64

    block_out = [1]
    req_src_blocks_out = [2]
    req_dst_values_out = [3]
    req_src_bytes_out = [4]
    req_dst_bytes_out = [5]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
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

    huge = 1 << 62
    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only(
        src_blocks,
        Q8_0_I64_MAX,
        1,
        huge,
        dst,
        Q8_0_I64_MAX,
        Q8_0_VALUES_PER_BLOCK,
        block_out,
        req_src_blocks_out,
        req_dst_values_out,
        req_src_bytes_out,
        req_dst_bytes_out,
    )
    assert err == Q8_0_ERR_OVERFLOW


def run() -> None:
    test_source_contains_iq951_commit_only_contract()
    test_null_and_alias_guards()
    test_success_and_atomic_publish()
    test_snapshot_mismatch_no_publish()
    test_overflow_and_capacity_failures()
    print(
        "q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only=ok"
    )


if __name__ == "__main__":
    run()

