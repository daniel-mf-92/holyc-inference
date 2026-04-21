#!/usr/bin/env python3
"""Harness for IQ-963 diagnostics-only preflight companion."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_COMMIT_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only.py"
)

_spec_commit = importlib.util.spec_from_file_location("q8_iq963_commit", _COMMIT_PATH)
assert _spec_commit and _spec_commit.loader
_commit = importlib.util.module_from_spec(_spec_commit)
sys.modules[_spec_commit.name] = _commit
_spec_commit.loader.exec_module(_commit)

Q8_0_OK = _commit.Q8_0_OK
Q8_0_ERR_NULL_PTR = _commit.Q8_0_ERR_NULL_PTR
Q8_0_ERR_BAD_DST_LEN = _commit.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_OVERFLOW = _commit.Q8_0_ERR_OVERFLOW
Q8_0_VALUES_PER_BLOCK = _commit.Q8_0_VALUES_PER_BLOCK
Q8_0_BLOCK_BYTES = _commit.Q8_0_BLOCK_BYTES
Q8_0_I64_MAX = _commit.Q8_0_I64_MAX

make_q8_block = _commit.make_q8_block
commit_only = (
    _commit.q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only
)
try_mul_i64_nonneg = _commit.try_mul_i64_nonneg


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
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
        or out_block_count is dst_q16
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

    err = commit_only(
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
        ok, src_last_offset = try_mul_i64_nonneg(snap_block_count - 1, snap_src_block_stride)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, dst_last_base = try_mul_i64_nonneg(snap_block_count - 1, snap_dst_block_stride_values)
        if not ok:
            return Q8_0_ERR_OVERFLOW

        if src_last_offset > Q8_0_I64_MAX - 1:
            return Q8_0_ERR_OVERFLOW
        recomputed_required_src_blocks = src_last_offset + 1

        if dst_last_base > Q8_0_I64_MAX - Q8_0_VALUES_PER_BLOCK:
            return Q8_0_ERR_OVERFLOW
        recomputed_required_dst_values = dst_last_base + Q8_0_VALUES_PER_BLOCK

    ok, recomputed_required_src_bytes = try_mul_i64_nonneg(recomputed_required_src_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, recomputed_required_dst_bytes = try_mul_i64_nonneg(recomputed_required_dst_values, 8)
    if not ok:
        return Q8_0_ERR_OVERFLOW

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


def test_source_contains_iq963_preflight_only_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1]

    assert (
        "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
        in body
    )
    assert "Diagnostics-only no-write companion over IQ-962 commit-only wrapper." in source
    assert "if (!Q8_0TryMulI64NonNeg(snapshot_block_count - 1," in body
    assert "if (!Q8_0TryMulI64NonNeg(recomputed_required_src_blocks," in body
    assert "if (staged_block_count != snapshot_block_count ||" in body
    assert "*out_required_dst_bytes = staged_required_dst_bytes;" in body


def test_success_and_no_write_publish() -> None:
    rng = random.Random(963)
    src_blocks = [make_q8_block(rng) for _ in range(12)]
    dst_q16 = [903] * 600
    before = dst_q16[:]

    out_block_count = [11]
    out_required_src_blocks = [12]
    out_required_dst_values = [13]
    out_required_src_bytes = [14]
    out_required_dst_bytes = [15]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        src_blocks,
        12,
        2,
        6,
        dst_q16,
        len(dst_q16),
        48,
        out_block_count,
        out_required_src_blocks,
        out_required_dst_values,
        out_required_src_bytes,
        out_required_dst_bytes,
    )
    assert err == Q8_0_OK
    assert out_block_count == [6]
    assert out_required_src_blocks == [11]
    assert out_required_dst_values == [272]
    assert out_required_src_bytes == [11 * Q8_0_BLOCK_BYTES]
    assert out_required_dst_bytes == [272 * 8]
    assert dst_q16 == before


def test_snapshot_mismatch_no_publish() -> None:
    rng = random.Random(1963)
    src_blocks = [make_q8_block(rng) for _ in range(8)]
    dst_q16 = [803] * 320

    out_block_count = [91]
    out_required_src_blocks = [92]
    out_required_dst_values = [93]
    out_required_src_bytes = [94]
    out_required_dst_bytes = [95]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
        src_blocks,
        8,
        1,
        4,
        dst_q16,
        len(dst_q16),
        32,
        out_block_count,
        out_required_src_blocks,
        out_required_dst_values,
        out_required_src_bytes,
        out_required_dst_bytes,
        mutate_snapshot=True,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert out_block_count == [91]
    assert out_required_src_blocks == [92]
    assert out_required_dst_values == [93]
    assert out_required_src_bytes == [94]
    assert out_required_dst_bytes == [95]


def test_randomized_adversarial_vectors() -> None:
    rng = random.Random(963000)
    for _ in range(200):
        stride_blocks = rng.randint(1, 4)
        block_count = rng.randint(0, 22)
        src_capacity = block_count * stride_blocks + rng.randint(0, 8)

        dst_stride = rng.randint(Q8_0_VALUES_PER_BLOCK, Q8_0_VALUES_PER_BLOCK + 48)
        required_dst = 0 if block_count == 0 else ((block_count - 1) * dst_stride + Q8_0_VALUES_PER_BLOCK)
        dst_capacity = required_dst + rng.randint(0, 24)

        src_blocks = [make_q8_block(rng) for _ in range(max(src_capacity, 1))]
        dst_q16 = [703] * max(dst_capacity, 1)

        out_block_count = [0]
        out_required_src_blocks = [0]
        out_required_dst_values = [0]
        out_required_src_bytes = [0]
        out_required_dst_bytes = [0]

        err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only(
            src_blocks,
            src_capacity,
            stride_blocks,
            block_count,
            dst_q16,
            dst_capacity,
            dst_stride,
            out_block_count,
            out_required_src_blocks,
            out_required_dst_values,
            out_required_src_bytes,
            out_required_dst_bytes,
        )

        assert err == Q8_0_OK
        assert out_block_count == [block_count]
        expected_src_blocks = 0 if block_count == 0 else ((block_count - 1) * stride_blocks + 1)
        expected_dst_values = required_dst
        assert out_required_src_blocks == [expected_src_blocks]
        assert out_required_dst_values == [expected_dst_values]
        assert out_required_src_bytes == [expected_src_blocks * Q8_0_BLOCK_BYTES]
        assert out_required_dst_bytes == [expected_dst_values * 8]


def run() -> None:
    test_source_contains_iq963_preflight_only_contract()
    test_success_and_no_write_publish()
    test_snapshot_mismatch_no_publish()
    test_randomized_adversarial_vectors()
    print("iq963_preflight_only=ok")


if __name__ == "__main__":
    run()
