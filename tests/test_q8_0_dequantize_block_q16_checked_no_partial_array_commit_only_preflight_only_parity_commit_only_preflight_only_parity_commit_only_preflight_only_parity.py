#!/usr/bin/env python3
"""Harness for IQ-961 parity gate over CommitOnlyPreflightOnly + ParityCommitOnlyPreflightOnly."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_PREV_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only.py"
)

_spec_prev = importlib.util.spec_from_file_location("q8_iq961_prev", _PREV_PATH)
assert _spec_prev and _spec_prev.loader
_prev = importlib.util.module_from_spec(_spec_prev)
sys.modules[_spec_prev.name] = _prev
_spec_prev.loader.exec_module(_prev)

Q8_0_OK = _prev.Q8_0_OK
Q8_0_ERR_NULL_PTR = _prev.Q8_0_ERR_NULL_PTR
Q8_0_ERR_BAD_DST_LEN = _prev.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_OVERFLOW = _prev.Q8_0_ERR_OVERFLOW
Q8_0_VALUES_PER_BLOCK = _prev.Q8_0_VALUES_PER_BLOCK
Q8_0_BLOCK_BYTES = _prev.Q8_0_BLOCK_BYTES
Q8_0_I64_MAX = _prev.Q8_0_I64_MAX

make_q8_block = _prev.make_q8_block
try_mul_i64_nonneg = _prev.try_mul_i64_nonneg
commit_only_preflight = (
    _prev.q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only
)


def preflight_only(
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
        required_src_blocks = 0
        required_dst_values = 0
    else:
        ok, src_last_offset = try_mul_i64_nonneg(block_count - 1, src_block_stride)
        if not ok:
            return Q8_0_ERR_OVERFLOW
        ok, dst_last_base = try_mul_i64_nonneg(block_count - 1, dst_block_stride_values)
        if not ok:
            return Q8_0_ERR_OVERFLOW

        if src_last_offset > Q8_0_I64_MAX - 1:
            return Q8_0_ERR_OVERFLOW
        required_src_blocks = src_last_offset + 1

        if dst_last_base > Q8_0_I64_MAX - Q8_0_VALUES_PER_BLOCK:
            return Q8_0_ERR_OVERFLOW
        required_dst_values = dst_last_base + Q8_0_VALUES_PER_BLOCK

    ok, required_src_bytes = try_mul_i64_nonneg(required_src_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return Q8_0_ERR_OVERFLOW
    ok, required_dst_bytes = try_mul_i64_nonneg(required_dst_values, 8)
    if not ok:
        return Q8_0_ERR_OVERFLOW

    if required_src_blocks > src_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if required_dst_values > dst_q16_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    out_block_count[0] = block_count
    out_required_src_blocks[0] = required_src_blocks
    out_required_dst_values[0] = required_dst_values
    out_required_src_bytes[0] = required_src_bytes
    out_required_dst_bytes[0] = required_dst_bytes
    return Q8_0_OK


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    parity_block_count = [0]
    parity_required_src_blocks = [0]
    parity_required_dst_values = [0]
    parity_required_src_bytes = [0]
    parity_required_dst_bytes = [0]

    err = commit_only_preflight(
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

    err = preflight_only(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        parity_block_count,
        parity_required_src_blocks,
        parity_required_dst_values,
        parity_required_src_bytes,
        parity_required_dst_bytes,
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

    if (
        staged_required_src_blocks[0] < 0
        or staged_required_dst_values[0] < 0
        or staged_required_src_bytes[0] < 0
        or staged_required_dst_bytes[0] < 0
        or parity_required_src_blocks[0] < 0
        or parity_required_dst_values[0] < 0
        or parity_required_src_bytes[0] < 0
        or parity_required_dst_bytes[0] < 0
    ):
        return Q8_0_ERR_OVERFLOW

    if staged_required_src_blocks[0] > snap_src_block_capacity:
        return Q8_0_ERR_BAD_DST_LEN
    if staged_required_dst_values[0] > snap_dst_q16_capacity:
        return Q8_0_ERR_BAD_DST_LEN

    if (
        staged_block_count[0] != snap_block_count
        or staged_block_count[0] != parity_block_count[0]
        or staged_required_src_blocks[0] != parity_required_src_blocks[0]
        or staged_required_dst_values[0] != parity_required_dst_values[0]
        or staged_required_src_bytes[0] != parity_required_src_bytes[0]
        or staged_required_dst_bytes[0] != parity_required_dst_bytes[0]
    ):
        return Q8_0_ERR_BAD_DST_LEN

    out_block_count[0] = staged_block_count[0]
    out_required_src_blocks[0] = staged_required_src_blocks[0]
    out_required_dst_values[0] = staged_required_dst_values[0]
    out_required_src_bytes[0] = staged_required_src_bytes[0]
    out_required_dst_bytes[0] = staged_required_dst_bytes[0]
    return Q8_0_OK


def test_source_contains_iq961_parity_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1]

    assert (
        "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in body
    )
    assert (
        "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in body
    )
    assert "staged_required_src_blocks != parity_required_src_blocks" in body
    assert "staged_required_dst_bytes != parity_required_dst_bytes" in body


def test_success_publish_and_no_write() -> None:
    rng = random.Random(961)
    src_blocks = [make_q8_block(rng) for _ in range(16)]
    dst_q16 = [777] * 2048
    before = dst_q16[:]

    out_block_count = [11]
    out_required_src_blocks = [12]
    out_required_dst_values = [13]
    out_required_src_bytes = [14]
    out_required_dst_bytes = [15]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        src_blocks,
        16,
        2,
        7,
        dst_q16,
        len(dst_q16),
        64,
        out_block_count,
        out_required_src_blocks,
        out_required_dst_values,
        out_required_src_bytes,
        out_required_dst_bytes,
    )
    assert err == Q8_0_OK
    assert out_block_count == [7]
    assert out_required_src_blocks == [13]
    assert out_required_dst_values == [416]
    assert out_required_src_bytes == [13 * Q8_0_BLOCK_BYTES]
    assert out_required_dst_bytes == [416 * 8]
    assert dst_q16 == before


def test_snapshot_mismatch_no_publish() -> None:
    rng = random.Random(1961)
    src_blocks = [make_q8_block(rng) for _ in range(10)]
    dst_q16 = [888] * 1024

    out_block_count = [91]
    out_required_src_blocks = [92]
    out_required_dst_values = [93]
    out_required_src_bytes = [94]
    out_required_dst_bytes = [95]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
        src_blocks,
        10,
        1,
        5,
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
    rng = random.Random(961000)
    for _ in range(200):
        stride_blocks = rng.randint(1, 5)
        block_count = rng.randint(0, 24)
        src_capacity = block_count * stride_blocks + rng.randint(0, 10)

        dst_stride = rng.randint(Q8_0_VALUES_PER_BLOCK, Q8_0_VALUES_PER_BLOCK + 64)
        required_dst = 0 if block_count == 0 else ((block_count - 1) * dst_stride + Q8_0_VALUES_PER_BLOCK)
        dst_capacity = required_dst + rng.randint(0, 40)

        src_blocks = [make_q8_block(rng) for _ in range(max(src_capacity, 1))]
        dst_q16 = [505] * max(dst_capacity, 1)

        out_block_count = [0]
        out_required_src_blocks = [0]
        out_required_dst_values = [0]
        out_required_src_bytes = [0]
        out_required_dst_bytes = [0]

        err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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
        assert out_required_src_blocks == [expected_src_blocks]
        assert out_required_dst_values == [required_dst]
        assert out_required_src_bytes == [expected_src_blocks * Q8_0_BLOCK_BYTES]
        assert out_required_dst_bytes == [required_dst * 8]


def run() -> None:
    test_source_contains_iq961_parity_contract()
    test_success_publish_and_no_write()
    test_snapshot_mismatch_no_publish()
    test_randomized_adversarial_vectors()
    print("iq961_parity=ok")


if __name__ == "__main__":
    run()
