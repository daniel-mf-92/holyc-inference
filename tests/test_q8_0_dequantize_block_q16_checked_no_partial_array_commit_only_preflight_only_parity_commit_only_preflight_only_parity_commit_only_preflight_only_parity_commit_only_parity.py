#!/usr/bin/env python3
"""Harness for IQ-969 strict parity gate over commit-only and preflight-only tuple producers."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_COMMIT_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only.py"
)
_PREFLIGHT_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only.py"
)

_spec_commit = importlib.util.spec_from_file_location("q8_iq969_commit", _COMMIT_PATH)
assert _spec_commit and _spec_commit.loader
_commit = importlib.util.module_from_spec(_spec_commit)
sys.modules[_spec_commit.name] = _commit
_spec_commit.loader.exec_module(_commit)

_spec_preflight = importlib.util.spec_from_file_location("q8_iq969_preflight", _PREFLIGHT_PATH)
assert _spec_preflight and _spec_preflight.loader
_preflight = importlib.util.module_from_spec(_spec_preflight)
sys.modules[_spec_preflight.name] = _preflight
_spec_preflight.loader.exec_module(_preflight)

Q8_0_OK = _commit.Q8_0_OK
Q8_0_ERR_NULL_PTR = _commit.Q8_0_ERR_NULL_PTR
Q8_0_ERR_BAD_DST_LEN = _commit.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_OVERFLOW = _commit.Q8_0_ERR_OVERFLOW
Q8_0_VALUES_PER_BLOCK = _commit.Q8_0_VALUES_PER_BLOCK
Q8_0_BLOCK_BYTES = _commit.Q8_0_BLOCK_BYTES

make_q8_block = _commit.make_q8_block
commit_only = (
    _commit.q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only
)
preflight_only = (
    _preflight.q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only
)


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_parity(
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

    staged_commit_block_count = [0]
    staged_commit_required_src_blocks = [0]
    staged_commit_required_dst_values = [0]
    staged_commit_required_src_bytes = [0]
    staged_commit_required_dst_bytes = [0]

    staged_preflight_block_count = [0]
    staged_preflight_required_src_blocks = [0]
    staged_preflight_required_dst_values = [0]
    staged_preflight_required_src_bytes = [0]
    staged_preflight_required_dst_bytes = [0]

    err = commit_only(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        staged_commit_block_count,
        staged_commit_required_src_blocks,
        staged_commit_required_dst_values,
        staged_commit_required_src_bytes,
        staged_commit_required_dst_bytes,
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
        staged_preflight_block_count,
        staged_preflight_required_src_blocks,
        staged_preflight_required_dst_values,
        staged_preflight_required_src_bytes,
        staged_preflight_required_dst_bytes,
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
        snap_src_block_capacity < 0
        or snap_dst_q16_capacity < 0
        or snap_block_count < 0
        or snap_src_block_stride <= 0
        or snap_dst_block_stride_values <= 0
        or snap_dst_block_stride_values < Q8_0_VALUES_PER_BLOCK
    ):
        return Q8_0_ERR_BAD_DST_LEN

    if (
        staged_commit_required_src_blocks[0] < 0
        or staged_commit_required_dst_values[0] < 0
        or staged_commit_required_src_bytes[0] < 0
        or staged_commit_required_dst_bytes[0] < 0
        or staged_preflight_required_src_blocks[0] < 0
        or staged_preflight_required_dst_values[0] < 0
        or staged_preflight_required_src_bytes[0] < 0
        or staged_preflight_required_dst_bytes[0] < 0
    ):
        return Q8_0_ERR_OVERFLOW

    if (
        staged_commit_required_src_blocks[0] > snap_src_block_capacity
        or staged_preflight_required_src_blocks[0] > snap_src_block_capacity
        or staged_commit_required_dst_values[0] > snap_dst_q16_capacity
        or staged_preflight_required_dst_values[0] > snap_dst_q16_capacity
    ):
        return Q8_0_ERR_BAD_DST_LEN

    if (
        staged_commit_block_count[0] != snap_block_count
        or staged_preflight_block_count[0] != snap_block_count
        or staged_commit_block_count[0] != staged_preflight_block_count[0]
        or staged_commit_required_src_blocks[0] != staged_preflight_required_src_blocks[0]
        or staged_commit_required_dst_values[0] != staged_preflight_required_dst_values[0]
        or staged_commit_required_src_bytes[0] != staged_preflight_required_src_bytes[0]
        or staged_commit_required_dst_bytes[0] != staged_preflight_required_dst_bytes[0]
    ):
        return Q8_0_ERR_BAD_DST_LEN

    out_block_count[0] = staged_commit_block_count[0]
    out_required_src_blocks[0] = staged_commit_required_src_blocks[0]
    out_required_dst_values[0] = staged_commit_required_dst_values[0]
    out_required_src_bytes[0] = staged_commit_required_src_bytes[0]
    out_required_dst_bytes[0] = staged_commit_required_dst_bytes[0]
    return Q8_0_OK


def test_source_contains_iq969_parity_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert (
        "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnly("
        in body
    )
    assert (
        "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly("
        in body
    )
    assert "if (snapshot_src_block_capacity < 0 || snapshot_dst_q16_capacity < 0)" in body
    assert "if (staged_commit_block_count != snapshot_block_count ||" in body
    assert "*out_required_dst_bytes = staged_commit_required_dst_bytes;" in body


def test_success_and_snapshot_failure() -> None:
    rng = random.Random(969)
    src_blocks = [make_q8_block(rng) for _ in range(10)]
    dst_q16 = [1003] * 600
    before = dst_q16[:]

    out_block_count = [1]
    out_required_src_blocks = [2]
    out_required_dst_values = [3]
    out_required_src_bytes = [4]
    out_required_dst_bytes = [5]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_parity(
        src_blocks,
        10,
        2,
        5,
        dst_q16,
        len(dst_q16),
        40,
        out_block_count,
        out_required_src_blocks,
        out_required_dst_values,
        out_required_src_bytes,
        out_required_dst_bytes,
    )
    assert err == Q8_0_OK
    assert out_block_count == [5]
    assert out_required_src_blocks == [9]
    assert out_required_dst_values == [192]
    assert out_required_src_bytes == [9 * Q8_0_BLOCK_BYTES]
    assert out_required_dst_bytes == [192 * 8]
    assert dst_q16 == before

    fail_block = [91]
    fail_src = [92]
    fail_dst = [93]
    fail_src_b = [94]
    fail_dst_b = [95]

    err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_parity(
        src_blocks,
        10,
        2,
        5,
        dst_q16,
        len(dst_q16),
        40,
        fail_block,
        fail_src,
        fail_dst,
        fail_src_b,
        fail_dst_b,
        mutate_snapshot=True,
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert fail_block == [91]
    assert fail_src == [92]
    assert fail_dst == [93]
    assert fail_src_b == [94]
    assert fail_dst_b == [95]


def test_randomized_adversarial_vectors() -> None:
    rng = random.Random(969000)

    for _ in range(220):
        src_stride = rng.randint(1, 5)
        block_count = rng.randint(0, 24)
        src_capacity = block_count * src_stride + rng.randint(0, 8)

        dst_stride = rng.randint(Q8_0_VALUES_PER_BLOCK, Q8_0_VALUES_PER_BLOCK + 64)
        required_dst_values = 0 if block_count == 0 else ((block_count - 1) * dst_stride + Q8_0_VALUES_PER_BLOCK)
        dst_capacity = required_dst_values + rng.randint(0, 40)

        src_blocks = [make_q8_block(rng) for _ in range(max(src_capacity, 1))]
        dst_q16 = [777] * max(dst_capacity, 1)

        out_block_count = [0]
        out_required_src_blocks = [0]
        out_required_dst_values = [0]
        out_required_src_bytes = [0]
        out_required_dst_bytes = [0]

        err = q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_preflight_only_parity_commit_only_parity(
            src_blocks,
            src_capacity,
            src_stride,
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

        expected_src_blocks = 0 if block_count == 0 else ((block_count - 1) * src_stride + 1)
        assert out_block_count == [block_count]
        assert out_required_src_blocks == [expected_src_blocks]
        assert out_required_dst_values == [required_dst_values]
        assert out_required_src_bytes == [expected_src_blocks * Q8_0_BLOCK_BYTES]
        assert out_required_dst_bytes == [required_dst_values * 8]


def run() -> None:
    test_source_contains_iq969_parity_contract()
    test_success_and_snapshot_failure()
    test_randomized_adversarial_vectors()
    print(
        "q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_"
        "parity_commit_only_preflight_only_parity_commit_only_preflight_only_"
        "parity_commit_only_parity=ok"
    )


if __name__ == "__main__":
    run()
