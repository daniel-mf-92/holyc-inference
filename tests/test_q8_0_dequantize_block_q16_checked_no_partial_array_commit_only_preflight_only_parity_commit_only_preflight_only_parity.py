#!/usr/bin/env python3
"""Parity harness for Q8_0...PreflightOnlyParityCommitOnlyPreflightOnlyParity (IQ-958)."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

_PRE_PARITY_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only.py"
)
_COMMIT_PATH = Path(
    "tests/test_q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only.py"
)

_spec_pre_parity = importlib.util.spec_from_file_location("q8_preflight_parity_base", _PRE_PARITY_PATH)
assert _spec_pre_parity and _spec_pre_parity.loader
_pre_parity = importlib.util.module_from_spec(_spec_pre_parity)
sys.modules[_spec_pre_parity.name] = _pre_parity
_spec_pre_parity.loader.exec_module(_pre_parity)

_spec_commit = importlib.util.spec_from_file_location("q8_parity_commit_base", _COMMIT_PATH)
assert _spec_commit and _spec_commit.loader
_commit = importlib.util.module_from_spec(_spec_commit)
sys.modules[_spec_commit.name] = _commit
_spec_commit.loader.exec_module(_commit)

Q8_0_OK = _commit.Q8_0_OK
Q8_0_ERR_NULL_PTR = _commit.Q8_0_ERR_NULL_PTR
Q8_0_ERR_BAD_DST_LEN = _commit.Q8_0_ERR_BAD_DST_LEN
Q8_0_ERR_OVERFLOW = _commit.Q8_0_ERR_OVERFLOW

make_q8_block = _commit.make_q8_block
parity_commit_only = _commit.q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only


def preflight_only_parity_commit_only_preflight_only(
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
    return parity_commit_only(
        src_blocks,
        src_block_capacity,
        src_block_stride,
        block_count,
        dst_q16,
        dst_q16_capacity,
        dst_block_stride_values,
        out_block_count,
        out_required_src_blocks,
        out_required_dst_values,
        out_required_src_bytes,
        out_required_dst_bytes,
    )


def q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
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

    err = preflight_only_parity_commit_only_preflight_only(
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

    err = parity_commit_only(
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


def test_source_contains_iq958_parity_contract() -> None:
    source = Path("src/quant/q8_0.HC").read_text(encoding="utf-8")
    sig = "I32 Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly(" in body
    assert "status = Q8_0DequantizeBlockQ16CheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly(" in body
    assert "staged_block_count != parity_block_count" in body
    assert "staged_required_src_blocks != parity_required_src_blocks" in body
    assert "staged_required_dst_bytes != parity_required_dst_bytes" in body


def test_parity_publish_and_no_write() -> None:
    rng = random.Random(958)
    src_blocks = [make_q8_block(rng) for _ in range(8)]
    dst_q16 = [333] * 256
    before = dst_q16[:]

    out_block_count = [0]
    out_required_src_blocks = [0]
    out_required_dst_values = [0]
    out_required_src_bytes = [0]
    out_required_dst_bytes = [0]

    err = (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            src_blocks,
            8,
            2,
            3,
            dst_q16,
            len(dst_q16),
            40,
            out_block_count,
            out_required_src_blocks,
            out_required_dst_values,
            out_required_src_bytes,
            out_required_dst_bytes,
        )
    )
    assert err == Q8_0_OK
    assert dst_q16 == before
    assert out_block_count == [3]
    assert out_required_src_blocks == [5]
    assert out_required_dst_values == [112]
    assert out_required_src_bytes == [5 * 34]
    assert out_required_dst_bytes == [112 * 8]


def test_failure_keeps_publish_cells_unchanged() -> None:
    rng = random.Random(959)
    src_blocks = [make_q8_block(rng) for _ in range(4)]
    dst_q16 = [0] * 64

    out_block_count = [71]
    out_required_src_blocks = [72]
    out_required_dst_values = [73]
    out_required_src_bytes = [74]
    out_required_dst_bytes = [75]

    err = (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            src_blocks,
            4,
            1,
            3,
            dst_q16,
            len(dst_q16),
            20,
            out_block_count,
            out_required_src_blocks,
            out_required_dst_values,
            out_required_src_bytes,
            out_required_dst_bytes,
        )
    )
    assert err == Q8_0_ERR_BAD_DST_LEN
    assert out_block_count == [71]
    assert out_required_src_blocks == [72]
    assert out_required_dst_values == [73]
    assert out_required_src_bytes == [74]
    assert out_required_dst_bytes == [75]


def test_snapshot_mutation_rejected() -> None:
    rng = random.Random(960)
    src_blocks = [make_q8_block(rng) for _ in range(5)]
    dst_q16 = [1] * 200

    out_block_count = [0]
    out_required_src_blocks = [0]
    out_required_dst_values = [0]
    out_required_src_bytes = [0]
    out_required_dst_bytes = [0]

    err = (
        q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity(
            src_blocks,
            5,
            1,
            2,
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
    )
    assert err == Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_source_contains_iq958_parity_contract()
    test_parity_publish_and_no_write()
    test_failure_keeps_publish_cells_unchanged()
    test_snapshot_mutation_rejected()
    print(
        "q8_0_dequantize_block_q16_checked_no_partial_array_commit_only_preflight_only_parity_commit_only_preflight_only_parity=ok"
    )


if __name__ == "__main__":
    run()
