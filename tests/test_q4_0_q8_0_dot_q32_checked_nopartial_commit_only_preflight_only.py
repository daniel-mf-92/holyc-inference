#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartialCommitOnlyPreflightOnly (IQ-1001)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref
from test_q4_0_q8_0_dot_q32_checked_nopartial import (
    make_q4_block,
    make_q8_block,
)
from test_q4_0_q8_0_dot_q32_checked_nopartial_commit_only import (
    q4_0_q8_0_dot_q32_checked_nopartial_commit_only,
)

Q4_0_BLOCK_BYTES = 18
Q8_0_BLOCK_BYTES = 34


def _try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > ref.Q4_0_Q8_0_I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def _try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > ref.Q4_0_Q8_0_I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < ref.Q4_0_Q8_0_I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def expected_diagnostics(
    lhs_block_capacity: int,
    lhs_block_stride: int,
    rhs_block_capacity: int,
    rhs_block_stride: int,
    block_count: int,
) -> tuple[int, tuple[int, int, int, int]]:
    if block_count == 0:
        return ref.Q4_0_Q8_0_OK, (0, 0, 0, 0)

    ok, lhs_last_offset = _try_mul_i64_nonneg(block_count - 1, lhs_block_stride)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0)
    ok, rhs_last_offset = _try_mul_i64_nonneg(block_count - 1, rhs_block_stride)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0)

    ok, lhs_blocks = _try_add_i64(lhs_last_offset, 1)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0)
    ok, rhs_blocks = _try_add_i64(rhs_last_offset, 1)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0)

    if lhs_blocks > lhs_block_capacity or rhs_blocks > rhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN, (0, 0, 0, 0)

    ok, lhs_bytes = _try_mul_i64_nonneg(lhs_blocks, Q4_0_BLOCK_BYTES)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0)
    ok, rhs_bytes = _try_mul_i64_nonneg(rhs_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0)

    return ref.Q4_0_Q8_0_OK, (lhs_blocks, rhs_blocks, lhs_bytes, rhs_bytes)


def q4_0_q8_0_dot_q32_checked_nopartial_commit_only_preflight_only(
    lhs_blocks,
    lhs_block_capacity: int,
    lhs_block_stride: int,
    rhs_blocks,
    rhs_block_capacity: int,
    rhs_block_stride: int,
    block_count: int,
    out_dot_q32,
    out_required_lhs_blocks,
    out_required_rhs_blocks,
    out_required_lhs_bytes,
    out_required_rhs_bytes,
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot_q32 is None:
        return ref.Q4_0_Q8_0_ERR_NULL_PTR
    if (
        out_required_lhs_blocks is None
        or out_required_rhs_blocks is None
        or out_required_lhs_bytes is None
        or out_required_rhs_bytes is None
    ):
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if (
        out_required_lhs_blocks is out_required_rhs_blocks
        or out_required_lhs_blocks is out_required_lhs_bytes
        or out_required_lhs_blocks is out_required_rhs_bytes
        or out_required_rhs_blocks is out_required_lhs_bytes
        or out_required_rhs_blocks is out_required_rhs_bytes
        or out_required_lhs_bytes is out_required_rhs_bytes
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if (
        out_required_lhs_blocks is lhs_blocks
        or out_required_lhs_blocks is rhs_blocks
        or out_required_lhs_blocks is out_dot_q32
        or out_required_rhs_blocks is lhs_blocks
        or out_required_rhs_blocks is rhs_blocks
        or out_required_rhs_blocks is out_dot_q32
        or out_required_lhs_bytes is lhs_blocks
        or out_required_lhs_bytes is rhs_blocks
        or out_required_lhs_bytes is out_dot_q32
        or out_required_rhs_bytes is lhs_blocks
        or out_required_rhs_bytes is rhs_blocks
        or out_required_rhs_bytes is out_dot_q32
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if lhs_block_capacity < 0 or rhs_block_capacity < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_block_stride <= 0 or rhs_block_stride <= 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    snapshot = (
        lhs_blocks,
        rhs_blocks,
        out_dot_q32,
        lhs_block_capacity,
        lhs_block_stride,
        rhs_block_capacity,
        rhs_block_stride,
        block_count,
    )

    staged_out_dot_q32 = [0x6A6A6A6A]
    staged_required_lhs_blocks = [0]
    staged_required_rhs_blocks = [0]
    staged_required_lhs_bytes = [0]
    staged_required_rhs_bytes = [0]

    status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs_blocks,
        lhs_block_capacity,
        lhs_block_stride,
        rhs_blocks,
        rhs_block_capacity,
        rhs_block_stride,
        block_count,
        staged_out_dot_q32,
        staged_required_lhs_blocks,
        staged_required_rhs_blocks,
        staged_required_lhs_bytes,
        staged_required_rhs_bytes,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

    canonical_status, canonical_diag = expected_diagnostics(
        lhs_block_capacity,
        lhs_block_stride,
        rhs_block_capacity,
        rhs_block_stride,
        block_count,
    )
    if canonical_status != ref.Q4_0_Q8_0_OK:
        return canonical_status

    if snapshot != (
        lhs_blocks,
        rhs_blocks,
        out_dot_q32,
        lhs_block_capacity,
        lhs_block_stride,
        rhs_block_capacity,
        rhs_block_stride,
        block_count,
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if (
        staged_required_lhs_blocks[0],
        staged_required_rhs_blocks[0],
        staged_required_lhs_bytes[0],
        staged_required_rhs_bytes[0],
    ) != canonical_diag:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    out_required_lhs_blocks[0] = staged_required_lhs_blocks[0]
    out_required_rhs_blocks[0] = staged_required_rhs_blocks[0]
    out_required_lhs_bytes[0] = staged_required_lhs_bytes[0]
    out_required_rhs_bytes[0] = staged_required_rhs_bytes[0]
    return ref.Q4_0_Q8_0_OK


def test_source_contains_iq1001_signature_and_preflight_only_contract() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialCommitOnlyPreflightOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 Q4_0Q8_0DotQ32CheckedNoPartialArray(", 1)[0]

    assert "status = Q4_0Q8_0DotQ32CheckedNoPartialCommitOnly(" in body
    assert "snapshot_lhs_ptr" in body
    assert "if (staged_required_lhs_blocks != canonical_required_lhs_blocks" in body
    assert "*out_required_lhs_blocks = staged_required_lhs_blocks;" in body


def test_no_partial_publish_and_out_dot_is_not_touched() -> None:
    rng = random.Random(1001)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]

    out_dot_q32 = [0x4444]
    req_lhs_blocks = [0x51]
    req_rhs_blocks = [0x52]
    req_lhs_bytes = [0x53]
    req_rhs_bytes = [0x54]

    status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only_preflight_only(
        lhs,
        1,
        1,
        rhs,
        1,
        1,
        1,
        out_dot_q32,
        None,
        req_rhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert status == ref.Q4_0_Q8_0_ERR_NULL_PTR
    assert out_dot_q32 == [0x4444]
    assert req_lhs_blocks == [0x51]

    status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only_preflight_only(
        lhs,
        1,
        1,
        rhs,
        1,
        1,
        1,
        out_dot_q32,
        req_lhs_blocks,
        req_lhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert status == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out_dot_q32 == [0x4444]
    assert req_lhs_blocks == [0x51]
    assert req_lhs_bytes == [0x53]


def test_known_vectors_and_randomized_parity() -> None:
    rng = random.Random(202604221001)

    for _ in range(500):
        block_count = rng.randint(0, 9)
        lhs_stride = rng.randint(1, 4)
        rhs_stride = rng.randint(1, 4)

        lhs_required = 0 if block_count == 0 else (block_count - 1) * lhs_stride + 1
        rhs_required = 0 if block_count == 0 else (block_count - 1) * rhs_stride + 1

        # Inject both valid and invalid capacities.
        lhs_capacity = lhs_required + rng.randint(-1, 3)
        rhs_capacity = rhs_required + rng.randint(-1, 3)
        lhs_capacity = max(lhs_capacity, 0)
        rhs_capacity = max(rhs_capacity, 0)

        lhs = [make_q4_block(rng) for _ in range(max(lhs_capacity, 1))]
        rhs = [make_q8_block(rng) for _ in range(max(rhs_capacity, 1))]

        out_dot_q32 = [0xABCD]
        req_lhs_blocks = [0x71]
        req_rhs_blocks = [0x72]
        req_lhs_bytes = [0x73]
        req_rhs_bytes = [0x74]

        status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only_preflight_only(
            lhs,
            lhs_capacity,
            lhs_stride,
            rhs,
            rhs_capacity,
            rhs_stride,
            block_count,
            out_dot_q32,
            req_lhs_blocks,
            req_rhs_blocks,
            req_lhs_bytes,
            req_rhs_bytes,
        )

        baseline_dot = [0]
        commit_lhs = [0]
        commit_rhs = [0]
        commit_lhs_bytes = [0]
        commit_rhs_bytes = [0]
        baseline_status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
            lhs,
            lhs_capacity,
            lhs_stride,
            rhs,
            rhs_capacity,
            rhs_stride,
            block_count,
            baseline_dot,
            commit_lhs,
            commit_rhs,
            commit_lhs_bytes,
            commit_rhs_bytes,
        )

        assert status == baseline_status

        if status == ref.Q4_0_Q8_0_OK:
            assert out_dot_q32 == [0xABCD]
            assert req_lhs_blocks[0] == commit_lhs[0]
            assert req_rhs_blocks[0] == commit_rhs[0]
            assert req_lhs_bytes[0] == commit_lhs_bytes[0]
            assert req_rhs_bytes[0] == commit_rhs_bytes[0]
        else:
            assert out_dot_q32 == [0xABCD]
            assert req_lhs_blocks == [0x71]
            assert req_rhs_blocks == [0x72]
            assert req_lhs_bytes == [0x73]
            assert req_rhs_bytes == [0x74]


if __name__ == "__main__":
    test_source_contains_iq1001_signature_and_preflight_only_contract()
    test_no_partial_publish_and_out_dot_is_not_touched()
    test_known_vectors_and_randomized_parity()
    print("q4_0_q8_0_dot_q32_checked_nopartial_commit_only_preflight_only=ok")
