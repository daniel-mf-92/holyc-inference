#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartialCommitOnly (IQ-1000)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref
from test_q4_0_q8_0_dot_q32_checked_nopartial import (
    make_q4_block,
    make_q8_block,
    q4_0_q8_0_dot_q32_checked_nopartial,
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


def q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
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

    status = q4_0_q8_0_dot_q32_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity,
        lhs_block_stride,
        rhs_blocks,
        rhs_block_capacity,
        rhs_block_stride,
        block_count,
        out_dot_q32,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

    if block_count == 0:
        staged_required_lhs_blocks = 0
        staged_required_rhs_blocks = 0
        staged_required_lhs_bytes = 0
        staged_required_rhs_bytes = 0
    else:
        ok, lhs_last_offset = _try_mul_i64_nonneg(block_count - 1, lhs_block_stride)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, rhs_last_offset = _try_mul_i64_nonneg(block_count - 1, rhs_block_stride)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, staged_required_lhs_blocks = _try_add_i64(lhs_last_offset, 1)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, staged_required_rhs_blocks = _try_add_i64(rhs_last_offset, 1)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        if staged_required_lhs_blocks > lhs_block_capacity:
            return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
        if staged_required_rhs_blocks > rhs_block_capacity:
            return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

        ok, staged_required_lhs_bytes = _try_mul_i64_nonneg(staged_required_lhs_blocks, Q4_0_BLOCK_BYTES)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, staged_required_rhs_bytes = _try_mul_i64_nonneg(staged_required_rhs_blocks, Q8_0_BLOCK_BYTES)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

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

    out_required_lhs_blocks[0] = staged_required_lhs_blocks
    out_required_rhs_blocks[0] = staged_required_rhs_blocks
    out_required_lhs_bytes[0] = staged_required_lhs_bytes
    out_required_rhs_bytes[0] = staged_required_rhs_bytes
    return ref.Q4_0_Q8_0_OK


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


def test_source_contains_iq1000_signature_and_commit_wrapper_contract() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 Q4_0Q8_0DotQ32CheckedNoPartialArray(", 1)[0]

    assert "status = Q4_0Q8_0DotQ32CheckedNoPartial(" in body
    assert "snapshot_lhs_ptr" in body
    assert "*out_required_lhs_blocks = staged_required_lhs_blocks;" in body


def test_null_and_alias_guards_keep_outputs_unchanged() -> None:
    rng = random.Random(1000)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]
    out = [123]

    req_lhs_blocks = [10]
    req_rhs_blocks = [11]
    req_lhs_bytes = [12]
    req_rhs_bytes = [13]

    status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        1,
        1,
        rhs,
        1,
        1,
        1,
        out,
        None,
        req_rhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert status == ref.Q4_0_Q8_0_ERR_NULL_PTR
    assert out == [123]
    assert req_lhs_blocks == [10]

    status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        1,
        1,
        rhs,
        1,
        1,
        1,
        out,
        req_lhs_blocks,
        req_lhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert status == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out == [123]
    assert req_lhs_blocks == [10]
    assert req_lhs_bytes == [12]


def test_known_vector_and_randomized_parity() -> None:
    rng = random.Random(202604221000)

    for _ in range(450):
        block_count = rng.randint(0, 7)
        lhs_stride = rng.randint(1, 4)
        rhs_stride = rng.randint(1, 4)

        lhs_required = 0 if block_count == 0 else (block_count - 1) * lhs_stride + 1
        rhs_required = 0 if block_count == 0 else (block_count - 1) * rhs_stride + 1
        lhs_capacity = lhs_required + rng.randint(0, 3)
        rhs_capacity = rhs_required + rng.randint(0, 3)

        lhs = [make_q4_block(rng) for _ in range(max(lhs_capacity, 1))]
        rhs = [make_q8_block(rng) for _ in range(max(rhs_capacity, 1))]

        out = [0x1234]
        req_lhs_blocks = [0x51]
        req_rhs_blocks = [0x52]
        req_lhs_bytes = [0x53]
        req_rhs_bytes = [0x54]

        status = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
            lhs,
            lhs_capacity,
            lhs_stride,
            rhs,
            rhs_capacity,
            rhs_stride,
            block_count,
            out,
            req_lhs_blocks,
            req_rhs_blocks,
            req_lhs_bytes,
            req_rhs_bytes,
        )

        expected_status = q4_0_q8_0_dot_q32_checked_nopartial(
            lhs,
            lhs_capacity,
            lhs_stride,
            rhs,
            rhs_capacity,
            rhs_stride,
            block_count,
            [0],
        )

        assert status == expected_status

        if status == ref.Q4_0_Q8_0_OK:
            diag_status, expected = expected_diagnostics(
                lhs_capacity,
                lhs_stride,
                rhs_capacity,
                rhs_stride,
                block_count,
            )
            assert diag_status == ref.Q4_0_Q8_0_OK
            assert (
                req_lhs_blocks[0],
                req_rhs_blocks[0],
                req_lhs_bytes[0],
                req_rhs_bytes[0],
            ) == expected
        else:
            assert req_lhs_blocks == [0x51]
            assert req_rhs_blocks == [0x52]
            assert req_lhs_bytes == [0x53]
            assert req_rhs_bytes == [0x54]


if __name__ == "__main__":
    test_source_contains_iq1000_signature_and_commit_wrapper_contract()
    test_null_and_alias_guards_keep_outputs_unchanged()
    test_known_vector_and_randomized_parity()
    print("q4_0_q8_0_dot_q32_checked_nopartial_commit_only=ok")

