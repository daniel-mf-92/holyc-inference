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

I64_MAX = (1 << 63) - 1


def try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < -I64_MAX - 1 - rhs:
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
    out_dot: list[int] | None,
    out_required_lhs_blocks: list[int] | None,
    out_required_rhs_blocks: list[int] | None,
    out_required_lhs_bytes: list[int] | None,
    out_required_rhs_bytes: list[int] | None,
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot is None:
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
        or out_required_lhs_blocks is out_dot
        or out_required_rhs_blocks is lhs_blocks
        or out_required_rhs_blocks is rhs_blocks
        or out_required_rhs_blocks is out_dot
        or out_required_lhs_bytes is lhs_blocks
        or out_required_lhs_bytes is rhs_blocks
        or out_required_lhs_bytes is out_dot
        or out_required_rhs_bytes is lhs_blocks
        or out_required_rhs_bytes is rhs_blocks
        or out_required_rhs_bytes is out_dot
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if lhs_block_capacity < 0 or rhs_block_capacity < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_block_stride <= 0 or rhs_block_stride <= 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    snapshot_lhs = lhs_blocks
    snapshot_rhs = rhs_blocks
    snapshot_out = out_dot
    snapshot_lhs_capacity = lhs_block_capacity
    snapshot_lhs_stride = lhs_block_stride
    snapshot_rhs_capacity = rhs_block_capacity
    snapshot_rhs_stride = rhs_block_stride
    snapshot_block_count = block_count

    status = q4_0_q8_0_dot_q32_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity,
        lhs_block_stride,
        rhs_blocks,
        rhs_block_capacity,
        rhs_block_stride,
        block_count,
        out_dot,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

    if block_count == 0:
        staged_required_lhs_blocks = 0
        staged_required_rhs_blocks = 0
        staged_required_lhs_bytes = 0
        staged_required_rhs_bytes = 0
    else:
        ok, lhs_last_offset = try_mul_i64_nonneg(block_count - 1, lhs_block_stride)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, rhs_last_offset = try_mul_i64_nonneg(block_count - 1, rhs_block_stride)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, staged_required_lhs_blocks = try_add_i64(lhs_last_offset, 1)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, staged_required_rhs_blocks = try_add_i64(rhs_last_offset, 1)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        if staged_required_lhs_blocks > lhs_block_capacity:
            return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
        if staged_required_rhs_blocks > rhs_block_capacity:
            return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

        ok, staged_required_lhs_bytes = try_mul_i64_nonneg(staged_required_lhs_blocks, 18)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, staged_required_rhs_bytes = try_mul_i64_nonneg(staged_required_rhs_blocks, 34)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

    if (
        snapshot_lhs is not lhs_blocks
        or snapshot_rhs is not rhs_blocks
        or snapshot_out is not out_dot
        or snapshot_lhs_capacity != lhs_block_capacity
        or snapshot_lhs_stride != lhs_block_stride
        or snapshot_rhs_capacity != rhs_block_capacity
        or snapshot_rhs_stride != rhs_block_stride
        or snapshot_block_count != block_count
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    out_required_lhs_blocks[0] = staged_required_lhs_blocks
    out_required_rhs_blocks[0] = staged_required_rhs_blocks
    out_required_lhs_bytes[0] = staged_required_lhs_bytes
    out_required_rhs_bytes[0] = staged_required_rhs_bytes
    return ref.Q4_0_Q8_0_OK


def test_source_contains_iq1000_signature_and_atomic_publish() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 Q4_0Q8_0DotQ32CheckedNoPartialArray(",
        1,
    )[0]

    assert "status = Q4_0Q8_0DotQ32CheckedNoPartial(" in body
    assert "snapshot_block_count" in body
    assert "Q4_0Q8_0TryMulI64NonNeg(staged_required_lhs_blocks" in body
    assert "Q4_0Q8_0TryMulI64NonNeg(staged_required_rhs_blocks" in body
    assert "*out_required_lhs_blocks = staged_required_lhs_blocks;" in body
    assert "*out_required_rhs_blocks = staged_required_rhs_blocks;" in body
    assert "*out_required_lhs_bytes = staged_required_lhs_bytes;" in body
    assert "*out_required_rhs_bytes = staged_required_rhs_bytes;" in body


def test_alias_and_shape_guard_vectors_keep_outputs_unchanged() -> None:
    rng = random.Random(202604221000)

    lhs = [make_q4_block(rng) for _ in range(4)]
    rhs = [make_q8_block(rng) for _ in range(4)]
    out = [0x1111]

    req_lhs_blocks = [0xA1]
    req_rhs_blocks = [0xA2]
    req_lhs_bytes = [0xA3]
    req_rhs_bytes = [0xA4]

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        len(lhs),
        1,
        rhs,
        len(rhs),
        1,
        2,
        out,
        req_lhs_blocks,
        req_lhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert req_lhs_blocks == [0xA1]
    assert req_lhs_bytes == [0xA3]
    assert req_rhs_bytes == [0xA4]

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        len(lhs),
        1,
        rhs,
        len(rhs),
        1,
        2,
        out,
        out,
        req_rhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out == [0x1111]

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        -1,
        1,
        rhs,
        len(rhs),
        1,
        2,
        out,
        req_lhs_blocks,
        req_rhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        len(lhs),
        0,
        rhs,
        len(rhs),
        1,
        2,
        out,
        req_lhs_blocks,
        req_rhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_known_vector_and_required_tuple_publish() -> None:
    rng = random.Random(202604221001)

    block_count = 3
    lhs_stride = 2
    rhs_stride = 1

    lhs_required = (block_count - 1) * lhs_stride + 1
    rhs_required = (block_count - 1) * rhs_stride + 1

    lhs = [make_q4_block(rng) for _ in range(lhs_required + 2)]
    rhs = [make_q8_block(rng) for _ in range(rhs_required + 1)]

    out = [0]
    req_lhs_blocks = [123]
    req_rhs_blocks = [124]
    req_lhs_bytes = [125]
    req_rhs_bytes = [126]

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        len(lhs),
        lhs_stride,
        rhs,
        len(rhs),
        rhs_stride,
        block_count,
        out,
        req_lhs_blocks,
        req_rhs_blocks,
        req_lhs_bytes,
        req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_OK

    expected = [0]
    err_base = q4_0_q8_0_dot_q32_checked_nopartial(
        lhs,
        len(lhs),
        lhs_stride,
        rhs,
        len(rhs),
        rhs_stride,
        block_count,
        expected,
    )
    assert err_base == ref.Q4_0_Q8_0_OK
    assert out == expected
    assert req_lhs_blocks == [lhs_required]
    assert req_rhs_blocks == [rhs_required]
    assert req_lhs_bytes == [lhs_required * 18]
    assert req_rhs_bytes == [rhs_required * 34]


def test_no_partial_diagnostics_when_base_or_capacity_fails() -> None:
    rng = random.Random(202604221002)

    lhs = [make_q4_block(rng) for _ in range(6)]
    rhs = [make_q8_block(rng) for _ in range(6)]

    out = [0x5151]
    req_lhs_blocks = [0x61]
    req_rhs_blocks = [0x62]
    req_lhs_bytes = [0x63]
    req_rhs_bytes = [0x64]

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        lhs_block_capacity=6,
        lhs_block_stride=3,
        rhs_blocks=rhs,
        rhs_block_capacity=6,
        rhs_block_stride=3,
        block_count=3,
        out_dot=out,
        out_required_lhs_blocks=req_lhs_blocks,
        out_required_rhs_blocks=req_rhs_blocks,
        out_required_lhs_bytes=req_lhs_bytes,
        out_required_rhs_bytes=req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out == [0x5151]
    assert req_lhs_blocks == [0x61]
    assert req_rhs_blocks == [0x62]
    assert req_lhs_bytes == [0x63]
    assert req_rhs_bytes == [0x64]

    out_ok = [0xAAAA]
    req_lhs_blocks_ok = [0x71]
    req_rhs_blocks_ok = [0x72]
    req_lhs_bytes_ok = [0x73]
    req_rhs_bytes_ok = [0x74]
    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        lhs_block_capacity=6,
        lhs_block_stride=2,
        rhs_blocks=rhs,
        rhs_block_capacity=6,
        rhs_block_stride=2,
        block_count=3,
        out_dot=out_ok,
        out_required_lhs_blocks=req_lhs_blocks_ok,
        out_required_rhs_blocks=req_rhs_blocks_ok,
        out_required_lhs_bytes=req_lhs_bytes_ok,
        out_required_rhs_bytes=req_rhs_bytes_ok,
    )
    assert err == ref.Q4_0_Q8_0_OK

    out_fail = [0xBBBB]
    req_lhs_blocks_fail = [0x81]
    req_rhs_blocks_fail = [0x82]
    req_lhs_bytes_fail = [0x83]
    req_rhs_bytes_fail = [0x84]
    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        lhs_block_capacity=4,
        lhs_block_stride=2,
        rhs_blocks=rhs,
        rhs_block_capacity=6,
        rhs_block_stride=2,
        block_count=3,
        out_dot=out_fail,
        out_required_lhs_blocks=req_lhs_blocks_fail,
        out_required_rhs_blocks=req_rhs_blocks_fail,
        out_required_lhs_bytes=req_lhs_bytes_fail,
        out_required_rhs_bytes=req_rhs_bytes_fail,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert req_lhs_blocks_fail == [0x81]
    assert req_rhs_blocks_fail == [0x82]
    assert req_lhs_bytes_fail == [0x83]
    assert req_rhs_bytes_fail == [0x84]


def test_randomized_stride_capacity_alias_vectors() -> None:
    rng = random.Random(202604221003)

    for i in range(900):
        block_count = rng.randint(0, 9)
        lhs_stride = rng.randint(1, 4)
        rhs_stride = rng.randint(1, 4)

        lhs_required = 0 if block_count == 0 else (block_count - 1) * lhs_stride + 1
        rhs_required = 0 if block_count == 0 else (block_count - 1) * rhs_stride + 1

        lhs_capacity = lhs_required + rng.randint(0, 3)
        rhs_capacity = rhs_required + rng.randint(0, 3)

        local_rng = random.Random(20260422100300 + i)
        lhs = [make_q4_block(local_rng) for _ in range(max(lhs_capacity, 1))]
        rhs = [make_q8_block(local_rng) for _ in range(max(rhs_capacity, 1))]

        out = [0x1234]
        req_lhs_blocks = [0x201]
        req_rhs_blocks = [0x202]
        req_lhs_bytes = [0x203]
        req_rhs_bytes = [0x204]

        err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
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

        expected = [0x1234]
        base_err = q4_0_q8_0_dot_q32_checked_nopartial(
            lhs,
            lhs_capacity,
            lhs_stride,
            rhs,
            rhs_capacity,
            rhs_stride,
            block_count,
            expected,
        )

        if base_err != ref.Q4_0_Q8_0_OK:
            assert err == base_err
            assert out == [0x1234]
            assert req_lhs_blocks == [0x201]
            assert req_rhs_blocks == [0x202]
            assert req_lhs_bytes == [0x203]
            assert req_rhs_bytes == [0x204]
            continue

        assert err == ref.Q4_0_Q8_0_OK
        assert out == expected
        assert req_lhs_blocks == [lhs_required]
        assert req_rhs_blocks == [rhs_required]
        assert req_lhs_bytes == [lhs_required * 18]
        assert req_rhs_bytes == [rhs_required * 34]


def test_zero_block_count_publishes_zero_diagnostics() -> None:
    rng = random.Random(202604221004)
    lhs = [make_q4_block(rng)]
    rhs = [make_q8_block(rng)]

    out = [999]
    req_lhs_blocks = [1]
    req_rhs_blocks = [2]
    req_lhs_bytes = [3]
    req_rhs_bytes = [4]

    err = q4_0_q8_0_dot_q32_checked_nopartial_commit_only(
        lhs,
        lhs_block_capacity=0,
        lhs_block_stride=1,
        rhs_blocks=rhs,
        rhs_block_capacity=0,
        rhs_block_stride=1,
        block_count=0,
        out_dot=out,
        out_required_lhs_blocks=req_lhs_blocks,
        out_required_rhs_blocks=req_rhs_blocks,
        out_required_lhs_bytes=req_lhs_bytes,
        out_required_rhs_bytes=req_rhs_bytes,
    )
    assert err == ref.Q4_0_Q8_0_OK
    assert out == [0]
    assert req_lhs_blocks == [0]
    assert req_rhs_blocks == [0]
    assert req_lhs_bytes == [0]
    assert req_rhs_bytes == [0]


if __name__ == "__main__":
    test_source_contains_iq1000_signature_and_atomic_publish()
    test_alias_and_shape_guard_vectors_keep_outputs_unchanged()
    test_known_vector_and_required_tuple_publish()
    test_no_partial_diagnostics_when_base_or_capacity_fails()
    test_randomized_stride_capacity_alias_vectors()
    test_zero_block_count_publishes_zero_diagnostics()
    print("q4_0_q8_0_dot_q32_checked_nopartial_commit_only=ok")
