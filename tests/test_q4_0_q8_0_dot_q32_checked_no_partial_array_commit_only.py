#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly (IQ-931)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref
import test_q4_0_q8_0_dot_q32_checked_no_partial_array as base_array

Q4_0_BLOCK_BYTES = 2 + 16
Q8_0_BLOCK_BYTES = 2 + 32
I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


def _try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def _try_mul_i64_nonneg(lhs: int, rhs: int) -> tuple[bool, int]:
    if lhs < 0 or rhs < 0:
        return False, 0
    if lhs == 0 or rhs == 0:
        return True, 0
    if lhs > I64_MAX // rhs:
        return False, 0
    return True, lhs * rhs


def q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
    lhs_blocks,
    lhs_block_capacity: int,
    lhs_pair_stride_blocks: int,
    lhs_block_stride: int,
    rhs_blocks,
    rhs_block_capacity: int,
    rhs_pair_stride_blocks: int,
    rhs_block_stride: int,
    pair_count: int,
    blocks_per_dot: int,
    out_dot_q32,
    out_dot_capacity: int,
    out_pair_count,
    out_required_lhs_blocks,
    out_required_rhs_blocks,
    out_required_out_cells,
    out_required_lhs_bytes,
    out_required_rhs_bytes,
    out_required_out_bytes,
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot_q32 is None:
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    outputs = [
        out_pair_count,
        out_required_lhs_blocks,
        out_required_rhs_blocks,
        out_required_out_cells,
        out_required_lhs_bytes,
        out_required_rhs_bytes,
        out_required_out_bytes,
    ]
    if any(out is None for out in outputs):
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if len({id(x) for x in outputs}) != len(outputs):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    for out in outputs:
        if out is out_dot_q32 or out is lhs_blocks or out is rhs_blocks:
            return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    snapshot = (
        lhs_blocks,
        rhs_blocks,
        out_dot_q32,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        out_dot_capacity,
    )

    err = base_array.q4_0_q8_0_dot_q32_checked_no_partial_array(
        lhs_blocks,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_blocks,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        out_dot_q32,
        out_dot_capacity,
    )
    if err != ref.Q4_0_Q8_0_OK:
        return err

    staged_pair_count = pair_count
    staged_required_out_cells = pair_count

    ok, staged_required_out_bytes = _try_mul_i64_nonneg(staged_required_out_cells, 8)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW

    if pair_count == 0 or blocks_per_dot == 0:
        staged_required_lhs_blocks = 0
        staged_required_rhs_blocks = 0
        staged_required_lhs_bytes = 0
        staged_required_rhs_bytes = 0
    else:
        ok, lhs_last_offset = _try_mul_i64_nonneg(blocks_per_dot - 1, lhs_block_stride)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, rhs_last_offset = _try_mul_i64_nonneg(blocks_per_dot - 1, rhs_block_stride)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, lhs_last_pair_base = _try_mul_i64_nonneg(pair_count - 1, lhs_pair_stride_blocks)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, rhs_last_pair_base = _try_mul_i64_nonneg(pair_count - 1, rhs_pair_stride_blocks)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, staged_required_lhs_blocks = _try_add_i64(lhs_last_pair_base, lhs_last_offset)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, staged_required_rhs_blocks = _try_add_i64(rhs_last_pair_base, rhs_last_offset)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW

        ok, staged_required_lhs_blocks = _try_add_i64(staged_required_lhs_blocks, 1)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        ok, staged_required_rhs_blocks = _try_add_i64(staged_required_rhs_blocks, 1)
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
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        out_dot_capacity,
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    out_pair_count[0] = staged_pair_count
    out_required_lhs_blocks[0] = staged_required_lhs_blocks
    out_required_rhs_blocks[0] = staged_required_rhs_blocks
    out_required_out_cells[0] = staged_required_out_cells
    out_required_lhs_bytes[0] = staged_required_lhs_bytes
    out_required_rhs_bytes[0] = staged_required_rhs_bytes
    out_required_out_bytes[0] = staged_required_out_bytes
    return ref.Q4_0_Q8_0_OK


def explicit_commit_only_composition(
    lhs_blocks,
    lhs_block_capacity: int,
    lhs_pair_stride_blocks: int,
    lhs_block_stride: int,
    rhs_blocks,
    rhs_block_capacity: int,
    rhs_pair_stride_blocks: int,
    rhs_block_stride: int,
    pair_count: int,
    blocks_per_dot: int,
    out_dot_capacity: int,
) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    staged_out = [0] * max(out_dot_capacity, 1)
    err = base_array.q4_0_q8_0_dot_q32_checked_no_partial_array(
        lhs_blocks,
        lhs_block_capacity,
        lhs_pair_stride_blocks,
        lhs_block_stride,
        rhs_blocks,
        rhs_block_capacity,
        rhs_pair_stride_blocks,
        rhs_block_stride,
        pair_count,
        blocks_per_dot,
        staged_out,
        out_dot_capacity,
    )
    if err != ref.Q4_0_Q8_0_OK:
        return err, (0, 0, 0, 0, 0, 0, 0)

    ok, required_out_bytes = _try_mul_i64_nonneg(pair_count, 8)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    if pair_count == 0 or blocks_per_dot == 0:
        return ref.Q4_0_Q8_0_OK, (pair_count, 0, 0, pair_count, 0, 0, required_out_bytes)

    ok, lhs_last_offset = _try_mul_i64_nonneg(blocks_per_dot - 1, lhs_block_stride)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, rhs_last_offset = _try_mul_i64_nonneg(blocks_per_dot - 1, rhs_block_stride)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    ok, lhs_last_pair_base = _try_mul_i64_nonneg(pair_count - 1, lhs_pair_stride_blocks)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, rhs_last_pair_base = _try_mul_i64_nonneg(pair_count - 1, rhs_pair_stride_blocks)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    ok, lhs_required_blocks = _try_add_i64(lhs_last_pair_base, lhs_last_offset)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, rhs_required_blocks = _try_add_i64(rhs_last_pair_base, rhs_last_offset)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    ok, lhs_required_blocks = _try_add_i64(lhs_required_blocks, 1)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, rhs_required_blocks = _try_add_i64(rhs_required_blocks, 1)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    ok, lhs_required_bytes = _try_mul_i64_nonneg(lhs_required_blocks, Q4_0_BLOCK_BYTES)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, rhs_required_bytes = _try_mul_i64_nonneg(rhs_required_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    return ref.Q4_0_Q8_0_OK, (
        pair_count,
        lhs_required_blocks,
        rhs_required_blocks,
        pair_count,
        lhs_required_bytes,
        rhs_required_bytes,
        required_out_bytes,
    )


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q4_from_signed(vals)


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q8_signed(vals)


def test_source_contains_iq931_function() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "Q4_0Q8_0DotQ32CheckedNoPartialArray(" in body
    assert "snapshot_lhs_pair_stride_blocks = lhs_pair_stride_blocks;" in body
    assert "staged_required_lhs_blocks = 0;" in body
    assert "*out_required_rhs_bytes = staged_required_rhs_bytes;" in body


def test_null_and_alias_guards() -> None:
    rng = random.Random(931)
    lhs_blocks = [make_q4_block(rng) for _ in range(8)]
    rhs_blocks = [make_q8_block(rng) for _ in range(8)]
    out_dots = [0, 0]

    out_pair_count = [0]
    out_lhs_blocks = [0]
    out_rhs_blocks = [0]
    out_out_cells = [0]
    out_lhs_bytes = [0]
    out_rhs_bytes = [0]
    out_out_bytes = [0]

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
            None,
            len(lhs_blocks),
            1,
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            1,
            2,
            1,
            out_dots,
            len(out_dots),
            out_pair_count,
            out_lhs_blocks,
            out_rhs_blocks,
            out_out_cells,
            out_lhs_bytes,
            out_rhs_bytes,
            out_out_bytes,
        )
        == ref.Q4_0_Q8_0_ERR_NULL_PTR
    )

    shared = [7]
    assert (
        q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
            lhs_blocks,
            len(lhs_blocks),
            1,
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            1,
            2,
            1,
            out_dots,
            len(out_dots),
            shared,
            shared,
            out_rhs_blocks,
            out_out_cells,
            out_lhs_bytes,
            out_rhs_bytes,
            out_out_bytes,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )


def test_known_vector_success_and_failure_no_partial() -> None:
    rng = random.Random(93101)
    lhs_blocks = [make_q4_block(rng) for _ in range(16)]
    rhs_blocks = [make_q8_block(rng) for _ in range(16)]

    out_dots = [111, 222, 333]
    out_pair_count = [9001]
    out_lhs_blocks = [9002]
    out_rhs_blocks = [9003]
    out_out_cells = [9004]
    out_lhs_bytes = [9005]
    out_rhs_bytes = [9006]
    out_out_bytes = [9007]

    err = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
        lhs_blocks,
        len(lhs_blocks),
        4,
        2,
        rhs_blocks,
        len(rhs_blocks),
        4,
        2,
        3,
        2,
        out_dots,
        len(out_dots),
        out_pair_count,
        out_lhs_blocks,
        out_rhs_blocks,
        out_out_cells,
        out_lhs_bytes,
        out_rhs_bytes,
        out_out_bytes,
    )
    assert err == ref.Q4_0_Q8_0_OK
    assert out_pair_count == [3]
    assert out_lhs_blocks == [11]
    assert out_rhs_blocks == [11]
    assert out_out_cells == [3]
    assert out_lhs_bytes == [11 * Q4_0_BLOCK_BYTES]
    assert out_rhs_bytes == [11 * Q8_0_BLOCK_BYTES]
    assert out_out_bytes == [24]

    fail_pair_count = [41]
    fail_lhs_blocks = [42]
    fail_rhs_blocks = [43]
    fail_out_cells = [44]
    fail_lhs_bytes = [45]
    fail_rhs_bytes = [46]
    fail_out_bytes = [47]
    before_out = out_dots[:]
    err = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
        lhs_blocks,
        len(lhs_blocks),
        4,
        2,
        rhs_blocks,
        len(rhs_blocks),
        4,
        2,
        3,
        2,
        out_dots,
        2,
        fail_pair_count,
        fail_lhs_blocks,
        fail_rhs_blocks,
        fail_out_cells,
        fail_lhs_bytes,
        fail_rhs_bytes,
        fail_out_bytes,
    )
    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert fail_pair_count == [41]
    assert fail_lhs_blocks == [42]
    assert fail_rhs_blocks == [43]
    assert fail_out_cells == [44]
    assert fail_lhs_bytes == [45]
    assert fail_rhs_bytes == [46]
    assert fail_out_bytes == [47]
    assert out_dots == before_out


def test_randomized_parity() -> None:
    rng = random.Random(20260421_931)

    for _ in range(2600):
        pair_count = rng.randint(0, 8)
        blocks_per_dot = rng.randint(0, 5)
        lhs_pair_stride = rng.randint(1, 5)
        rhs_pair_stride = rng.randint(1, 5)
        lhs_block_stride = rng.randint(1, 4)
        rhs_block_stride = rng.randint(1, 4)

        lhs_required = 1
        rhs_required = 1
        if pair_count > 0 and blocks_per_dot > 0:
            lhs_required = (pair_count - 1) * lhs_pair_stride + (blocks_per_dot - 1) * lhs_block_stride + 1
            rhs_required = (pair_count - 1) * rhs_pair_stride + (blocks_per_dot - 1) * rhs_block_stride + 1

        lhs_capacity = lhs_required + rng.randint(0, 3)
        rhs_capacity = rhs_required + rng.randint(0, 3)
        lhs_blocks = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs_blocks = [make_q8_block(rng) for _ in range(rhs_capacity)]

        if rng.random() < 0.17 and pair_count > 0 and blocks_per_dot > 0:
            lhs_capacity = max(0, lhs_required - rng.randint(1, min(2, lhs_required)))
        if rng.random() < 0.17 and pair_count > 0 and blocks_per_dot > 0:
            rhs_capacity = max(0, rhs_required - rng.randint(1, min(2, rhs_required)))

        out_capacity = max(0, pair_count + rng.randint(0, 2))
        if rng.random() < 0.16 and pair_count > 0:
            out_capacity = pair_count - 1

        out_dots = [rng.randint(-2_000_000, 2_000_000) for _ in range(max(out_capacity, 1))]
        out_pair_count = [rng.randint(-1000, 1000)]
        out_lhs_blocks = [rng.randint(-1000, 1000)]
        out_rhs_blocks = [rng.randint(-1000, 1000)]
        out_out_cells = [rng.randint(-1000, 1000)]
        out_lhs_bytes = [rng.randint(-1000, 1000)]
        out_rhs_bytes = [rng.randint(-1000, 1000)]
        out_out_bytes = [rng.randint(-1000, 1000)]

        err_impl = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
            lhs_blocks,
            lhs_capacity,
            lhs_pair_stride,
            lhs_block_stride,
            rhs_blocks,
            rhs_capacity,
            rhs_pair_stride,
            rhs_block_stride,
            pair_count,
            blocks_per_dot,
            out_dots,
            out_capacity,
            out_pair_count,
            out_lhs_blocks,
            out_rhs_blocks,
            out_out_cells,
            out_lhs_bytes,
            out_rhs_bytes,
            out_out_bytes,
        )

        err_ref, tup_ref = explicit_commit_only_composition(
            lhs_blocks,
            lhs_capacity,
            lhs_pair_stride,
            lhs_block_stride,
            rhs_blocks,
            rhs_capacity,
            rhs_pair_stride,
            rhs_block_stride,
            pair_count,
            blocks_per_dot,
            out_capacity,
        )

        assert err_impl == err_ref
        if err_impl == ref.Q4_0_Q8_0_OK:
            assert (
                out_pair_count[0],
                out_lhs_blocks[0],
                out_rhs_blocks[0],
                out_out_cells[0],
                out_lhs_bytes[0],
                out_rhs_bytes[0],
                out_out_bytes[0],
            ) == tup_ref


def main() -> None:
    test_source_contains_iq931_function()
    test_null_and_alias_guards()
    test_known_vector_success_and_failure_no_partial()
    test_randomized_parity()
    print("ok")


if __name__ == "__main__":
    main()
