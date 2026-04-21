#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnlyPreflightOnlyParity (IQ-936)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref
from test_q4_0_q8_0_dot_q32_checked_no_partial_array import (
    q4_0_q8_0_dot_q32_checked_no_partial_array,
)
from test_q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only import (
    q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only,
)


def q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only_parity(
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
    out_count,
    out_required_in_blocks,
    out_required_out_cells,
    out_required_in_bytes,
    out_required_out_bytes,
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot_q32 is None:
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if (
        out_count is None
        or out_required_in_blocks is None
        or out_required_out_cells is None
        or out_required_in_bytes is None
        or out_required_out_bytes is None
    ):
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if (
        out_count is out_required_in_blocks
        or out_count is out_required_out_cells
        or out_count is out_required_in_bytes
        or out_count is out_required_out_bytes
        or out_required_in_blocks is out_required_out_cells
        or out_required_in_blocks is out_required_in_bytes
        or out_required_in_blocks is out_required_out_bytes
        or out_required_out_cells is out_required_in_bytes
        or out_required_out_cells is out_required_out_bytes
        or out_required_in_bytes is out_required_out_bytes
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if (
        out_count is lhs_blocks
        or out_count is rhs_blocks
        or out_count is out_dot_q32
        or out_required_in_blocks is lhs_blocks
        or out_required_in_blocks is rhs_blocks
        or out_required_in_blocks is out_dot_q32
        or out_required_out_cells is lhs_blocks
        or out_required_out_cells is rhs_blocks
        or out_required_out_cells is out_dot_q32
        or out_required_in_bytes is lhs_blocks
        or out_required_in_bytes is rhs_blocks
        or out_required_in_bytes is out_dot_q32
        or out_required_out_bytes is lhs_blocks
        or out_required_out_bytes is rhs_blocks
        or out_required_out_bytes is out_dot_q32
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if lhs_block_capacity < 0 or rhs_block_capacity < 0 or out_dot_capacity < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if pair_count < 0 or blocks_per_dot < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_pair_stride_blocks <= 0 or rhs_pair_stride_blocks <= 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_block_stride <= 0 or rhs_block_stride <= 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if pair_count > out_dot_capacity:
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

    staged_count = [0]
    staged_required_in_blocks = [0]
    staged_required_out_cells = [0]
    staged_required_in_bytes = [0]
    staged_required_out_bytes = [0]

    status = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only(
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
        staged_count,
        staged_required_in_blocks,
        staged_required_out_cells,
        staged_required_in_bytes,
        staged_required_out_bytes,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

    # Canonical checked-array composition runs against a scratch output vector so
    # this diagnostics helper never mutates caller out_dot buffers.
    scratch_out = [0] * pair_count if pair_count > 0 else [0]
    status = q4_0_q8_0_dot_q32_checked_no_partial_array(
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
        scratch_out,
        pair_count,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

    recomputed_count = [0]
    recomputed_required_in_blocks = [0]
    recomputed_required_out_cells = [0]
    recomputed_required_in_bytes = [0]
    recomputed_required_out_bytes = [0]

    status = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only(
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
        recomputed_count,
        recomputed_required_in_blocks,
        recomputed_required_out_cells,
        recomputed_required_in_bytes,
        recomputed_required_out_bytes,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

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

    if staged_count[0] != recomputed_count[0]:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if staged_required_in_blocks[0] != recomputed_required_in_blocks[0]:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if staged_required_out_cells[0] != recomputed_required_out_cells[0]:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if staged_required_in_bytes[0] != recomputed_required_in_bytes[0]:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if staged_required_out_bytes[0] != recomputed_required_out_bytes[0]:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    out_count[0] = staged_count[0]
    out_required_in_blocks[0] = staged_required_in_blocks[0]
    out_required_out_cells[0] = staged_required_out_cells[0]
    out_required_in_bytes[0] = staged_required_in_bytes[0]
    out_required_out_bytes[0] = staged_required_out_bytes[0]
    return ref.Q4_0_Q8_0_OK


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q4_from_signed(vals)


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q8_signed(vals)


def explicit_parity_composition(
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
) -> tuple[int, tuple[int, int, int, int, int]]:
    staged_count = [0]
    staged_required_in_blocks = [0]
    staged_required_out_cells = [0]
    staged_required_in_bytes = [0]
    staged_required_out_bytes = [0]

    status = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only(
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
        staged_count,
        staged_required_in_blocks,
        staged_required_out_cells,
        staged_required_in_bytes,
        staged_required_out_bytes,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status, (0, 0, 0, 0, 0)

    scratch_out = [0] * pair_count if pair_count > 0 else [0]
    status = q4_0_q8_0_dot_q32_checked_no_partial_array(
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
        scratch_out,
        pair_count,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status, (0, 0, 0, 0, 0)

    recomputed_count = [0]
    recomputed_required_in_blocks = [0]
    recomputed_required_out_cells = [0]
    recomputed_required_in_bytes = [0]
    recomputed_required_out_bytes = [0]

    status = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only(
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
        recomputed_count,
        recomputed_required_in_blocks,
        recomputed_required_out_cells,
        recomputed_required_in_bytes,
        recomputed_required_out_bytes,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status, (0, 0, 0, 0, 0)

    if (
        staged_count[0] != recomputed_count[0]
        or staged_required_in_blocks[0] != recomputed_required_in_blocks[0]
        or staged_required_out_cells[0] != recomputed_required_out_cells[0]
        or staged_required_in_bytes[0] != recomputed_required_in_bytes[0]
        or staged_required_out_bytes[0] != recomputed_required_out_bytes[0]
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN, (0, 0, 0, 0, 0)

    return ref.Q4_0_Q8_0_OK, (
        staged_count[0],
        staged_required_in_blocks[0],
        staged_required_out_cells[0],
        staged_required_in_bytes[0],
        staged_required_out_bytes[0],
    )


def test_source_contains_iq936_helper() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnlyPreflightOnlyParity("
    assert sig in source
    body = source.split(sig, 1)[1].split(
        "I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly(", 1
    )[0]

    assert "if (out_count == (I64 *)lhs_q4" in body
    assert "scratch_out_q32 = MAlloc(scratch_out_bytes);" in body
    assert "pair_count > 0 ? scratch_out_q32 : out_dot_q32" in body
    assert "if (scratch_out_q32)" in body
    assert "Free(scratch_out_q32);" in body


def test_known_vector_success_and_no_write() -> None:
    rng = random.Random(93601)
    lhs_blocks = [make_q4_block(rng) for _ in range(14)]
    rhs_blocks = [make_q8_block(rng) for _ in range(14)]

    out_dots = [123456, -222222]
    before_dots = out_dots[:]
    out_count = [11]
    out_in_blocks = [22]
    out_out_cells = [33]
    out_in_bytes = [44]
    out_out_bytes = [55]

    status = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only_parity(
        lhs_blocks,
        len(lhs_blocks),
        3,
        1,
        rhs_blocks,
        len(rhs_blocks),
        3,
        1,
        2,
        2,
        out_dots,
        len(out_dots),
        out_count,
        out_in_blocks,
        out_out_cells,
        out_in_bytes,
        out_out_bytes,
    )
    assert status == ref.Q4_0_Q8_0_OK
    assert out_dots == before_dots
    assert out_count == [2]
    assert out_in_blocks == [5]
    assert out_out_cells == [2]
    assert out_in_bytes == [5 * 18 + 5 * 34]
    assert out_out_bytes == [16]


def test_alias_guard_and_no_publish_on_fail() -> None:
    rng = random.Random(93602)
    lhs_blocks = [make_q4_block(rng) for _ in range(8)]
    rhs_blocks = [make_q8_block(rng) for _ in range(8)]
    out_dots = [0, 0]

    shared = [9]
    out_cells = [10]
    out_in_bytes = [11]
    out_out_bytes = [12]

    status = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only_parity(
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
        out_cells,
        out_in_bytes,
        out_out_bytes,
    )
    assert status == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert shared == [9]
    assert out_cells == [10]
    assert out_in_bytes == [11]
    assert out_out_bytes == [12]


def test_randomized_parity_adversarial_vectors() -> None:
    rng = random.Random(20260421_936)

    for _ in range(2800):
        pair_count = rng.randint(0, 8)
        blocks_per_dot = rng.randint(0, 6)
        lhs_pair_stride = rng.randint(1, 6)
        rhs_pair_stride = rng.randint(1, 6)
        lhs_block_stride = rng.randint(1, 4)
        rhs_block_stride = rng.randint(1, 4)

        lhs_required = 1
        rhs_required = 1
        if pair_count > 0 and blocks_per_dot > 0:
            lhs_required = (
                (pair_count - 1) * lhs_pair_stride
                + (blocks_per_dot - 1) * lhs_block_stride
                + 1
            )
            rhs_required = (
                (pair_count - 1) * rhs_pair_stride
                + (blocks_per_dot - 1) * rhs_block_stride
                + 1
            )

        lhs_capacity = lhs_required + rng.randint(0, 2)
        rhs_capacity = rhs_required + rng.randint(0, 2)

        if rng.random() < 0.20 and pair_count > 0 and blocks_per_dot > 0:
            lhs_capacity = max(0, lhs_required - rng.randint(1, min(3, lhs_required)))
        if rng.random() < 0.20 and pair_count > 0 and blocks_per_dot > 0:
            rhs_capacity = max(0, rhs_required - rng.randint(1, min(3, rhs_required)))

        lhs_blocks = [make_q4_block(rng) for _ in range(lhs_capacity)]
        rhs_blocks = [make_q8_block(rng) for _ in range(rhs_capacity)]

        out_capacity = max(0, pair_count + rng.randint(0, 2))
        if rng.random() < 0.15 and pair_count > 0:
            out_capacity = pair_count - 1

        out_dots = [rng.randint(-1_000_000, 1_000_000) for _ in range(max(out_capacity, 1))]
        before_dots = out_dots[:]

        out_count = [rng.randint(-1000, 1000)]
        out_in_blocks = [rng.randint(-1000, 1000)]
        out_out_cells = [rng.randint(-1000, 1000)]
        out_in_bytes = [rng.randint(-1000, 1000)]
        out_out_bytes = [rng.randint(-1000, 1000)]
        before_diag = (
            out_count[0],
            out_in_blocks[0],
            out_out_cells[0],
            out_in_bytes[0],
            out_out_bytes[0],
        )

        status_impl = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only_preflight_only_parity(
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
            out_count,
            out_in_blocks,
            out_out_cells,
            out_in_bytes,
            out_out_bytes,
        )

        status_ref, tup_ref = explicit_parity_composition(
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
        )

        assert status_impl == status_ref

        if status_impl == ref.Q4_0_Q8_0_OK:
            assert (
                out_count[0],
                out_in_blocks[0],
                out_out_cells[0],
                out_in_bytes[0],
                out_out_bytes[0],
            ) == tup_ref
            assert out_dots == before_dots
        else:
            assert (
                out_count[0],
                out_in_blocks[0],
                out_out_cells[0],
                out_in_bytes[0],
                out_out_bytes[0],
            ) == before_diag
            assert out_dots == before_dots


if __name__ == "__main__":
    test_source_contains_iq936_helper()
    test_known_vector_success_and_no_write()
    test_alias_guard_and_no_publish_on_fail()
    test_randomized_parity_adversarial_vectors()
    print("ok")
