#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly (IQ-931)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref
from test_q4_0_q8_0_dot_q32_checked_no_partial_array import (
    q4_0_q8_0_dot_q32_checked_no_partial_array,
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

    if (
        out_pair_count is None
        or out_required_lhs_blocks is None
        or out_required_rhs_blocks is None
        or out_required_out_cells is None
        or out_required_lhs_bytes is None
        or out_required_rhs_bytes is None
        or out_required_out_bytes is None
    ):
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if (
        out_pair_count is out_required_lhs_blocks
        or out_pair_count is out_required_rhs_blocks
        or out_pair_count is out_required_out_cells
        or out_pair_count is out_required_lhs_bytes
        or out_pair_count is out_required_rhs_bytes
        or out_pair_count is out_required_out_bytes
        or out_required_lhs_blocks is out_required_rhs_blocks
        or out_required_lhs_blocks is out_required_out_cells
        or out_required_lhs_blocks is out_required_lhs_bytes
        or out_required_lhs_blocks is out_required_rhs_bytes
        or out_required_lhs_blocks is out_required_out_bytes
        or out_required_rhs_blocks is out_required_out_cells
        or out_required_rhs_blocks is out_required_lhs_bytes
        or out_required_rhs_blocks is out_required_rhs_bytes
        or out_required_rhs_blocks is out_required_out_bytes
        or out_required_out_cells is out_required_lhs_bytes
        or out_required_out_cells is out_required_rhs_bytes
        or out_required_out_cells is out_required_out_bytes
        or out_required_lhs_bytes is out_required_rhs_bytes
        or out_required_lhs_bytes is out_required_out_bytes
        or out_required_rhs_bytes is out_required_out_bytes
    ):
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if (
        out_pair_count is lhs_blocks
        or out_pair_count is rhs_blocks
        or out_pair_count is out_dot_q32
        or out_required_lhs_blocks is lhs_blocks
        or out_required_lhs_blocks is rhs_blocks
        or out_required_lhs_blocks is out_dot_q32
        or out_required_rhs_blocks is lhs_blocks
        or out_required_rhs_blocks is rhs_blocks
        or out_required_rhs_blocks is out_dot_q32
        or out_required_out_cells is lhs_blocks
        or out_required_out_cells is rhs_blocks
        or out_required_out_cells is out_dot_q32
        or out_required_lhs_bytes is lhs_blocks
        or out_required_lhs_bytes is rhs_blocks
        or out_required_lhs_bytes is out_dot_q32
        or out_required_rhs_bytes is lhs_blocks
        or out_required_rhs_bytes is rhs_blocks
        or out_required_rhs_bytes is out_dot_q32
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
        out_dot_q32,
        out_dot_capacity,
    )
    if status != ref.Q4_0_Q8_0_OK:
        return status

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


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q4_from_signed(vals)


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q8_signed(vals)


def expected_diagnostics(
    lhs_block_capacity: int,
    lhs_pair_stride_blocks: int,
    lhs_block_stride: int,
    rhs_block_capacity: int,
    rhs_pair_stride_blocks: int,
    rhs_block_stride: int,
    pair_count: int,
    blocks_per_dot: int,
) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
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

    ok, required_lhs_blocks = _try_add_i64(lhs_last_pair_base, lhs_last_offset)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, required_rhs_blocks = _try_add_i64(rhs_last_pair_base, rhs_last_offset)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    ok, required_lhs_blocks = _try_add_i64(required_lhs_blocks, 1)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, required_rhs_blocks = _try_add_i64(required_rhs_blocks, 1)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    if required_lhs_blocks > lhs_block_capacity or required_rhs_blocks > rhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN, (0, 0, 0, 0, 0, 0, 0)

    ok, required_lhs_bytes = _try_mul_i64_nonneg(required_lhs_blocks, Q4_0_BLOCK_BYTES)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)
    ok, required_rhs_bytes = _try_mul_i64_nonneg(required_rhs_blocks, Q8_0_BLOCK_BYTES)
    if not ok:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW, (0, 0, 0, 0, 0, 0, 0)

    return ref.Q4_0_Q8_0_OK, (
        pair_count,
        required_lhs_blocks,
        required_rhs_blocks,
        pair_count,
        required_lhs_bytes,
        required_rhs_bytes,
        required_out_bytes,
    )


def test_source_contains_iq931_function() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialArrayCommitOnly("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = Q4_0Q8_0DotQ32CheckedNoPartialArray(" in body
    assert "staged_pair_count = pair_count;" in body
    assert "if (!pair_count || !blocks_per_dot)" in body
    assert "if (!Q4_0Q8_0TryMulI64NonNeg(staged_required_lhs_blocks," in body
    assert "*out_required_out_bytes = staged_required_out_bytes;" in body


def test_null_and_alias_guards() -> None:
    rng = random.Random(931)
    lhs_blocks = [make_q4_block(rng) for _ in range(8)]
    rhs_blocks = [make_q8_block(rng) for _ in range(8)]
    out_dots = [0, 0]

    out_pair_count = [11]
    out_lhs_blocks = [22]
    out_rhs_blocks = [33]
    out_out_cells = [44]
    out_lhs_bytes = [55]
    out_rhs_bytes = [66]
    out_out_bytes = [77]

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

    shared = [123]
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


def test_preflight_error_preserves_diagnostics_and_outputs() -> None:
    rng = random.Random(202604210931)
    lhs_blocks = [make_q4_block(rng) for _ in range(10)]
    rhs_blocks = [make_q8_block(rng) for _ in range(10)]

    out_dots = [111111, -222222, 333333]
    before_dots = out_dots[:]

    out_pair_count = [9001]
    out_lhs_blocks = [9002]
    out_rhs_blocks = [9003]
    out_out_cells = [9004]
    out_lhs_bytes = [9005]
    out_rhs_bytes = [9006]
    out_out_bytes = [9007]
    before_diag = (
        out_pair_count[0],
        out_lhs_blocks[0],
        out_rhs_blocks[0],
        out_out_cells[0],
        out_lhs_bytes[0],
        out_rhs_bytes[0],
        out_out_bytes[0],
    )

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

    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out_dots == before_dots
    assert (
        out_pair_count[0],
        out_lhs_blocks[0],
        out_rhs_blocks[0],
        out_out_cells[0],
        out_lhs_bytes[0],
        out_rhs_bytes[0],
        out_out_bytes[0],
    ) == before_diag


def test_known_vector_and_random_parity() -> None:
    lhs_blocks = [
        (ref.half_bits(1.0), ref.pack_q4_from_signed([((i % 16) - 8) for i in range(32)])),
        (ref.half_bits(-0.5), ref.pack_q4_from_signed([7 - (i % 16) for i in range(32)])),
        (ref.half_bits(0.25), ref.pack_q4_from_signed([((i % 8) - 4) for i in range(32)])),
        (ref.half_bits(1.5), ref.pack_q4_from_signed([3 - (i % 8) for i in range(32)])),
    ]
    rhs_blocks = [
        (ref.half_bits(0.75), ref.pack_q8_signed([((i % 11) - 5) * 3 for i in range(32)])),
        (ref.half_bits(-1.25), ref.pack_q8_signed([7 - (i % 13) for i in range(32)])),
        (ref.half_bits(2.0), ref.pack_q8_signed([((i % 9) - 4) * 4 for i in range(32)])),
        (ref.half_bits(-0.25), ref.pack_q8_signed([((i % 7) - 3) * 5 for i in range(32)])),
    ]

    out_dots = [0, 0]
    out_pair_count = [0]
    out_lhs_blocks = [0]
    out_rhs_blocks = [0]
    out_out_cells = [0]
    out_lhs_bytes = [0]
    out_rhs_bytes = [0]
    out_out_bytes = [0]

    err = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
        lhs_blocks,
        len(lhs_blocks),
        2,
        1,
        rhs_blocks,
        len(rhs_blocks),
        2,
        1,
        2,
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

    expected_err, expected_diag = expected_diagnostics(
        len(lhs_blocks),
        2,
        1,
        len(rhs_blocks),
        2,
        1,
        2,
        2,
    )
    assert expected_err == ref.Q4_0_Q8_0_OK
    assert (
        out_pair_count[0],
        out_lhs_blocks[0],
        out_rhs_blocks[0],
        out_out_cells[0],
        out_lhs_bytes[0],
        out_rhs_bytes[0],
        out_out_bytes[0],
    ) == expected_diag

    rng = random.Random(931931)
    for _ in range(80):
        pair_count = rng.randrange(0, 6)
        blocks_per_dot = rng.randrange(0, 5)
        lhs_pair_stride = rng.randrange(1, 6)
        rhs_pair_stride = rng.randrange(1, 6)
        lhs_block_stride = rng.randrange(1, 4)
        rhs_block_stride = rng.randrange(1, 4)

        if pair_count == 0 or blocks_per_dot == 0:
            lhs_need = 0
            rhs_need = 0
        else:
            lhs_need = (pair_count - 1) * lhs_pair_stride + (blocks_per_dot - 1) * lhs_block_stride + 1
            rhs_need = (pair_count - 1) * rhs_pair_stride + (blocks_per_dot - 1) * rhs_block_stride + 1

        lhs_cap = lhs_need + rng.randrange(0, 3)
        rhs_cap = rhs_need + rng.randrange(0, 3)
        out_cap = max(pair_count, 1)

        lhs = [make_q4_block(rng) for _ in range(lhs_cap)]
        rhs = [make_q8_block(rng) for _ in range(rhs_cap)]
        out = [0x1111] * out_cap

        out_pair_count = [0x2001]
        out_lhs_blocks = [0x2002]
        out_rhs_blocks = [0x2003]
        out_out_cells = [0x2004]
        out_lhs_bytes = [0x2005]
        out_rhs_bytes = [0x2006]
        out_out_bytes = [0x2007]

        err = q4_0_q8_0_dot_q32_checked_no_partial_array_commit_only(
            lhs,
            lhs_cap,
            lhs_pair_stride,
            lhs_block_stride,
            rhs,
            rhs_cap,
            rhs_pair_stride,
            rhs_block_stride,
            pair_count,
            blocks_per_dot,
            out,
            out_cap,
            out_pair_count,
            out_lhs_blocks,
            out_rhs_blocks,
            out_out_cells,
            out_lhs_bytes,
            out_rhs_bytes,
            out_out_bytes,
        )

        expected_err, expected_diag = expected_diagnostics(
            lhs_cap,
            lhs_pair_stride,
            lhs_block_stride,
            rhs_cap,
            rhs_pair_stride,
            rhs_block_stride,
            pair_count,
            blocks_per_dot,
        )
        assert err == expected_err

        if err == ref.Q4_0_Q8_0_OK:
            assert (
                out_pair_count[0],
                out_lhs_blocks[0],
                out_rhs_blocks[0],
                out_out_cells[0],
                out_lhs_bytes[0],
                out_rhs_bytes[0],
                out_out_bytes[0],
            ) == expected_diag


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
