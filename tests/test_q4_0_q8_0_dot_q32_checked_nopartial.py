#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartial (IQ-989)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref

Q4_BLOCK_BYTES = 18
Q8_BLOCK_BYTES = 34
I64_BYTES = 8


def q4_0_q8_0_dot_q32_checked_no_partial(
    lhs_blocks,
    lhs_block_capacity: int,
    lhs_block_stride: int,
    rhs_blocks,
    rhs_block_capacity: int,
    rhs_block_stride: int,
    block_count: int,
    out_dot_q32,
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot_q32 is None:
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if lhs_block_capacity < 0 or rhs_block_capacity < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_block_stride <= 0 or rhs_block_stride <= 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if block_count == 0:
        out_dot_q32[0] = 0
        return ref.Q4_0_Q8_0_OK

    lhs_required_blocks = (block_count - 1) * lhs_block_stride + 1
    rhs_required_blocks = (block_count - 1) * rhs_block_stride + 1

    if lhs_required_blocks > ref.Q4_0_Q8_0_I64_MAX:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW
    if rhs_required_blocks > ref.Q4_0_Q8_0_I64_MAX:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW

    if lhs_required_blocks > lhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if rhs_required_blocks > rhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    lhs_start = 0
    lhs_end = lhs_required_blocks * Q4_BLOCK_BYTES
    rhs_start = 0
    rhs_end = rhs_required_blocks * Q8_BLOCK_BYTES
    out_start = out_dot_q32[1]
    out_end = out_start + I64_BYTES

    if lhs_start < out_end and out_start < lhs_end:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if rhs_start < out_end and out_start < rhs_end:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    lhs_slice = []
    rhs_slice = []
    for block_idx in range(block_count):
        lhs_slice.append(lhs_blocks[block_idx * lhs_block_stride])
        rhs_slice.append(rhs_blocks[block_idx * rhs_block_stride])

    err, staged = ref.dot_product_blocks_q32(lhs_slice, rhs_slice)
    if err != ref.Q4_0_Q8_0_OK:
        return err

    out_dot_q32[0] = staged
    return ref.Q4_0_Q8_0_OK


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-3.0, 3.0)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q4_from_signed(vals)


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-3.0, 3.0)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q8_signed(vals)


def expected_single_dot(lhs_blocks, lhs_stride: int, rhs_blocks, rhs_stride: int, block_count: int) -> int:
    lhs = [lhs_blocks[i * lhs_stride] for i in range(block_count)]
    rhs = [rhs_blocks[i * rhs_stride] for i in range(block_count)]
    err, dot_q32 = ref.dot_product_blocks_q32(lhs, rhs)
    assert err == ref.Q4_0_Q8_0_OK
    return dot_q32


def test_source_contains_iq989_function_and_alias_guards() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    assert "I32 Q4_0Q8_0DotQ32CheckedNoPartial(" in source
    assert "Bool Q4_0Q8_0ByteSpanEndChecked(" in source
    assert "Bool Q4_0Q8_0ByteSpansOverlap(" in source
    assert "if (Q4_0Q8_0ByteSpansOverlap(lhs_start, lhs_end, out_start, out_end))" in source
    assert "if (Q4_0Q8_0ByteSpansOverlap(rhs_start, rhs_end, out_start, out_end))" in source


def test_null_and_shape_guards() -> None:
    rng = random.Random(989)
    lhs = [make_q4_block(rng) for _ in range(6)]
    rhs = [make_q8_block(rng) for _ in range(6)]
    out = [123, 10_000]

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            None,
            len(lhs),
            1,
            rhs,
            len(rhs),
            1,
            1,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_NULL_PTR
    )

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            -1,
            1,
            rhs,
            len(rhs),
            1,
            1,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            len(lhs),
            0,
            rhs,
            len(rhs),
            1,
            1,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            len(lhs),
            1,
            rhs,
            len(rhs),
            1,
            -1,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )


def test_capacity_and_stride_bounds() -> None:
    rng = random.Random(9891)
    lhs = [make_q4_block(rng) for _ in range(16)]
    rhs = [make_q8_block(rng) for _ in range(16)]
    out = [11, 99_000]

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            8,
            2,
            rhs,
            len(rhs),
            1,
            5,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            len(lhs),
            1,
            rhs,
            3,
            2,
            2,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )


def test_alias_span_rejection() -> None:
    rng = random.Random(9892)
    lhs = [make_q4_block(rng) for _ in range(8)]
    rhs = [make_q8_block(rng) for _ in range(8)]

    out_inside_lhs = [55, 12]
    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            len(lhs),
            1,
            rhs,
            len(rhs),
            1,
            1,
            out_inside_lhs,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    out_inside_rhs = [66, 20]
    assert (
        q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            len(lhs),
            1,
            rhs,
            len(rhs),
            1,
            1,
            out_inside_rhs,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )


def test_zero_block_count_commits_zero() -> None:
    rng = random.Random(9893)
    lhs = [make_q4_block(rng) for _ in range(4)]
    rhs = [make_q8_block(rng) for _ in range(4)]
    out = [777, 999_999]

    status = q4_0_q8_0_dot_q32_checked_no_partial(
        lhs,
        len(lhs),
        1,
        rhs,
        len(rhs),
        1,
        0,
        out,
    )
    assert status == ref.Q4_0_Q8_0_OK
    assert out[0] == 0


def test_random_parity_against_explicit_reference() -> None:
    rng = random.Random(9894)

    for _ in range(250):
        lhs_stride = rng.choice((1, 2, 3))
        rhs_stride = rng.choice((1, 2, 3))
        block_count = rng.randint(1, 8)

        lhs_required = (block_count - 1) * lhs_stride + 1
        rhs_required = (block_count - 1) * rhs_stride + 1

        lhs_cap = lhs_required + rng.randint(0, 3)
        rhs_cap = rhs_required + rng.randint(0, 3)

        lhs = [make_q4_block(rng) for _ in range(lhs_cap)]
        rhs = [make_q8_block(rng) for _ in range(rhs_cap)]

        out = [-(1 << 50), 100_000]
        status = q4_0_q8_0_dot_q32_checked_no_partial(
            lhs,
            lhs_cap,
            lhs_stride,
            rhs,
            rhs_cap,
            rhs_stride,
            block_count,
            out,
        )
        assert status == ref.Q4_0_Q8_0_OK

        expected = expected_single_dot(lhs, lhs_stride, rhs, rhs_stride, block_count)
        assert out[0] == expected


def main() -> None:
    test_source_contains_iq989_function_and_alias_guards()
    test_null_and_shape_guards()
    test_capacity_and_stride_bounds()
    test_alias_span_rejection()
    test_zero_block_count_commits_zero()
    test_random_parity_against_explicit_reference()
    print("q4_0_q8_0_dot_q32_checked_nopartial=ok")


if __name__ == "__main__":
    main()
