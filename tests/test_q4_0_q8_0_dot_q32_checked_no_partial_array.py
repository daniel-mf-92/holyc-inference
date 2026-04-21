#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartialArray (IQ-924)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref


def q4_0_q8_0_dot_q32_checked_no_partial_array(
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
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot_q32 is None:
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

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

    if pair_count == 0:
        return ref.Q4_0_Q8_0_OK

    if blocks_per_dot == 0:
        for idx in range(pair_count):
            out_dot_q32[idx] = 0
        return ref.Q4_0_Q8_0_OK

    lhs_last_offset = (blocks_per_dot - 1) * lhs_block_stride
    rhs_last_offset = (blocks_per_dot - 1) * rhs_block_stride
    lhs_last_pair_base = (pair_count - 1) * lhs_pair_stride_blocks
    rhs_last_pair_base = (pair_count - 1) * rhs_pair_stride_blocks

    lhs_required_blocks = lhs_last_pair_base + lhs_last_offset + 1
    rhs_required_blocks = rhs_last_pair_base + rhs_last_offset + 1

    if lhs_required_blocks > ref.Q4_0_Q8_0_I64_MAX:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW
    if rhs_required_blocks > ref.Q4_0_Q8_0_I64_MAX:
        return ref.Q4_0_Q8_0_ERR_OVERFLOW

    if lhs_required_blocks > lhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if rhs_required_blocks > rhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    staged: list[int] = [0] * pair_count
    for pair_idx in range(pair_count):
        lhs_pair_base = pair_idx * lhs_pair_stride_blocks
        rhs_pair_base = pair_idx * rhs_pair_stride_blocks

        lhs_slice = []
        rhs_slice = []
        for block_idx in range(blocks_per_dot):
            lhs_index = lhs_pair_base + block_idx * lhs_block_stride
            rhs_index = rhs_pair_base + block_idx * rhs_block_stride
            lhs_slice.append(lhs_blocks[lhs_index])
            rhs_slice.append(rhs_blocks[rhs_index])

        err, dot_q32 = ref.dot_product_blocks_q32(lhs_slice, rhs_slice)
        if err != ref.Q4_0_Q8_0_OK:
            return err
        staged[pair_idx] = dot_q32

    for pair_idx in range(pair_count):
        out_dot_q32[pair_idx] = staged[pair_idx]

    return ref.Q4_0_Q8_0_OK


def explicit_expected_tuple(
    lhs_blocks,
    lhs_pair_stride_blocks: int,
    lhs_block_stride: int,
    rhs_blocks,
    rhs_pair_stride_blocks: int,
    rhs_block_stride: int,
    pair_count: int,
    blocks_per_dot: int,
) -> tuple[int, list[int]]:
    out: list[int] = []
    for pair_idx in range(pair_count):
        lhs_pair_base = pair_idx * lhs_pair_stride_blocks
        rhs_pair_base = pair_idx * rhs_pair_stride_blocks

        lhs_slice = []
        rhs_slice = []
        for block_idx in range(blocks_per_dot):
            lhs_slice.append(lhs_blocks[lhs_pair_base + block_idx * lhs_block_stride])
            rhs_slice.append(rhs_blocks[rhs_pair_base + block_idx * rhs_block_stride])

        err, dot_q32 = ref.dot_product_blocks_q32(lhs_slice, rhs_slice)
        if err != ref.Q4_0_Q8_0_OK:
            return err, []
        out.append(dot_q32)

    return ref.Q4_0_Q8_0_OK, out


def make_q4_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-8, 8) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q4_from_signed(vals)


def make_q8_block(rng: random.Random) -> tuple[int, bytes]:
    scale = rng.uniform(-2.5, 2.5)
    vals = [rng.randrange(-128, 128) for _ in range(32)]
    return ref.half_bits(scale), ref.pack_q8_signed(vals)


def test_source_contains_iq924_function() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")
    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartialArray("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "Q4_0Q8_0DotQ32CheckedNoPartialArrayDotOne(" in body
    assert "if (!pair_count)" in body
    assert "if (!blocks_per_dot)" in body
    assert "// Preflight pass: compute every pair and discard staged results." in body
    assert "// Commit pass: repeat computations and publish only after full preflight." in body
    assert "out_dot_q32[pair_idx] = staged_dot_q32;" in body


def test_null_and_shape_guards() -> None:
    rng = random.Random(924)
    lhs_blocks = [make_q4_block(rng) for _ in range(8)]
    rhs_blocks = [make_q8_block(rng) for _ in range(8)]

    out = [777, 888]
    assert (
        q4_0_q8_0_dot_q32_checked_no_partial_array(
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
            out,
            len(out),
        )
        == ref.Q4_0_Q8_0_ERR_NULL_PTR
    )

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial_array(
            lhs_blocks,
            len(lhs_blocks),
            1,
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            1,
            3,
            1,
            out,
            len(out),
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_q8_0_dot_q32_checked_no_partial_array(
            lhs_blocks,
            len(lhs_blocks),
            0,
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            1,
            2,
            1,
            out,
            len(out),
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )


def test_preflight_failure_preserves_output() -> None:
    rng = random.Random(20260421)
    lhs_blocks = [make_q4_block(rng) for _ in range(10)]
    rhs_blocks = [make_q8_block(rng) for _ in range(10)]

    out = [1234567, -7654321, 111]
    before = out[:]

    err = q4_0_q8_0_dot_q32_checked_no_partial_array(
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
        out,
        len(out),
    )

    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out == before


def test_known_vector_and_strided_random_parity() -> None:
    rng = random.Random(818181)

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

    out = [0, 0]
    err = q4_0_q8_0_dot_q32_checked_no_partial_array(
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
        out,
        len(out),
    )
    assert err == ref.Q4_0_Q8_0_OK

    expected_err, expected = explicit_expected_tuple(
        lhs_blocks,
        2,
        1,
        rhs_blocks,
        2,
        1,
        2,
        2,
    )
    assert expected_err == ref.Q4_0_Q8_0_OK
    assert out == expected

    for _ in range(220):
        pair_count = rng.randint(1, 6)
        blocks_per_dot = rng.randint(1, 5)
        lhs_pair_stride = blocks_per_dot + rng.randint(0, 3)
        rhs_pair_stride = blocks_per_dot + rng.randint(0, 3)
        lhs_block_stride = rng.randint(1, 3)
        rhs_block_stride = rng.randint(1, 3)

        lhs_required = (pair_count - 1) * lhs_pair_stride + (blocks_per_dot - 1) * lhs_block_stride + 1
        rhs_required = (pair_count - 1) * rhs_pair_stride + (blocks_per_dot - 1) * rhs_block_stride + 1

        lhs_blocks = [make_q4_block(rng) for _ in range(lhs_required + rng.randint(0, 2))]
        rhs_blocks = [make_q8_block(rng) for _ in range(rhs_required + rng.randint(0, 2))]

        out = [999999] * pair_count
        err = q4_0_q8_0_dot_q32_checked_no_partial_array(
            lhs_blocks,
            len(lhs_blocks),
            lhs_pair_stride,
            lhs_block_stride,
            rhs_blocks,
            len(rhs_blocks),
            rhs_pair_stride,
            rhs_block_stride,
            pair_count,
            blocks_per_dot,
            out,
            len(out),
        )
        assert err == ref.Q4_0_Q8_0_OK

        expected_err, expected = explicit_expected_tuple(
            lhs_blocks,
            lhs_pair_stride,
            lhs_block_stride,
            rhs_blocks,
            rhs_pair_stride,
            rhs_block_stride,
            pair_count,
            blocks_per_dot,
        )
        assert expected_err == ref.Q4_0_Q8_0_OK
        assert out == expected


def main() -> None:
    test_source_contains_iq924_function()
    test_null_and_shape_guards()
    test_preflight_failure_preserves_output()
    test_known_vector_and_strided_random_parity()
    print("ok")


if __name__ == "__main__":
    main()
