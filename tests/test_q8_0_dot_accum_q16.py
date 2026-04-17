#!/usr/bin/env python3
"""Focused parity harness for Q8_0 Q16 blockwise accumulation semantics."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_q8_0_dot import (
    Q8_0_ERR_BAD_DST_LEN,
    Q8_0_ERR_OVERFLOW,
    Q8_0_I64_MAX,
    Q8_0_I64_MIN,
    Q8_0_OK,
    Q8_0_VALUES_PER_BLOCK,
    dot_product_block_q32,
    dot_product_blocks_q16_accumulate,
    dot_product_blocks_q16_accumulate_checked,
    dot_product_blocks_q32,
    dot_q32_to_q16,
    half_bits,
    pack_signed,
)


def make_block(rng: random.Random, *, scale: float | None = None) -> tuple[int, bytes]:
    if scale is None:
        scale = rng.uniform(-2.5, 2.5)
    return (
        half_bits(scale),
        pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]),
    )


def expected_blockwise_q16(
    lhs_blocks: list[tuple[int, bytes]],
    rhs_blocks: list[tuple[int, bytes]],
    initial_q16: int,
) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q8_0_ERR_BAD_DST_LEN, 0

    total_q16 = initial_q16
    for (lhs_scale, lhs_qs), (rhs_scale, rhs_qs) in zip(lhs_blocks, rhs_blocks):
        err, block_dot_q32 = dot_product_block_q32(lhs_scale, lhs_qs, rhs_scale, rhs_qs)
        if err != Q8_0_OK:
            return err, 0
        total_q16 += dot_q32_to_q16(block_dot_q32)

    return Q8_0_OK, total_q16


def test_seeded_blockwise_rounding_parity_randomized() -> None:
    rng = random.Random(20260417035)

    for _ in range(320):
        block_count = rng.randint(1, 10)
        initial_q16 = rng.randint(-(1 << 41), (1 << 41))

        lhs_blocks = [make_block(rng) for _ in range(block_count)]
        rhs_blocks = [make_block(rng) for _ in range(block_count)]

        err_expected, expected_q16 = expected_blockwise_q16(lhs_blocks, rhs_blocks, initial_q16)
        err_got, got_q16 = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, initial_q16)

        assert err_expected == Q8_0_OK
        assert err_got == Q8_0_OK
        assert got_q16 == expected_q16


def test_seeded_blockwise_rounding_not_global_rounding() -> None:
    # Each block contributes +0.5 Q16 unit before rounding:
    # Round per block => 0 + 0; round at end => 1.
    scale_fp16 = half_bits(181.0 / 65536.0)
    lhs_qs = pack_signed([1] + [0] * (Q8_0_VALUES_PER_BLOCK - 1))
    rhs_qs = pack_signed([1] + [0] * (Q8_0_VALUES_PER_BLOCK - 1))

    lhs_blocks = [(scale_fp16, lhs_qs), (scale_fp16, lhs_qs)]
    rhs_blocks = [(scale_fp16, rhs_qs), (scale_fp16, rhs_qs)]

    err, got_q16 = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)
    assert err == Q8_0_OK

    err, total_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
    assert err == Q8_0_OK
    global_round_q16 = dot_q32_to_q16(total_q32)

    assert got_q16 == 0
    assert global_round_q16 == 1
    assert got_q16 != global_round_q16


def test_checked_matches_unchecked_seeded_safe_ranges() -> None:
    rng = random.Random(20260417036)

    for _ in range(260):
        block_count = rng.randint(1, 9)
        initial_q16 = rng.randint(-(1 << 40), (1 << 40))

        lhs_blocks = [make_block(rng, scale=rng.uniform(-1.5, 1.5)) for _ in range(block_count)]
        rhs_blocks = [make_block(rng, scale=rng.uniform(-1.5, 1.5)) for _ in range(block_count)]

        err_unchecked, accum_unchecked = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, initial_q16)
        err_checked, accum_checked = dot_product_blocks_q16_accumulate_checked(lhs_blocks, rhs_blocks, initial_q16)

        assert err_unchecked == Q8_0_OK
        assert err_checked == Q8_0_OK
        assert accum_checked == accum_unchecked


def test_checked_overflow_edges_from_initial_accumulator() -> None:
    pos_block = (half_bits(1.0), pack_signed([127] * Q8_0_VALUES_PER_BLOCK))
    neg_block = (half_bits(-1.0), pack_signed([127] * Q8_0_VALUES_PER_BLOCK))

    err, block_q32_pos = dot_product_block_q32(pos_block[0], pos_block[1], pos_block[0], pos_block[1])
    assert err == Q8_0_OK
    block_q16_pos = dot_q32_to_q16(block_q32_pos)
    assert block_q16_pos > 0

    err, block_q32_neg = dot_product_block_q32(neg_block[0], neg_block[1], pos_block[0], pos_block[1])
    assert err == Q8_0_OK
    block_q16_neg = dot_q32_to_q16(block_q32_neg)
    assert block_q16_neg < 0

    err, _ = dot_product_blocks_q16_accumulate_checked([pos_block], [pos_block], Q8_0_I64_MAX - block_q16_pos + 1)
    assert err == Q8_0_ERR_OVERFLOW

    err, _ = dot_product_blocks_q16_accumulate_checked([neg_block], [pos_block], Q8_0_I64_MIN - block_q16_neg - 1)
    assert err == Q8_0_ERR_OVERFLOW


def test_bad_length_contracts() -> None:
    lhs_blocks = [make_block(random.Random(1))]
    rhs_blocks: list[tuple[int, bytes]] = []

    err, _ = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)
    assert err == Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_product_blocks_q16_accumulate_checked(lhs_blocks, rhs_blocks, 0)
    assert err == Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_seeded_blockwise_rounding_parity_randomized()
    test_seeded_blockwise_rounding_not_global_rounding()
    test_checked_matches_unchecked_seeded_safe_ranges()
    test_checked_overflow_edges_from_initial_accumulator()
    test_bad_length_contracts()
    print("q8_0_dot_accum_q16_checks=ok")


if __name__ == "__main__":
    run()
