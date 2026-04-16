#!/usr/bin/env python3
"""GGML-bounded parity checks for mixed Q4_0 x Q8_0 dot helpers."""

from __future__ import annotations

import math
import random
import struct
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import test_q4_0_q8_0_dot_kernel as ref


Q16_ONE = 1 << 16
Q32_ONE = 1 << 32


def half_to_float(fp16_bits: int) -> float:
    return float(struct.unpack("<e", struct.pack("<H", fp16_bits))[0])


def q4_block_qdot_q0(lhs_q4: bytes, rhs_q8: bytes) -> int:
    lhs_signed = ref.unpack_q4_signed(lhs_q4)
    rhs_signed = ref.unpack_q8_signed(rhs_q8)
    return sum(a * b for a, b in zip(lhs_signed, rhs_signed))


def ggml_float_block_q32(lhs_scale_fp16: int, lhs_q4: bytes, rhs_scale_fp16: int, rhs_q8: bytes) -> tuple[int, int]:
    qdot_q0 = q4_block_qdot_q0(lhs_q4, rhs_q8)
    lhs_scale = half_to_float(lhs_scale_fp16)
    rhs_scale = half_to_float(rhs_scale_fp16)
    return qdot_q0, int(round(lhs_scale * rhs_scale * qdot_q0 * Q32_ONE))


def ggml_scale_rounding_bound_q32(lhs_scale_fp16: int, rhs_scale_fp16: int, qdot_q0: int) -> int:
    lhs_scale = half_to_float(lhs_scale_fp16)
    rhs_scale = half_to_float(rhs_scale_fp16)

    # HolyC path rounds each fp16 scale to Q16 once before multiplying scales.
    # Q16 rounding error per scale is <= 0.5 / 2^16.
    eps = 0.5 / Q16_ONE
    qdot_abs = abs(qdot_q0)

    scale_err = qdot_abs * (abs(lhs_scale) * eps + abs(rhs_scale) * eps + eps * eps)
    q32_err = scale_err * Q32_ONE

    # Add one output-rounding ulp and a tiny integer slack.
    return int(math.ceil(q32_err)) + 2


def ggml_scale_rounding_bound_q16(lhs_scale_fp16: int, rhs_scale_fp16: int, qdot_q0: int) -> int:
    lhs_scale = half_to_float(lhs_scale_fp16)
    rhs_scale = half_to_float(rhs_scale_fp16)

    eps = 0.5 / Q16_ONE
    qdot_abs = abs(qdot_q0)

    scale_err = qdot_abs * (abs(lhs_scale) * eps + abs(rhs_scale) * eps + eps * eps)
    q16_err = scale_err * Q16_ONE

    # +1 for Q32->Q16 half-up rounding and +1 slack for integer guard margin.
    return int(math.ceil(q16_err)) + 2


def q32_to_q16_round_half_up_signed(value_q32: int) -> int:
    return ref.round_shift_right_signed(value_q32, 16)


def test_mixed_block_matches_ggml_with_scale_rounding_bounds() -> None:
    rng = random.Random(20260416)

    for _ in range(1200):
        lhs_scale_fp16 = ref.half_bits(rng.uniform(-6.0, 6.0))
        rhs_scale_fp16 = ref.half_bits(rng.uniform(-6.0, 6.0))

        lhs_q4 = ref.pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
        rhs_q8 = ref.pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])

        err, got_q32 = ref.dot_product_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
        assert err == ref.Q4_0_Q8_0_OK

        qdot_q0, ggml_q32 = ggml_float_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
        bound_q32 = ggml_scale_rounding_bound_q32(lhs_scale_fp16, rhs_scale_fp16, qdot_q0)
        assert abs(got_q32 - ggml_q32) <= bound_q32


def test_multiblock_q32_sum_stays_within_bound_envelope() -> None:
    rng = random.Random(88442211)

    for block_count in (1, 2, 3, 7, 13):
        for _ in range(120):
            lhs_blocks = []
            rhs_blocks = []
            ggml_sum_q32 = 0
            bound_sum_q32 = 0

            for _block in range(block_count):
                lhs_scale_fp16 = ref.half_bits(rng.uniform(-4.0, 4.0))
                rhs_scale_fp16 = ref.half_bits(rng.uniform(-4.0, 4.0))
                lhs_q4 = ref.pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
                rhs_q8 = ref.pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])

                lhs_blocks.append((lhs_scale_fp16, lhs_q4))
                rhs_blocks.append((rhs_scale_fp16, rhs_q8))

                qdot_q0, ggml_q32 = ggml_float_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
                ggml_sum_q32 += ggml_q32
                bound_sum_q32 += ggml_scale_rounding_bound_q32(lhs_scale_fp16, rhs_scale_fp16, qdot_q0)

            err, got_sum_q32 = ref.dot_product_blocks_q32(lhs_blocks, rhs_blocks)
            assert err == ref.Q4_0_Q8_0_OK
            assert abs(got_sum_q32 - ggml_sum_q32) <= bound_sum_q32


def test_q16_accumulate_matches_blockwise_q32_rounding() -> None:
    rng = random.Random(517733)

    for block_count in (1, 5, 9):
        for _ in range(200):
            lhs_blocks = []
            rhs_blocks = []

            for _block in range(block_count):
                lhs_scale_fp16 = ref.half_bits(rng.uniform(-3.0, 3.0))
                rhs_scale_fp16 = ref.half_bits(rng.uniform(-3.0, 3.0))
                lhs_q4 = ref.pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
                rhs_q8 = ref.pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])
                lhs_blocks.append((lhs_scale_fp16, lhs_q4))
                rhs_blocks.append((rhs_scale_fp16, rhs_q8))

            seed_accum_q16 = rng.randrange(-(1 << 27), 1 << 27)

            err, got_accum_q16 = ref.dot_product_blocks_q16_accumulate_checked(lhs_blocks, rhs_blocks, seed_accum_q16)
            assert err == ref.Q4_0_Q8_0_OK

            expected_accum_q16 = seed_accum_q16
            bound_accum_q16 = 0
            for (lhs_scale_fp16, lhs_q4), (rhs_scale_fp16, rhs_q8) in zip(lhs_blocks, rhs_blocks):
                qdot_q0, ggml_q32 = ggml_float_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
                expected_accum_q16 += q32_to_q16_round_half_up_signed(ggml_q32)
                bound_accum_q16 += ggml_scale_rounding_bound_q16(lhs_scale_fp16, rhs_scale_fp16, qdot_q0)

            assert abs(got_accum_q16 - expected_accum_q16) <= bound_accum_q16


def main() -> None:
    test_mixed_block_matches_ggml_with_scale_rounding_bounds()
    test_multiblock_q32_sum_stays_within_bound_envelope()
    test_q16_accumulate_matches_blockwise_q32_rounding()
    print("q4_0_q8_0_dot_reference_checks=ok")


if __name__ == "__main__":
    main()
