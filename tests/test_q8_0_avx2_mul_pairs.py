#!/usr/bin/env python3
"""Reference checks for Q8_0 AVX2-style pairwise I16 lane multiply-add."""

from __future__ import annotations

import random

Q8_0_AVX2_VALUES_PER_BLOCK = 32
Q8_0_AVX2_PAIR_COUNT = 16

Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2


def q8_0_mul_i16_lanes_to_i32_pairs_avx2(lhs_i16, rhs_i16):
    if lhs_i16 is None or rhs_i16 is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []
    if len(lhs_i16) < Q8_0_AVX2_VALUES_PER_BLOCK:
        return Q8_0_AVX2_ERR_BAD_LEN, []
    if len(rhs_i16) < Q8_0_AVX2_VALUES_PER_BLOCK:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    out = [0] * Q8_0_AVX2_PAIR_COUNT
    for pair_index in range(Q8_0_AVX2_PAIR_COUNT):
        lane_base = pair_index * 2
        out[pair_index] = (
            int(lhs_i16[lane_base]) * int(rhs_i16[lane_base])
            + int(lhs_i16[lane_base + 1]) * int(rhs_i16[lane_base + 1])
        )
    return Q8_0_AVX2_OK, out


def test_known_pattern_signed_edges() -> None:
    lhs = [
        -32768,
        -30000,
        -16384,
        -1024,
        -512,
        -255,
        -128,
        -1,
        0,
        1,
        2,
        7,
        8,
        15,
        16,
        31,
        32,
        63,
        64,
        95,
        96,
        127,
        128,
        255,
        256,
        511,
        1024,
        4096,
        8192,
        16384,
        30000,
        32767,
    ]
    rhs = list(reversed(lhs))

    err, out = q8_0_mul_i16_lanes_to_i32_pairs_avx2(lhs, rhs)
    assert err == Q8_0_AVX2_OK

    expected = []
    for i in range(0, 32, 2):
        expected.append(lhs[i] * rhs[i] + lhs[i + 1] * rhs[i + 1])
    assert out == expected


def test_randomized_pairwise_contract() -> None:
    rng = random.Random(20260416)

    for _ in range(2000):
        lhs = [rng.randint(-32768, 32767) for _ in range(32)]
        rhs = [rng.randint(-32768, 32767) for _ in range(32)]

        err, out = q8_0_mul_i16_lanes_to_i32_pairs_avx2(lhs, rhs)
        assert err == Q8_0_AVX2_OK

        for pair_index in range(16):
            lane = pair_index * 2
            want = lhs[lane] * rhs[lane] + lhs[lane + 1] * rhs[lane + 1]
            assert out[pair_index] == want


def test_len_and_null_errors() -> None:
    err, _ = q8_0_mul_i16_lanes_to_i32_pairs_avx2(None, [0] * 32)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_mul_i16_lanes_to_i32_pairs_avx2([0] * 32, None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_mul_i16_lanes_to_i32_pairs_avx2([0] * 31, [0] * 32)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_mul_i16_lanes_to_i32_pairs_avx2([0] * 32, [0] * 31)
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def run() -> None:
    test_known_pattern_signed_edges()
    test_randomized_pairwise_contract()
    test_len_and_null_errors()
    print("q8_0_avx2_mul_pairs_reference_checks=ok")


if __name__ == "__main__":
    run()
