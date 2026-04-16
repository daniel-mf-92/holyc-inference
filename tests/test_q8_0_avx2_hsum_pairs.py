#!/usr/bin/env python3
"""Reference checks for Q8_0 AVX2-style horizontal sum over 16 I32 pairs."""

from __future__ import annotations

import random

Q8_0_AVX2_PAIR_COUNT = 16

Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2
Q8_0_AVX2_ERR_OVERFLOW = 3

Q8_0_AVX2_I64_MAX = (1 << 63) - 1
Q8_0_AVX2_I64_MIN = -(1 << 63)


def q8_0_hsum_i32_pairs_avx2(pairs_i32):
    if pairs_i32 is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0
    if len(pairs_i32) < Q8_0_AVX2_PAIR_COUNT:
        return Q8_0_AVX2_ERR_BAD_LEN, 0

    running_sum = 0
    for pair_index in range(Q8_0_AVX2_PAIR_COUNT):
        term = int(pairs_i32[pair_index])
        if term > 0 and running_sum > Q8_0_AVX2_I64_MAX - term:
            return Q8_0_AVX2_ERR_OVERFLOW, 0
        if term < 0 and running_sum < Q8_0_AVX2_I64_MIN - term:
            return Q8_0_AVX2_ERR_OVERFLOW, 0
        running_sum += term
    return Q8_0_AVX2_OK, running_sum


def test_known_pattern_zero_sum() -> None:
    # Symmetric cancellation keeps strict lane-order deterministic.
    pairs = [
        -1_000_000,
        1_000_000,
        -10,
        10,
        -32768,
        32768,
        -123456,
        123456,
        -999,
        999,
        -1,
        1,
        -42,
        42,
        -7,
        7,
    ]
    err, got = q8_0_hsum_i32_pairs_avx2(pairs)
    assert err == Q8_0_AVX2_OK
    assert got == 0


def test_randomized_matches_python_sum() -> None:
    rng = random.Random(20260416)

    for _ in range(4000):
        pairs = [rng.randint(-(1 << 31), (1 << 31) - 1) for _ in range(16)]
        err, got = q8_0_hsum_i32_pairs_avx2(pairs)
        assert err == Q8_0_AVX2_OK
        assert got == sum(pairs)


def test_len_and_null_errors() -> None:
    err, _ = q8_0_hsum_i32_pairs_avx2(None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_hsum_i32_pairs_avx2([1] * 15)
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def test_overflow_guards_trigger() -> None:
    # Force signed I64 overflow at pair 16.
    near_max = [0] * 16
    near_max[0] = Q8_0_AVX2_I64_MAX
    near_max[1] = 1
    err, _ = q8_0_hsum_i32_pairs_avx2(near_max)
    assert err == Q8_0_AVX2_ERR_OVERFLOW

    near_min = [0] * 16
    near_min[0] = Q8_0_AVX2_I64_MIN
    near_min[1] = -1
    err, _ = q8_0_hsum_i32_pairs_avx2(near_min)
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def run() -> None:
    test_known_pattern_zero_sum()
    test_randomized_matches_python_sum()
    test_len_and_null_errors()
    test_overflow_guards_trigger()
    print("q8_0_avx2_hsum_pairs_reference_checks=ok")


if __name__ == "__main__":
    run()
