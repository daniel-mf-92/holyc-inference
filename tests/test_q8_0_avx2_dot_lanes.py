#!/usr/bin/env python3
"""Reference checks for Q8_0 AVX2-style composed 32-lane I16 dot helper."""

from __future__ import annotations

import random

Q8_0_AVX2_VALUES_PER_BLOCK = 32
Q8_0_AVX2_PAIR_COUNT = 16

Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2
Q8_0_AVX2_ERR_OVERFLOW = 3

Q8_0_AVX2_I64_MAX = (1 << 63) - 1
Q8_0_AVX2_I64_MIN = -(1 << 63)


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


def q8_0_dot_i16_lanes_avx2(lhs_i16, rhs_i16):
    if lhs_i16 is None or rhs_i16 is None:
        return Q8_0_AVX2_ERR_NULL_PTR, 0

    err_pairs, pairs = q8_0_mul_i16_lanes_to_i32_pairs_avx2(lhs_i16, rhs_i16)
    if err_pairs != Q8_0_AVX2_OK:
        return err_pairs, 0

    return q8_0_hsum_i32_pairs_avx2(pairs)


def test_known_signed_edge_contract() -> None:
    lhs = [
        -32768,
        -30000,
        -20000,
        -16384,
        -8192,
        -4096,
        -2048,
        -1024,
        -512,
        -255,
        -128,
        -64,
        -1,
        0,
        1,
        2,
        3,
        7,
        15,
        31,
        63,
        127,
        255,
        511,
        1023,
        2047,
        4095,
        8191,
        12000,
        16384,
        25000,
        32767,
    ]
    rhs = list(reversed(lhs))

    err, got = q8_0_dot_i16_lanes_avx2(lhs, rhs)
    assert err == Q8_0_AVX2_OK
    assert got == sum(int(a) * int(b) for a, b in zip(lhs, rhs))


def test_randomized_matches_scalar_reference() -> None:
    rng = random.Random(20260416)

    for _ in range(4000):
        lhs = [rng.randint(-32768, 32767) for _ in range(32)]
        rhs = [rng.randint(-32768, 32767) for _ in range(32)]

        err, got = q8_0_dot_i16_lanes_avx2(lhs, rhs)
        assert err == Q8_0_AVX2_OK
        assert got == sum(int(a) * int(b) for a, b in zip(lhs, rhs))


def test_lane_order_invariant_pair_sensitivity() -> None:
    lhs = [0] * 32
    rhs = [0] * 32

    lhs[6] = 1234
    rhs[6] = -321

    err, got = q8_0_dot_i16_lanes_avx2(lhs, rhs)
    assert err == Q8_0_AVX2_OK
    assert got == -396114

    # Move same magnitudes to adjacent lane: result should be identical scalar dot.
    lhs2 = [0] * 32
    rhs2 = [0] * 32
    lhs2[7] = 1234
    rhs2[7] = -321

    err, got2 = q8_0_dot_i16_lanes_avx2(lhs2, rhs2)
    assert err == Q8_0_AVX2_OK
    assert got2 == -396114


def test_pair_grouping_contract_matches_mul_then_hsum() -> None:
    # Adversarial signed edge pattern keeps every pair distinct so lane grouping
    # bugs show up immediately (wrong lane stride/order changes pair sequence).
    lhs = [
        -32768,
        32767,
        -30000,
        29999,
        -20000,
        19999,
        -16384,
        16383,
        -8192,
        8191,
        -4096,
        4095,
        -2048,
        2047,
        -1024,
        1023,
        -512,
        511,
        -255,
        254,
        -128,
        127,
        -64,
        63,
        -32,
        31,
        -16,
        15,
        -8,
        7,
        -2,
        1,
    ]
    rhs = [
        1,
        -2,
        7,
        -8,
        15,
        -16,
        31,
        -32,
        63,
        -64,
        127,
        -128,
        254,
        -255,
        511,
        -512,
        1023,
        -1024,
        2047,
        -2048,
        4095,
        -4096,
        8191,
        -8192,
        16383,
        -16384,
        19999,
        -20000,
        29999,
        -30000,
        32767,
        -32768,
    ]

    err_pairs, pair_terms = q8_0_mul_i16_lanes_to_i32_pairs_avx2(lhs, rhs)
    assert err_pairs == Q8_0_AVX2_OK

    err_hsum, reduced = q8_0_hsum_i32_pairs_avx2(pair_terms)
    assert err_hsum == Q8_0_AVX2_OK

    err_dot, direct_dot = q8_0_dot_i16_lanes_avx2(lhs, rhs)
    assert err_dot == Q8_0_AVX2_OK

    assert reduced == direct_dot
    assert direct_dot == sum(int(a) * int(b) for a, b in zip(lhs, rhs))


def test_lane_order_mismatch_changes_dot_predictably() -> None:
    # Dot helper must preserve lane order: rotating one side changes output.
    lhs = list(range(-16, 16))
    rhs = [v * 3 for v in range(-16, 16)]

    err_base, base_dot = q8_0_dot_i16_lanes_avx2(lhs, rhs)
    assert err_base == Q8_0_AVX2_OK

    rotated_rhs = rhs[1:] + rhs[:1]
    err_rot, rot_dot = q8_0_dot_i16_lanes_avx2(lhs, rotated_rhs)
    assert err_rot == Q8_0_AVX2_OK

    assert base_dot == sum(int(a) * int(b) for a, b in zip(lhs, rhs))
    assert rot_dot == sum(int(a) * int(b) for a, b in zip(lhs, rotated_rhs))
    assert rot_dot != base_dot


def test_reduction_overflow_probe_path() -> None:
    # Lane dot itself cannot overflow I64 with 16 I32 terms, but the reduction
    # helper is designed for future wider streams and must keep guards correct.
    probe = [0] * Q8_0_AVX2_PAIR_COUNT
    probe[0] = Q8_0_AVX2_I64_MAX
    probe[1] = 1

    err, _ = q8_0_hsum_i32_pairs_avx2(probe)
    assert err == Q8_0_AVX2_ERR_OVERFLOW

    probe[0] = Q8_0_AVX2_I64_MIN
    probe[1] = -1
    err, _ = q8_0_hsum_i32_pairs_avx2(probe)
    assert err == Q8_0_AVX2_ERR_OVERFLOW


def test_len_and_null_errors() -> None:
    err, _ = q8_0_dot_i16_lanes_avx2(None, [0] * 32)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_i16_lanes_avx2([0] * 32, None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_dot_i16_lanes_avx2([0] * 31, [0] * 32)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_dot_i16_lanes_avx2([0] * 32, [0] * 31)
    assert err == Q8_0_AVX2_ERR_BAD_LEN


def run() -> None:
    test_known_signed_edge_contract()
    test_randomized_matches_scalar_reference()
    test_lane_order_invariant_pair_sensitivity()
    test_pair_grouping_contract_matches_mul_then_hsum()
    test_lane_order_mismatch_changes_dot_predictably()
    test_reduction_overflow_probe_path()
    test_len_and_null_errors()
    print("q8_0_avx2_dot_lanes_reference_checks=ok")


if __name__ == "__main__":
    run()
