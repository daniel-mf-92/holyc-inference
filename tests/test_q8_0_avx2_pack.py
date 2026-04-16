#!/usr/bin/env python3
"""Reference checks for Q8_0 AVX2-prep 32-byte -> 32xI16 lane packing."""

from __future__ import annotations

import random
import struct

Q8_0_AVX2_VALUES_PER_BLOCK = 32

Q8_0_AVX2_OK = 0
Q8_0_AVX2_ERR_NULL_PTR = 1
Q8_0_AVX2_ERR_BAD_LEN = 2


def as_i8(byte_value: int) -> int:
    return struct.unpack("<b", bytes([byte_value & 0xFF]))[0]


def q8_0_pack32_to_i16_lanes_avx2(src_q8: bytes):
    if src_q8 is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []
    if len(src_q8) < Q8_0_AVX2_VALUES_PER_BLOCK:
        return Q8_0_AVX2_ERR_BAD_LEN, []

    out = [0] * Q8_0_AVX2_VALUES_PER_BLOCK
    for lane in range(Q8_0_AVX2_VALUES_PER_BLOCK):
        out[lane] = as_i8(src_q8[lane])
    return Q8_0_AVX2_OK, out


def q8_0_pack_block_to_i16_lanes_avx2(block_qs: bytes):
    if block_qs is None:
        return Q8_0_AVX2_ERR_NULL_PTR, []
    return q8_0_pack32_to_i16_lanes_avx2(block_qs)


def test_known_edge_pattern() -> None:
    signed = [
        -128,
        -127,
        -64,
        -33,
        -32,
        -17,
        -16,
        -9,
        -8,
        -2,
        -1,
        0,
        1,
        2,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        95,
        96,
        97,
        126,
        127,
        -5,
    ]
    src = bytes((v + 256) % 256 for v in signed)

    err, packed = q8_0_pack32_to_i16_lanes_avx2(src)
    assert err == Q8_0_AVX2_OK
    assert packed == signed


def test_randomized_matches_sign_extend_reference() -> None:
    rng = random.Random(20260416)

    for _ in range(1200):
        src = bytes(rng.randrange(0, 256) for _ in range(Q8_0_AVX2_VALUES_PER_BLOCK))
        err, packed = q8_0_pack32_to_i16_lanes_avx2(src)
        assert err == Q8_0_AVX2_OK

        expected = [as_i8(value) for value in src]
        assert packed == expected


def test_block_wrapper_is_identical_to_direct_pack() -> None:
    rng = random.Random(101)

    for _ in range(256):
        src = bytes(rng.randrange(0, 256) for _ in range(Q8_0_AVX2_VALUES_PER_BLOCK))

        err_direct, out_direct = q8_0_pack32_to_i16_lanes_avx2(src)
        err_wrap, out_wrap = q8_0_pack_block_to_i16_lanes_avx2(src)

        assert err_direct == Q8_0_AVX2_OK
        assert err_wrap == Q8_0_AVX2_OK
        assert out_wrap == out_direct


def test_len_and_null_errors() -> None:
    err, _ = q8_0_pack32_to_i16_lanes_avx2(None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR

    err, _ = q8_0_pack32_to_i16_lanes_avx2(b"\x00" * 31)
    assert err == Q8_0_AVX2_ERR_BAD_LEN

    err, _ = q8_0_pack_block_to_i16_lanes_avx2(None)
    assert err == Q8_0_AVX2_ERR_NULL_PTR


def run() -> None:
    test_known_edge_pattern()
    test_randomized_matches_sign_extend_reference()
    test_block_wrapper_is_identical_to_direct_pack()
    test_len_and_null_errors()
    print("q8_0_avx2_pack_reference_checks=ok")


if __name__ == "__main__":
    run()
