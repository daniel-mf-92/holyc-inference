#!/usr/bin/env python3
"""Reference checks for Q8_0 integer-only dot product semantics."""

from __future__ import annotations

import random
import struct

Q8_0_VALUES_PER_BLOCK = 32
Q8_0_PACKED_BYTES = 32

Q8_0_OK = 0
Q8_0_ERR_NULL_PTR = 1
Q8_0_ERR_BAD_DST_LEN = 2


def round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    round_bias = 1 << (shift - 1)
    return (value + round_bias) >> shift


def round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return round_shift_right_unsigned(value, shift)
    return -round_shift_right_unsigned(-value, shift)


def f16_to_q16(fp16_bits: int) -> int:
    sign_bit = (fp16_bits >> 15) & 1
    exponent_bits = (fp16_bits >> 10) & 0x1F
    fraction_bits = fp16_bits & 0x03FF

    if exponent_bits == 0:
        if fraction_bits == 0:
            return 0
        magnitude_q16 = round_shift_right_unsigned(fraction_bits, 8)
        return -magnitude_q16 if sign_bit else magnitude_q16

    if exponent_bits == 0x1F:
        return -(0x3FFFFFFFFFFFFFFF) if sign_bit else 0x3FFFFFFFFFFFFFFF

    mantissa = 1024 + fraction_bits
    shift_amount = exponent_bits - 9
    if shift_amount >= 0:
        magnitude_q16 = mantissa << shift_amount
    else:
        magnitude_q16 = round_shift_right_unsigned(mantissa, -shift_amount)

    return -magnitude_q16 if sign_bit else magnitude_q16


def half_bits(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", value))[0]


def pack_signed(values: list[int]) -> bytes:
    assert len(values) == Q8_0_PACKED_BYTES
    return bytes((v + 256) % 256 for v in values)


def unpack_signed(qs: bytes) -> list[int]:
    return [struct.unpack("<b", bytes([byte]))[0] for byte in qs]


def dot_product_block_q32(
    lhs_scale_fp16: int,
    lhs_qs: bytes,
    rhs_scale_fp16: int,
    rhs_qs: bytes,
) -> tuple[int, int]:
    if len(lhs_qs) != Q8_0_PACKED_BYTES or len(rhs_qs) != Q8_0_PACKED_BYTES:
        return Q8_0_ERR_BAD_DST_LEN, 0

    lhs_signed = unpack_signed(lhs_qs)
    rhs_signed = unpack_signed(rhs_qs)

    q_dot_q0 = 0
    for lhs_q, rhs_q in zip(lhs_signed, rhs_signed):
        q_dot_q0 += lhs_q * rhs_q

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    dot_q32 = lhs_scale_q16 * rhs_scale_q16 * q_dot_q0
    return Q8_0_OK, dot_q32


def dot_product_blocks_q32(
    lhs_blocks: list[tuple[int, bytes]],
    rhs_blocks: list[tuple[int, bytes]],
) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q8_0_ERR_BAD_DST_LEN, 0

    total = 0
    for (lhs_scale, lhs_qs), (rhs_scale, rhs_qs) in zip(lhs_blocks, rhs_blocks):
        err, block_dot = dot_product_block_q32(lhs_scale, lhs_qs, rhs_scale, rhs_qs)
        if err != Q8_0_OK:
            return err, 0
        total += block_dot

    return Q8_0_OK, total


def dot_q32_to_q16(dot_q32: int) -> int:
    return round_shift_right_signed(dot_q32, 16)


def dot_product_blocks_q16_accumulate(
    lhs_blocks: list[tuple[int, bytes]],
    rhs_blocks: list[tuple[int, bytes]],
    initial_accum_q16: int,
) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q8_0_ERR_BAD_DST_LEN, 0

    total_q16 = initial_accum_q16
    for (lhs_scale, lhs_qs), (rhs_scale, rhs_qs) in zip(lhs_blocks, rhs_blocks):
        err, block_dot_q32 = dot_product_block_q32(lhs_scale, lhs_qs, rhs_scale, rhs_qs)
        if err != Q8_0_OK:
            return err, 0
        total_q16 += dot_q32_to_q16(block_dot_q32)

    return Q8_0_OK, total_q16


def dot_row_blocks_q16(
    lhs_blocks: list[tuple[int, bytes]],
    rhs_blocks: list[tuple[int, bytes]],
) -> tuple[int, int]:
    return dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)


def dot_rows_q16_matrix_vector(
    matrix_blocks: list[tuple[int, bytes]],
    row_count: int,
    row_stride_blocks: int,
    vec_blocks: list[tuple[int, bytes]],
    vec_block_count: int,
) -> tuple[int, list[int]]:
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q8_0_ERR_BAD_DST_LEN, []
    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q8_0_ERR_BAD_DST_LEN, []

    out_rows_q16 = [0] * row_count
    for row_index in range(row_count):
        row_base = row_index * row_stride_blocks
        row_slice = matrix_blocks[row_base : row_base + vec_block_count]
        if len(row_slice) != vec_block_count:
            return Q8_0_ERR_BAD_DST_LEN, []

        err, row_dot_q16 = dot_row_blocks_q16(row_slice, vec_blocks[:vec_block_count])
        if err != Q8_0_OK:
            return err, []

        out_rows_q16[row_index] = row_dot_q16

    return Q8_0_OK, out_rows_q16


def dot_product_blocks_q32_to_q16(
    lhs_blocks: list[tuple[int, bytes]],
    rhs_blocks: list[tuple[int, bytes]],
) -> tuple[int, int]:
    err, total_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
    if err != Q8_0_OK:
        return err, 0
    return Q8_0_OK, dot_q32_to_q16(total_q32)


def test_identity_block() -> None:
    lhs_scale_fp16 = half_bits(1.0)
    rhs_scale_fp16 = half_bits(1.0)
    signed = list(range(-16, 16))
    packed = pack_signed(signed)

    err, got_q32 = dot_product_block_q32(lhs_scale_fp16, packed, rhs_scale_fp16, packed)
    assert err == Q8_0_OK

    expected_q0 = sum(v * v for v in signed)
    expected_q32 = expected_q0 * (1 << 32)
    assert got_q32 == expected_q32
    assert dot_q32_to_q16(got_q32) == expected_q0 * (1 << 16)


def test_negative_scale_sign() -> None:
    lhs_scale_fp16 = half_bits(-0.5)
    rhs_scale_fp16 = half_bits(2.0)

    lhs_signed = [5] * Q8_0_VALUES_PER_BLOCK
    rhs_signed = [3] * Q8_0_VALUES_PER_BLOCK

    err, got_q32 = dot_product_block_q32(
        lhs_scale_fp16,
        pack_signed(lhs_signed),
        rhs_scale_fp16,
        pack_signed(rhs_signed),
    )
    assert err == Q8_0_OK

    expected_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))
    expected_q32 = -expected_q0 * (1 << 32)
    assert got_q32 == expected_q32


def test_random_blocks_match_float_reference_bounds() -> None:
    rng = random.Random(992244)

    for _ in range(400):
        lhs_scale = rng.uniform(-3.0, 3.0)
        rhs_scale = rng.uniform(-3.0, 3.0)
        lhs_scale_fp16 = half_bits(lhs_scale)
        rhs_scale_fp16 = half_bits(rhs_scale)

        lhs_signed = [rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]
        rhs_signed = [rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)]

        err, got_q32 = dot_product_block_q32(
            lhs_scale_fp16,
            pack_signed(lhs_signed),
            rhs_scale_fp16,
            pack_signed(rhs_signed),
        )
        assert err == Q8_0_OK

        dot_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))
        lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
        rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
        expected_int_q32 = lhs_scale_q16 * rhs_scale_q16 * dot_q0
        assert got_q32 == expected_int_q32

        lhs_scale_half = struct.unpack("<e", struct.pack("<H", lhs_scale_fp16))[0]
        rhs_scale_half = struct.unpack("<e", struct.pack("<H", rhs_scale_fp16))[0]
        expected_float_q32 = round((lhs_scale_half * rhs_scale_half) * dot_q0 * (1 << 32))

        # Integer path rounds scales once to Q16. Tight deterministic bound.
        assert abs(got_q32 - expected_float_q32) <= 700_000_000


def test_multiblock_accumulation() -> None:
    lhs_blocks = [
        (half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK)),
        (half_bits(0.5), pack_signed([2] * Q8_0_VALUES_PER_BLOCK)),
    ]
    rhs_blocks = [
        (half_bits(1.0), pack_signed([3] * Q8_0_VALUES_PER_BLOCK)),
        (half_bits(-2.0), pack_signed([4] * Q8_0_VALUES_PER_BLOCK)),
    ]

    err, got_total_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
    assert err == Q8_0_OK

    _, block0 = dot_product_block_q32(*lhs_blocks[0], *rhs_blocks[0])
    _, block1 = dot_product_block_q32(*lhs_blocks[1], *rhs_blocks[1])
    assert got_total_q32 == block0 + block1


def test_error_on_bad_input_length() -> None:
    err, _ = dot_product_block_q32(half_bits(1.0), b"\x00" * 31, half_bits(1.0), b"\x00" * 32)
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_q16_accumulator_helper_matches_blockwise_rounding() -> None:
    rng = random.Random(662900)

    for _ in range(250):
        block_count = rng.randint(1, 6)
        lhs_blocks: list[tuple[int, bytes]] = []
        rhs_blocks: list[tuple[int, bytes]] = []

        for _ in range(block_count):
            lhs_scale = half_bits(rng.uniform(-2.5, 2.5))
            rhs_scale = half_bits(rng.uniform(-2.5, 2.5))
            lhs_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            rhs_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            lhs_blocks.append((lhs_scale, lhs_qs))
            rhs_blocks.append((rhs_scale, rhs_qs))

        initial_accum = rng.randint(-(1 << 28), (1 << 28))
        err, got_q16 = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, initial_accum)
        assert err == Q8_0_OK

        expected_q16 = initial_accum
        for lhs_block, rhs_block in zip(lhs_blocks, rhs_blocks):
            err, block_q32 = dot_product_block_q32(*lhs_block, *rhs_block)
            assert err == Q8_0_OK
            expected_q16 += dot_q32_to_q16(block_q32)

        assert got_q16 == expected_q16


def test_q16_accumulator_bad_length_error() -> None:
    lhs_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]
    rhs_blocks: list[tuple[int, bytes]] = []
    err, _ = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_row_blocks_q16_matches_per_block_rounding() -> None:
    rng = random.Random(321909)

    for _ in range(250):
        block_count = rng.randint(1, 8)
        lhs_blocks: list[tuple[int, bytes]] = []
        rhs_blocks: list[tuple[int, bytes]] = []

        for _ in range(block_count):
            lhs_scale = half_bits(rng.uniform(-2.5, 2.5))
            rhs_scale = half_bits(rng.uniform(-2.5, 2.5))
            lhs_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            rhs_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            lhs_blocks.append((lhs_scale, lhs_qs))
            rhs_blocks.append((rhs_scale, rhs_qs))

        err, got_row_q16 = dot_row_blocks_q16(lhs_blocks, rhs_blocks)
        assert err == Q8_0_OK

        expected_q16 = 0
        for lhs_block, rhs_block in zip(lhs_blocks, rhs_blocks):
            err, block_q32 = dot_product_block_q32(*lhs_block, *rhs_block)
            assert err == Q8_0_OK
            expected_q16 += dot_q32_to_q16(block_q32)

        assert got_row_q16 == expected_q16


def test_row_blocks_q16_distinct_from_round_at_end() -> None:
    # Make each block contribute +0.5 Q16 units:
    # q_dot_q0=1 and scale_q16=181 so (181*181)>>16 rounds to 0 per block,
    # while summing two blocks before rounding gives 1.
    scale_fp16 = half_bits(181.0 / 65536.0)
    lhs_qs = pack_signed([1] + [0] * 31)
    rhs_qs = pack_signed([1] + [0] * 31)

    lhs_blocks = [(scale_fp16, lhs_qs), (scale_fp16, lhs_qs)]
    rhs_blocks = [(scale_fp16, rhs_qs), (scale_fp16, rhs_qs)]

    err, got_row_q16 = dot_row_blocks_q16(lhs_blocks, rhs_blocks)
    assert err == Q8_0_OK

    err, total_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
    assert err == Q8_0_OK
    round_at_end_q16 = dot_q32_to_q16(total_q32)

    assert got_row_q16 != round_at_end_q16
    assert got_row_q16 == 0
    assert round_at_end_q16 == 1


def test_row_blocks_q16_bad_length_error() -> None:
    lhs_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]
    rhs_blocks: list[tuple[int, bytes]] = []
    err, _ = dot_row_blocks_q16(lhs_blocks, rhs_blocks)
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_q32_to_q16_single_rounding_matches_total_q32_rounding() -> None:
    rng = random.Random(20260415)

    for _ in range(250):
        block_count = rng.randint(1, 10)
        lhs_blocks: list[tuple[int, bytes]] = []
        rhs_blocks: list[tuple[int, bytes]] = []

        for _ in range(block_count):
            lhs_scale = half_bits(rng.uniform(-2.5, 2.5))
            rhs_scale = half_bits(rng.uniform(-2.5, 2.5))
            lhs_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            rhs_qs = pack_signed([rng.randint(-128, 127) for _ in range(Q8_0_VALUES_PER_BLOCK)])
            lhs_blocks.append((lhs_scale, lhs_qs))
            rhs_blocks.append((rhs_scale, rhs_qs))

        err, got_q16 = dot_product_blocks_q32_to_q16(lhs_blocks, rhs_blocks)
        assert err == Q8_0_OK

        err, total_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
        assert err == Q8_0_OK
        assert got_q16 == dot_q32_to_q16(total_q32)


def test_q32_to_q16_length_mismatch_error() -> None:
    lhs_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]
    rhs_blocks: list[tuple[int, bytes]] = []
    err, _ = dot_product_blocks_q32_to_q16(lhs_blocks, rhs_blocks)
    assert err == Q8_0_ERR_BAD_DST_LEN


def test_matrix_vector_rows_stride_and_values() -> None:
    row_count = 3
    vec_block_count = 2
    row_stride_blocks = 3

    vec_blocks = [
        (half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK)),
        (half_bits(-0.5), pack_signed([2] * Q8_0_VALUES_PER_BLOCK)),
    ]

    matrix_blocks: list[tuple[int, bytes]] = []
    expected_rows: list[int] = []
    for row_index in range(row_count):
        active_blocks: list[tuple[int, bytes]] = []
        for block_index in range(vec_block_count):
            scale = half_bits(1.0 + 0.25 * (row_index + block_index))
            q = (row_index + 1) * (block_index + 1)
            packed = pack_signed([q] * Q8_0_VALUES_PER_BLOCK)
            block = (scale, packed)
            matrix_blocks.append(block)
            active_blocks.append(block)

        # Padding block that must never affect row result.
        matrix_blocks.append((half_bits(2.0), pack_signed([127] * Q8_0_VALUES_PER_BLOCK)))

        err, expected_q16 = dot_row_blocks_q16(active_blocks, vec_blocks)
        assert err == Q8_0_OK
        expected_rows.append(expected_q16)

    err, got_rows = dot_rows_q16_matrix_vector(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
    )
    assert err == Q8_0_OK
    assert got_rows == expected_rows


def test_matrix_vector_rejects_stride_smaller_than_vector_span() -> None:
    matrix_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]
    vec_blocks = [(half_bits(1.0), pack_signed([1] * Q8_0_VALUES_PER_BLOCK))]

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, 0, vec_blocks, 1)
    assert err == Q8_0_ERR_BAD_DST_LEN


def run() -> None:
    test_identity_block()
    test_negative_scale_sign()
    test_random_blocks_match_float_reference_bounds()
    test_multiblock_accumulation()
    test_error_on_bad_input_length()
    test_q16_accumulator_helper_matches_blockwise_rounding()
    test_q16_accumulator_bad_length_error()
    test_row_blocks_q16_matches_per_block_rounding()
    test_row_blocks_q16_distinct_from_round_at_end()
    test_row_blocks_q16_bad_length_error()
    test_q32_to_q16_single_rounding_matches_total_q32_rounding()
    test_q32_to_q16_length_mismatch_error()
    test_matrix_vector_rows_stride_and_values()
    test_matrix_vector_rejects_stride_smaller_than_vector_span()
    print("q8_0_dot_reference_checks=ok")


if __name__ == "__main__":
    run()
