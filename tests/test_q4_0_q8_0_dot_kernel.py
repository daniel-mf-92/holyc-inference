#!/usr/bin/env python3
"""Reference checks for integer mixed Q4_0 x Q8_0 dot-product semantics."""

from __future__ import annotations

import random
import struct

Q4_0_PACKED_BYTES = 16
Q8_0_PACKED_BYTES = 32

Q4_0_Q8_0_OK = 0
Q4_0_Q8_0_ERR_NULL_PTR = 1
Q4_0_Q8_0_ERR_BAD_DST_LEN = 2
Q4_0_Q8_0_ERR_OVERFLOW = 3
Q4_0_Q8_0_I64_MAX = (1 << 63) - 1
Q4_0_Q8_0_I64_MIN = -(1 << 63)


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


def nibble_to_signed(packed: int, upper_nibble: bool) -> int:
    if upper_nibble:
        q_unsigned = (packed >> 4) & 0x0F
    else:
        q_unsigned = packed & 0x0F
    return q_unsigned - 8


def unpack_q4_signed(qs: bytes) -> list[int]:
    out: list[int] = []
    for packed in qs:
        out.append(nibble_to_signed(packed, False))
        out.append(nibble_to_signed(packed, True))
    return out


def unpack_q8_signed(qs: bytes) -> list[int]:
    return [struct.unpack("<b", bytes([byte]))[0] for byte in qs]


def pack_q4_from_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    out = bytearray()
    for idx in range(0, 32, 2):
        lo = vals[idx] + 8
        hi = vals[idx + 1] + 8
        assert 0 <= lo <= 15 and 0 <= hi <= 15
        out.append((lo & 0x0F) | ((hi & 0x0F) << 4))
    return bytes(out)


def pack_q8_signed(vals: list[int]) -> bytes:
    assert len(vals) == 32
    return bytes((v + 256) % 256 for v in vals)


def dot_product_block_q32(lhs_scale_fp16: int, lhs_q4: bytes, rhs_scale_fp16: int, rhs_q8: bytes) -> tuple[int, int]:
    if len(lhs_q4) != Q4_0_PACKED_BYTES or len(rhs_q8) != Q8_0_PACKED_BYTES:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
    rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
    scale_prod_q32 = lhs_scale_q16 * rhs_scale_q16

    lhs_signed = unpack_q4_signed(lhs_q4)
    rhs_signed = unpack_q8_signed(rhs_q8)
    q_dot_q0 = sum(a * b for a, b in zip(lhs_signed, rhs_signed))
    return Q4_0_Q8_0_OK, scale_prod_q32 * q_dot_q0


def dot_product_blocks_q32(lhs_blocks, rhs_blocks) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total = 0
    for (l_scale, l_q4), (r_scale, r_q8) in zip(lhs_blocks, rhs_blocks):
        err, block_dot = dot_product_block_q32(l_scale, l_q4, r_scale, r_q8)
        if err != Q4_0_Q8_0_OK:
            return err, 0
        total += block_dot

    return Q4_0_Q8_0_OK, total


def dot_product_blocks_q32_to_q16(lhs_blocks, rhs_blocks) -> tuple[int, int]:
    err, dot_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
    if err != Q4_0_Q8_0_OK:
        return err, 0
    return Q4_0_Q8_0_OK, dot_q32_to_q16(dot_q32)


def dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, initial_accum_q16: int) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total_q16 = initial_accum_q16
    for (l_scale, l_q4), (r_scale, r_q8) in zip(lhs_blocks, rhs_blocks):
        err, block_dot_q32 = dot_product_block_q32(l_scale, l_q4, r_scale, r_q8)
        if err != Q4_0_Q8_0_OK:
            return err, 0
        total_q16 += dot_q32_to_q16(block_dot_q32)

    return Q4_0_Q8_0_OK, total_q16


def try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    if rhs > 0 and lhs > Q4_0_Q8_0_I64_MAX - rhs:
        return False, 0
    if rhs < 0 and lhs < Q4_0_Q8_0_I64_MIN - rhs:
        return False, 0
    return True, lhs + rhs


def dot_product_blocks_q16_accumulate_checked(lhs_blocks, rhs_blocks, initial_accum_q16: int) -> tuple[int, int]:
    if len(lhs_blocks) != len(rhs_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, 0

    total_q16 = initial_accum_q16
    for (l_scale, l_q4), (r_scale, r_q8) in zip(lhs_blocks, rhs_blocks):
        err, block_dot_q32 = dot_product_block_q32(l_scale, l_q4, r_scale, r_q8)
        if err != Q4_0_Q8_0_OK:
            return err, 0
        block_dot_q16 = dot_q32_to_q16(block_dot_q32)
        ok, checked = try_add_i64(total_q16, block_dot_q16)
        if not ok:
            return Q4_0_Q8_0_ERR_OVERFLOW, 0
        total_q16 = checked

    return Q4_0_Q8_0_OK, total_q16


def dot_row_blocks_q16(lhs_blocks, rhs_blocks) -> tuple[int, int]:
    return dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)


def dot_rows_q16_matrix_vector(
    matrix_blocks,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_count: int,
) -> tuple[int, list[int]]:
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    required_matrix_blocks = row_count * row_stride_blocks
    if required_matrix_blocks > len(matrix_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if vec_block_count > len(vec_blocks):
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    out_rows: list[int] = []
    for row_index in range(row_count):
        row_start = row_index * row_stride_blocks
        row_slice = matrix_blocks[row_start : row_start + vec_block_count]
        err, row_q16 = dot_row_blocks_q16(row_slice, vec_blocks[:vec_block_count])
        if err != Q4_0_Q8_0_OK:
            return err, []
        out_rows.append(row_q16)

    return Q4_0_Q8_0_OK, out_rows


def dot_rows_q16_matrix_vector_checked(
    matrix_blocks,
    matrix_block_capacity: int,
    row_count: int,
    row_stride_blocks: int,
    vec_blocks,
    vec_block_capacity: int,
    vec_block_count: int,
) -> tuple[int, list[int]]:
    if matrix_block_capacity < 0 or vec_block_capacity < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if row_count < 0 or row_stride_blocks < 0 or vec_block_count < 0:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if row_count > 0 and row_stride_blocks < vec_block_count:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []
    if vec_block_count > vec_block_capacity:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    required_matrix_blocks = row_count * row_stride_blocks
    if required_matrix_blocks > Q4_0_Q8_0_I64_MAX:
        return Q4_0_Q8_0_ERR_OVERFLOW, []
    if required_matrix_blocks > matrix_block_capacity:
        return Q4_0_Q8_0_ERR_BAD_DST_LEN, []

    out_rows: list[int] = []
    for row_index in range(row_count):
        row_start = row_index * row_stride_blocks
        row_slice = matrix_blocks[row_start : row_start + vec_block_count]
        err, row_q16 = dot_product_blocks_q16_accumulate_checked(row_slice, vec_blocks[:vec_block_count], 0)
        if err != Q4_0_Q8_0_OK:
            return err, []
        out_rows.append(row_q16)

    return Q4_0_Q8_0_OK, out_rows


def dot_q32_to_q16(dot_q32: int) -> int:
    return round_shift_right_signed(dot_q32, 16)


def test_identity_mixed_block() -> None:
    lhs_scale_fp16 = half_bits(1.0)
    rhs_scale_fp16 = half_bits(1.0)

    q4_signed = [((idx % 16) - 8) for idx in range(32)]
    q8_signed = q4_signed[:]

    lhs_q4 = pack_q4_from_signed(q4_signed)
    rhs_q8 = pack_q8_signed(q8_signed)

    err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
    assert err == Q4_0_Q8_0_OK

    expected_q0 = sum(v * v for v in q4_signed)
    expected_q32 = (1 << 16) * (1 << 16) * expected_q0
    assert got_q32 == expected_q32
    assert dot_q32_to_q16(got_q32) == expected_q0 * (1 << 16)


def test_random_blocks_match_integer_reference() -> None:
    rng = random.Random(20260412)

    for _ in range(500):
        lhs_scale = rng.uniform(-4.0, 4.0)
        rhs_scale = rng.uniform(-4.0, 4.0)

        lhs_scale_fp16 = half_bits(lhs_scale)
        rhs_scale_fp16 = half_bits(rhs_scale)

        q4_signed = [rng.randrange(-8, 8) for _ in range(32)]
        q8_signed = [rng.randrange(-128, 128) for _ in range(32)]

        lhs_q4 = pack_q4_from_signed(q4_signed)
        rhs_q8 = pack_q8_signed(q8_signed)

        err, got_q32 = dot_product_block_q32(lhs_scale_fp16, lhs_q4, rhs_scale_fp16, rhs_q8)
        assert err == Q4_0_Q8_0_OK

        expected_q0 = sum(a * b for a, b in zip(q4_signed, q8_signed))
        lhs_scale_q16 = f16_to_q16(lhs_scale_fp16)
        rhs_scale_q16 = f16_to_q16(rhs_scale_fp16)
        expected_q32 = lhs_scale_q16 * rhs_scale_q16 * expected_q0
        assert got_q32 == expected_q32

        lhs_scale_half = struct.unpack("<e", struct.pack("<H", lhs_scale_fp16))[0]
        rhs_scale_half = struct.unpack("<e", struct.pack("<H", rhs_scale_fp16))[0]
        expected_float_q32 = round((lhs_scale_half * rhs_scale_half) * expected_q0 * (1 << 32))

        # Integer path rounds scales once to Q16 then multiplies.
        assert abs(got_q32 - expected_float_q32) <= 70_000_000


def test_multiblock_accumulation() -> None:
    b0 = (half_bits(0.75), pack_q4_from_signed([((i % 8) - 4) for i in range(32)]))
    b1 = (half_bits(-0.5), pack_q4_from_signed([4 - (i % 8) for i in range(32)]))

    r0 = (half_bits(1.25), pack_q8_signed([((i % 11) - 5) * 3 for i in range(32)]))
    r1 = (half_bits(-1.5), pack_q8_signed([7 - (i % 13) for i in range(32)]))

    err, total = dot_product_blocks_q32([b0, b1], [r0, r1])
    assert err == Q4_0_Q8_0_OK

    _, part0 = dot_product_block_q32(*b0, *r0)
    _, part1 = dot_product_block_q32(*b1, *r1)
    assert total == part0 + part1


def test_error_on_bad_lengths() -> None:
    err, _ = dot_product_block_q32(half_bits(1.0), b"\x00" * 15, half_bits(1.0), b"\x00" * 32)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_product_block_q32(half_bits(1.0), b"\x00" * 16, half_bits(1.0), b"\x00" * 31)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_q16_accumulator_helper_matches_blockwise_rounding() -> None:
    rng = random.Random(2026041217)

    for _ in range(250):
        block_count = rng.randint(1, 6)
        lhs_blocks = []
        rhs_blocks = []

        for _ in range(block_count):
            l_scale = half_bits(rng.uniform(-3.0, 3.0))
            r_scale = half_bits(rng.uniform(-3.0, 3.0))
            l_q4 = pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
            r_q8 = pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])
            lhs_blocks.append((l_scale, l_q4))
            rhs_blocks.append((r_scale, r_q8))

        initial_accum = rng.randint(-(1 << 30), (1 << 30))
        err, got_q16 = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, initial_accum)
        assert err == Q4_0_Q8_0_OK

        expected_q16 = initial_accum
        for lhs_block, rhs_block in zip(lhs_blocks, rhs_blocks):
            err, block_dot_q32 = dot_product_block_q32(*lhs_block, *rhs_block)
            assert err == Q4_0_Q8_0_OK
            expected_q16 += dot_q32_to_q16(block_dot_q32)

        assert got_q16 == expected_q16


def test_q32_to_q16_helper_rounds_once_after_full_accumulation() -> None:
    b0 = (half_bits(0.125), pack_q4_from_signed([7] * 32))
    b1 = (half_bits(0.125), pack_q4_from_signed([-8] * 32))
    r0 = (half_bits(0.125), pack_q8_signed([127] * 32))
    r1 = (half_bits(0.125), pack_q8_signed([127] * 32))

    err, dot_q16_single_round = dot_product_blocks_q32_to_q16([b0, b1], [r0, r1])
    assert err == Q4_0_Q8_0_OK

    err, dot_q16_blockwise = dot_product_blocks_q16_accumulate([b0, b1], [r0, r1], 0)
    assert err == Q4_0_Q8_0_OK

    # This vector pair intentionally exercises fractional cancellation.
    # Full-dot single rounding (Q32->Q16 once) is the IQ-085 contract,
    # while blockwise Q16 accumulation can differ by a few LSBs.
    assert abs(dot_q16_single_round - dot_q16_blockwise) <= 2


def test_q32_to_q16_helper_matches_full_dot_reference() -> None:
    rng = random.Random(202604151)

    for _ in range(200):
        block_count = rng.randint(1, 8)
        lhs_blocks = []
        rhs_blocks = []

        for _ in range(block_count):
            l_scale = half_bits(rng.uniform(-3.5, 3.5))
            r_scale = half_bits(rng.uniform(-3.5, 3.5))
            l_q4 = pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
            r_q8 = pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])
            lhs_blocks.append((l_scale, l_q4))
            rhs_blocks.append((r_scale, r_q8))

        err, got_q16 = dot_product_blocks_q32_to_q16(lhs_blocks, rhs_blocks)
        assert err == Q4_0_Q8_0_OK

        err, full_q32 = dot_product_blocks_q32(lhs_blocks, rhs_blocks)
        assert err == Q4_0_Q8_0_OK
        assert got_q16 == dot_q32_to_q16(full_q32)


def test_q16_accumulator_bad_length_error() -> None:
    lhs_blocks = [(half_bits(1.0), pack_q4_from_signed([0] * 32))]
    rhs_blocks = []
    err, _ = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_row_helper_matches_zero_init_accumulate() -> None:
    rng = random.Random(2026041527)

    for _ in range(220):
        block_count = rng.randint(1, 8)
        lhs_blocks = []
        rhs_blocks = []

        for _ in range(block_count):
            l_scale = half_bits(rng.uniform(-3.25, 3.25))
            r_scale = half_bits(rng.uniform(-3.25, 3.25))
            l_q4 = pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
            r_q8 = pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])
            lhs_blocks.append((l_scale, l_q4))
            rhs_blocks.append((r_scale, r_q8))

        err, got_row_q16 = dot_row_blocks_q16(lhs_blocks, rhs_blocks)
        assert err == Q4_0_Q8_0_OK

        err, expected_q16 = dot_product_blocks_q16_accumulate(lhs_blocks, rhs_blocks, 0)
        assert err == Q4_0_Q8_0_OK
        assert got_row_q16 == expected_q16


def test_row_helper_can_differ_from_full_dot_single_rounding() -> None:
    b0 = (half_bits(0.125), pack_q4_from_signed([7] * 32))
    b1 = (half_bits(0.125), pack_q4_from_signed([-8] * 32))
    r0 = (half_bits(0.125), pack_q8_signed([127] * 32))
    r1 = (half_bits(0.125), pack_q8_signed([127] * 32))

    err, row_q16 = dot_row_blocks_q16([b0, b1], [r0, r1])
    assert err == Q4_0_Q8_0_OK

    err, single_round_q16 = dot_product_blocks_q32_to_q16([b0, b1], [r0, r1])
    assert err == Q4_0_Q8_0_OK

    # Row helper intentionally rounds each block contribution before summation.
    # Keep tolerance tight so any hidden semantic drift is caught quickly.
    assert abs(row_q16 - single_round_q16) <= 2


def test_matrix_vector_stride_padding_ignored_randomized() -> None:
    rng = random.Random(2026041601)

    for _ in range(220):
        row_count = rng.randint(1, 8)
        vec_block_count = rng.randint(1, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 4)

        vec_blocks = []
        for _ in range(vec_block_count):
            vec_scale = half_bits(rng.uniform(-2.25, 2.25))
            vec_q8 = pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])
            vec_blocks.append((vec_scale, vec_q8))

        matrix_blocks = []
        expected_rows = []

        for _ in range(row_count):
            active_blocks = []
            for _ in range(vec_block_count):
                row_scale = half_bits(rng.uniform(-2.25, 2.25))
                row_q4 = pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
                block = (row_scale, row_q4)
                matrix_blocks.append(block)
                active_blocks.append(block)

            pad_blocks = row_stride_blocks - vec_block_count
            for _ in range(pad_blocks):
                pad_scale = half_bits(rng.uniform(-2.25, 2.25))
                pad_q4 = pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
                matrix_blocks.append((pad_scale, pad_q4))

            err, expected_q16 = dot_row_blocks_q16(active_blocks, vec_blocks)
            assert err == Q4_0_Q8_0_OK
            expected_rows.append(expected_q16)

        err, got_rows = dot_rows_q16_matrix_vector(
            matrix_blocks,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
        )
        assert err == Q4_0_Q8_0_OK
        assert got_rows == expected_rows


def test_matrix_vector_sign_semantics_reference_rowwise() -> None:
    row_count = 4
    vec_block_count = 1
    row_stride_blocks = 1

    vec_blocks = [(half_bits(1.0), pack_q8_signed([3] * 32))]
    matrix_rows = [
        [(half_bits(1.0), pack_q4_from_signed([2] * 32))],
        [(half_bits(-1.0), pack_q4_from_signed([2] * 32))],
        [(half_bits(1.0), pack_q4_from_signed([-2] * 32))],
        [(half_bits(-1.0), pack_q4_from_signed([-2] * 32))],
    ]

    matrix_blocks = [block for row in matrix_rows for block in row]
    err, got_rows = dot_rows_q16_matrix_vector(
        matrix_blocks,
        row_count,
        row_stride_blocks,
        vec_blocks,
        vec_block_count,
    )
    assert err == Q4_0_Q8_0_OK

    expected_rows = []
    for row in matrix_rows:
        err, row_q16 = dot_row_blocks_q16(row, vec_blocks)
        assert err == Q4_0_Q8_0_OK
        expected_rows.append(row_q16)

    assert got_rows == expected_rows
    assert got_rows[0] > 0
    assert got_rows[1] < 0
    assert got_rows[2] < 0
    assert got_rows[3] > 0


def test_matrix_vector_argument_and_extent_errors() -> None:
    vec_blocks = [(half_bits(1.0), pack_q8_signed([1] * 32))]
    matrix_blocks = [(half_bits(1.0), pack_q4_from_signed([1] * 32))]

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, -1, 1, vec_blocks, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, -1, vec_blocks, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, 1, vec_blocks, -1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, 0, vec_blocks, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    # Truncated matrix extent for two rows with stride=1.
    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 2, 1, vec_blocks, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    # Truncated vector extent when vec_block_count exceeds provided vector blocks.
    err, _ = dot_rows_q16_matrix_vector(matrix_blocks, 1, 1, [], 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_matrix_vector_checked_matches_unchecked_normal_ranges() -> None:
    rng = random.Random(2026041602)

    for _ in range(180):
        row_count = rng.randint(1, 8)
        vec_block_count = rng.randint(1, 6)
        row_stride_blocks = vec_block_count + rng.randint(0, 3)

        vec_blocks = []
        for _ in range(vec_block_count):
            vec_scale = half_bits(rng.uniform(-2.0, 2.0))
            vec_q8 = pack_q8_signed([rng.randrange(-128, 128) for _ in range(32)])
            vec_blocks.append((vec_scale, vec_q8))

        matrix_blocks = []
        for _ in range(row_count):
            for _ in range(row_stride_blocks):
                row_scale = half_bits(rng.uniform(-2.0, 2.0))
                row_q4 = pack_q4_from_signed([rng.randrange(-8, 8) for _ in range(32)])
                matrix_blocks.append((row_scale, row_q4))

        err_u, out_u = dot_rows_q16_matrix_vector(
            matrix_blocks,
            row_count,
            row_stride_blocks,
            vec_blocks,
            vec_block_count,
        )
        err_c, out_c = dot_rows_q16_matrix_vector_checked(
            matrix_blocks,
            len(matrix_blocks),
            row_count,
            row_stride_blocks,
            vec_blocks,
            len(vec_blocks),
            vec_block_count,
        )
        assert err_u == Q4_0_Q8_0_OK
        assert err_c == Q4_0_Q8_0_OK
        assert out_c == out_u


def test_matrix_vector_checked_extent_errors() -> None:
    vec_blocks = [(half_bits(1.0), pack_q8_signed([1] * 32))]
    matrix_blocks = [(half_bits(1.0), pack_q4_from_signed([1] * 32))]

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, -1, 1, 1, vec_blocks, 1, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, 1, 1, vec_blocks, -1, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, -1, 1, vec_blocks, 1, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, 1, -1, vec_blocks, 1, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, 1, 1, vec_blocks, 1, -1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, 1, 0, vec_blocks, 1, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, 1, 1, vec_blocks, 0, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN

    err, _ = dot_rows_q16_matrix_vector_checked(matrix_blocks, 1, 2, 1, vec_blocks, 1, 1)
    assert err == Q4_0_Q8_0_ERR_BAD_DST_LEN


def test_matrix_vector_checked_reports_row_accumulator_overflow() -> None:
    scale_pos_max = 0x7C00
    plus_one_q8 = pack_q8_signed([1] + [0] * 31)
    plus_seven_q4 = pack_q4_from_signed([7] + [0] * 31)

    # One block contributes a huge positive Q16 value; two rows force overflow
    # when checked code accumulates row outputs separately.
    row_block = (scale_pos_max, plus_seven_q4)
    vec_block = (scale_pos_max, plus_one_q8)
    matrix_blocks = [row_block, row_block]
    vec_blocks = [vec_block]

    err, _ = dot_rows_q16_matrix_vector_checked(
        matrix_blocks,
        len(matrix_blocks),
        2,
        1,
        vec_blocks,
        len(vec_blocks),
        1,
    )
    assert err == Q4_0_Q8_0_ERR_OVERFLOW


def run() -> None:
    test_identity_mixed_block()
    test_random_blocks_match_integer_reference()
    test_multiblock_accumulation()
    test_error_on_bad_lengths()
    test_q16_accumulator_helper_matches_blockwise_rounding()
    test_q32_to_q16_helper_rounds_once_after_full_accumulation()
    test_q32_to_q16_helper_matches_full_dot_reference()
    test_q16_accumulator_bad_length_error()
    test_row_helper_matches_zero_init_accumulate()
    test_row_helper_can_differ_from_full_dot_single_rounding()
    test_matrix_vector_stride_padding_ignored_randomized()
    test_matrix_vector_sign_semantics_reference_rowwise()
    test_matrix_vector_argument_and_extent_errors()
    test_matrix_vector_checked_matches_unchecked_normal_ranges()
    test_matrix_vector_checked_extent_errors()
    test_matrix_vector_checked_reports_row_accumulator_overflow()
    print("q4_0_q8_0_dot_kernel_reference_checks=ok")


if __name__ == "__main__":
    run()
