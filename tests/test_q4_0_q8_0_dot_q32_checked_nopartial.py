#!/usr/bin/env python3
"""Parity harness for Q4_0Q8_0DotQ32CheckedNoPartial (IQ-989)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

import test_q4_0_q8_0_dot_kernel as ref


def q4_0_q8_0_dot_q32_checked_nopartial(
    lhs_blocks,
    lhs_block_capacity: int,
    lhs_block_stride: int,
    rhs_blocks,
    rhs_block_capacity: int,
    rhs_block_stride: int,
    block_count: int,
    out_dot: list[int],
) -> int:
    if lhs_blocks is None or rhs_blocks is None or out_dot is None:
        return ref.Q4_0_Q8_0_ERR_NULL_PTR

    if lhs_block_capacity < 0 or rhs_block_capacity < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_block_stride <= 0 or rhs_block_stride <= 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if block_count < 0:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    if lhs_blocks is out_dot or rhs_blocks is out_dot:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    if block_count == 0:
        out_dot[0] = 0
        return ref.Q4_0_Q8_0_OK

    lhs_required = (block_count - 1) * lhs_block_stride + 1
    rhs_required = (block_count - 1) * rhs_block_stride + 1
    if lhs_required > lhs_block_capacity or rhs_required > rhs_block_capacity:
        return ref.Q4_0_Q8_0_ERR_BAD_DST_LEN

    staged = 0
    for block_idx in range(block_count):
        lhs_index = block_idx * lhs_block_stride
        rhs_index = block_idx * rhs_block_stride

        err, block_dot_q32 = ref.dot_product_block_q32(
            lhs_blocks[lhs_index][0],
            lhs_blocks[lhs_index][1],
            rhs_blocks[rhs_index][0],
            rhs_blocks[rhs_index][1],
        )
        if err != ref.Q4_0_Q8_0_OK:
            return err

        ok, next_total = ref.try_add_i64(staged, block_dot_q32)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW
        staged = next_total

    out_dot[0] = staged
    return ref.Q4_0_Q8_0_OK


def explicit_expected(lhs_blocks, lhs_stride: int, rhs_blocks, rhs_stride: int, block_count: int):
    total = 0
    for block_idx in range(block_count):
        lhs_index = block_idx * lhs_stride
        rhs_index = block_idx * rhs_stride

        err, block_dot_q32 = ref.dot_product_block_q32(
            lhs_blocks[lhs_index][0],
            lhs_blocks[lhs_index][1],
            rhs_blocks[rhs_index][0],
            rhs_blocks[rhs_index][1],
        )
        if err != ref.Q4_0_Q8_0_OK:
            return err, 0

        ok, total = ref.try_add_i64(total, block_dot_q32)
        if not ok:
            return ref.Q4_0_Q8_0_ERR_OVERFLOW, 0

    return ref.Q4_0_Q8_0_OK, total


def make_q4_block(rng: random.Random):
    scale = rng.uniform(-2.0, 2.0)
    scale_fp16 = ref.half_bits(scale)
    q_signed = [rng.randint(-8, 7) for _ in range(32)]
    return scale_fp16, ref.pack_q4_from_signed(q_signed)


def make_q8_block(rng: random.Random):
    scale = rng.uniform(-2.0, 2.0)
    scale_fp16 = ref.half_bits(scale)
    q_signed = [rng.randint(-127, 127) for _ in range(32)]
    return scale_fp16, ref.pack_q8_signed(q_signed)


def test_source_contains_iq989_checked_single_dot() -> None:
    source = Path("src/quant/q4_0_q8_0_dot.HC").read_text(encoding="utf-8")

    sig = "I32 Q4_0Q8_0DotQ32CheckedNoPartial("
    assert sig in source
    body = source.split(sig, 1)[1]

    assert "status = Q4_0Q8_0DotQ32CheckedNoPartialArrayDotOne(" in body
    assert "*out_dot_q32 = staged_dot_q32;" in body
    assert "if (!block_count)" in body
    assert "if ((I64 *)lhs_q4 == out_dot_q32 || (I64 *)rhs_q8 == out_dot_q32)" in body


def test_null_and_shape_guards() -> None:
    rng = random.Random(989)
    lhs_blocks = [make_q4_block(rng) for _ in range(8)]
    rhs_blocks = [make_q8_block(rng) for _ in range(8)]
    out = [12345]

    assert (
        q4_0_q8_0_dot_q32_checked_nopartial(
            None,
            len(lhs_blocks),
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            2,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_NULL_PTR
    )

    assert (
        q4_0_q8_0_dot_q32_checked_nopartial(
            lhs_blocks,
            len(lhs_blocks),
            0,
            rhs_blocks,
            len(rhs_blocks),
            1,
            2,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_q8_0_dot_q32_checked_nopartial(
            lhs_blocks,
            len(lhs_blocks),
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            -1,
            out,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )

    assert (
        q4_0_q8_0_dot_q32_checked_nopartial(
            lhs_blocks,
            len(lhs_blocks),
            1,
            rhs_blocks,
            len(rhs_blocks),
            1,
            0,
            lhs_blocks,
        )
        == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    )


def test_preflight_failure_keeps_output_unchanged() -> None:
    rng = random.Random(20260422)
    lhs_blocks = [make_q4_block(rng) for _ in range(4)]
    rhs_blocks = [make_q8_block(rng) for _ in range(4)]
    out = [777777]
    before = out[:]

    err = q4_0_q8_0_dot_q32_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity=4,
        lhs_block_stride=2,
        rhs_blocks=rhs_blocks,
        rhs_block_capacity=4,
        rhs_block_stride=2,
        block_count=3,
        out_dot=out,
    )

    assert err == ref.Q4_0_Q8_0_ERR_BAD_DST_LEN
    assert out == before


def test_overflow_preflight_keeps_output_unchanged() -> None:
    lhs_overflow = (0x7C00, ref.pack_q4_from_signed([7] * 32))
    rhs_overflow = (0x7C00, ref.pack_q8_signed([127] * 32))

    out = [24680]
    before = out[:]

    err = q4_0_q8_0_dot_q32_checked_nopartial(
        [lhs_overflow],
        lhs_block_capacity=1,
        lhs_block_stride=1,
        rhs_blocks=[rhs_overflow],
        rhs_block_capacity=1,
        rhs_block_stride=1,
        block_count=1,
        out_dot=out,
    )

    assert err == ref.Q4_0_Q8_0_ERR_OVERFLOW
    assert out == before


def test_known_vector_and_random_parity() -> None:
    rng = random.Random(909989)

    lhs_blocks = [
        (ref.half_bits(1.0), ref.pack_q4_from_signed([((i % 16) - 8) for i in range(32)])),
        (ref.half_bits(-0.5), ref.pack_q4_from_signed([7 - (i % 16) for i in range(32)])),
        (ref.half_bits(0.25), ref.pack_q4_from_signed([((i % 8) - 4) for i in range(32)])),
    ]
    rhs_blocks = [
        (ref.half_bits(0.75), ref.pack_q8_signed([((i % 11) - 5) * 3 for i in range(32)])),
        (ref.half_bits(-1.25), ref.pack_q8_signed([7 - (i % 13) for i in range(32)])),
        (ref.half_bits(2.0), ref.pack_q8_signed([((i % 9) - 4) * 4 for i in range(32)])),
    ]

    out = [0]
    err = q4_0_q8_0_dot_q32_checked_nopartial(
        lhs_blocks,
        lhs_block_capacity=3,
        lhs_block_stride=1,
        rhs_blocks=rhs_blocks,
        rhs_block_capacity=3,
        rhs_block_stride=1,
        block_count=3,
        out_dot=out,
    )
    assert err == ref.Q4_0_Q8_0_OK

    expected_err, expected = explicit_expected(lhs_blocks, 1, rhs_blocks, 1, 3)
    assert expected_err == ref.Q4_0_Q8_0_OK
    assert out[0] == expected

    for _ in range(260):
        block_count = rng.randint(0, 6)
        lhs_stride = rng.randint(1, 3)
        rhs_stride = rng.randint(1, 3)

        lhs_required = 0 if block_count == 0 else (block_count - 1) * lhs_stride + 1
        rhs_required = 0 if block_count == 0 else (block_count - 1) * rhs_stride + 1

        lhs_capacity = lhs_required + rng.randint(0, 3)
        rhs_capacity = rhs_required + rng.randint(0, 3)

        lhs_blocks = [make_q4_block(rng) for _ in range(max(lhs_capacity, 1))]
        rhs_blocks = [make_q8_block(rng) for _ in range(max(rhs_capacity, 1))]

        out = [123456789]
        out_before = out[:]

        err = q4_0_q8_0_dot_q32_checked_nopartial(
            lhs_blocks,
            lhs_block_capacity=lhs_capacity,
            lhs_block_stride=lhs_stride,
            rhs_blocks=rhs_blocks,
            rhs_block_capacity=rhs_capacity,
            rhs_block_stride=rhs_stride,
            block_count=block_count,
            out_dot=out,
        )

        if block_count == 0:
            assert err == ref.Q4_0_Q8_0_OK
            assert out[0] == 0
            continue

        expected_err, expected = explicit_expected(lhs_blocks, lhs_stride, rhs_blocks, rhs_stride, block_count)
        assert err == expected_err

        if err == ref.Q4_0_Q8_0_OK:
            assert out[0] == expected
        else:
            assert out == out_before


def main() -> None:
    test_source_contains_iq989_checked_single_dot()
    test_null_and_shape_guards()
    test_preflight_failure_keeps_output_unchanged()
    test_overflow_preflight_keeps_output_unchanged()
    test_known_vector_and_random_parity()
    print("q4_0_q8_0_dot_q32_checked_nopartial=ok")


if __name__ == "__main__":
    main()
