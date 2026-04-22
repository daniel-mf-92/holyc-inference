#!/usr/bin/env python3
"""Parity harness for Q4_0DotBlocksQ32CheckedNoPartial (IQ-1140)."""

from __future__ import annotations

from pathlib import Path

Q4_0_OK = 0
Q4_0_ERR_NULL_PTR = 1
Q4_0_ERR_BAD_DST_LEN = 2
Q4_0_ERR_OVERFLOW = 3

I64_MIN = -(1 << 63)
I64_MAX = (1 << 63) - 1


def q4_0_round_shift_right_unsigned(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return (value + (1 << (shift - 1))) >> shift


def q4_0_round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if value >= 0:
        return q4_0_round_shift_right_unsigned(value, shift)
    return -q4_0_round_shift_right_unsigned(-value, shift)


def q4_0_dot_q32_to_q16(dot_q32: int) -> int:
    return q4_0_round_shift_right_signed(dot_q32, 16)


def q4_0_try_add_i64(lhs: int, rhs: int) -> tuple[bool, int]:
    total = lhs + rhs
    if total < I64_MIN or total > I64_MAX:
        return False, 0
    return True, total


def q4_0_dot_blocks_q32_checked_nopartial(
    block_dots_q32: list[int] | None,
    block_count: int,
    out_dot_q32: list[int] | None,
    out_dot_q16: list[int] | None,
) -> int:
    if block_dots_q32 is None or out_dot_q32 is None or out_dot_q16 is None:
        return Q4_0_ERR_NULL_PTR
    if block_count < 0:
        return Q4_0_ERR_BAD_DST_LEN

    staged_total = 0
    for index in range(block_count):
        ok, next_total = q4_0_try_add_i64(staged_total, block_dots_q32[index])
        if not ok:
            return Q4_0_ERR_OVERFLOW
        staged_total = next_total

    staged_q16 = q4_0_dot_q32_to_q16(staged_total)
    out_dot_q32[0] = staged_total
    out_dot_q16[0] = staged_q16
    return Q4_0_OK


def test_source_contains_iq1140_kernel_and_checked_add() -> None:
    source = Path("src/quant/q4_0_dot.HC").read_text(encoding="utf-8")

    assert "#define Q4_0_ERR_OVERFLOW     3" in source
    assert "Bool Q4_0TryAddI64(I64 lhs, I64 rhs, I64 *out_sum)" in source

    sig = "I32 Q4_0DotBlocksQ32CheckedNoPartial(Q4_0Block *lhs,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 Q4_0DotQ32ToQ16(", 1)[0]

    assert "if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)" in body
    assert "if (!Q4_0TryAddI64(staged_total_q32, block_dot_q32, &next_total_q32))" in body
    assert "staged_total_q16 = Q4_0DotQ32ToQ16(staged_total_q32);" in body
    assert "*out_dot_q32 = staged_total_q32;" in body
    assert "*out_dot_q16 = staged_total_q16;" in body


def test_bad_inputs_and_no_partial_publish() -> None:
    out_q32 = [123]
    out_q16 = [456]

    err = q4_0_dot_blocks_q32_checked_nopartial(None, 1, out_q32, out_q16)
    assert err == Q4_0_ERR_NULL_PTR
    assert out_q32 == [123]
    assert out_q16 == [456]

    err = q4_0_dot_blocks_q32_checked_nopartial([1], -1, out_q32, out_q16)
    assert err == Q4_0_ERR_BAD_DST_LEN
    assert out_q32 == [123]
    assert out_q16 == [456]


def test_happy_path_and_single_rounding_downshift() -> None:
    block_dots = [1 << 15, 1 << 16, -(1 << 15), 7 << 16]
    out_q32 = [0]
    out_q16 = [0]

    err = q4_0_dot_blocks_q32_checked_nopartial(
        block_dots,
        len(block_dots),
        out_q32,
        out_q16,
    )
    assert err == Q4_0_OK

    expected_q32 = sum(block_dots)
    assert out_q32[0] == expected_q32
    assert out_q16[0] == q4_0_dot_q32_to_q16(expected_q32)


def test_checked_accumulator_overflow_rejected_without_publish() -> None:
    out_q32 = [999]
    out_q16 = [888]

    err = q4_0_dot_blocks_q32_checked_nopartial(
        [I64_MAX - 10, 42],
        2,
        out_q32,
        out_q16,
    )
    assert err == Q4_0_ERR_OVERFLOW
    assert out_q32 == [999]
    assert out_q16 == [888]


def test_empty_block_count_is_valid_zero_dot() -> None:
    out_q32 = [111]
    out_q16 = [222]

    err = q4_0_dot_blocks_q32_checked_nopartial([], 0, out_q32, out_q16)
    assert err == Q4_0_OK
    assert out_q32[0] == 0
    assert out_q16[0] == 0


if __name__ == "__main__":
    test_source_contains_iq1140_kernel_and_checked_add()
    test_bad_inputs_and_no_partial_publish()
    test_happy_path_and_single_rounding_downshift()
    test_checked_accumulator_overflow_rejected_without_publish()
    test_empty_block_count_is_valid_zero_dot()
    print("q4_0_dot_blocks_q32_checked_nopartial=ok")
