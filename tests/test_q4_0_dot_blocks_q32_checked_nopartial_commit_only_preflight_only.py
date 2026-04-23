#!/usr/bin/env python3
"""Parity harness for Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnly (IQ-1198)."""

from pathlib import Path

Q4_0_OK = 0
Q4_0_ERR_NULL_PTR = 1
Q4_0_ERR_BAD_DST_LEN = 2
Q4_0_ERR_OVERFLOW = 3

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)


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

    out_dot_q32[0] = staged_total
    out_dot_q16[0] = q4_0_dot_q32_to_q16(staged_total)
    return Q4_0_OK


def q4_0_dot_blocks_q32_checked_nopartial_commit_only(
    block_dots_q32: list[int] | None,
    block_count: int,
    out_dot_q32: list[int] | None,
    out_dot_q16: list[int] | None,
) -> int:
    if block_dots_q32 is None or out_dot_q32 is None or out_dot_q16 is None:
        return Q4_0_ERR_NULL_PTR
    if block_count < 0:
        return Q4_0_ERR_BAD_DST_LEN
    if out_dot_q32 is out_dot_q16:
        return Q4_0_ERR_BAD_DST_LEN

    staged_q32 = [0]
    staged_q16 = [0]
    status = q4_0_dot_blocks_q32_checked_nopartial(block_dots_q32, block_count, staged_q32, staged_q16)
    if status != Q4_0_OK:
        return status

    out_dot_q32[0] = staged_q32[0]
    out_dot_q16[0] = staged_q16[0]
    return Q4_0_OK


def q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
    block_dots_q32: list[int] | None,
    block_count: int,
    out_dot_q32: list[int] | None,
    out_dot_q16: list[int] | None,
) -> int:
    if block_dots_q32 is None or out_dot_q32 is None or out_dot_q16 is None:
        return Q4_0_ERR_NULL_PTR
    if block_count < 0:
        return Q4_0_ERR_BAD_DST_LEN
    if out_dot_q32 is out_dot_q16:
        return Q4_0_ERR_BAD_DST_LEN

    staged_q32 = [0]
    staged_q16 = [0]
    status = q4_0_dot_blocks_q32_checked_nopartial_commit_only(
        block_dots_q32,
        block_count,
        staged_q32,
        staged_q16,
    )
    if status != Q4_0_OK:
        return status

    canonical_q32 = [0]
    canonical_q16 = [0]
    status = q4_0_dot_blocks_q32_checked_nopartial(
        block_dots_q32,
        block_count,
        canonical_q32,
        canonical_q16,
    )
    if status != Q4_0_OK:
        return status

    if canonical_q32[0] != staged_q32[0] or canonical_q16[0] != staged_q16[0]:
        return Q4_0_ERR_BAD_DST_LEN

    return Q4_0_OK


def test_source_contains_iq1198_preflight_only_wrapper() -> None:
    source = Path("src/quant/q4_0_dot.HC").read_text(encoding="utf-8")

    sig = "I32 Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnly(Q4_0Block *lhs,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 Q4_0DotQ32ToQ16(", 1)[0]

    assert "if (!lhs || !rhs || !out_dot_q32 || !out_dot_q16)" in body
    assert "if (block_count < 0)" in body
    assert "if (out_dot_q32 == out_dot_q16)" in body
    assert "snapshot_lhs_ptr = lhs;" in body
    assert "snapshot_rhs_ptr = rhs;" in body
    assert "snapshot_block_count = block_count;" in body
    assert "status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnly(lhs," in body
    assert "status = Q4_0DotBlocksQ32CheckedNoPartial(lhs," in body
    assert "if (canonical_dot_q32 != staged_dot_q32 ||" in body
    assert "canonical_dot_q16 != staged_dot_q16)" in body


def test_rejects_null_and_alias_without_partial_write() -> None:
    out_q32 = [111]
    out_q16 = [222]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
        None, 1, out_q32, out_q16
    )
    assert err == Q4_0_ERR_NULL_PTR
    assert out_q32 == [111]
    assert out_q16 == [222]

    shared = [333]
    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
        [1], 1, shared, shared
    )
    assert err == Q4_0_ERR_BAD_DST_LEN
    assert shared == [333]


def test_overflow_and_no_publish() -> None:
    out_q32 = [555]
    out_q16 = [666]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
        [I64_MAX - 3, 9],
        2,
        out_q32,
        out_q16,
    )
    assert err == Q4_0_ERR_OVERFLOW
    assert out_q32 == [555]
    assert out_q16 == [666]


def test_success_parity_vectors() -> None:
    out_q32 = [444]
    out_q16 = [555]
    block_dots = [7 << 16, -(3 << 15), 5, -(9 << 16)]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
        block_dots,
        len(block_dots),
        out_q32,
        out_q16,
    )
    assert err == Q4_0_OK
    assert out_q32 == [444]
    assert out_q16 == [555]


def test_zero_block_count_ok_no_publish() -> None:
    out_q32 = [888]
    out_q16 = [999]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
        [],
        0,
        out_q32,
        out_q16,
    )
    assert err == Q4_0_OK
    assert out_q32 == [888]
    assert out_q16 == [999]


if __name__ == "__main__":
    test_source_contains_iq1198_preflight_only_wrapper()
    test_rejects_null_and_alias_without_partial_write()
    test_overflow_and_no_publish()
    test_success_parity_vectors()
    test_zero_block_count_ok_no_publish()
    print("q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only=ok")
