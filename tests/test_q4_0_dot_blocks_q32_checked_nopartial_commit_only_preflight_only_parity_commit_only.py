#!/usr/bin/env python3
"""Parity harness for IQ-1275 Q4_0 commit-only preflight-only parity commit-only gate."""

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


def q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity(
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
    status = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only(
        block_dots_q32,
        block_count,
        staged_q32,
        staged_q16,
    )
    if status != Q4_0_OK:
        return status

    canonical_q32 = [0]
    canonical_q16 = [0]
    status = q4_0_dot_blocks_q32_checked_nopartial_commit_only(
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


def q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
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
    status = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity(
        block_dots_q32,
        block_count,
        staged_q32,
        staged_q16,
    )
    if status != Q4_0_OK:
        return status

    canonical_q32 = [0]
    canonical_q16 = [0]
    status = q4_0_dot_blocks_q32_checked_nopartial_commit_only(
        block_dots_q32,
        block_count,
        canonical_q32,
        canonical_q16,
    )
    if status != Q4_0_OK:
        return status

    if canonical_q32[0] != staged_q32[0] or canonical_q16[0] != staged_q16[0]:
        return Q4_0_ERR_BAD_DST_LEN

    out_dot_q32[0] = canonical_q32[0]
    out_dot_q16[0] = canonical_q16[0]
    return Q4_0_OK


def test_source_contains_iq1275_commit_only_parity_gate() -> None:
    source = Path("src/quant/q4_0_dot.HC").read_text(encoding="utf-8")

    sig = "I32 Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly(Q4_0Block *lhs,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I64 Q4_0DotQ32ToQ16(", 1)[0]

    assert "status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnlyPreflightOnlyParity(lhs," in body
    assert "status = Q4_0DotBlocksQ32CheckedNoPartialCommitOnly(lhs," in body
    assert "if (canonical_dot_q32 != staged_dot_q32 ||" in body
    assert "canonical_dot_q16 != staged_dot_q16)" in body
    assert "*out_dot_q32 = canonical_dot_q32;" in body
    assert "*out_dot_q16 = canonical_dot_q16;" in body


def test_rejects_null_alias_and_negative_without_partial_write() -> None:
    out_q32 = [11]
    out_q16 = [22]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        None, 1, out_q32, out_q16
    )
    assert err == Q4_0_ERR_NULL_PTR
    assert out_q32 == [11]
    assert out_q16 == [22]

    shared = [33]
    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        [1], 1, shared, shared
    )
    assert err == Q4_0_ERR_BAD_DST_LEN
    assert shared == [33]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        [1], -1, out_q32, out_q16
    )
    assert err == Q4_0_ERR_BAD_DST_LEN
    assert out_q32 == [11]
    assert out_q16 == [22]


def test_overflow_no_publish() -> None:
    out_q32 = [77]
    out_q16 = [88]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        [I64_MAX - 2, 7],
        2,
        out_q32,
        out_q16,
    )
    assert err == Q4_0_ERR_OVERFLOW
    assert out_q32 == [77]
    assert out_q16 == [88]


def test_success_publishes_tuple_once() -> None:
    out_q32 = [101]
    out_q16 = [202]
    block_dots = [5 << 16, -(2 << 16), 99, -(4 << 16), 1 << 16]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        block_dots,
        len(block_dots),
        out_q32,
        out_q16,
    )
    assert err == Q4_0_OK

    expect_q32 = [0]
    expect_q16 = [0]
    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only(
        block_dots,
        len(block_dots),
        expect_q32,
        expect_q16,
    )
    assert err == Q4_0_OK

    assert out_q32 == [expect_q32[0]]
    assert out_q16 == [expect_q16[0]]


def test_zero_blocks_publish_zero_tuple() -> None:
    out_q32 = [303]
    out_q16 = [404]

    err = q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only(
        [],
        0,
        out_q32,
        out_q16,
    )
    assert err == Q4_0_OK
    assert out_q32 == [0]
    assert out_q16 == [0]


if __name__ == "__main__":
    test_source_contains_iq1275_commit_only_parity_gate()
    test_rejects_null_alias_and_negative_without_partial_write()
    test_overflow_no_publish()
    test_success_publishes_tuple_once()
    test_zero_blocks_publish_zero_tuple()
    print("q4_0_dot_blocks_q32_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok")
