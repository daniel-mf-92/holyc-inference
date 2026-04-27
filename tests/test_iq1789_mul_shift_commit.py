#!/usr/bin/env python3
"""IQ-1789 harness for mul-shift diagnostics commit/parity wrappers."""

from __future__ import annotations

from pathlib import Path

FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4
I64_MAX_VALUE = (1 << 63) - 1


def mul_shift_round_checked(lhs: int, rhs: int, shift: int) -> tuple[int, int]:
    if shift < 0 or shift > 62:
        return FP_Q16_ERR_BAD_PARAM, 0
    if lhs == 0 or rhs == 0:
        return FP_Q16_OK, 0

    mag_lhs = abs(lhs)
    mag_rhs = abs(rhs)
    if mag_lhs > ((1 << 64) - 1) // mag_rhs:
        return FP_Q16_ERR_OVERFLOW, 0

    product = mag_lhs * mag_rhs
    quot = product >> shift if shift else product
    if shift:
        rem = product & (((1 << shift) - 1))
        half = 1 << (shift - 1)
        if rem >= half:
            quot += 1

    sign_neg = (lhs < 0) ^ (rhs < 0)
    signed_limit = 1 << 63 if sign_neg else I64_MAX_VALUE
    if quot > signed_limit:
        return FP_Q16_ERR_OVERFLOW, 0

    return FP_Q16_OK, -quot if sign_neg else quot


def mul_shift_commit_only_preflight_only_model(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    out_required_lhs: list[int],
    out_required_rhs: list[int],
    out_required_out: list[int],
) -> int:
    if count < 0 or count > len(lhs) or count > len(rhs) or count > len(out):
        return FP_Q16_ERR_BAD_PARAM

    staged_out = list(out)
    for lane in range(count):
        status, value = mul_shift_round_checked(lhs[lane], rhs[lane], shift)
        if status != FP_Q16_OK:
            return status
        staged_out[lane] = value

    if out_required_lhs is out_required_rhs or out_required_lhs is out_required_out or out_required_rhs is out_required_out:
        return FP_Q16_ERR_BAD_PARAM

    out_required_lhs[0] = count
    out_required_rhs[0] = count
    out_required_out[0] = count
    return FP_Q16_OK


def mul_shift_commit_only_model(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    out_required_lhs: list[int],
    out_required_rhs: list[int],
    out_required_out: list[int],
) -> int:
    if count < 0 or count > len(lhs) or count > len(rhs) or count > len(out):
        return FP_Q16_ERR_BAD_PARAM

    staged = list(out)
    for lane in range(count):
        status, value = mul_shift_round_checked(lhs[lane], rhs[lane], shift)
        if status != FP_Q16_OK:
            return status
        staged[lane] = value

    out[:count] = staged[:count]
    out_required_lhs[0] = count
    out_required_rhs[0] = count
    out_required_out[0] = count
    return FP_Q16_OK


def mul_shift_parity_model(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    out_required_lhs: list[int],
    out_required_rhs: list[int],
    out_required_out: list[int],
) -> int:
    a_l = [0]
    a_r = [0]
    a_o = [0]
    status = mul_shift_commit_only_preflight_only_model(lhs, rhs, out, count, shift, a_l, a_r, a_o)
    if status != FP_Q16_OK:
        return status

    b_l = [0]
    b_r = [0]
    b_o = [0]
    status = mul_shift_commit_only_model(lhs, rhs, out, count, shift, b_l, b_r, b_o)
    if status != FP_Q16_OK:
        return status

    if (a_l[0], a_r[0], a_o[0]) != (b_l[0], b_r[0], b_o[0]):
        return FP_Q16_ERR_BAD_PARAM

    out_required_lhs[0] = a_l[0]
    out_required_rhs[0] = a_r[0]
    out_required_out[0] = a_o[0]
    return FP_Q16_OK


def mul_shift_commit_parity_model(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    out_required_lhs: list[int],
    out_required_rhs: list[int],
    out_required_out: list[int],
) -> int:
    p_l = [0]
    p_r = [0]
    p_o = [0]
    status = mul_shift_parity_model(lhs, rhs, out, count, shift, p_l, p_r, p_o)
    if status != FP_Q16_OK:
        return status

    f_l = [0]
    f_r = [0]
    f_o = [0]
    status = mul_shift_commit_only_preflight_only_model(lhs, rhs, out, count, shift, f_l, f_r, f_o)
    if status != FP_Q16_OK:
        return status

    if (p_l[0], p_r[0], p_o[0]) != (f_l[0], f_r[0], f_o[0]):
        return FP_Q16_ERR_BAD_PARAM

    out_required_lhs[0] = p_l[0]
    out_required_rhs[0] = p_r[0]
    out_required_out[0] = p_o[0]
    return FP_Q16_OK


def _extract_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    idx = brace + 1
    while depth:
        ch = source[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    return source[brace + 1 : idx - 1]


def test_source_contains_iq1789_aliases_and_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    assert "#define FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnlyParity FPMulShiftRoundParity" in source
    assert "#define FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnly FPMulShiftRoundCommitParity" in source
    body = _extract_body(source, "I32 FPMulShiftRoundCommitParity(")
    assert "status = FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnlyParity(" in body
    assert "status = FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnly(" in body
    assert "if (staged_parity_required_lhs != staged_preflight_required_lhs ||" in body


def test_capacity_underflow_rejected() -> None:
    lhs = [1, 2]
    rhs = [3, 4]
    out = [0, 0]
    req_l = [99]
    req_r = [99]
    req_o = [99]

    status = mul_shift_commit_parity_model(lhs, rhs, out, 3, 1, req_l, req_r, req_o)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert req_l == [99] and req_r == [99] and req_o == [99]


def test_shift_domain_rejected() -> None:
    lhs = [5]
    rhs = [6]
    out = [777]
    req_l = [0]
    req_r = [0]
    req_o = [0]

    status = mul_shift_commit_parity_model(lhs, rhs, out, 1, -1, req_l, req_r, req_o)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [777]


def test_overflow_rejected_without_publish() -> None:
    lhs = [1 << 62]
    rhs = [8]
    out = [1234]
    req_l = [55]
    req_r = [66]
    req_o = [77]

    status = mul_shift_commit_parity_model(lhs, rhs, out, 1, 0, req_l, req_r, req_o)
    assert status == FP_Q16_ERR_OVERFLOW
    assert req_l == [55] and req_r == [66] and req_o == [77]


def test_deterministic_tuple_parity_success() -> None:
    lhs = [13, -11, 7, -9]
    rhs = [5, 3, -12, -8]
    out = [900, 901, 902, 903]
    req_l = [0]
    req_r = [0]
    req_o = [0]

    status = mul_shift_commit_parity_model(lhs, rhs, out, 4, 2, req_l, req_r, req_o)
    assert status == FP_Q16_OK
    assert (req_l[0], req_r[0], req_o[0]) == (4, 4, 4)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
