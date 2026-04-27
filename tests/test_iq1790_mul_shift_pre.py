#!/usr/bin/env python3
"""IQ-1790 harness for mul-shift zero-write preflight wrapper."""

from __future__ import annotations

from pathlib import Path

FP_Q16_OK = 0
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


def model_commit_parity(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    req_l: list[int],
    req_r: list[int],
    req_o: list[int],
) -> int:
    if count < 0 or count > len(lhs) or count > len(rhs) or count > len(out):
        return FP_Q16_ERR_BAD_PARAM

    staged = list(out)
    for lane in range(count):
        status, value = mul_shift_round_checked(lhs[lane], rhs[lane], shift)
        if status != FP_Q16_OK:
            return status
        staged[lane] = value

    req_l[0] = count
    req_r[0] = count
    req_o[0] = count
    return FP_Q16_OK


def model_parity(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    req_l: list[int],
    req_r: list[int],
    req_o: list[int],
) -> int:
    return model_commit_parity(lhs, rhs, out, count, shift, req_l, req_r, req_o)


def model_commit_pre(
    lhs: list[int],
    rhs: list[int],
    out: list[int],
    count: int,
    shift: int,
    req_l: list[int],
    req_r: list[int],
    req_o: list[int],
) -> int:
    if req_l is req_r or req_l is req_o or req_r is req_o:
        return FP_Q16_ERR_BAD_PARAM

    out_before = list(out)

    c_l = [0]
    c_r = [0]
    c_o = [0]
    status = model_commit_parity(lhs, rhs, list(out), count, shift, c_l, c_r, c_o)
    if status != FP_Q16_OK:
        return status

    p_l = [0]
    p_r = [0]
    p_o = [0]
    status = model_parity(lhs, rhs, list(out), count, shift, p_l, p_r, p_o)
    if status != FP_Q16_OK:
        return status

    if (c_l[0], c_r[0], c_o[0]) != (p_l[0], p_r[0], p_o[0]):
        return FP_Q16_ERR_BAD_PARAM

    if out != out_before:
        return FP_Q16_ERR_BAD_PARAM

    req_l[0] = c_l[0]
    req_r[0] = c_r[0]
    req_o[0] = c_o[0]
    return FP_Q16_OK


def _extract_body(source: str, sig: str) -> str:
    start = source.index(sig)
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


def test_source_has_iq1790_wrapper() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    assert "#define FPQ16MulShiftRoundCheckedNoPartialArrayCommitOnlyPreflightOnlyParityCommitOnlyPreflightOnly FPMulShiftRoundCommitPre" in source
    body = _extract_body(source, "I32 FPMulShiftRoundCommitPre(")
    assert "status = FPMulShiftRoundCommitParity(" in body
    assert "status = FPMulShiftRoundParity(" in body
    assert "staged_commit_out_values = MAlloc(stage_bytes);" in body


def test_alias_rejected() -> None:
    lhs = [1, 2]
    rhs = [3, 4]
    out = [9, 9]
    req = [0]

    status = model_commit_pre(lhs, rhs, out, 2, 1, req, req, [0])
    assert status == FP_Q16_ERR_BAD_PARAM


def test_capacity_underflow_rejected() -> None:
    lhs = [1, 2]
    rhs = [3, 4]
    out = [0, 0]
    req_l = [9]
    req_r = [8]
    req_o = [7]

    status = model_commit_pre(lhs, rhs, out, 3, 1, req_l, req_r, req_o)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert (req_l[0], req_r[0], req_o[0]) == (9, 8, 7)


def test_shift_domain_rejected() -> None:
    lhs = [5]
    rhs = [6]
    out = [77]
    req_l = [0]
    req_r = [0]
    req_o = [0]

    status = model_commit_pre(lhs, rhs, out, 1, -1, req_l, req_r, req_o)
    assert status == FP_Q16_ERR_BAD_PARAM
    assert out == [77]


def test_overflow_rejected_and_no_write() -> None:
    lhs = [1 << 62]
    rhs = [8]
    out = [1234]
    req_l = [55]
    req_r = [66]
    req_o = [77]

    status = model_commit_pre(lhs, rhs, out, 1, 0, req_l, req_r, req_o)
    assert status == FP_Q16_ERR_OVERFLOW
    assert out == [1234]
    assert (req_l[0], req_r[0], req_o[0]) == (55, 66, 77)


def test_det_no_write_success() -> None:
    lhs = [13, -11, 7, -9]
    rhs = [5, 3, -12, -8]
    out = [900, 901, 902, 903]
    out_before = list(out)
    req_l = [0]
    req_r = [0]
    req_o = [0]

    status = model_commit_pre(lhs, rhs, out, 4, 2, req_l, req_r, req_o)
    assert status == FP_Q16_OK
    assert out == out_before
    assert (req_l[0], req_r[0], req_o[0]) == (4, 4, 4)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
