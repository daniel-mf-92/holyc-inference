#!/usr/bin/env python3
"""Parity harness for FPQ16MulCheckedNoPartialArrayRequiredBytes (IQ-966)."""

from __future__ import annotations

import random
from pathlib import Path

FP_Q16_SHIFT = 16
FP_Q16_OK = 0
FP_Q16_ERR_NULL_PTR = 1
FP_Q16_ERR_BAD_PARAM = 2
FP_Q16_ERR_OVERFLOW = 4

I64_MAX_VALUE = (1 << 63) - 1
I64_MIN_VALUE = -(1 << 63)
U64_MAX_VALUE = (1 << 64) - 1


def fp_abs_to_u64(x: int) -> int:
    if x >= 0:
        return x
    return (-(x + 1)) + 1


def fp_try_apply_sign_from_u64_checked(mag: int, is_negative: bool) -> tuple[int, int]:
    limit = 1 << 63 if is_negative else I64_MAX_VALUE
    if mag > limit:
        return FP_Q16_ERR_OVERFLOW, 0
    if is_negative:
        if mag >= (1 << 63):
            return FP_Q16_OK, I64_MIN_VALUE
        return FP_Q16_OK, -mag
    return FP_Q16_OK, mag


def fpq16_mul_checked(a_q16: int, b_q16: int) -> tuple[int, int]:
    if a_q16 == 0 or b_q16 == 0:
        return FP_Q16_OK, 0

    abs_a = fp_abs_to_u64(a_q16)
    abs_b = fp_abs_to_u64(b_q16)
    is_negative = (a_q16 < 0) ^ (b_q16 < 0)

    if abs_a > (U64_MAX_VALUE // abs_b):
        return FP_Q16_ERR_OVERFLOW, 0

    abs_prod = abs_a * abs_b
    round_bias = 1 << (FP_Q16_SHIFT - 1)

    if abs_prod > U64_MAX_VALUE - round_bias:
        rounded_mag = U64_MAX_VALUE >> FP_Q16_SHIFT
    else:
        rounded_mag = (abs_prod + round_bias) >> FP_Q16_SHIFT

    return fp_try_apply_sign_from_u64_checked(rounded_mag, is_negative)


def fpq16_mul_checked_no_partial_array_required_bytes(
    lhs_q16: list[int] | None,
    rhs_q16: list[int] | None,
    out_q16: list[int] | None,
    count: int,
    out_required_cells_present: bool,
    out_required_bytes_present: bool,
    *,
    alias_lhs_rhs: bool = False,
    alias_lhs_out: bool = False,
    alias_rhs_out: bool = False,
    alias_cells_bytes: bool = False,
    diag_alias_lhs: bool = False,
    diag_alias_rhs: bool = False,
    diag_alias_out: bool = False,
) -> tuple[int, int, int]:
    if (
        lhs_q16 is None
        or rhs_q16 is None
        or out_q16 is None
        or not out_required_cells_present
        or not out_required_bytes_present
    ):
        return FP_Q16_ERR_NULL_PTR, -1, -1

    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, -1, -1

    if alias_lhs_rhs or alias_lhs_out or alias_rhs_out:
        return FP_Q16_ERR_BAD_PARAM, -1, -1

    if alias_cells_bytes or diag_alias_lhs or diag_alias_rhs or diag_alias_out:
        return FP_Q16_ERR_BAD_PARAM, -1, -1

    if count > len(lhs_q16) or count > len(rhs_q16) or count > len(out_q16):
        return FP_Q16_ERR_BAD_PARAM, -1, -1

    for i in range(count):
        err, _ = fpq16_mul_checked(lhs_q16[i], rhs_q16[i])
        if err != FP_Q16_OK:
            return err, -1, -1

    required_cells = count
    if required_cells > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, -1, -1
    required_bytes = required_cells << 3

    return FP_Q16_OK, required_cells, required_bytes


def test_source_contains_required_bytes_helper_contract() -> None:
    source = Path("src/math/fixedpoint.HC").read_text(encoding="utf-8")
    sig = "I32 FPQ16MulCheckedNoPartialArrayRequiredBytes(I64 *lhs_q16,"
    assert sig in source
    body = source.split(sig, 1)[1].split("I32 FPQ16MulSatCheckedNoPartialArray", 1)[0]
    assert "if (!lhs_q16 || !rhs_q16 || !out_q16 ||" in body
    assert "!out_required_cells || !out_required_bytes" in body
    assert "if (lhs_q16 == rhs_q16 || lhs_q16 == out_q16 || rhs_q16 == out_q16)" in body
    assert "status = FPArrayI64SpanChecked(lhs_q16, count, &lhs_base, &lhs_end);" in body
    assert "FPAddressRangesOverlap(lhs_base, lhs_end, rhs_base, rhs_end)" in body
    assert "status = FPQ16MulChecked(lhs_q16[i]," in body
    assert "*out_required_cells = staged_required_cells;" in body
    assert "*out_required_bytes = staged_required_bytes;" in body


def test_null_bad_param_alias_and_no_write_on_hard_fail() -> None:
    lhs = [1 << FP_Q16_SHIFT, 2 << FP_Q16_SHIFT]
    rhs = [3 << FP_Q16_SHIFT, 4 << FP_Q16_SHIFT]
    out = [0x55, 0x66]

    status, req_cells, req_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
        None, rhs, out, 2, True, True
    )
    assert status == FP_Q16_ERR_NULL_PTR
    assert req_cells == -1 and req_bytes == -1
    assert out == [0x55, 0x66]

    status, req_cells, req_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
        lhs, rhs, out, -1, True, True
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert req_cells == -1 and req_bytes == -1
    assert out == [0x55, 0x66]

    status, req_cells, req_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
        lhs, rhs, out, 2, True, True, alias_lhs_out=True
    )
    assert status == FP_Q16_ERR_BAD_PARAM
    assert req_cells == -1 and req_bytes == -1
    assert out == [0x55, 0x66]


def test_known_vectors_required_tuple_and_overflow_passthrough() -> None:
    lhs_ok = [1 << FP_Q16_SHIFT, -(2 << FP_Q16_SHIFT), 3 << FP_Q16_SHIFT]
    rhs_ok = [3 << FP_Q16_SHIFT, 4 << FP_Q16_SHIFT, -(5 << FP_Q16_SHIFT)]
    out_ok = [0xAB, 0xCD, 0xEF]

    status_ok, required_cells_ok, required_bytes_ok = fpq16_mul_checked_no_partial_array_required_bytes(
        lhs_ok, rhs_ok, out_ok, 3, True, True
    )
    assert status_ok == FP_Q16_OK
    assert required_cells_ok == 3
    assert required_bytes_ok == 24
    assert out_ok == [0xAB, 0xCD, 0xEF]

    lhs_bad = [I64_MAX_VALUE]
    rhs_bad = [I64_MAX_VALUE]
    out_bad = [0x77]
    status_bad, required_cells_bad, required_bytes_bad = fpq16_mul_checked_no_partial_array_required_bytes(
        lhs_bad, rhs_bad, out_bad, 1, True, True
    )
    assert status_bad == FP_Q16_ERR_OVERFLOW
    assert required_cells_bad == -1
    assert required_bytes_bad == -1
    assert out_bad == [0x77]


def test_randomized_reference_parity_alias_and_overflow_vectors() -> None:
    rng = random.Random(20260421_966)

    for _ in range(6000):
        count = rng.randint(0, 64)
        lhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        rhs = [rng.randint(I64_MIN_VALUE, I64_MAX_VALUE) for _ in range(count)]
        out = [0x7A7A7A7A] * count

        alias_mode = rng.randint(0, 6)
        kwargs = {
            "alias_lhs_rhs": alias_mode == 1,
            "alias_lhs_out": alias_mode == 2,
            "alias_rhs_out": alias_mode == 3,
            "alias_cells_bytes": alias_mode == 4,
            "diag_alias_lhs": alias_mode == 5,
            "diag_alias_rhs": alias_mode == 6,
        }

        exp_status, exp_cells, exp_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
            lhs.copy(), rhs.copy(), out.copy(), count, True, True, **kwargs
        )
        got_status, got_cells, got_bytes = fpq16_mul_checked_no_partial_array_required_bytes(
            lhs.copy(), rhs.copy(), out, count, True, True, **kwargs
        )

        assert got_status == exp_status
        assert got_cells == exp_cells
        assert got_bytes == exp_bytes
        assert out == [0x7A7A7A7A] * count


def run() -> None:
    test_source_contains_required_bytes_helper_contract()
    test_null_bad_param_alias_and_no_write_on_hard_fail()
    test_known_vectors_required_tuple_and_overflow_passthrough()
    test_randomized_reference_parity_alias_and_overflow_vectors()
    print("fixedpoint_q16_mul_checked_no_partial_array_required_bytes=ok")


if __name__ == "__main__":
    run()
