#!/usr/bin/env python3
"""Parity checks for FPQ16ExpArrayCheckedNoPartialPreflight."""

from __future__ import annotations

import pathlib
import random
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from test_intexp_exp_array_checked import (
    FP_Q16_ERR_BAD_PARAM,
    FP_Q16_ERR_NULL_PTR,
    FP_Q16_ERR_OVERFLOW,
    FP_Q16_OK,
    I64_MAX_VALUE,
    U64_MAX_VALUE,
)


def fpq16_exp_array_checked_no_partial_preflight(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1600,
    synthetic_output_addr: int = 0x2700,
    out_staged_bytes_present: bool = True,
) -> tuple[int, int]:
    if not out_staged_bytes_present:
        return FP_Q16_ERR_NULL_PTR, 0

    staged_bytes = 0
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, 0
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, 0
    if count == 0:
        return FP_Q16_OK, 0

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW, 0
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW, 0

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0

    staged_bytes = count << 3
    if staged_bytes <= 0:
        return FP_Q16_ERR_OVERFLOW, 0

    return FP_Q16_OK, staged_bytes


def inline_no_partial_preflight_equivalent(
    input_q16: list[int] | None,
    output_q16: list[int] | None,
    count: int,
    *,
    synthetic_input_addr: int = 0x1600,
    synthetic_output_addr: int = 0x2700,
) -> tuple[int, int]:
    if input_q16 is None or output_q16 is None:
        return FP_Q16_ERR_NULL_PTR, 0
    if count < 0:
        return FP_Q16_ERR_BAD_PARAM, 0
    if count == 0:
        return FP_Q16_OK, 0

    last_index = count - 1
    if last_index > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0

    last_byte_offset = last_index << 3
    if synthetic_input_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW, 0
    if synthetic_output_addr > (U64_MAX_VALUE - last_byte_offset):
        return FP_Q16_ERR_OVERFLOW, 0

    if count > (I64_MAX_VALUE >> 3):
        return FP_Q16_ERR_OVERFLOW, 0

    staged_bytes = count << 3
    if staged_bytes <= 0:
        return FP_Q16_ERR_OVERFLOW, 0

    return FP_Q16_OK, staged_bytes


def test_source_contains_preflight_shape_and_wrapper_delegation() -> None:
    source = pathlib.Path("src/math/intexp.HC").read_text(encoding="utf-8")
    assert "FPQ16ExpArrayCheckedNoPartialPreflight" in source
    assert "if (!out_staged_bytes)" in source
    assert "status = FPQ16ExpArrayCheckedNoPartialPreflight(input_q16," in source


def test_null_bad_count_and_zero_surfaces() -> None:
    assert fpq16_exp_array_checked_no_partial_preflight(None, [0], 1)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_no_partial_preflight([0], None, 1)[0] == FP_Q16_ERR_NULL_PTR
    assert fpq16_exp_array_checked_no_partial_preflight([0], [0], -1)[0] == FP_Q16_ERR_BAD_PARAM

    err, staged = fpq16_exp_array_checked_no_partial_preflight([1], [0], 0)
    assert err == FP_Q16_OK
    assert staged == 0

    err, staged = fpq16_exp_array_checked_no_partial_preflight(
        [1],
        [0],
        1,
        out_staged_bytes_present=False,
    )
    assert err == FP_Q16_ERR_NULL_PTR
    assert staged == 0


def test_overflow_and_pointer_guard_parity() -> None:
    inp = [0, 1]
    out = [8, 9]

    huge_count = (I64_MAX_VALUE >> 3) + 2
    got = fpq16_exp_array_checked_no_partial_preflight(inp, out, huge_count)
    ref = inline_no_partial_preflight_equivalent(inp, out, huge_count)
    assert got == ref == (FP_Q16_ERR_OVERFLOW, 0)

    near_end = U64_MAX_VALUE - ((len(inp) - 1) << 3) + 1
    got = fpq16_exp_array_checked_no_partial_preflight(
        inp,
        out,
        len(inp),
        synthetic_output_addr=near_end,
    )
    ref = inline_no_partial_preflight_equivalent(
        inp,
        out,
        len(inp),
        synthetic_output_addr=near_end,
    )
    assert got == ref == (FP_Q16_ERR_OVERFLOW, 0)


def test_boundary_and_randomized_parity_vs_inline_preflight() -> None:
    for count in [1, 2, 31, 32, 33, 64, 127, 1024]:
        inp = [0] * count
        out = [0] * count
        got = fpq16_exp_array_checked_no_partial_preflight(inp, out, count)
        ref = inline_no_partial_preflight_equivalent(inp, out, count)
        assert got == ref
        assert got[0] == FP_Q16_OK
        assert got[1] == count << 3

    rng = random.Random(20260419_517)
    for _ in range(2000):
        count = rng.randint(-8, 1024)
        valid_count = max(0, count)
        inp = [rng.randint(-(1 << 20), (1 << 20)) for _ in range(valid_count)]
        out = [0x44] * valid_count
        in_addr = rng.randrange(0, 1 << 16)
        out_addr = rng.randrange(0, 1 << 16)

        got = fpq16_exp_array_checked_no_partial_preflight(
            inp,
            out,
            count,
            synthetic_input_addr=in_addr,
            synthetic_output_addr=out_addr,
        )
        ref = inline_no_partial_preflight_equivalent(
            inp,
            out,
            count,
            synthetic_input_addr=in_addr,
            synthetic_output_addr=out_addr,
        )

        assert got == ref


def run() -> None:
    test_source_contains_preflight_shape_and_wrapper_delegation()
    test_null_bad_count_and_zero_surfaces()
    test_overflow_and_pointer_guard_parity()
    test_boundary_and_randomized_parity_vs_inline_preflight()
    print("intexp_exp_array_checked_no_partial_preflight=ok")


if __name__ == "__main__":
    run()
