#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ValidateCodepointScalarChecked."""

from __future__ import annotations

import random

TOKENIZER_UTF8_OK = 0
TOKENIZER_UTF8_ERR_NULL_PTR = 1
TOKENIZER_UTF8_ERR_BAD_PARAM = 2
TOKENIZER_UTF8_ERR_OVERFLOW = 3
TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS = 4
TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE = 5
TOKENIZER_UTF8_ERR_BAD_CONTINUATION = 6
TOKENIZER_UTF8_ERR_BAD_CODEPOINT = 7
TOKENIZER_UTF8_ERR_TRUNCATED = 8


def tokenizer_utf8_validate_codepoint_scalar_checked(
    codepoint: int,
    out_codepoint: list[int] | None,
) -> int:
    if out_codepoint is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if 0xD800 <= codepoint <= 0xDFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    if codepoint > 0x10FFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT

    out_codepoint[0] = codepoint
    return TOKENIZER_UTF8_OK


def ref_validate_scalar(codepoint: int) -> tuple[int, int]:
    if 0xD800 <= codepoint <= 0xDFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT, -1
    if codepoint > 0x10FFFF:
        return TOKENIZER_UTF8_ERR_BAD_CODEPOINT, -1
    return TOKENIZER_UTF8_OK, codepoint


def test_null_ptr_contract() -> None:
    assert (
        tokenizer_utf8_validate_codepoint_scalar_checked(0x41, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )


def test_boundary_vectors() -> None:
    cases = [
        (0x00000000, TOKENIZER_UTF8_OK, 0x1234),
        (0x00000041, TOKENIZER_UTF8_OK, 0x1234),
        (0x0000D7FF, TOKENIZER_UTF8_OK, 0x1234),
        (0x0000D800, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
        (0x0000DBFF, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
        (0x0000DC00, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
        (0x0000DFFF, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
        (0x0000E000, TOKENIZER_UTF8_OK, 0x1234),
        (0x0010FFFF, TOKENIZER_UTF8_OK, 0x1234),
        (0x00110000, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
        (0x7FFFFFFF, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
        (0xFFFFFFFF, TOKENIZER_UTF8_ERR_BAD_CODEPOINT, 0x1234),
    ]

    for codepoint, expected_err, sentinel in cases:
        out = [sentinel]
        err = tokenizer_utf8_validate_codepoint_scalar_checked(codepoint, out)
        assert err == expected_err
        if err == TOKENIZER_UTF8_OK:
            assert out[0] == codepoint
        else:
            assert out[0] == sentinel


def test_exhaustive_core_planes() -> None:
    # Fully sweep BMP and first supplementary boundary behavior.
    for codepoint in range(0x0000, 0x10000):
        out = [0xABCD]
        err = tokenizer_utf8_validate_codepoint_scalar_checked(codepoint, out)
        ref_err, ref_out = ref_validate_scalar(codepoint)
        assert err == ref_err
        if err == TOKENIZER_UTF8_OK:
            assert out[0] == ref_out
        else:
            assert out[0] == 0xABCD

    # Sweep tail window around Unicode max.
    for codepoint in range(0x10FF00, 0x110100):
        out = [0x5555]
        err = tokenizer_utf8_validate_codepoint_scalar_checked(codepoint, out)
        ref_err, ref_out = ref_validate_scalar(codepoint)
        assert err == ref_err
        if err == TOKENIZER_UTF8_OK:
            assert out[0] == ref_out
        else:
            assert out[0] == 0x5555


def test_randomized_full_u32_domain() -> None:
    rng = random.Random(20260418_343)

    for _ in range(120000):
        codepoint = rng.getrandbits(32)
        out = [0xCAFEBABE]
        err = tokenizer_utf8_validate_codepoint_scalar_checked(codepoint, out)
        ref_err, ref_out = ref_validate_scalar(codepoint)

        assert err == ref_err
        if err == TOKENIZER_UTF8_OK:
            assert out[0] == ref_out
        else:
            assert out[0] == 0xCAFEBABE


if __name__ == "__main__":
    test_null_ptr_contract()
    test_boundary_vectors()
    test_exhaustive_core_planes()
    test_randomized_full_u32_domain()
    print("tokenizer_utf8_validate_codepoint_scalar_checked_reference_checks=ok")
