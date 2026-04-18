#!/usr/bin/env python3
"""Parity harness for TokenizerUTF8ExpectedLengthFromLeadByteChecked."""

from __future__ import annotations

TOKENIZER_UTF8_OK = 0
TOKENIZER_UTF8_ERR_NULL_PTR = 1
TOKENIZER_UTF8_ERR_BAD_PARAM = 2
TOKENIZER_UTF8_ERR_OVERFLOW = 3
TOKENIZER_UTF8_ERR_OUT_OF_BOUNDS = 4
TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE = 5
TOKENIZER_UTF8_ERR_BAD_CONTINUATION = 6
TOKENIZER_UTF8_ERR_BAD_CODEPOINT = 7
TOKENIZER_UTF8_ERR_TRUNCATED = 8


def tokenizer_utf8_expected_length_from_lead_byte_checked(
    lead: int,
    out_expected_length: list[int] | None,
) -> int:
    if out_expected_length is None:
        return TOKENIZER_UTF8_ERR_NULL_PTR

    if lead <= 0x7F:
        need = 1
    elif 0xC2 <= lead <= 0xDF:
        need = 2
    elif 0xE0 <= lead <= 0xEF:
        need = 3
    elif 0xF0 <= lead <= 0xF4:
        need = 4
    else:
        return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE

    out_expected_length[0] = need
    return TOKENIZER_UTF8_OK


def ref_expected_len_or_err(lead: int) -> tuple[int, int]:
    if lead <= 0x7F:
        return TOKENIZER_UTF8_OK, 1
    if 0xC2 <= lead <= 0xDF:
        return TOKENIZER_UTF8_OK, 2
    if 0xE0 <= lead <= 0xEF:
        return TOKENIZER_UTF8_OK, 3
    if 0xF0 <= lead <= 0xF4:
        return TOKENIZER_UTF8_OK, 4
    return TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, -1


def test_null_ptr_contract() -> None:
    assert (
        tokenizer_utf8_expected_length_from_lead_byte_checked(0x41, None)
        == TOKENIZER_UTF8_ERR_NULL_PTR
    )


def test_exhaustive_lead_domain() -> None:
    for lead in range(256):
        out = [0xAB]
        err = tokenizer_utf8_expected_length_from_lead_byte_checked(lead, out)
        ref_err, ref_len = ref_expected_len_or_err(lead)

        assert err == ref_err, f"lead=0x{lead:02X}"
        if err == TOKENIZER_UTF8_OK:
            assert out[0] == ref_len, f"lead=0x{lead:02X}"
            assert 1 <= out[0] <= 4
        else:
            assert out[0] == 0xAB, f"lead=0x{lead:02X}"


def test_boundary_vectors() -> None:
    cases = [
        (0x00, TOKENIZER_UTF8_OK, 1),
        (0x7F, TOKENIZER_UTF8_OK, 1),
        (0x80, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0xCC),
        (0xBF, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0xCC),
        (0xC0, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0xCC),
        (0xC1, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0xCC),
        (0xC2, TOKENIZER_UTF8_OK, 2),
        (0xDF, TOKENIZER_UTF8_OK, 2),
        (0xE0, TOKENIZER_UTF8_OK, 3),
        (0xEF, TOKENIZER_UTF8_OK, 3),
        (0xF0, TOKENIZER_UTF8_OK, 4),
        (0xF4, TOKENIZER_UTF8_OK, 4),
        (0xF5, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0xCC),
        (0xFF, TOKENIZER_UTF8_ERR_BAD_LEAD_BYTE, 0xCC),
    ]

    for lead, expected_err, expected_out in cases:
        out = [0xCC]
        err = tokenizer_utf8_expected_length_from_lead_byte_checked(lead, out)
        assert err == expected_err
        assert out[0] == expected_out


if __name__ == "__main__":
    test_null_ptr_contract()
    test_exhaustive_lead_domain()
    test_boundary_vectors()
    print("tokenizer_utf8_expected_length_from_lead_byte_checked_reference_checks=ok")
