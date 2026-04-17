#!/usr/bin/env python3
"""Reference checks for GGUFMetadataArrayFixedPayloadBytesChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_BAD_PARAM = 2
GGUF_META_TABLE_ERR_OVERFLOW = 3
GGUF_META_TABLE_ERR_OUT_OF_BOUNDS = 4

I64_MAX = (1 << 63) - 1
U64_MAX = (1 << 64) - 1


def gguf_metadata_mul_u64_checked(
    lhs: int,
    rhs: int,
    out_product_ref: list[int] | None,
) -> int:
    if out_product_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    if lhs > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW
    if rhs > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    if rhs != 0 and lhs > (U64_MAX // rhs):
        return GGUF_META_TABLE_ERR_OVERFLOW

    product = lhs * rhs
    if product > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW

    out_product_ref[0] = product
    return GGUF_META_TABLE_OK


def gguf_metadata_array_fixed_payload_bytes_checked(
    elem_count: int,
    elem_width_bytes: int,
    out_payload_bytes_ref: list[int] | None,
) -> int:
    return gguf_metadata_mul_u64_checked(
        elem_count,
        elem_width_bytes,
        out_payload_bytes_ref,
    )


def expected_mul_status(lhs: int, rhs: int) -> tuple[int, int | None]:
    if lhs > I64_MAX or rhs > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None
    if rhs != 0 and lhs > (U64_MAX // rhs):
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    product = lhs * rhs
    if product > I64_MAX:
        return GGUF_META_TABLE_ERR_OVERFLOW, None

    return GGUF_META_TABLE_OK, product


def test_null_out_ptr() -> None:
    assert (
        gguf_metadata_array_fixed_payload_bytes_checked(
            7,
            4,
            None,
        )
        == GGUF_META_TABLE_ERR_NULL_PTR
    )


def test_zero_factor_cases() -> None:
    out_payload = [999]

    for elem_count, elem_width in ((0, 0), (0, 4), (99, 0), (0, I64_MAX)):
        out_payload[0] = 999
        err = gguf_metadata_array_fixed_payload_bytes_checked(
            elem_count,
            elem_width,
            out_payload,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_payload[0] == 0


def test_fixed_width_payload_products() -> None:
    out_payload = [111]
    fixed_widths = (1, 2, 4, 8)
    counts = (1, 3, 31, 128, 4096, 65535)

    for elem_width in fixed_widths:
        for elem_count in counts:
            out_payload[0] = 111
            err = gguf_metadata_array_fixed_payload_bytes_checked(
                elem_count,
                elem_width,
                out_payload,
            )
            assert err == GGUF_META_TABLE_OK
            assert out_payload[0] == elem_count * elem_width


def test_i64_boundary_products() -> None:
    out_payload = [222]

    safe_cases = [
        (I64_MAX, 1, I64_MAX),
        (I64_MAX // 2, 2, (I64_MAX // 2) * 2),
        (I64_MAX // 4, 4, (I64_MAX // 4) * 4),
        (I64_MAX // 8, 8, (I64_MAX // 8) * 8),
    ]

    for elem_count, elem_width, expected_payload in safe_cases:
        out_payload[0] = 222
        err = gguf_metadata_array_fixed_payload_bytes_checked(
            elem_count,
            elem_width,
            out_payload,
        )
        assert err == GGUF_META_TABLE_OK
        assert out_payload[0] == expected_payload


def test_overflow_cases_no_partial_write() -> None:
    out_payload = [777]

    overflow_cases = [
        (I64_MAX + 1, 1),
        (1, I64_MAX + 1),
        (I64_MAX, 2),
        ((I64_MAX // 2) + 1, 2),
        ((U64_MAX // 3) + 1, 3),
        ((U64_MAX // 8) + 1, 8),
    ]

    for elem_count, elem_width in overflow_cases:
        out_payload[0] = 777
        err = gguf_metadata_array_fixed_payload_bytes_checked(
            elem_count,
            elem_width,
            out_payload,
        )
        assert err == GGUF_META_TABLE_ERR_OVERFLOW
        assert out_payload[0] == 777


def test_randomized_oracle_parity() -> None:
    random.seed(0xB16B00B5)
    out_payload = [333]

    for _ in range(1000):
        lhs = random.randrange(0, U64_MAX + 1)
        rhs = random.randrange(0, U64_MAX + 1)
        out_payload[0] = 333

        expected_err, expected_product = expected_mul_status(lhs, rhs)
        err = gguf_metadata_array_fixed_payload_bytes_checked(lhs, rhs, out_payload)
        assert err == expected_err

        if expected_err == GGUF_META_TABLE_OK:
            assert out_payload[0] == expected_product
        else:
            assert out_payload[0] == 333


if __name__ == "__main__":
    test_null_out_ptr()
    test_zero_factor_cases()
    test_fixed_width_payload_products()
    test_i64_boundary_products()
    test_overflow_cases_no_partial_write()
    test_randomized_oracle_parity()
    print("ok")
