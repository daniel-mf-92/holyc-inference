#!/usr/bin/env python3
"""Reference checks for GGUFMetadataArrayFixedPayloadBytesChecked semantics."""

from __future__ import annotations

import random

GGUF_META_TABLE_OK = 0
GGUF_META_TABLE_ERR_NULL_PTR = 1
GGUF_META_TABLE_ERR_OVERFLOW = 3

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
    if out_payload_bytes_ref is None:
        return GGUF_META_TABLE_ERR_NULL_PTR

    payload_bytes = [0]
    err = gguf_metadata_mul_u64_checked(elem_count, elem_width_bytes, payload_bytes)
    if err != GGUF_META_TABLE_OK:
        return err

    out_payload_bytes_ref[0] = payload_bytes[0]
    return GGUF_META_TABLE_OK


def test_null_out_and_no_partial_write() -> None:
    assert (
        gguf_metadata_array_fixed_payload_bytes_checked(1, 1, None)
        == GGUF_META_TABLE_ERR_NULL_PTR
    )


def test_zero_lanes_and_basic_products() -> None:
    out = [999]

    err = gguf_metadata_array_fixed_payload_bytes_checked(0, 8, out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == 0

    err = gguf_metadata_array_fixed_payload_bytes_checked(123, 0, out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == 0

    err = gguf_metadata_array_fixed_payload_bytes_checked(32, 2, out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == 64

    err = gguf_metadata_array_fixed_payload_bytes_checked(4096, 8, out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == 32768


def test_i64_domain_overflow_guards_no_commit() -> None:
    out = [123456]

    err = gguf_metadata_array_fixed_payload_bytes_checked(I64_MAX + 1, 1, out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out[0] == 123456

    err = gguf_metadata_array_fixed_payload_bytes_checked(1, I64_MAX + 1, out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out[0] == 123456


def test_product_overflow_guards_no_commit() -> None:
    out = [777]

    err = gguf_metadata_array_fixed_payload_bytes_checked(I64_MAX, 2, out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out[0] == 777

    err = gguf_metadata_array_fixed_payload_bytes_checked((1 << 62), 4, out)
    assert err == GGUF_META_TABLE_ERR_OVERFLOW
    assert out[0] == 777


def test_max_safe_pairs() -> None:
    out = [0]

    err = gguf_metadata_array_fixed_payload_bytes_checked(I64_MAX, 1, out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == I64_MAX

    err = gguf_metadata_array_fixed_payload_bytes_checked(I64_MAX // 8, 8, out)
    assert err == GGUF_META_TABLE_OK
    assert out[0] == (I64_MAX // 8) * 8


def test_randomized_safe_and_overflow_cases() -> None:
    rng = random.Random(0xA11CE)
    out = [0xDEADBEEF]

    for _ in range(200):
        lhs = rng.randrange(0, I64_MAX + 1)
        rhs = rng.choice((1, 2, 4, 8, 16, 31, 64, 127, 256, 1024))

        out[0] = 0xDEADBEEF
        err = gguf_metadata_array_fixed_payload_bytes_checked(lhs, rhs, out)

        product = lhs * rhs
        if product <= I64_MAX:
            assert err == GGUF_META_TABLE_OK
            assert out[0] == product
        else:
            assert err == GGUF_META_TABLE_ERR_OVERFLOW
            assert out[0] == 0xDEADBEEF


if __name__ == "__main__":
    test_null_out_and_no_partial_write()
    test_zero_lanes_and_basic_products()
    test_i64_domain_overflow_guards_no_commit()
    test_product_overflow_guards_no_commit()
    test_max_safe_pairs()
    test_randomized_safe_and_overflow_cases()
    print("ok")
