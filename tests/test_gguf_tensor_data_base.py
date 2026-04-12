#!/usr/bin/env python3
"""Reference checks for GGUF tensor-data base alignment semantics."""

from __future__ import annotations

import random

GGUF_TDBASE_OK = 0
GGUF_TDBASE_ERR_NULL_PTR = 1
GGUF_TDBASE_ERR_BAD_ALIGNMENT = 2
GGUF_TDBASE_ERR_OVERFLOW = 3

GGUF_DEFAULT_ALIGNMENT = 32
U64_MAX = (1 << 64) - 1


def is_pow2_u32(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def validate_alignment(alignment: int) -> int:
    if not is_pow2_u32(alignment):
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT
    return GGUF_TDBASE_OK


def align_up_u64(value: int, alignment: int):
    err = validate_alignment(alignment)
    if err != GGUF_TDBASE_OK:
        return err, 0

    mask = alignment - 1
    if value > U64_MAX - mask:
        return GGUF_TDBASE_ERR_OVERFLOW, 0

    out = (value + mask) & ~mask
    return GGUF_TDBASE_OK, out


def tensor_data_base_align(cursor_after_tensor_info: int, alignment: int):
    return align_up_u64(cursor_after_tensor_info, alignment)


def tensor_data_base_align_default(cursor_after_tensor_info: int):
    return tensor_data_base_align(cursor_after_tensor_info, GGUF_DEFAULT_ALIGNMENT)


def tensor_data_base_is_aligned(offset: int, alignment: int) -> bool:
    if validate_alignment(alignment) != GGUF_TDBASE_OK:
        return False
    return (offset & (alignment - 1)) == 0


def test_default_alignment_examples() -> None:
    assert tensor_data_base_align_default(0) == (GGUF_TDBASE_OK, 0)
    assert tensor_data_base_align_default(1) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(31) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(32) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(33) == (GGUF_TDBASE_OK, 64)


def test_custom_power_of_two_alignment() -> None:
    err, base = tensor_data_base_align(0x1234, 64)
    assert err == GGUF_TDBASE_OK
    assert base == 0x1240
    assert tensor_data_base_is_aligned(base, 64)


def test_reject_non_power_of_two_alignment() -> None:
    assert tensor_data_base_align(100, 0)[0] == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert tensor_data_base_align(100, 3)[0] == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert tensor_data_base_align(100, 24)[0] == GGUF_TDBASE_ERR_BAD_ALIGNMENT
    assert tensor_data_base_is_aligned(128, 24) is False


def test_overflow_guard_near_u64_max() -> None:
    err, _ = tensor_data_base_align(U64_MAX - 7, 16)
    assert err == GGUF_TDBASE_ERR_OVERFLOW

    err, base = tensor_data_base_align(U64_MAX - 15, 16)
    assert err == GGUF_TDBASE_OK
    assert base == U64_MAX - 15


def test_random_alignment_matches_reference_formula() -> None:
    rng = random.Random(760031)

    for _ in range(1000):
        alignment = 1 << rng.randint(0, 20)
        cursor = rng.randrange(0, 1 << 56)

        err, got = tensor_data_base_align(cursor, alignment)
        assert err == GGUF_TDBASE_OK

        reference = ((cursor + (alignment - 1)) // alignment) * alignment
        assert got == reference
        assert got >= cursor
        assert tensor_data_base_is_aligned(got, alignment)


def run() -> None:
    test_default_alignment_examples()
    test_custom_power_of_two_alignment()
    test_reject_non_power_of_two_alignment()
    test_overflow_guard_near_u64_max()
    test_random_alignment_matches_reference_formula()
    print("gguf_tensor_data_base_reference_checks=ok")


if __name__ == "__main__":
    run()

