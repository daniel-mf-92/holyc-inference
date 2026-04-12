#!/usr/bin/env python3
"""Reference checks for GGUF tensor-data base alignment and offset semantics."""

from __future__ import annotations

import random

GGUF_TDBASE_OK = 0
GGUF_TDBASE_ERR_NULL_PTR = 1
GGUF_TDBASE_ERR_BAD_ALIGNMENT = 2
GGUF_TDBASE_ERR_OVERFLOW = 3
GGUF_TDBASE_ERR_MISALIGNED_BASE = 4
GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET = 5

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


def tensor_data_base_offset(tensor_data_base: int, tensor_rel_offset: int, alignment: int):
    if validate_alignment(alignment) != GGUF_TDBASE_OK:
        return GGUF_TDBASE_ERR_BAD_ALIGNMENT, 0
    if not tensor_data_base_is_aligned(tensor_data_base, alignment):
        return GGUF_TDBASE_ERR_MISALIGNED_BASE, 0
    if not tensor_data_base_is_aligned(tensor_rel_offset, alignment):
        return GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET, 0
    if tensor_data_base > U64_MAX - tensor_rel_offset:
        return GGUF_TDBASE_ERR_OVERFLOW, 0
    return GGUF_TDBASE_OK, tensor_data_base + tensor_rel_offset


def tensor_data_base_offset_default(tensor_data_base: int, tensor_rel_offset: int):
    return tensor_data_base_offset(tensor_data_base, tensor_rel_offset, GGUF_DEFAULT_ALIGNMENT)


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


def test_tensor_data_base_offset_happy_path() -> None:
    err, abs_off = tensor_data_base_offset(0x2000, 0x80, 32)
    assert err == GGUF_TDBASE_OK
    assert abs_off == 0x2080



def test_tensor_data_base_offset_default_alignment() -> None:
    err, abs_off = tensor_data_base_offset_default(0x100, 0x40)
    assert err == GGUF_TDBASE_OK
    assert abs_off == 0x140



def test_tensor_data_base_offset_reject_bad_alignment() -> None:
    err, _ = tensor_data_base_offset(0x1000, 0x40, 24)
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT



def test_tensor_data_base_offset_reject_misaligned_base() -> None:
    err, _ = tensor_data_base_offset(0x1010, 0x40, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_BASE



def test_tensor_data_base_offset_reject_misaligned_rel() -> None:
    err, _ = tensor_data_base_offset(0x1000, 0x48, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET



def test_tensor_data_base_offset_overflow_guard() -> None:
    err, _ = tensor_data_base_offset(U64_MAX - 31, 32, 32)
    assert err == GGUF_TDBASE_ERR_OVERFLOW



def test_tensor_data_base_offset_random_aligned_cases() -> None:
    rng = random.Random(761113)

    for _ in range(1000):
        alignment = 1 << rng.randint(0, 15)
        base_units = rng.randrange(0, 1 << 24)
        rel_units = rng.randrange(0, 1 << 20)
        base = base_units * alignment
        rel = rel_units * alignment

        err, got = tensor_data_base_offset(base, rel, alignment)
        assert err == GGUF_TDBASE_OK
        assert got == base + rel
        assert tensor_data_base_is_aligned(got, alignment)


def run() -> None:
    test_default_alignment_examples()
    test_custom_power_of_two_alignment()
    test_reject_non_power_of_two_alignment()
    test_overflow_guard_near_u64_max()
    test_random_alignment_matches_reference_formula()
    test_tensor_data_base_offset_happy_path()
    test_tensor_data_base_offset_default_alignment()
    test_tensor_data_base_offset_reject_bad_alignment()
    test_tensor_data_base_offset_reject_misaligned_base()
    test_tensor_data_base_offset_reject_misaligned_rel()
    test_tensor_data_base_offset_overflow_guard()
    test_tensor_data_base_offset_random_aligned_cases()
    print("gguf_tensor_data_base_reference_checks=ok")


if __name__ == "__main__":
    run()
