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
GGUF_TDBASE_ERR_BAD_TYPE = 6
GGUF_TDBASE_ERR_OUT_OF_BOUNDS = 7

GGUF_DEFAULT_ALIGNMENT = 32
U64_MAX = (1 << 64) - 1

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8


def tensor_type_block_size(ggml_type: int):
    if ggml_type == GGML_TYPE_F32:
        return GGUF_TDBASE_OK, 1
    if ggml_type == GGML_TYPE_F16:
        return GGUF_TDBASE_OK, 1
    if ggml_type == GGML_TYPE_Q4_0:
        return GGUF_TDBASE_OK, 32
    if ggml_type == GGML_TYPE_Q8_0:
        return GGUF_TDBASE_OK, 32
    return GGUF_TDBASE_ERR_BAD_TYPE, 0


def tensor_type_block_bytes(ggml_type: int):
    if ggml_type == GGML_TYPE_F32:
        return GGUF_TDBASE_OK, 4
    if ggml_type == GGML_TYPE_F16:
        return GGUF_TDBASE_OK, 2
    if ggml_type == GGML_TYPE_Q4_0:
        return GGUF_TDBASE_OK, 18
    if ggml_type == GGML_TYPE_Q8_0:
        return GGUF_TDBASE_OK, 34
    return GGUF_TDBASE_ERR_BAD_TYPE, 0


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


def tensor_resolve_range(
    tensor_data_base: int,
    tensor_rel_offset: int,
    tensor_nbytes: int,
    gguf_file_nbytes: int,
    alignment: int,
):
    err, abs_start = tensor_data_base_offset(tensor_data_base, tensor_rel_offset, alignment)
    if err != GGUF_TDBASE_OK:
        return err, 0, 0

    if abs_start > U64_MAX - tensor_nbytes:
        return GGUF_TDBASE_ERR_OVERFLOW, 0, 0

    abs_end = abs_start + tensor_nbytes
    if abs_end > gguf_file_nbytes:
        return GGUF_TDBASE_ERR_OUT_OF_BOUNDS, 0, 0

    return GGUF_TDBASE_OK, abs_start, abs_end


def tensor_resolve_range_default(
    tensor_data_base: int,
    tensor_rel_offset: int,
    tensor_nbytes: int,
    gguf_file_nbytes: int,
):
    return tensor_resolve_range(
        tensor_data_base,
        tensor_rel_offset,
        tensor_nbytes,
        gguf_file_nbytes,
        GGUF_DEFAULT_ALIGNMENT,
    )


def test_default_alignment_examples() -> None:
    assert tensor_data_base_align_default(0) == (GGUF_TDBASE_OK, 0)
    assert tensor_data_base_align_default(1) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(31) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(32) == (GGUF_TDBASE_OK, 32)
    assert tensor_data_base_align_default(33) == (GGUF_TDBASE_OK, 64)


def test_tensor_type_block_size_known_types() -> None:
    assert tensor_type_block_size(GGML_TYPE_F32) == (GGUF_TDBASE_OK, 1)
    assert tensor_type_block_size(GGML_TYPE_F16) == (GGUF_TDBASE_OK, 1)
    assert tensor_type_block_size(GGML_TYPE_Q4_0) == (GGUF_TDBASE_OK, 32)
    assert tensor_type_block_size(GGML_TYPE_Q8_0) == (GGUF_TDBASE_OK, 32)


def test_tensor_type_block_bytes_known_types() -> None:
    assert tensor_type_block_bytes(GGML_TYPE_F32) == (GGUF_TDBASE_OK, 4)
    assert tensor_type_block_bytes(GGML_TYPE_F16) == (GGUF_TDBASE_OK, 2)
    assert tensor_type_block_bytes(GGML_TYPE_Q4_0) == (GGUF_TDBASE_OK, 18)
    assert tensor_type_block_bytes(GGML_TYPE_Q8_0) == (GGUF_TDBASE_OK, 34)


def test_tensor_type_helpers_reject_unknown_type() -> None:
    assert tensor_type_block_size(999) == (GGUF_TDBASE_ERR_BAD_TYPE, 0)
    assert tensor_type_block_bytes(999) == (GGUF_TDBASE_ERR_BAD_TYPE, 0)


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


def test_tensor_resolve_range_happy_path() -> None:
    err, abs_start, abs_end = tensor_resolve_range(0x2000, 0x80, 0x120, 0x3000, 32)
    assert err == GGUF_TDBASE_OK
    assert abs_start == 0x2080
    assert abs_end == 0x21A0


def test_tensor_resolve_range_default_alignment() -> None:
    err, abs_start, abs_end = tensor_resolve_range_default(0x100, 0x40, 0x10, 0x1000)
    assert err == GGUF_TDBASE_OK
    assert abs_start == 0x140
    assert abs_end == 0x150


def test_tensor_resolve_range_reject_bad_alignment() -> None:
    err, _, _ = tensor_resolve_range(0x100, 0x20, 0x10, 0x1000, 24)
    assert err == GGUF_TDBASE_ERR_BAD_ALIGNMENT


def test_tensor_resolve_range_reject_misaligned_base() -> None:
    err, _, _ = tensor_resolve_range(0x110, 0x20, 0x10, 0x1000, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_BASE


def test_tensor_resolve_range_reject_misaligned_offset() -> None:
    err, _, _ = tensor_resolve_range(0x100, 0x18, 0x10, 0x1000, 32)
    assert err == GGUF_TDBASE_ERR_MISALIGNED_TENSOR_OFFSET


def test_tensor_resolve_range_overflow_guard() -> None:
    err, _, _ = tensor_resolve_range(U64_MAX - 0x1F, 0x20, 0x40, U64_MAX, 32)
    assert err == GGUF_TDBASE_ERR_OVERFLOW


def test_tensor_resolve_range_out_of_bounds() -> None:
    err, _, _ = tensor_resolve_range(0x200, 0x40, 0xC0, 0x2F0, 32)
    assert err == GGUF_TDBASE_ERR_OUT_OF_BOUNDS


def test_tensor_resolve_range_allows_exact_eof() -> None:
    err, abs_start, abs_end = tensor_resolve_range(0x400, 0x80, 0x100, 0x580, 32)
    assert err == GGUF_TDBASE_OK
    assert abs_start == 0x480
    assert abs_end == 0x580


def test_tensor_resolve_range_random_valid_cases() -> None:
    rng = random.Random(761499)

    for _ in range(1000):
        alignment = 1 << rng.randint(0, 15)
        base_units = rng.randrange(0, 1 << 20)
        rel_units = rng.randrange(0, 1 << 20)
        tensor_nbytes = rng.randrange(0, 1 << 16)

        base = base_units * alignment
        rel = rel_units * alignment
        start = base + rel
        if start > U64_MAX - tensor_nbytes:
            continue
        end = start + tensor_nbytes

        file_pad = rng.randrange(0, 1 << 12)
        file_nbytes = end + file_pad

        err, got_start, got_end = tensor_resolve_range(
            base,
            rel,
            tensor_nbytes,
            file_nbytes,
            alignment,
        )
        assert err == GGUF_TDBASE_OK
        assert got_start == start
        assert got_end == end


def run() -> None:
    test_default_alignment_examples()
    test_tensor_type_block_size_known_types()
    test_tensor_type_block_bytes_known_types()
    test_tensor_type_helpers_reject_unknown_type()
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
    test_tensor_resolve_range_happy_path()
    test_tensor_resolve_range_default_alignment()
    test_tensor_resolve_range_reject_bad_alignment()
    test_tensor_resolve_range_reject_misaligned_base()
    test_tensor_resolve_range_reject_misaligned_offset()
    test_tensor_resolve_range_overflow_guard()
    test_tensor_resolve_range_out_of_bounds()
    test_tensor_resolve_range_allows_exact_eof()
    test_tensor_resolve_range_random_valid_cases()
    print("gguf_tensor_data_base_reference_checks=ok")


if __name__ == "__main__":
    run()
