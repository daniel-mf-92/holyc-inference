#!/usr/bin/env python3
"""Parity checks for GGUFTensorBytesForType payload sizing semantics."""

from __future__ import annotations

import random

GGUF_TDBASE_OK = 0
GGUF_TDBASE_ERR_NULL_PTR = 1
GGUF_TDBASE_ERR_OVERFLOW = 3
GGUF_TDBASE_ERR_BAD_TYPE = 6
GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE = 8

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

U64_MAX = 0xFFFFFFFFFFFFFFFF


def tensor_type_block_size(ggml_type: int) -> tuple[int, int]:
    if ggml_type == GGML_TYPE_F32:
        return GGUF_TDBASE_OK, 1
    if ggml_type == GGML_TYPE_F16:
        return GGUF_TDBASE_OK, 1
    if ggml_type == GGML_TYPE_Q4_0:
        return GGUF_TDBASE_OK, 32
    if ggml_type == GGML_TYPE_Q8_0:
        return GGUF_TDBASE_OK, 32
    return GGUF_TDBASE_ERR_BAD_TYPE, 0


def tensor_type_block_bytes(ggml_type: int) -> tuple[int, int]:
    if ggml_type == GGML_TYPE_F32:
        return GGUF_TDBASE_OK, 4
    if ggml_type == GGML_TYPE_F16:
        return GGUF_TDBASE_OK, 2
    if ggml_type == GGML_TYPE_Q4_0:
        return GGUF_TDBASE_OK, 18
    if ggml_type == GGML_TYPE_Q8_0:
        return GGUF_TDBASE_OK, 34
    return GGUF_TDBASE_ERR_BAD_TYPE, 0


def tensor_bytes_for_type(ggml_type: int, element_count: int) -> tuple[int, int]:
    err, block_size = tensor_type_block_size(ggml_type)
    if err != GGUF_TDBASE_OK:
        return err, 0

    err, block_bytes = tensor_type_block_bytes(ggml_type)
    if err != GGUF_TDBASE_OK:
        return err, 0

    if block_size == 0 or block_bytes == 0:
        return GGUF_TDBASE_ERR_BAD_TYPE, 0

    if element_count == 0:
        return GGUF_TDBASE_OK, 0

    if element_count % block_size != 0:
        return GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0

    block_count = element_count // block_size
    if block_count > (U64_MAX // block_bytes):
        return GGUF_TDBASE_ERR_OVERFLOW, 0

    return GGUF_TDBASE_OK, block_count * block_bytes


def test_known_payload_sizes() -> None:
    assert tensor_bytes_for_type(GGML_TYPE_F32, 1) == (GGUF_TDBASE_OK, 4)
    assert tensor_bytes_for_type(GGML_TYPE_F16, 7) == (GGUF_TDBASE_OK, 14)
    assert tensor_bytes_for_type(GGML_TYPE_Q4_0, 32) == (GGUF_TDBASE_OK, 18)
    assert tensor_bytes_for_type(GGML_TYPE_Q8_0, 64) == (GGUF_TDBASE_OK, 68)


def test_zero_elements_are_legal_for_all_supported_types() -> None:
    for ggml_type in (GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0):
        assert tensor_bytes_for_type(ggml_type, 0) == (GGUF_TDBASE_OK, 0)


def test_quant_block_multiples_are_enforced() -> None:
    assert tensor_bytes_for_type(GGML_TYPE_Q4_0, 31) == (GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0)
    assert tensor_bytes_for_type(GGML_TYPE_Q8_0, 63) == (GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0)


def test_overflow_boundaries_for_supported_types() -> None:
    for ggml_type, block_size, block_bytes in (
        (GGML_TYPE_F32, 1, 4),
        (GGML_TYPE_F16, 1, 2),
        (GGML_TYPE_Q4_0, 32, 18),
        (GGML_TYPE_Q8_0, 32, 34),
    ):
        max_block_count = U64_MAX // block_bytes
        max_elements = max_block_count * block_size
        err, nbytes = tensor_bytes_for_type(ggml_type, max_elements)
        assert err == GGUF_TDBASE_OK
        assert nbytes == max_block_count * block_bytes

        overflow_elements = max_elements + block_size
        err, nbytes = tensor_bytes_for_type(ggml_type, overflow_elements)
        assert err == GGUF_TDBASE_ERR_OVERFLOW
        assert nbytes == 0


def test_unknown_type_rejected() -> None:
    assert tensor_bytes_for_type(999, 64) == (GGUF_TDBASE_ERR_BAD_TYPE, 0)


def test_randomized_reference_equivalence() -> None:
    rng = random.Random(20260417)
    types = [GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0]

    for _ in range(4000):
        ggml_type = rng.choice(types)
        _, block_size = tensor_type_block_size(ggml_type)
        _, block_bytes = tensor_type_block_bytes(ggml_type)

        if rng.random() < 0.30:
            element_count = 0
        elif ggml_type in (GGML_TYPE_Q4_0, GGML_TYPE_Q8_0) and rng.random() < 0.25:
            element_count = rng.randrange(1, 10_000)
            if element_count % block_size == 0:
                element_count += 1
        else:
            max_blocks = 1_000_000
            block_count = rng.randrange(0, max_blocks + 1)
            element_count = block_count * block_size

        err, nbytes = tensor_bytes_for_type(ggml_type, element_count)

        if element_count == 0:
            assert (err, nbytes) == (GGUF_TDBASE_OK, 0)
            continue

        if element_count % block_size != 0:
            assert (err, nbytes) == (GGUF_TDBASE_ERR_BAD_BLOCK_MULTIPLE, 0)
            continue

        block_count = element_count // block_size
        if block_count > (U64_MAX // block_bytes):
            assert (err, nbytes) == (GGUF_TDBASE_ERR_OVERFLOW, 0)
            continue

        assert err == GGUF_TDBASE_OK
        assert nbytes == block_count * block_bytes


def run() -> None:
    test_known_payload_sizes()
    test_zero_elements_are_legal_for_all_supported_types()
    test_quant_block_multiples_are_enforced()
    test_overflow_boundaries_for_supported_types()
    test_unknown_type_rejected()
    test_randomized_reference_equivalence()
    print("gguf_tensor_bytes_for_type_reference_checks=ok")


if __name__ == "__main__":
    run()
