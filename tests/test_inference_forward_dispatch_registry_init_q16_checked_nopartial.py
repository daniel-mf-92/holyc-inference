#!/usr/bin/env python3
"""Parity harness for IQ-1098 dispatch registry initialization wrapper."""

from __future__ import annotations

from pathlib import Path
import random

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2

DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT = 4

DISPATCH_ARCH_Q16_ID_LLAMA = 1
DISPATCH_ARCH_Q16_ID_MISTRAL = 2
DISPATCH_ARCH_Q16_ID_QWEN2 = 3
DISPATCH_ARCH_Q16_ID_PHI3 = 4


EXPECTED_TAGS = [b"llama", b"mistral", b"qwen2", b"phi3"]
EXPECTED_LENS = [5, 7, 5, 4]
EXPECTED_IDS = [
    DISPATCH_ARCH_Q16_ID_LLAMA,
    DISPATCH_ARCH_Q16_ID_MISTRAL,
    DISPATCH_ARCH_Q16_ID_QWEN2,
    DISPATCH_ARCH_Q16_ID_PHI3,
]


def _parse_architecture_id(tag: bytes) -> tuple[int, int]:
    if tag == b"llama":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_LLAMA
    if tag == b"mistral":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_MISTRAL
    if tag in {b"qwen2", b"qwen"}:
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_QWEN2
    if tag == b"phi3":
        return SAMPLING_Q16_OK, DISPATCH_ARCH_Q16_ID_PHI3
    return SAMPLING_Q16_ERR_BAD_PARAM, 0


def registry_init_checked_nopartial_model(
    *,
    registry_slot_capacity: int,
    out_registry_tags: list[bytes],
    out_registry_tag_lens: list[int],
    out_registry_arch_ids: list[int],
    out_registry_count: list[int],
    out_default_arch_id: list[int],
    inject_duplicate_tag: bool = False,
    inject_unsupported_tag: bool = False,
) -> int:
    if registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snap_tags = out_registry_tags[:]
    snap_lens = out_registry_tag_lens[:]
    snap_ids = out_registry_arch_ids[:]
    snap_count = out_registry_count[0]
    snap_default = out_default_arch_id[0]

    staged_tags = EXPECTED_TAGS.copy()
    staged_lens = EXPECTED_LENS.copy()
    staged_ids = EXPECTED_IDS.copy()

    if inject_duplicate_tag:
        staged_tags[3] = staged_tags[0]
        staged_lens[3] = staged_lens[0]
    if inject_unsupported_tag:
        staged_tags[2] = b"gemma"
        staged_lens[2] = 5

    for index in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        status, parsed = _parse_architecture_id(staged_tags[index])
        if status != SAMPLING_Q16_OK:
            assert out_registry_tags == snap_tags
            assert out_registry_tag_lens == snap_lens
            assert out_registry_arch_ids == snap_ids
            assert out_registry_count[0] == snap_count
            assert out_default_arch_id[0] == snap_default
            return status
        if parsed != staged_ids[index]:
            return SAMPLING_Q16_ERR_BAD_PARAM

    for i in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        for j in range(i + 1, DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
            if staged_tags[i] == staged_tags[j]:
                assert out_registry_tags == snap_tags
                assert out_registry_tag_lens == snap_lens
                assert out_registry_arch_ids == snap_ids
                assert out_registry_count[0] == snap_count
                assert out_default_arch_id[0] == snap_default
                return SAMPLING_Q16_ERR_BAD_PARAM
            if staged_ids[i] == staged_ids[j]:
                return SAMPLING_Q16_ERR_BAD_PARAM

    for index in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        out_registry_tags[index] = staged_tags[index]
        out_registry_tag_lens[index] = staged_lens[index]
        out_registry_arch_ids[index] = staged_ids[index]

    out_registry_count[0] = DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT
    out_default_arch_id[0] = DISPATCH_ARCH_Q16_ID_LLAMA
    return SAMPLING_Q16_OK


def _extract_function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    index = brace + 1
    while depth:
        ch = source[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        index += 1
    return source[brace + 1 : index - 1]


def test_source_contains_registry_init_function_and_canonical_slots() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    signature = "I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartial("
    assert signature in source

    body = _extract_function_body(source, signature)
    assert "DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT" in source
    assert 'staged_tags[0] = (U8 *)"llama";' in body
    assert 'staged_tags[1] = (U8 *)"mistral";' in body
    assert 'staged_tags[2] = (U8 *)"qwen2";' in body
    assert 'staged_tags[3] = (U8 *)"phi3";' in body
    assert "InferenceDispatchParseArchitectureIdChecked(" in body
    assert "*out_registry_count = staged_registry_count;" in body
    assert "*out_default_arch_id = staged_default_arch_id;" in body


def test_registry_capacity_guard_and_no_partial_outputs() -> None:
    tags = [b"x"] * 4
    lens = [-1] * 4
    ids = [-1] * 4
    count = [123]
    default_arch = [456]

    status = registry_init_checked_nopartial_model(
        registry_slot_capacity=3,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=default_arch,
    )
    assert status == SAMPLING_Q16_ERR_BAD_PARAM
    assert tags == [b"x"] * 4
    assert lens == [-1] * 4
    assert ids == [-1] * 4
    assert count == [123]
    assert default_arch == [456]


def test_duplicate_and_unsupported_tags_rejected_without_publish() -> None:
    for duplicate, unsupported in [(True, False), (False, True)]:
        tags = [b"hold"] * 4
        lens = [99] * 4
        ids = [77] * 4
        count = [111]
        default_arch = [222]

        status = registry_init_checked_nopartial_model(
            registry_slot_capacity=4,
            out_registry_tags=tags,
            out_registry_tag_lens=lens,
            out_registry_arch_ids=ids,
            out_registry_count=count,
            out_default_arch_id=default_arch,
            inject_duplicate_tag=duplicate,
            inject_unsupported_tag=unsupported,
        )
        assert status == SAMPLING_Q16_ERR_BAD_PARAM
        assert tags == [b"hold"] * 4
        assert lens == [99] * 4
        assert ids == [77] * 4
        assert count == [111]
        assert default_arch == [222]


def test_success_publishes_deterministic_slot_order() -> None:
    tags = [b""] * 4
    lens = [0] * 4
    ids = [0] * 4
    count = [-1]
    default_arch = [-1]

    status = registry_init_checked_nopartial_model(
        registry_slot_capacity=4,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=default_arch,
    )
    assert status == SAMPLING_Q16_OK
    assert tags == EXPECTED_TAGS
    assert lens == EXPECTED_LENS
    assert ids == EXPECTED_IDS
    assert count == [DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT]
    assert default_arch == [DISPATCH_ARCH_Q16_ID_LLAMA]


def test_deterministic_vectors_keep_same_registry_layout() -> None:
    rng = random.Random(1098)
    for _ in range(120):
        tags = [bytes([rng.randint(0, 255)]) for _ in range(4)]
        lens = [rng.randint(-10, 10) for _ in range(4)]
        ids = [rng.randint(-10, 10) for _ in range(4)]
        count = [rng.randint(-1000, 1000)]
        default_arch = [rng.randint(-1000, 1000)]

        status = registry_init_checked_nopartial_model(
            registry_slot_capacity=4,
            out_registry_tags=tags,
            out_registry_tag_lens=lens,
            out_registry_arch_ids=ids,
            out_registry_count=count,
            out_default_arch_id=default_arch,
        )
        assert status == SAMPLING_Q16_OK
        assert tags == EXPECTED_TAGS
        assert lens == EXPECTED_LENS
        assert ids == EXPECTED_IDS
        assert count == [DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT]
        assert default_arch == [DISPATCH_ARCH_Q16_ID_LLAMA]


if __name__ == "__main__":
    test_source_contains_registry_init_function_and_canonical_slots()
    test_registry_capacity_guard_and_no_partial_outputs()
    test_duplicate_and_unsupported_tags_rejected_without_publish()
    test_success_publishes_deterministic_slot_order()
    test_deterministic_vectors_keep_same_registry_layout()
    print("ok")
