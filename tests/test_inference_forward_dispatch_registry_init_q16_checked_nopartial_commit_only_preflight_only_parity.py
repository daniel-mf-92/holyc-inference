#!/usr/bin/env python3
"""Harness for IQ-1130 registry-init preflight parity gate."""

from __future__ import annotations

from pathlib import Path

SAMPLING_Q16_OK = 0
SAMPLING_Q16_ERR_NULL_PTR = 1
SAMPLING_Q16_ERR_BAD_PARAM = 2

DISPATCH_ARCH_Q16_ID_LLAMA = 1
DISPATCH_ARCH_Q16_ID_MISTRAL = 2
DISPATCH_ARCH_Q16_ID_QWEN2 = 3
DISPATCH_ARCH_Q16_ID_PHI3 = 4
DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT = 4


def registry_init_checked_nopartial_reference(
    *,
    registry_slot_capacity: int,
    out_registry_tags: list[bytes] | None,
    out_registry_tag_lens: list[int] | None,
    out_registry_arch_ids: list[int] | None,
    out_registry_count: list[int] | None,
    out_default_arch_id: list[int] | None,
    inject_duplicate_tag: bool = False,
    inject_unsupported_tag: bool = False,
) -> int:
    if (
        out_registry_tags is None
        or out_registry_tag_lens is None
        or out_registry_arch_ids is None
        or out_registry_count is None
        or out_default_arch_id is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR
    if registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if out_registry_count is out_default_arch_id:
        return SAMPLING_Q16_ERR_BAD_PARAM

    tags = [b"llama", b"mistral", b"qwen2", b"phi3"]
    lens = [5, 7, 5, 4]
    ids = [
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    ]

    if inject_duplicate_tag:
        tags[3] = tags[0]
        lens[3] = lens[0]
    if inject_unsupported_tag:
        tags[2] = b"gemma"
        lens[2] = 5

    if len(set(tags)) != len(tags):
        return SAMPLING_Q16_ERR_BAD_PARAM

    accepted = {
        b"llama": DISPATCH_ARCH_Q16_ID_LLAMA,
        b"mistral": DISPATCH_ARCH_Q16_ID_MISTRAL,
        b"qwen2": DISPATCH_ARCH_Q16_ID_QWEN2,
        b"phi3": DISPATCH_ARCH_Q16_ID_PHI3,
    }
    for idx in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        parsed = accepted.get(tags[idx])
        if parsed is None or parsed != ids[idx]:
            return SAMPLING_Q16_ERR_BAD_PARAM

    for idx in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        out_registry_tags[idx] = tags[idx]
        out_registry_tag_lens[idx] = lens[idx]
        out_registry_arch_ids[idx] = ids[idx]

    out_registry_count[0] = DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT
    out_default_arch_id[0] = DISPATCH_ARCH_Q16_ID_LLAMA
    return SAMPLING_Q16_OK


def registry_init_checked_nopartial_commit_only_preflight_only_reference(
    *,
    registry_slot_capacity: int,
    out_registry_tags: list[bytes] | None,
    out_registry_tag_lens: list[int] | None,
    out_registry_arch_ids: list[int] | None,
    out_registry_count: list[int] | None,
    out_default_arch_id: list[int] | None,
    inject_duplicate_tag: bool = False,
    inject_unsupported_tag: bool = False,
) -> int:
    if (
        out_registry_tags is None
        or out_registry_tag_lens is None
        or out_registry_arch_ids is None
        or out_registry_count is None
        or out_default_arch_id is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR
    if registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if out_registry_count is out_default_arch_id:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snap_capacity = registry_slot_capacity
    snap_tags = out_registry_tags
    snap_lens = out_registry_tag_lens
    snap_ids = out_registry_arch_ids
    snap_count = out_registry_count
    snap_default = out_default_arch_id

    staged_tags = [b"", b"", b"", b""]
    staged_lens = [0, 0, 0, 0]
    staged_ids = [0, 0, 0, 0]
    staged_count = [0]
    staged_default = [0]

    rc = registry_init_checked_nopartial_reference(
        registry_slot_capacity=registry_slot_capacity,
        out_registry_tags=staged_tags,
        out_registry_tag_lens=staged_lens,
        out_registry_arch_ids=staged_ids,
        out_registry_count=staged_count,
        out_default_arch_id=staged_default,
        inject_duplicate_tag=inject_duplicate_tag,
        inject_unsupported_tag=inject_unsupported_tag,
    )
    if rc != SAMPLING_Q16_OK:
        return rc

    canonical_count = DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT
    canonical_default = DISPATCH_ARCH_Q16_ID_LLAMA

    if snap_capacity != registry_slot_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if snap_capacity < canonical_count:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if (
        snap_tags is not out_registry_tags
        or snap_lens is not out_registry_tag_lens
        or snap_ids is not out_registry_arch_ids
        or snap_count is not out_registry_count
        or snap_default is not out_default_arch_id
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_count[0] != canonical_count or staged_default[0] != canonical_default:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for idx in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        out_registry_tags[idx] = staged_tags[idx]
        out_registry_tag_lens[idx] = staged_lens[idx]
        out_registry_arch_ids[idx] = staged_ids[idx]

    out_registry_count[0] = staged_count[0]
    out_default_arch_id[0] = staged_default[0]
    return SAMPLING_Q16_OK


def registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
    *,
    registry_slot_capacity: int,
    out_registry_tags: list[bytes] | None,
    out_registry_tag_lens: list[int] | None,
    out_registry_arch_ids: list[int] | None,
    out_registry_count: list[int] | None,
    out_default_arch_id: list[int] | None,
    inject_duplicate_tag: bool = False,
    inject_unsupported_tag: bool = False,
) -> int:
    if (
        out_registry_tags is None
        or out_registry_tag_lens is None
        or out_registry_arch_ids is None
        or out_registry_count is None
        or out_default_arch_id is None
    ):
        return SAMPLING_Q16_ERR_NULL_PTR
    if registry_slot_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if out_registry_count is out_default_arch_id:
        return SAMPLING_Q16_ERR_BAD_PARAM

    snap_capacity = registry_slot_capacity
    snap_tags = out_registry_tags
    snap_lens = out_registry_tag_lens
    snap_ids = out_registry_arch_ids
    snap_count = out_registry_count
    snap_default = out_default_arch_id

    staged_tags = [b"", b"", b"", b""]
    staged_lens = [0, 0, 0, 0]
    staged_ids = [0, 0, 0, 0]
    staged_count = [0]
    staged_default = [0]

    canonical_tags = [b"", b"", b"", b""]
    canonical_lens = [0, 0, 0, 0]
    canonical_ids = [0, 0, 0, 0]
    canonical_count = [0]
    canonical_default = [0]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_reference(
        registry_slot_capacity=registry_slot_capacity,
        out_registry_tags=staged_tags,
        out_registry_tag_lens=staged_lens,
        out_registry_arch_ids=staged_ids,
        out_registry_count=staged_count,
        out_default_arch_id=staged_default,
        inject_duplicate_tag=inject_duplicate_tag,
        inject_unsupported_tag=inject_unsupported_tag,
    )
    if rc != SAMPLING_Q16_OK:
        return rc

    rc = registry_init_checked_nopartial_reference(
        registry_slot_capacity=registry_slot_capacity,
        out_registry_tags=canonical_tags,
        out_registry_tag_lens=canonical_lens,
        out_registry_arch_ids=canonical_ids,
        out_registry_count=canonical_count,
        out_default_arch_id=canonical_default,
        inject_duplicate_tag=inject_duplicate_tag,
        inject_unsupported_tag=inject_unsupported_tag,
    )
    if rc != SAMPLING_Q16_OK:
        return rc

    if snap_capacity != registry_slot_capacity:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if snap_capacity < DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT:
        return SAMPLING_Q16_ERR_BAD_PARAM
    if (
        snap_tags is not out_registry_tags
        or snap_lens is not out_registry_tag_lens
        or snap_ids is not out_registry_arch_ids
        or snap_count is not out_registry_count
        or snap_default is not out_default_arch_id
    ):
        return SAMPLING_Q16_ERR_BAD_PARAM

    if staged_count[0] != canonical_count[0] or staged_default[0] != canonical_default[0]:
        return SAMPLING_Q16_ERR_BAD_PARAM

    for idx in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        if staged_lens[idx] <= 0 or canonical_lens[idx] <= 0:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if staged_lens[idx] != canonical_lens[idx]:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if staged_ids[idx] != canonical_ids[idx]:
            return SAMPLING_Q16_ERR_BAD_PARAM
        if staged_tags[idx] != canonical_tags[idx]:
            return SAMPLING_Q16_ERR_BAD_PARAM

    for idx in range(DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT):
        out_registry_tags[idx] = staged_tags[idx]
        out_registry_tag_lens[idx] = staged_lens[idx]
        out_registry_arch_ids[idx] = staged_ids[idx]

    out_registry_count[0] = staged_count[0]
    out_default_arch_id[0] = staged_default[0]
    return SAMPLING_Q16_OK


def test_source_contains_iq1130_signature_and_parity_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParity("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split("I32 InferenceForwardDispatchModelArchQ16CheckedNoPartial(", 1)[0]

    assert "InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "InferenceForwardDispatchRegistryInitQ16CheckedNoPartial(" in body
    assert "snapshot_registry_slot_capacity != registry_slot_capacity" in body
    assert "staged_registry_count != canonical_registry_count" in body
    assert "staged_default_arch_id != canonical_default_arch_id" in body
    assert "InferenceDispatchParseArchitectureIdChecked(" in body
    assert "staged_default_matches != 1 || canonical_default_matches != 1" in body
    assert "staged_arch_ids[slot_index] == staged_arch_ids[compare_index]" in body
    assert "canonical_arch_ids[slot_index] == canonical_arch_ids[compare_index]" in body


def test_success_and_error_vectors_match_contract() -> None:
    tags = [b"X", b"Y", b"Z", b"W"]
    lens = [99, 99, 99, 99]
    ids = [88, 88, 88, 88]
    count = [77]
    default_arch = [66]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
        registry_slot_capacity=4,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=default_arch,
    )
    assert rc == SAMPLING_Q16_OK
    assert tags == [b"llama", b"mistral", b"qwen2", b"phi3"]
    assert lens == [5, 7, 5, 4]
    assert ids == [1, 2, 3, 4]
    assert count == [4]
    assert default_arch == [1]

    tags = [b"a", b"b", b"c", b"d"]
    lens = [1, 1, 1, 1]
    ids = [9, 9, 9, 9]
    count = [12]
    default_arch = [34]
    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
        registry_slot_capacity=3,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=default_arch,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM
    assert tags == [b"a", b"b", b"c", b"d"]
    assert lens == [1, 1, 1, 1]
    assert ids == [9, 9, 9, 9]
    assert count == [12]
    assert default_arch == [34]

    for duplicate, unsupported in ((True, False), (False, True)):
        tags = [b"a", b"b", b"c", b"d"]
        lens = [1, 1, 1, 1]
        ids = [9, 9, 9, 9]
        count = [12]
        default_arch = [34]
        rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
            registry_slot_capacity=4,
            out_registry_tags=tags,
            out_registry_tag_lens=lens,
            out_registry_arch_ids=ids,
            out_registry_count=count,
            out_default_arch_id=default_arch,
            inject_duplicate_tag=duplicate,
            inject_unsupported_tag=unsupported,
        )
        assert rc == SAMPLING_Q16_ERR_BAD_PARAM
        assert tags == [b"a", b"b", b"c", b"d"]
        assert lens == [1, 1, 1, 1]
        assert ids == [9, 9, 9, 9]
        assert count == [12]
        assert default_arch == [34]


def test_pointer_alias_rejected_with_no_partial_writes() -> None:
    tags = [b"a", b"b", b"c", b"d"]
    lens = [1, 1, 1, 1]
    ids = [9, 9, 9, 9]
    shared = [55]
    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
        registry_slot_capacity=4,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=shared,
        out_default_arch_id=shared,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM
    assert tags == [b"a", b"b", b"c", b"d"]
    assert lens == [1, 1, 1, 1]
    assert ids == [9, 9, 9, 9]
    assert shared == [55]


def run() -> None:
    test_source_contains_iq1130_signature_and_parity_contract()
    test_success_and_error_vectors_match_contract()
    print("inference_forward_dispatch_registry_init_q16_checked_nopartial_commit_only_preflight_only_parity=ok")


if __name__ == "__main__":
    run()
