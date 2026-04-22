#!/usr/bin/env python3
"""Harness for IQ-1115 registry-init preflight companion."""

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
    )
    if rc != SAMPLING_Q16_OK:
        return rc

    canonical_tags = [b"llama", b"mistral", b"qwen2", b"phi3"]
    canonical_lens = [5, 7, 5, 4]
    canonical_ids = [
        DISPATCH_ARCH_Q16_ID_LLAMA,
        DISPATCH_ARCH_Q16_ID_MISTRAL,
        DISPATCH_ARCH_Q16_ID_QWEN2,
        DISPATCH_ARCH_Q16_ID_PHI3,
    ]
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


def test_source_contains_iq1115_signature_and_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split("I32 InferenceForwardDispatchModelArchQ16CheckedNoPartial(", 1)[0]

    assert "InferenceForwardDispatchRegistryInitQ16CheckedNoPartial(" in body
    assert "snapshot_registry_slot_capacity != registry_slot_capacity" in body
    assert "staged_registry_count != canonical_registry_count" in body
    assert "InferenceDispatchByteSpanEquals(staged_tags[slot_index]" in body


def test_success_publishes_expected_registry() -> None:
    tags = [b"X", b"Y", b"Z", b"W"]
    lens = [99, 99, 99, 99]
    ids = [88, 88, 88, 88]
    count = [77]
    default_arch = [66]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_reference(
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


def test_capacity_underflow_and_alias_reject_without_publish() -> None:
    tags = [b"a", b"b", b"c", b"d"]
    lens = [1, 1, 1, 1]
    ids = [1, 2, 3, 4]
    count = [9]
    default_arch = [10]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_reference(
        registry_slot_capacity=3,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=default_arch,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM
    assert count == [9]
    assert default_arch == [10]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_reference(
        registry_slot_capacity=4,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=count,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM


def run() -> None:
    test_source_contains_iq1115_signature_and_contract()
    test_success_publishes_expected_registry()
    test_capacity_underflow_and_alias_reject_without_publish()
    print("inference_forward_dispatch_registry_init_q16_checked_nopartial_commit_only_preflight_only=ok")


if __name__ == "__main__":
    run()
