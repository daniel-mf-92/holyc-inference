#!/usr/bin/env python3
"""Harness for IQ-1129 registry-init parity commit-only wrapper."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests").resolve()))

from test_inference_forward_dispatch_registry_init_q16_checked_nopartial_commit_only_preflight_only_parity import (
    DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT,
    SAMPLING_Q16_ERR_BAD_PARAM,
    SAMPLING_Q16_ERR_NULL_PTR,
    SAMPLING_Q16_OK,
    registry_init_checked_nopartial_commit_only_preflight_only_parity_reference,
    registry_init_checked_nopartial_commit_only_preflight_only_reference,
)


def registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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

    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
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

    rc = registry_init_checked_nopartial_commit_only_preflight_only_reference(
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


def test_source_contains_iq1129_signature_and_contract() -> None:
    source = Path("src/model/inference.HC").read_text(encoding="utf-8")
    sig = "I32 InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParityCommitOnly("
    assert source.count(sig) == 1
    body = source.split(sig, 1)[1].split("I32 InferenceForwardDispatchModelArchQ16CheckedNoPartial(", 1)[0]

    assert "IQ-1129 commit-only diagnostics wrapper for registry-init parity." in source
    assert "InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnlyParity(" in body
    assert "InferenceForwardDispatchRegistryInitQ16CheckedNoPartialCommitOnlyPreflightOnly(" in body
    assert "snapshot_registry_slot_capacity != registry_slot_capacity" in body
    assert "if (snapshot_out_registry_tags != out_registry_tags ||" in body
    assert "if (staged_registry_count != canonical_registry_count ||" in body
    assert "staged_registry_count != DISPATCH_ARCH_Q16_REGISTRY_SLOT_COUNT" in body
    assert "staged_default_arch_id != DISPATCH_ARCH_Q16_ID_LLAMA" in body
    assert "*out_registry_count = staged_registry_count;" in body


def test_duplicate_unsupported_capacity_and_success_no_partial() -> None:
    tags = [b"x", b"y", b"z", b"w"]
    lens = [9, 9, 9, 9]
    ids = [8, 8, 8, 8]
    count = [7]
    default_arch = [6]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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

    for duplicate, unsupported in ((True, False), (False, True)):
        tags = [b"a", b"b", b"c", b"d"]
        lens = [1, 1, 1, 1]
        ids = [2, 2, 2, 2]
        count = [19]
        default_arch = [20]
        rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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
        assert ids == [2, 2, 2, 2]
        assert count == [19]
        assert default_arch == [20]

    tags = [b"a", b"b", b"c", b"d"]
    lens = [1, 1, 1, 1]
    ids = [2, 2, 2, 2]
    count = [19]
    default_arch = [20]
    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
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
    assert ids == [2, 2, 2, 2]
    assert count == [19]
    assert default_arch == [20]


def registry_init_commit_only_explicit_composition(
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

    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_reference(
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

    rc = registry_init_checked_nopartial_commit_only_preflight_only_reference(
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


def test_randomized_equivalence_with_explicit_composition() -> None:
    rng = random.Random(20260422_1129)

    for _ in range(500):
        cap = rng.randint(0, 6)
        dup = rng.choice([False, True])
        bad = rng.choice([False, True])

        tags_a = [b"m", b"n", b"o", b"p"]
        lens_a = [3, 3, 3, 3]
        ids_a = [7, 7, 7, 7]
        count_a = [22]
        default_a = [23]

        err_new = registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
            registry_slot_capacity=cap,
            out_registry_tags=tags_a,
            out_registry_tag_lens=lens_a,
            out_registry_arch_ids=ids_a,
            out_registry_count=count_a,
            out_default_arch_id=default_a,
            inject_duplicate_tag=dup,
            inject_unsupported_tag=bad,
        )

        tags_b = [b"m", b"n", b"o", b"p"]
        lens_b = [3, 3, 3, 3]
        ids_b = [7, 7, 7, 7]
        count_b = [22]
        default_b = [23]

        err_ref = registry_init_commit_only_explicit_composition(
            registry_slot_capacity=cap,
            out_registry_tags=tags_b,
            out_registry_tag_lens=lens_b,
            out_registry_arch_ids=ids_b,
            out_registry_count=count_b,
            out_default_arch_id=default_b,
            inject_duplicate_tag=dup,
            inject_unsupported_tag=bad,
        )

        assert err_new == err_ref
        if err_new == SAMPLING_Q16_OK:
            assert tags_a == tags_b
            assert lens_a == lens_b
            assert ids_a == ids_b
            assert count_a == count_b
            assert default_a == default_b
        else:
            assert tags_a == [b"m", b"n", b"o", b"p"]
            assert lens_a == [3, 3, 3, 3]
            assert ids_a == [7, 7, 7, 7]
            assert count_a == [22]
            assert default_a == [23]


def test_null_and_alias_guards() -> None:
    tags = [b"u", b"v", b"w", b"x"]
    lens = [4, 4, 4, 4]
    ids = [5, 5, 5, 5]
    count = [33]
    default_arch = [44]

    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        registry_slot_capacity=4,
        out_registry_tags=None,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=count,
        out_default_arch_id=default_arch,
    )
    assert rc == SAMPLING_Q16_ERR_NULL_PTR

    aliased = [99]
    rc = registry_init_checked_nopartial_commit_only_preflight_only_parity_commit_only_reference(
        registry_slot_capacity=4,
        out_registry_tags=tags,
        out_registry_tag_lens=lens,
        out_registry_arch_ids=ids,
        out_registry_count=aliased,
        out_default_arch_id=aliased,
    )
    assert rc == SAMPLING_Q16_ERR_BAD_PARAM


if __name__ == "__main__":
    test_source_contains_iq1129_signature_and_contract()
    test_duplicate_unsupported_capacity_and_success_no_partial()
    test_randomized_equivalence_with_explicit_composition()
    test_null_and_alias_guards()
    print(
        "inference_forward_dispatch_registry_init_q16_checked_nopartial_commit_only_preflight_only_parity_commit_only=ok"
    )
