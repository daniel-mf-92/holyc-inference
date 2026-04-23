#!/usr/bin/env python3
"""Harness for IQ-1254 GPU dispatch policy gate."""

from __future__ import annotations

from pathlib import Path

GPU_POLICY_OK = 0
GPU_POLICY_ERR_NULL_PTR = 1
GPU_POLICY_ERR_BAD_PARAM = 2
GPU_POLICY_ERR_PROFILE_GUARD = 3
GPU_POLICY_ERR_IOMMU_GUARD = 4
GPU_POLICY_ERR_BOOK_GUARD = 5

GPU_POLICY_REASON_ALLOW = 0
GPU_POLICY_REASON_PROFILE_REJECT = 1
GPU_POLICY_REASON_IOMMU_DISABLED = 2
GPU_POLICY_REASON_BOOK_HOOKS_MISS = 3

GPU_POLICY_PROFILE_SECURE_LOCAL = 1
GPU_POLICY_PROFILE_DEV_LOCAL = 2


def _is_binary(value: int) -> bool:
    return value in (0, 1)


def gpu_policy_allow_dispatch_checked(
    profile_id: int,
    iommu_enabled: int,
    bot_dma_log_enabled: int,
    bot_mmio_log_enabled: int,
    bot_dispatch_log_enabled: int,
) -> tuple[int, int, int, int]:
    allow_dispatch = 0
    reason_code = GPU_POLICY_REASON_PROFILE_REJECT
    out_profile_id = 0

    if not all(
        _is_binary(flag)
        for flag in (
            iommu_enabled,
            bot_dma_log_enabled,
            bot_mmio_log_enabled,
            bot_dispatch_log_enabled,
        )
    ):
        return GPU_POLICY_ERR_BAD_PARAM, allow_dispatch, reason_code, out_profile_id

    if profile_id not in (GPU_POLICY_PROFILE_SECURE_LOCAL, GPU_POLICY_PROFILE_DEV_LOCAL):
        return GPU_POLICY_ERR_PROFILE_GUARD, allow_dispatch, reason_code, out_profile_id

    out_profile_id = profile_id

    if not iommu_enabled:
        reason_code = GPU_POLICY_REASON_IOMMU_DISABLED
        return GPU_POLICY_ERR_IOMMU_GUARD, allow_dispatch, reason_code, out_profile_id

    if not bot_dma_log_enabled or not bot_mmio_log_enabled or not bot_dispatch_log_enabled:
        reason_code = GPU_POLICY_REASON_BOOK_HOOKS_MISS
        return GPU_POLICY_ERR_BOOK_GUARD, allow_dispatch, reason_code, out_profile_id

    allow_dispatch = 1
    reason_code = GPU_POLICY_REASON_ALLOW
    return GPU_POLICY_OK, allow_dispatch, reason_code, out_profile_id


def test_source_contains_iq1254_symbols() -> None:
    src = Path("src/gpu/policy.HC").read_text(encoding="utf-8")

    assert "I32 GPUPolicyAllowDispatchChecked(" in src
    assert "GPU_POLICY_ERR_PROFILE_GUARD" in src
    assert "GPU_POLICY_ERR_IOMMU_GUARD" in src
    assert "GPU_POLICY_ERR_BOOK_GUARD" in src
    assert "GPU_POLICY_REASON_IOMMU_DISABLED" in src
    assert "GPU_POLICY_REASON_BOOK_HOOKS_MISS" in src
    assert "InferenceProfileStatusChecked" in src


def test_secure_local_allows_only_when_all_guards_are_on() -> None:
    status, allow_dispatch, reason_code, out_profile_id = gpu_policy_allow_dispatch_checked(
        profile_id=GPU_POLICY_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )

    assert status == GPU_POLICY_OK
    assert allow_dispatch == 1
    assert reason_code == GPU_POLICY_REASON_ALLOW
    assert out_profile_id == GPU_POLICY_PROFILE_SECURE_LOCAL


def test_dev_local_still_requires_iommu_and_book_hooks() -> None:
    status, allow_dispatch, reason_code, out_profile_id = gpu_policy_allow_dispatch_checked(
        profile_id=GPU_POLICY_PROFILE_DEV_LOCAL,
        iommu_enabled=0,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )

    assert status == GPU_POLICY_ERR_IOMMU_GUARD
    assert allow_dispatch == 0
    assert reason_code == GPU_POLICY_REASON_IOMMU_DISABLED
    assert out_profile_id == GPU_POLICY_PROFILE_DEV_LOCAL


def test_book_hooks_missing_blocks_dispatch() -> None:
    status, allow_dispatch, reason_code, out_profile_id = gpu_policy_allow_dispatch_checked(
        profile_id=GPU_POLICY_PROFILE_SECURE_LOCAL,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=0,
        bot_dispatch_log_enabled=1,
    )

    assert status == GPU_POLICY_ERR_BOOK_GUARD
    assert allow_dispatch == 0
    assert reason_code == GPU_POLICY_REASON_BOOK_HOOKS_MISS
    assert out_profile_id == GPU_POLICY_PROFILE_SECURE_LOCAL


def test_invalid_profile_rejected() -> None:
    status, allow_dispatch, reason_code, out_profile_id = gpu_policy_allow_dispatch_checked(
        profile_id=99,
        iommu_enabled=1,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )

    assert status == GPU_POLICY_ERR_PROFILE_GUARD
    assert allow_dispatch == 0
    assert reason_code == GPU_POLICY_REASON_PROFILE_REJECT
    assert out_profile_id == 0


def test_non_binary_guard_inputs_rejected() -> None:
    status, allow_dispatch, reason_code, out_profile_id = gpu_policy_allow_dispatch_checked(
        profile_id=GPU_POLICY_PROFILE_SECURE_LOCAL,
        iommu_enabled=2,
        bot_dma_log_enabled=1,
        bot_mmio_log_enabled=1,
        bot_dispatch_log_enabled=1,
    )

    assert status == GPU_POLICY_ERR_BAD_PARAM
    assert allow_dispatch == 0
    assert reason_code == GPU_POLICY_REASON_PROFILE_REJECT
    assert out_profile_id == 0


if __name__ == "__main__":
    test_source_contains_iq1254_symbols()
    test_secure_local_allows_only_when_all_guards_are_on()
    test_dev_local_still_requires_iommu_and_book_hooks()
    test_book_hooks_missing_blocks_dispatch()
    test_invalid_profile_rejected()
    test_non_binary_guard_inputs_rejected()
    print("ok")

